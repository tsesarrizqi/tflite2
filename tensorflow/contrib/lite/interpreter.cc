/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/contrib/lite/interpreter.h"
#include <cassert>
#include <cstdarg>
#include <cstdint>
#include <cstring>
#include "tensorflow/contrib/lite/arena_planner.h"
#include "tensorflow/contrib/lite/context.h"
#include "tensorflow/contrib/lite/error_reporter.h"
#include "tensorflow/contrib/lite/graph_info.h"
#include "tensorflow/contrib/lite/kernels/gemm_support.h"
#include "tensorflow/contrib/lite/memory_planner.h"
#include "tensorflow/contrib/lite/nnapi_delegate.h"
//note: android opencl
#include "CL/cl.h"

//note: vulkan
#include "vulkan/vulkan.h"
// #include "vulkan/vk_platform.h"

//note: shaderc
// #include "shaderc/shaderc.hpp"

namespace {

// std::vector preallocation tuning.
constexpr const int kSlotsToReserve = 128;

}  // namespace

namespace tflite {

// A trivial implementation of GraphInfo around the Interpreter.
class InterpreterInfo : public GraphInfo {
 public:
  explicit InterpreterInfo(Interpreter* interpreter)
      : interpreter_(interpreter) {}

  size_t num_tensors() const override { return interpreter_->tensors_size(); }
  TfLiteTensor* tensor(size_t index) override {
    return interpreter_->tensor(index);
  }
  size_t num_nodes() const override { return interpreter_->nodes_size(); }
  const TfLiteNode& node(size_t index) const override {
    return interpreter_->node_and_registration(index)->first;
  }
  const std::vector<int>& inputs() const override {
    return interpreter_->inputs();
  }
  const std::vector<int>& outputs() const override {
    return interpreter_->outputs();
  }

 public:
  Interpreter* interpreter_;
};

Interpreter::Interpreter(ErrorReporter* error_reporter)
    : error_reporter_(error_reporter ? error_reporter
                                     : DefaultErrorReporter()) {
  context_.impl_ = static_cast<void*>(this);
  context_.ResizeTensor = ResizeTensor;
  context_.ReportError = ReportError;
  context_.AddTensors = AddTensors;
  context_.tensors = nullptr;
  context_.tensors_size = 0;
  context_.gemm_context = nullptr;
  // Reserve some space for the tensors to avoid excessive resizing.
  tensors_.reserve(kSlotsToReserve);
  nodes_and_registration_.reserve(kSlotsToReserve);
  next_node_to_prepare_ = 0;
  UseNNAPI(false);
}

Interpreter::~Interpreter() {
  for (auto& nodeAndReg : nodes_and_registration_) {
    TfLiteNode& node = nodeAndReg.first;
    TfLiteIntArrayFree(node.inputs);
    TfLiteIntArrayFree(node.outputs);
    TfLiteIntArrayFree(node.temporaries);
    if (node.builtin_data) free(node.builtin_data);
    OpFree(nodeAndReg.second, node.user_data);
    node.builtin_data = nullptr;
  }

  for (int i = 0; i < context_.tensors_size; i++) {
    TfLiteTensorFree(&context_.tensors[i]);
  }
}

TfLiteStatus Interpreter::SetInputs(std::vector<int> inputs) {
  TF_LITE_ENSURE_OK(&context_,
                    CheckTensorIndices("inputs", inputs.data(), inputs.size()));
  inputs_ = std::move(inputs);
  return kTfLiteOk;
}

TfLiteStatus Interpreter::SetOutputs(std::vector<int> outputs) {
  TF_LITE_ENSURE_OK(
      &context_, CheckTensorIndices("outputs", outputs.data(), outputs.size()));
  outputs_ = std::move(outputs);
  return kTfLiteOk;
}

TfLiteStatus Interpreter::CheckTensorIndices(const char* label,
                                             const int* indices, int length) {
  // Making sure kOptionalTensor is not re-defined to something other than -1.
  static_assert(kOptionalTensor == -1, "kOptionalTensor should be defined -1");

  for (int i = 0; i < length; i++) {
    int index = indices[i];
    if (index < kOptionalTensor || index >= context_.tensors_size) {
      ReportError(&context_, "Invalid tensor index %d in %s\n", index, label);
      consistent_ = false;
      return kTfLiteError;
    }
  }
  return kTfLiteOk;
}

TfLiteStatus Interpreter::BytesRequired(TfLiteType type, const int* dims,
                                        int dims_size, size_t* bytes) {
  // TODO(aselle): Check for overflow here using overflow.h in TensorFlow
  // MultiplyWithoutOverflow.
  TF_LITE_ENSURE(&context_, bytes != nullptr);
  size_t count = 1;
  for (int k = 0; k < dims_size; k++) count *= dims[k];
  switch (type) {
    case kTfLiteFloat32:
      *bytes = sizeof(float) * count;
      break;
    case kTfLiteInt32:
      *bytes = sizeof(int32_t) * count;
      break;
    case kTfLiteUInt8:
      *bytes = sizeof(uint8_t) * count;
      break;
    case kTfLiteInt64:
      *bytes = sizeof(int64_t) * count;
      break;
    default:
      ReportError(&context_,
                  "Only float32, int32, int64, uint8 supported currently.");
      return kTfLiteError;
  }
  return kTfLiteOk;
}

namespace {
TfLiteIntArray* convertVectorToTfLiteIntArray(const std::vector<int>& x) {
  TfLiteIntArray* lite = TfLiteIntArrayCreate(x.size());
  for (size_t i = 0; i < x.size(); i++) lite->data[i] = x[i];
  return lite;
}
}  // namespace

TfLiteStatus Interpreter::AllocateTensors() {
  next_node_to_prepare_ = 0;
  if (memory_planner_) {
    TF_LITE_ENSURE_STATUS(memory_planner_->ResetAllocations());
  }

  if (!consistent_) {
    ReportError(&context_, "AllocateTensors() called on inconsistent model.");
    return kTfLiteError;
  }

  TF_LITE_ENSURE_STATUS(PrepareOpsAndTensors());
  invokable_ = true;
  return kTfLiteOk;
}

TfLiteStatus Interpreter::AddNodeWithParameters(
    const std::vector<int>& inputs, const std::vector<int>& outputs,
    const char* init_data, size_t init_data_size, void* builtin_data,
    const TfLiteRegistration* registration, int* node_index) {
  invokable_ = false;

  std::unique_ptr<void, decltype(free)*> builtin_data_deleter(builtin_data,
                                                              free);

  TF_LITE_ENSURE_OK(&context_, CheckTensorIndices("node inputs", inputs.data(),
                                                  inputs.size()));
  TF_LITE_ENSURE_OK(
      &context_,
      CheckTensorIndices("node outputs", outputs.data(), outputs.size()));

  if (node_index) *node_index = nodes_and_registration_.size();
  nodes_and_registration_.resize(nodes_and_registration_.size() + 1);
  auto& node_and_reg = nodes_and_registration_.back();
  TfLiteNode& node = node_and_reg.first;
  if (node.inputs) TfLiteIntArrayFree(node.inputs);
  if (node.outputs) TfLiteIntArrayFree(node.outputs);
  if (node.temporaries) TfLiteIntArrayFree(node.temporaries);

  // NOTE, here we are not using move semantics yet, since our internal
  // representation isn't std::vector, but in the future we would like to avoid
  // copies, so we want the interface to take r-value references now.
  node.inputs = convertVectorToTfLiteIntArray(inputs);
  node.outputs = convertVectorToTfLiteIntArray(outputs);
  node.temporaries = TfLiteIntArrayCreate(0);
  if (init_data) {
    node.user_data = OpInit(*registration, init_data, init_data_size);
  } else {
    node.user_data =
        OpInit(*registration,
               reinterpret_cast<const char*>(builtin_data_deleter.get()), 0);
  }
  node.builtin_data = builtin_data_deleter.release();
  node_and_reg.second = *registration;
  return kTfLiteOk;
}

TfLiteStatus Interpreter::AddNodeWithParametersOpenCL(
    const std::vector<int>& inputs, const std::vector<int>& outputs,
    const char* init_data, size_t init_data_size, void* builtin_data,
    const TfLiteRegistration* registration,
    cl_context context_cl, cl_command_queue queue, cl_program program, cl_mem cl_mem_arr[6],
    VkPhysicalDevice physicalDevice, VkDevice device, VkPipeline pipelineConv, VkPipeline pipelineMatmul, VkPipelineLayout pipelineLayoutConv, VkPipelineLayout pipelineLayoutMatmul, VkPipeline pipelineConvMatmul, VkPipelineLayout pipelineLayoutConvMatmul,
    VkDescriptorSetLayout descriptorSetLayoutConv, VkDescriptorSetLayout descriptorSetLayoutMatmul, VkQueue queueV, uint32_t queueFamilyIndex,
    VkCommandPool conv_commandPool, VkCommandBuffer conv_commandBuffer, VkBuffer conv_matrixA, VkBuffer conv_matrixB, VkBuffer conv_matrixC, VkBuffer conv_matrixSizes, VkDeviceMemory conv_bufferMemory,
    int* node_index) {
  invokable_ = false;

  std::unique_ptr<void, decltype(free)*> builtin_data_deleter(builtin_data,
                                                              free);

  TF_LITE_ENSURE_OK(&context_, CheckTensorIndices("node inputs", inputs.data(),
                                                  inputs.size()));
  TF_LITE_ENSURE_OK(
      &context_,
      CheckTensorIndices("node outputs", outputs.data(), outputs.size()));

  if (node_index) *node_index = nodes_and_registration_.size();
  nodes_and_registration_.resize(nodes_and_registration_.size() + 1);
  auto& node_and_reg = nodes_and_registration_.back();
  TfLiteNode& node = node_and_reg.first;
  if (node.inputs) TfLiteIntArrayFree(node.inputs);
  if (node.outputs) TfLiteIntArrayFree(node.outputs);
  if (node.temporaries) TfLiteIntArrayFree(node.temporaries);

  // NOTE, here we are not using move semantics yet, since our internal
  // representation isn't std::vector, but in the future we would like to avoid
  // copies, so we want the interface to take r-value references now.
  node.inputs = convertVectorToTfLiteIntArray(inputs);
  node.outputs = convertVectorToTfLiteIntArray(outputs);
  node.temporaries = TfLiteIntArrayCreate(0);
  __android_log_print(ANDROID_LOG_INFO, "Ngising", "masuk interpreter");
  if (init_data) {
    node.user_data = OpInitOpenCL(*registration, init_data, init_data_size, context_cl, queue, program, cl_mem_arr,
      physicalDevice, device, pipelineConv, pipelineMatmul, pipelineLayoutConv, pipelineLayoutMatmul, pipelineConvMatmul, pipelineLayoutConvMatmul, descriptorSetLayoutConv, 
      descriptorSetLayoutMatmul, queueV, queueFamilyIndex,
      conv_commandPool, conv_commandBuffer, conv_matrixA, conv_matrixB, conv_matrixC, conv_matrixSizes, conv_bufferMemory);
  } else {
    node.user_data =
        OpInitOpenCL(*registration,
               reinterpret_cast<const char*>(builtin_data_deleter.get()), 0,
               context_cl, queue, program, cl_mem_arr,
               physicalDevice, device, pipelineConv, pipelineMatmul, pipelineLayoutConv, pipelineLayoutMatmul, pipelineConvMatmul, pipelineLayoutConvMatmul, 
               descriptorSetLayoutConv, descriptorSetLayoutMatmul, queueV, queueFamilyIndex,
               conv_commandPool, conv_commandBuffer, conv_matrixA, conv_matrixB, conv_matrixC, conv_matrixSizes, conv_bufferMemory);
  }
  node.builtin_data = builtin_data_deleter.release();
  node_and_reg.second = *registration;
  return kTfLiteOk;
}

TfLiteStatus Interpreter::ResizeInputTensor(int tensor_index,
                                            const std::vector<int>& dims) {
  // TODO(aselle): All bounds checks can be implemented as one-sided bounds
  // checks by casting to unsigned for efficiency. Profile before doing this.

  TF_LITE_ENSURE(&context_,
                 tensor_index < context_.tensors_size && tensor_index >= 0);
  invokable_ = false;
  TfLiteIntArray* dims_lite = convertVectorToTfLiteIntArray(dims);
  return ResizeTensorImpl(&context_.tensors[tensor_index], dims_lite);
}

// Returns true if at least one tensor in the given list is kTfLiteDynamic.
bool HasDynamicTensor(const TfLiteContext& context,
                      const TfLiteIntArray* tensors) {
  for (int i = 0; i < tensors->size; ++i) {
    const TfLiteTensor& tensor = context.tensors[tensors->data[i]];
    if (tensor.allocation_type == kTfLiteDynamic) {
      return true;
    }
  }
  return false;
}

TfLiteStatus Interpreter::PrepareOpsStartingAt(int first_node,
                                               int* last_node_prepared) {
  for (int i = first_node; i < nodes_and_registration_.size(); i++) {
    TfLiteNode& node = nodes_and_registration_[i].first;
    const TfLiteRegistration& registration = nodes_and_registration_[i].second;
    if (OpPrepare(registration, &node) == kTfLiteError) {
      return kTfLiteError;
    }

    *last_node_prepared = i;

    // Discontinue if the node has dynamic outputs. Note that we don't
    // stop for dynamic temporary tensors since they won't affect the
    // sizes of other tensors in the graph.
    if (HasDynamicTensor(context_, node.outputs)) {
      break;
    }
  }
  return kTfLiteOk;
}

TfLiteStatus Interpreter::PrepareOpsAndTensors() {
  if (!memory_planner_) {
    memory_planner_.reset(new ArenaPlanner(
        &context_, std::unique_ptr<GraphInfo>(new InterpreterInfo(this))));
    memory_planner_->PlanAllocations();
  }

  int last_node_prepared = 0;

  TF_LITE_ENSURE_STATUS(
      PrepareOpsStartingAt(next_node_to_prepare_, &last_node_prepared));
  TF_LITE_ENSURE_STATUS(memory_planner_->ExecuteAllocations(
      next_node_to_prepare_, last_node_prepared));

  next_node_to_prepare_ = last_node_prepared + 1;
  return kTfLiteOk;
}

TfLiteStatus Interpreter::Invoke() {
  if (!consistent_) {
    ReportError(&context_, "Invoke called on model that is not consistent.");
    return kTfLiteError;
  }
  if (!invokable_) {
    ReportError(&context_, "Invoke called on model that is not ready.");
    return kTfLiteError;
  }

  TfLiteStatus status = kTfLiteOk;
  if (nnapi_delegate_) {
    if (next_node_to_prepare_ == nodes_and_registration_.size()) {
      TF_LITE_ENSURE_OK(&context_, nnapi_delegate_->Invoke(this));
      return kTfLiteOk;
    } else {
      // TODO(aselle): In the future, we would like this to be an
      // automatic tflite CPU fallback.
      ReportError(&context_,
                  "NNAPI was requested, but dependent sized tensors "
                  "being used.\n");
      return kTfLiteError;
    }
  }

  // Invocations are always done in node order.
  // Note that calling Invoke repeatedly will cause the original memory plan to
  // be reused, unless either ResizeInputTensor() or AllocateTensors() has been
  // called.
  // TODO(b/71913981): we should force recalculation in the presence of dynamic
  // tensors, because they may have new value which in turn may affect shapes
  // and allocations.
  for (int i = 0; i < nodes_and_registration_.size(); i++) {
    if (i == next_node_to_prepare_) {
      TF_LITE_ENSURE_STATUS(PrepareOpsAndTensors());
      TF_LITE_ENSURE(&context_, next_node_to_prepare_ >= i);
    }
    TfLiteNode& node = nodes_and_registration_[i].first;
    const TfLiteRegistration& registration = nodes_and_registration_[i].second;
    if (OpInvoke(registration, &node) == kTfLiteError) {
      status = kTfLiteError;
    }
  }
  return status;
}

TfLiteStatus Interpreter::ResizeTensor(TfLiteContext* context,
                                       TfLiteTensor* tensor,
                                       TfLiteIntArray* new_size) {
  // Note here that context->impl_ is recovering the this pointer for an
  // instance of Interpreter to call into the member function ResizeTensorImpl
  // (this function is static).
  return static_cast<Interpreter*>(context->impl_)
      ->ResizeTensorImpl(tensor, new_size);
}

void Interpreter::ReportErrorImpl(const char* format, va_list args) {
  error_reporter_->Report(format, args);
}

void Interpreter::ReportError(TfLiteContext* context, const char* format, ...) {
  va_list args;
  va_start(args, format);
  auto* f = static_cast<Interpreter*>(context->impl_);
  // Note here that context->impl_ is recovering the this pointer for an
  // instance of Interpreter to call into the member function ReportErrorImpl
  // (this function is static).
  f->ReportErrorImpl(format, args);
  va_end(args);
}

TfLiteStatus Interpreter::AddTensors(int tensors_to_add,
                                     int* first_new_tensor_index) {
  int base_index = tensors_.size();
  if (first_new_tensor_index) *first_new_tensor_index = base_index;
  tensors_.resize(tensors_.size() + tensors_to_add);
  for (int i = base_index; i < tensors_.size(); i++) {
    memset(&tensors_[i], 0, sizeof(tensors_[i]));
  }
  context_.tensors = tensors_.data();
  context_.tensors_size = tensors_.size();
  return kTfLiteOk;
}

TfLiteStatus Interpreter::AddTensors(TfLiteContext* context, int tensors_to_add,
                                     int* first_new_tensor_index) {
  // Note here that context->impl_ is recovering the this pointer for an
  // instance of Interpreter to call into the member function AddTensors
  // (this function is static).
  return static_cast<Interpreter*>(context->impl_)
      ->AddTensors(tensors_to_add, first_new_tensor_index);
}

TfLiteStatus Interpreter::SetTensorParametersReadOnly(
    int tensor_index, TfLiteType type, const char* name,
    const std::vector<int>& dims, TfLiteQuantizationParams quantization,
    const char* buffer, size_t bytes, const Allocation* allocation) {
  TF_LITE_ENSURE(&context_,
                 tensor_index < context_.tensors_size && tensor_index >= 0);
  // For most tensors we know exactly how much memory is necessary so we can
  // ensure the buffer is large enough. However, we need to skip string tensors
  // because their sizes change with the contents of the individual strings.
  if (type != kTfLiteString) {
    size_t required_bytes;
    TF_LITE_ENSURE_OK(&context_, BytesRequired(type, dims.data(), dims.size(),
                                               &required_bytes));
    TF_LITE_ENSURE_EQ(&context_, required_bytes, bytes);
  }
  invokable_ = false;
  TfLiteTensorReset(type, name, convertVectorToTfLiteIntArray(dims),
                    quantization, const_cast<char*>(buffer), bytes,
                    kTfLiteMmapRo, allocation, &context_.tensors[tensor_index]);
  return kTfLiteOk;
}

// Set description of inputs/outputs/data/fptrs for node `node_index`.
// This variant assumes an external buffer has been allocated of size
// bytes. The lifetime of buffer must be ensured to be greater or equal
// to Interpreter.
TfLiteStatus Interpreter::SetTensorParametersReadWrite(
    int tensor_index, TfLiteType type, const char* name,
    const std::vector<int>& dims, TfLiteQuantizationParams quantization) {
  invokable_ = false;
  TF_LITE_ENSURE(&context_,
                 tensor_index < context_.tensors_size && tensor_index >= 0);
  size_t required_bytes = 0;
  if (type != kTfLiteString) {
    // These types will be allocated in our arena so we need to record how
    // many bytes we will need based on the dimensions. String tensors are
    // allocated dynamically and we can't know ahead of time how much space
    // they will require.
    TF_LITE_ENSURE_OK(&context_, BytesRequired(type, dims.data(), dims.size(),
                                               &required_bytes));
  }
  TfLiteTensorReset(type, name, convertVectorToTfLiteIntArray(dims),
                    quantization,
                    /*buffer=*/nullptr, required_bytes,
                    type == kTfLiteString ? kTfLiteDynamic : kTfLiteArenaRw,
                    nullptr, &context_.tensors[tensor_index]);
  return kTfLiteOk;
}

TfLiteStatus Interpreter::ResizeTensorImpl(TfLiteTensor* tensor,
                                           TfLiteIntArray* new_size) {
  // Note that in theory we could resize kTfLiteArenaRwPersistent tensors too.
  if (tensor->allocation_type == kTfLiteArenaRw ||
      tensor->allocation_type == kTfLiteDynamic) {
    if (tensor->type != kTfLiteString) {
      size_t bytesRequired;
      TfLiteStatus status = BytesRequired(tensor->type, new_size->data,
                                          new_size->size, &bytesRequired);
      if (status != kTfLiteOk) {
        TfLiteIntArrayFree(new_size);
        return kTfLiteError;
      }
      tensor->bytes = bytesRequired;
    }
    if (tensor->dims) TfLiteIntArrayFree(tensor->dims);
    tensor->dims = new_size;

    if (tensor->allocation_type != kTfLiteDynamic) {
      tensor->data.raw = nullptr;
    }
  } else {
    // kTfLiteMmapRo tensors are stored in the flatbuffer and are therefore
    // of fixed size.
    TfLiteIntArrayFree(new_size);
    ReportError(&context_, "Attempting to resize a fixed-size tensor.");
    return kTfLiteError;
  }
  return kTfLiteOk;
}

void Interpreter::UseNNAPI(bool enable) {
  // TODO(aselle): This is a workaround for finding if NNAPI exists.
  // We also need to make sure getLibraryHandle() is renamed to be NNAPI
  // prefixed.
  if (!NNAPIExists()) enable = false;
  if (!enable) {
    nnapi_delegate_.reset();
  } else if (!nnapi_delegate_) {
    nnapi_delegate_.reset(new NNAPIDelegate);
  }
}

void Interpreter::SetNumThreads(int num_threads) {
  // TODO(ahentz): this forces us to link against gemmlowp even when the ops
  // don't use it. We should implement some dynamic mechanism for this sort of
  // library-specific initialization.
  tflite::gemm_support::SetMaxNumThreads(&context_, num_threads);
}

}  // namespace tflite
