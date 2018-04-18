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
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include "tensorflow/contrib/lite/allocation.h"
#include "tensorflow/contrib/lite/builtin_op_data.h"
#include "tensorflow/contrib/lite/error_reporter.h"
#include "tensorflow/contrib/lite/model.h"
#include "tensorflow/contrib/lite/nnapi_delegate.h"
#include "tensorflow/contrib/lite/version.h"

//note: android opencl
#include "CL/cl.h"

//note: vulkan
#include "vulkan/vulkan.h"
#include "vulkan/vk_platform.h"
#include <vector>
#include <string.h>
#include <assert.h>
#include <stdexcept>
#include <cmath>

//note: shaderc
#include "shaderc/shaderc.hpp"

//note: string
#include <string>
#include <iostream>

//note: android log
#include <android/log.h> 
#include <stdio.h> 

#define VK_CHECK_RESULT(f)                                        \
{                                                   \
    VkResult res = (f);                                         \
    if (res != VK_SUCCESS)                                        \
    {                                                 \
        __android_log_print(ANDROID_LOG_INFO, "VulkanTest", "Fatal : VkResult is %d in %s at line %d\n", res,  __FILE__, __LINE__); \
        assert(res == VK_SUCCESS);                                    \
    }                                                 \
}

const char *kernelSource =           "\n" \
"#pragma OPENCL EXTENSION cl_khr_fp16 : enable \n" \
"__kernel void convhalf(__global half4* input_data,    \n" \
"          __constant half4* filter_data,    \n" \
"          __global half* bias_data,    \n" \
"          __global half* output_data,   \n" \
"          int stride_width, int stride_height,    \n" \
"          int pad_width, int pad_height,    \n" \
"          __global int16* dim_sizes0, __global int16* dim_strides0,   \n" \
"          half output_activation_min, half output_activation_max) {   \n" \
"     \n" \
"    int batchdepth = get_global_id(0);   \n" \
"    int widthheight = get_global_id(1);   \n" \
"    int16 dim_sizes = dim_sizes0[0];   \n" \
"    int output_depth = dim_sizes.s7;  \n" \
"    int output_height = dim_sizes.se;   \n" \
"    int batch = batchdepth/output_depth;   \n" \
"    int out_channel = batchdepth\%output_depth;   \n" \
"    int out_x = widthheight/output_height;   \n" \
"    int out_y = widthheight\%output_height;   \n" \
"    if((batch < dim_sizes.s3) && (out_x < dim_sizes.sd) && (out_y < output_height) && (out_channel < output_depth)) {   \n" \
"            int16 dim_strides = dim_strides0[0];   \n" \
"            half total = 0.0;   \n" \
"            for (int filter_y = 0; filter_y < dim_sizes.s6; ++filter_y) {   \n" \
"              for (int filter_x = 0; filter_x < dim_sizes.s5; ++filter_x) {   \n" \
"                for (int in_channel = 0; in_channel < dim_sizes.s0/4; ++in_channel) {   \n" \
"                  int in_x = (out_x * stride_width) - pad_width + filter_x;   \n" \
"                  int in_y = (out_y * stride_height) - pad_height + filter_y;   \n" \
"                  if ((in_x >= 0) && (in_x < dim_sizes.s1) && (in_y >= 0) &&   \n" \
"                      (in_y < dim_sizes.s2)) {   \n" \
"                    half4 input_value = input_data[in_channel*dim_strides.s0 + in_x*dim_strides.s1/4 + in_y*dim_strides.s2/4 + batch*dim_strides.s3/4];   \n" \
"                    half4 filter_value = filter_data[in_channel*dim_strides.s4 + filter_x*dim_strides.s5/4 + filter_y*dim_strides.s6/4 + out_channel*dim_strides.s7/4];  \n" \
"                    total += dot(input_value,filter_value);   \n" \
"                  }   \n" \
"                }   \n" \
"              }   \n" \
"            }   \n" \
"            half bias_value = 0.0;   \n" \
"            if (bias_data) {   \n" \
"              bias_value = bias_data[out_channel*dim_strides.s8];   \n" \
"            } \n" \
"            output_data[out_channel*dim_strides.sc + out_x*dim_strides.sd + out_y*dim_strides.se + batch*dim_strides.sf] = min(max(total + bias_value, output_activation_min), output_activation_max); \n" \
"    }  \n" \
"}   \n" \
"  \n" \
"__kernel void conv(__global float4* input_data,    \n" \
"          __constant float4* filter_data,    \n" \
"          __global float* bias_data,    \n" \
"          __global float* output_data,   \n" \
"          int stride_width, int stride_height,    \n" \
"          int pad_width, int pad_height,    \n" \
"          __global int16* dim_sizes0, __global int16* dim_strides0,   \n" \
"          float output_activation_min, float output_activation_max) {   \n" \
"     \n" \
"    int batchdepth = get_global_id(0);   \n" \
"    int widthheight = get_global_id(1);   \n" \
"    int16 dim_sizes = dim_sizes0[0];   \n" \
"    int output_depth = dim_sizes.s7;  \n" \
"    int output_height = dim_sizes.se;   \n" \
"    int batch = batchdepth/output_depth;   \n" \
"    int out_channel = batchdepth\%output_depth;   \n" \
"    int out_x = widthheight/output_height;   \n" \
"    int out_y = widthheight\%output_height;   \n" \
"    if((batch < dim_sizes.s3) && (out_x < dim_sizes.sd) && (out_y < output_height) && (out_channel < output_depth)) {   \n" \
"            int16 dim_strides = dim_strides0[0];   \n" \
"            float total = 0.0;   \n" \
"            for (int filter_y = 0; filter_y < dim_sizes.s6; ++filter_y) {   \n" \
"              for (int filter_x = 0; filter_x < dim_sizes.s5; ++filter_x) {   \n" \
"                for (int in_channel = 0; in_channel < dim_sizes.s0/4; ++in_channel) {   \n" \
"                  int in_x = (out_x * stride_width) - pad_width + filter_x;   \n" \
"                  int in_y = (out_y * stride_height) - pad_height + filter_y;   \n" \
"                  if ((in_x >= 0) && (in_x < dim_sizes.s1) && (in_y >= 0) &&   \n" \
"                      (in_y < dim_sizes.s2)) {   \n" \
"                    float4 input_value = input_data[in_channel*dim_strides.s0 + in_x*dim_strides.s1/4 + in_y*dim_strides.s2/4 + batch*dim_strides.s3/4];   \n" \
"                    float4 filter_value = filter_data[in_channel*dim_strides.s4 + filter_x*dim_strides.s5/4 + filter_y*dim_strides.s6/4 + out_channel*dim_strides.s7/4];  \n" \
"                    total += dot(input_value,filter_value);   \n" \
"                  }   \n" \
"                }   \n" \
"              }   \n" \
"            }   \n" \
"            float bias_value = 0.0;   \n" \
"            if (bias_data) {   \n" \
"              bias_value = bias_data[out_channel*dim_strides.s8];   \n" \
"            } \n" \
"            output_data[out_channel*dim_strides.sc + out_x*dim_strides.sd + out_y*dim_strides.se + batch*dim_strides.sf] = min(max(total + bias_value, output_activation_min), output_activation_max); \n" \
"    }  \n" \
"}   \n" \
"__kernel void transpose(__global float4* input, __global float* output,\n" \
"    int rows, int cols) {         \n" \
"   int row = get_global_id(0);                                      \n" \
"   int col4 = get_global_id(1);                                      \n" \
"   const float4 in_value = input[row*(cols/4)+col4];\n" \
"   output[(col4*4+0)*rows + row] = in_value.x;\n" \
"   output[(col4*4+1)*rows + row] = in_value.y;\n" \
"   output[(col4*4+2)*rows + row] = in_value.z;\n" \
"   output[(col4*4+3)*rows + row] = in_value.w;\n" \
"}      \n" \
"__kernel void matrixVectorMulF4(__global half4* result,    \n" \
"    const __global half4* matrix,    \n" \
"    const __global half4* vector,     \n" \
"    int m_cols,    \n" \
"    int m_rows,    \n" \
"    int n_batch)    \n" \
"{  \n" \
"    int row = get_global_id(0)*4; \n" \
"    int localidx = get_local_id(1); \n" \
"    __local half4 Aacc[32]; \n" \
"    if (row < m_rows) { \n" \
"        half4 sum = {0.0, 0.0, 0.0, 0.0}; \n" \
"        int starti = localidx*(m_cols/128); \n" \
"        for(int i = starti; i < starti+(m_cols/128); i++){ \n" \
"            half4 currb = vector[i];\n" \
"            sum.x += dot(matrix[(row*m_cols/4) + i],currb);\n" \
"            sum.y += dot(matrix[((row+1)*m_cols/4) + i],currb); \n" \
"            sum.z += dot(matrix[((row+2)*m_cols/4) + i],currb);\n" \
"            sum.w += dot(matrix[((row+3)*m_cols/4) + i],currb);\n" \
"        } \n" \
"        Aacc[localidx] = sum; \n" \
"        barrier(CLK_LOCAL_MEM_FENCE);      \n" \
"        if(localidx == 0) { \n" \
"            result[row/4] = Aacc[0] + Aacc[1] + Aacc[2] + Aacc[3] + Aacc[4] + Aacc[5] + Aacc[6] + Aacc[7]+ \n" \
"            Aacc[8] + Aacc[9] + Aacc[10] + Aacc[11] + Aacc[12] + Aacc[13] + Aacc[14] + Aacc[15]+ \n" \
"            Aacc[16] + Aacc[17] + Aacc[18] + Aacc[19] + Aacc[20] + Aacc[21] + Aacc[22] + Aacc[23]+ \n" \
"            Aacc[24] + Aacc[25] + Aacc[26] + Aacc[27] + Aacc[28] + Aacc[29] + Aacc[30] + Aacc[31]; \n" \
"        } \n" \
"    } \n" \
"} \n" \
"__kernel void matrixVectorMulF4float(__global float4* result,    \n" \
"    const __global float4* matrix,    \n" \
"    const __global float4* vector,     \n" \
"    int m_cols,    \n" \
"    int m_rows,    \n" \
"    int n_batch)    \n" \
"{  \n" \
"    int row = get_global_id(0)*4; \n" \
"    int localidx = get_local_id(1); \n" \
"    __local float4 Aacc[32]; \n" \
"    if (row < m_rows) { \n" \
"        float4 sum = {0.0, 0.0, 0.0, 0.0}; \n" \
"        int starti = localidx*(m_cols/128); \n" \
"        for(int i = starti; i < starti+(m_cols/128); i++){ \n" \
"            float4 currb = vector[i];\n" \
"            sum.x += dot(matrix[(row*m_cols/4) + i],currb);\n" \
"            sum.y += dot(matrix[((row+1)*m_cols/4) + i],currb); \n" \
"            sum.z += dot(matrix[((row+2)*m_cols/4) + i],currb);\n" \
"            sum.w += dot(matrix[((row+3)*m_cols/4) + i],currb);\n" \
"        } \n" \
"        Aacc[localidx] = sum; \n" \
"        barrier(CLK_LOCAL_MEM_FENCE);      \n" \
"        if(localidx == 0) { \n" \
"            result[row/4] = Aacc[0] + Aacc[1] + Aacc[2] + Aacc[3] + Aacc[4] + Aacc[5] + Aacc[6] + Aacc[7]+ \n" \
"            Aacc[8] + Aacc[9] + Aacc[10] + Aacc[11] + Aacc[12] + Aacc[13] + Aacc[14] + Aacc[15]+ \n" \
"            Aacc[16] + Aacc[17] + Aacc[18] + Aacc[19] + Aacc[20] + Aacc[21] + Aacc[22] + Aacc[23]+ \n" \
"            Aacc[24] + Aacc[25] + Aacc[26] + Aacc[27] + Aacc[28] + Aacc[29] + Aacc[30] + Aacc[31]; \n" \
"        } \n" \
"    } \n" \
"} \n" \
"\n";


//OpenCL
cl_platform_id cpPlatform = NULL;
cl_device_id device_id = NULL;    
cl_context context_cl = NULL;       
cl_command_queue queueCL = NULL;
cl_program program = NULL;

cl_mem d_conv_input = NULL;
cl_mem d_conv_filter = NULL;
cl_mem d_conv_bias = NULL;
cl_mem d_conv_output = NULL;
cl_mem d_conv_dim_sizes = NULL;
cl_mem d_conv_dim_strides = NULL;
VkCommandPool conv_commandPool = NULL;
VkCommandBuffer conv_commandBuffer = NULL;
VkBuffer conv_matrixA = NULL;
VkBuffer conv_matrixSizes = NULL;
VkDeviceMemory conv_bufferMemory = NULL;

int buffsizes[4] = {710432, 1548288, 1024, 1382976};

// cl_mem cl_mem_arr[6], VkCommandPool conv_commandPool, VkCommandBuffer conv_commandBuffer, VkBuffer conv_matrixA, VkBuffer conv_matrixSizes, VkDeviceMemory conv_bufferMemory
// cl_mem_arr,conv_commandPool, conv_commandBuffer, conv_matrixA, conv_matrixSizes, conv_bufferMemory

//mobilenet
// 04-17 12:13:31.648 24739-24794/android.example.com.tflitecamerademo I/VectorSize: inputSizeconv.cc: 401408
// 04-17 12:13:31.648 24739-24794/android.example.com.tflitecamerademo I/VectorSize: filterSizeconv.cc: 1048576
// 04-17 12:13:31.648 24739-24794/android.example.com.tflitecamerademo I/VectorSize: biasSizeconv.cc: 1024
// 04-17 12:13:31.648 24739-24794/android.example.com.tflitecamerademo I/VectorSize: outputSizeconv.cc: 802816

//inception
// 04-17 12:18:46.322 27363-27479/android.example.com.tflitecamerademo I/VectorSize: inputSizeconv.cc: 710432
// 04-17 12:18:46.322 27363-27479/android.example.com.tflitecamerademo I/VectorSize: filterSizeconv.cc: 1548288
// 04-17 12:18:46.322 27363-27479/android.example.com.tflitecamerademo I/VectorSize: biasSizeconv.cc: 448
// 04-17 12:18:46.322 27363-27479/android.example.com.tflitecamerademo I/VectorSize: outputSizeconv.cc: 1382976

//buffer sizes
// int buffsizes[4] = {0,0,0,0};


// device, pipelineConv, pipelineMatmul, pipelineLayoutConv, pipelineLayoutMatmul, descriptorSetLayoutConv, descriptorSetLayoutMatmul, queue, queueFamilyIndex

namespace tflite {

const char* kEmptyTensorName = "";

std::unique_ptr<FlatBufferModel> FlatBufferModel::BuildFromFile(
    const char* filename, ErrorReporter* error_reporter) {
  std::unique_ptr<FlatBufferModel> model;
  model.reset(new FlatBufferModel(filename, /*mmap_file=*/true, error_reporter,
                                  /*use_nnapi=*/true));
  if (!model->initialized()) model.reset();
  return model;
}

std::unique_ptr<FlatBufferModel> FlatBufferModel::BuildFromBuffer(
    const char* buffer, size_t buffer_size, ErrorReporter* error_reporter) {
  std::unique_ptr<FlatBufferModel> model;
  model.reset(new FlatBufferModel(buffer, buffer_size, error_reporter));
  if (!model->initialized()) model.reset();
  return model;
}

std::unique_ptr<FlatBufferModel> FlatBufferModel::BuildFromModel(
    const tflite::Model* model_spec, ErrorReporter* error_reporter) {
  std::unique_ptr<FlatBufferModel> model;
  model.reset(new FlatBufferModel(model_spec, error_reporter));
  if (!model->initialized()) model.reset();
  return model;
}

FlatBufferModel::FlatBufferModel(const char* filename, bool mmap_file,
                                 ErrorReporter* error_reporter, bool use_nnapi)
    : error_reporter_(error_reporter ? error_reporter
                                     : DefaultErrorReporter()) {
  if (mmap_file) {
    if (use_nnapi && NNAPIExists())
      allocation_ = new NNAPIAllocation(filename, error_reporter);
    else
      allocation_ = new MMAPAllocation(filename, error_reporter);
  } else {
    allocation_ = new FileCopyAllocation(filename, error_reporter);
  }
  if (!allocation_->valid() || !CheckModelIdentifier()) return;

  model_ = ::tflite::GetModel(allocation_->base());
}

bool FlatBufferModel::CheckModelIdentifier() const {
  if (!tflite::ModelBufferHasIdentifier(allocation_->base())) {
    const char* ident = flatbuffers::GetBufferIdentifier(allocation_->base());
    error_reporter_->Report(
        "Model provided has model identifier '%c%c%c%c', should be '%s'\n",
        ident[0], ident[1], ident[2], ident[3], tflite::ModelIdentifier());
    return false;
  }
  return true;
}

FlatBufferModel::FlatBufferModel(const char* ptr, size_t num_bytes,
                                 ErrorReporter* error_reporter)
    : error_reporter_(error_reporter ? error_reporter
                                     : DefaultErrorReporter()) {
  allocation_ = new MemoryAllocation(ptr, num_bytes, error_reporter);
  if (!allocation_->valid()) return;

  model_ = ::tflite::GetModel(allocation_->base());
}

FlatBufferModel::FlatBufferModel(const Model* model,
                                 ErrorReporter* error_reporter)
    : error_reporter_(error_reporter ? error_reporter
                                     : DefaultErrorReporter()) {
  model_ = model;
}

FlatBufferModel::~FlatBufferModel() { delete allocation_; }

InterpreterBuilder::InterpreterBuilder(const FlatBufferModel& model,
                                       const OpResolver& op_resolver)
    : model_(model.GetModel()),
      op_resolver_(op_resolver),
      error_reporter_(model.error_reporter()),
      allocation_(model.allocation()) {}

InterpreterBuilder::InterpreterBuilder(const ::tflite::Model* model,
                                       const OpResolver& op_resolver,
                                       ErrorReporter* error_reporter)
    : model_(model),
      op_resolver_(op_resolver),
      error_reporter_(error_reporter ? error_reporter
                                     : DefaultErrorReporter()) {}

TfLiteStatus InterpreterBuilder::BuildLocalIndexToRegistrationMapping() {
  TfLiteStatus status = kTfLiteOk;
  auto opcodes = model_->operator_codes();
  for (const OperatorCode* opcode : *opcodes) {
    TfLiteRegistration* registration = nullptr;

    if (opcode->builtin_code() != BuiltinOperator_CUSTOM) {
      auto x = opcode->builtin_code();
      flatbuffer_op_index_to_registration_types_.push_back(x);
      registration = op_resolver_.FindOp(x);
      if (registration == nullptr) {
        error_reporter_->Report("Didn't find op for builtin opcode '%s'\n",
                                EnumNameBuiltinOperator(x));
        status = kTfLiteError;
      }
    } else if (!opcode->custom_code()) {
      error_reporter_->Report(
          "Operator with builtin_code==0 has no custom_code.\n");
      status = kTfLiteError;
    } else {
      const char* name = opcode->custom_code()->c_str();
      registration = op_resolver_.FindOp(name);
      flatbuffer_op_index_to_registration_types_.push_back(
          BuiltinOperator_CUSTOM);
      if (registration == nullptr) {
        error_reporter_->Report("Didn't find custom op for name '%s'\n", name);
        status = kTfLiteError;
      }
    }
    flatbuffer_op_index_to_registration_.push_back(registration);
  }
  return status;
}

namespace {
template <class T>
std::vector<int> FlatBufferIntArrayToVector(T* flat_array) {
  std::vector<int> ret(flat_array->Length());
  for (int i = 0; i < flat_array->Length(); i++) {
    ret[i] = flat_array->Get(i);
  }
  return ret;
}

// Copies the contents from the flatbuffer int vector `flatbuffer` into the
// int array `buffer`. `flat_vector` and `buffer` represent the same
// configuration operation for a given operation.
void FlatBufferIntVectorToArray(int max_size_of_buffer,
                                const flatbuffers::Vector<int32_t>* flat_vector,
                                int* buffer, ErrorReporter* error_reporter) {
  if (!flat_vector) {
    error_reporter->Report("Input array not provided for operation.\n");
  } else {
    int num_dimensions = flat_vector->Length();
    if (num_dimensions > max_size_of_buffer / sizeof(int)) {
      error_reporter->Report(
          "Found too many dimensions in the operation's input array.\n");
    } else {
      for (int i = 0; i < num_dimensions; ++i) {
        buffer[i] = flat_vector->Get(i);
      }
    }
  }
}

// Allocate a structure using C malloc, but make sure the structure is a
// POD structure that doesn't require constructors to run. The reason we do
// this, is that Interpreter's C extension part will take ownership and wants
// to use malloc() and free().
template <class T>
T* MallocPOD() {
  static_assert(std::is_pod<T>::value, "Builtin data structure must be POD.");
  return static_cast<T*>(malloc(sizeof(T)));
}

// Parse the appropriate data out of the op.
//
// This handles builtin data explicitly as there are flatbuffer schemas.
//
// Returns memory that must be feed.
//
// TODO(nupurgarg): Pass in void ** and return TfLiteStatus to ensure program
// crashes if error reporter is called.
void* ParseOpData(const Operator* op, BuiltinOperator op_type,
                  ErrorReporter* error_reporter) {
  auto parse_padding = [](Padding padding) {
    switch (padding) {
      case Padding_SAME:
        return kTfLitePaddingSame;
      case Padding_VALID:
        return kTfLitePaddingValid;
    }
    return kTfLitePaddingUnknown;
  };
  auto parse_activation = [](ActivationFunctionType activation) {
    switch (activation) {
      case ActivationFunctionType_NONE:
        return kTfLiteActNone;
      case ActivationFunctionType_RELU:
        return kTfLiteActRelu;
      case ActivationFunctionType_RELU_N1_TO_1:
        return kTfLiteActRelu1;
      case ActivationFunctionType_RELU6:
        return kTfLiteActRelu6;
      case ActivationFunctionType_TANH:
        return kTfLiteActTanh;
      case ActivationFunctionType_SIGN_BIT:
        return kTfLiteActSignBit;
    }
    return kTfLiteActNone;
  };
  auto parseLSHProjectionType = [](LSHProjectionType type) {
    switch (type) {
      case LSHProjectionType_SPARSE:
        return kTfLiteLshProjectionSparse;
      case LSHProjectionType_DENSE:
        return kTfLiteLshProjectionDense;
      default:
        return kTfLiteLshProjectionUnknown;
    }
  };
  auto parseCombinerType = [](CombinerType type) {
    switch (type) {
      case CombinerType_MEAN:
        return kTfLiteCombinerTypeMean;
      case CombinerType_SQRTN:
        return kTfLiteCombinerTypeSqrtn;
      case CombinerType_SUM:
      default:
        return kTfLiteCombinerTypeSum;
    }
  };

  void* builtin_data = nullptr;
  switch (op_type) {
    case BuiltinOperator_CALL:
      // TODO(aselle): Implement call in BuiltinOptions, but nullptrs are
      // ok for now, since there is no call implementation either.
      break;
    case BuiltinOperator_CUSTOM:
      break;
    case BuiltinOperator_CONV_2D: {
      TfLiteConvParams* params = MallocPOD<TfLiteConvParams>();
      if (auto* conv_params = op->builtin_options_as_Conv2DOptions()) {
        params->padding = parse_padding(conv_params->padding());
        params->stride_width = conv_params->stride_w();
        params->stride_height = conv_params->stride_h();
        params->activation =
            parse_activation(conv_params->fused_activation_function());
      }
      builtin_data = reinterpret_cast<void*>(params);
      break;
    }
    case BuiltinOperator_TANH:
    case BuiltinOperator_LOGISTIC:
    case BuiltinOperator_RELU:
    case BuiltinOperator_RELU_N1_TO_1:
    case BuiltinOperator_RELU6:
    case BuiltinOperator_CONCAT_EMBEDDINGS:
      break;
    case BuiltinOperator_LSH_PROJECTION: {
      TfLiteLSHProjectionParams* params =
          MallocPOD<TfLiteLSHProjectionParams>();
      if (auto* lshParams = op->builtin_options_as_LSHProjectionOptions()) {
        params->type = parseLSHProjectionType(lshParams->type());
      }
      builtin_data = reinterpret_cast<void*>(params);
      break;
    }
    case BuiltinOperator_AVERAGE_POOL_2D:
    case BuiltinOperator_MAX_POOL_2D:
    case BuiltinOperator_L2_POOL_2D: {
      TfLitePoolParams* params = MallocPOD<TfLitePoolParams>();
      if (auto* pool_params = op->builtin_options_as_Pool2DOptions()) {
        params->padding = parse_padding(pool_params->padding());
        params->stride_width = pool_params->stride_w();
        params->stride_height = pool_params->stride_h();
        params->filter_width = pool_params->filter_width();
        params->filter_height = pool_params->filter_height();
        params->activation =
            parse_activation(pool_params->fused_activation_function());
      }
      builtin_data = reinterpret_cast<void*>(params);
      break;
    }
    case BuiltinOperator_DEPTHWISE_CONV_2D: {
      TfLiteDepthwiseConvParams* params =
          MallocPOD<TfLiteDepthwiseConvParams>();
      if (auto* conv_params = op->builtin_options_as_DepthwiseConv2DOptions()) {
        params->padding = parse_padding(conv_params->padding());
        params->stride_width = conv_params->stride_w();
        params->stride_height = conv_params->stride_h();
        params->depth_multiplier = conv_params->depth_multiplier();
        params->activation =
            parse_activation(conv_params->fused_activation_function());
      }
      builtin_data = reinterpret_cast<void*>(params);
      break;
    }
    case BuiltinOperator_SVDF: {
      TfLiteSVDFParams* params = MallocPOD<TfLiteSVDFParams>();
      if (auto* svdf_params = op->builtin_options_as_SVDFOptions()) {
        params->rank = svdf_params->rank();
        params->activation =
            parse_activation(svdf_params->fused_activation_function());
      }
      builtin_data = reinterpret_cast<void*>(params);
      break;
    }
    case BuiltinOperator_BIDIRECTIONAL_SEQUENCE_RNN:
    case BuiltinOperator_UNIDIRECTIONAL_SEQUENCE_RNN: {
      TfLiteSequenceRNNParams* params = MallocPOD<TfLiteSequenceRNNParams>();
      if (auto* sequence_rnn_params =
              op->builtin_options_as_SequenceRNNOptions()) {
        params->activation =
            parse_activation(sequence_rnn_params->fused_activation_function());
        params->time_major = sequence_rnn_params->time_major();
      }
      builtin_data = reinterpret_cast<void*>(params);
      break;
    }
    case BuiltinOperator_RNN: {
      TfLiteRNNParams* params = MallocPOD<TfLiteRNNParams>();
      if (auto* rnn_params = op->builtin_options_as_RNNOptions()) {
        params->activation =
            parse_activation(rnn_params->fused_activation_function());
      }
      builtin_data = reinterpret_cast<void*>(params);
      break;
    }
    case BuiltinOperator_EMBEDDING_LOOKUP:
      // no-op.
      break;
    case BuiltinOperator_EMBEDDING_LOOKUP_SPARSE: {
      TfLiteEmbeddingLookupSparseParams* params =
          MallocPOD<TfLiteEmbeddingLookupSparseParams>();
      if (auto* embedding_params =
              op->builtin_options_as_EmbeddingLookupSparseOptions()) {
        params->combiner = parseCombinerType(embedding_params->combiner());
      }
      builtin_data = reinterpret_cast<void*>(params);
      break;
    }
    case BuiltinOperator_FULLY_CONNECTED: {
      TfLiteFullyConnectedParams* params =
          MallocPOD<TfLiteFullyConnectedParams>();
      if (auto* fully_connected_params =
              op->builtin_options_as_FullyConnectedOptions()) {
        params->activation = parse_activation(
            fully_connected_params->fused_activation_function());
      }
      builtin_data = reinterpret_cast<void*>(params);
      break;
    }
    case BuiltinOperator_HASHTABLE_LOOKUP:
      // no-op.
      break;
    case BuiltinOperator_SOFTMAX: {
      TfLiteSoftmaxParams* params = MallocPOD<TfLiteSoftmaxParams>();
      if (auto* softmax_params = op->builtin_options_as_SoftmaxOptions()) {
        params->beta = softmax_params->beta();
      }
      builtin_data = reinterpret_cast<void*>(params);
      break;
    }
    case BuiltinOperator_CONCATENATION: {
      TfLiteConcatenationParams* params =
          MallocPOD<TfLiteConcatenationParams>();
      if (auto* concatenation_params =
              op->builtin_options_as_ConcatenationOptions()) {
        params->activation =
            parse_activation(concatenation_params->fused_activation_function());
        params->axis = concatenation_params->axis();
      }
      builtin_data = reinterpret_cast<void*>(params);
      break;
    }
    case BuiltinOperator_MUL: {
      auto* params = MallocPOD<TfLiteMulParams>();
      if (auto* schema_params = op->builtin_options_as_MulOptions()) {
        params->activation =
            parse_activation(schema_params->fused_activation_function());
      }
      builtin_data = reinterpret_cast<void*>(params);
      break;
    }
    case BuiltinOperator_ADD: {
      auto* params = MallocPOD<TfLiteAddParams>();
      if (auto* schema_params = op->builtin_options_as_AddOptions()) {
        params->activation =
            parse_activation(schema_params->fused_activation_function());
      }
      builtin_data = reinterpret_cast<void*>(params);
      break;
    }
    case BuiltinOperator_DIV: {
      auto* params = MallocPOD<TfLiteDivParams>();
      if (auto* schema_params = op->builtin_options_as_DivOptions()) {
        params->activation =
            parse_activation(schema_params->fused_activation_function());
      }
      builtin_data = reinterpret_cast<void*>(params);
      break;
    }
    case BuiltinOperator_SUB: {
      auto* params = MallocPOD<TfLiteSubParams>();
      if (auto* schema_params = op->builtin_options_as_SubOptions()) {
        params->activation =
            parse_activation(schema_params->fused_activation_function());
      }
      builtin_data = reinterpret_cast<void*>(params);
      break;
    }
    case BuiltinOperator_L2_NORMALIZATION: {
      auto* params = MallocPOD<TfLiteL2NormParams>();
      if (auto* schema_params = op->builtin_options_as_L2NormOptions()) {
        params->activation =
            parse_activation(schema_params->fused_activation_function());
      }
      builtin_data = reinterpret_cast<void*>(params);
      break;
    }
    case BuiltinOperator_LOCAL_RESPONSE_NORMALIZATION: {
      auto* params = MallocPOD<TfLiteLocalResponseNormParams>();
      if (auto* schema_params =
              op->builtin_options_as_LocalResponseNormalizationOptions()) {
        params->radius = schema_params->radius();
        params->bias = schema_params->bias();
        params->alpha = schema_params->alpha();
        params->beta = schema_params->beta();
      }
      builtin_data = reinterpret_cast<void*>(params);
      break;
    }
    case BuiltinOperator_UNIDIRECTIONAL_SEQUENCE_LSTM:
    case BuiltinOperator_LSTM: {
      TfLiteLSTMParams* params = MallocPOD<TfLiteLSTMParams>();
      if (auto* lstm_params = op->builtin_options_as_LSTMOptions()) {
        params->activation =
            parse_activation(lstm_params->fused_activation_function());
        params->cell_clip = lstm_params->cell_clip();
        params->proj_clip = lstm_params->proj_clip();
      }
      builtin_data = reinterpret_cast<void*>(params);
      break;
    }
    case BuiltinOperator_RESIZE_BILINEAR: {
      auto* params = MallocPOD<TfLiteResizeBilinearParams>();
      if (auto* schema_params =
              op->builtin_options_as_ResizeBilinearOptions()) {
      }
      builtin_data = reinterpret_cast<void*>(params);
      break;
    }
    case BuiltinOperator_PAD: {
      break;
    }
    case BuiltinOperator_RESHAPE: {
      auto* params = MallocPOD<TfLiteReshapeParams>();
      if (auto* schema_params = op->builtin_options_as_ReshapeOptions()) {
        auto* new_shape = schema_params->new_shape();
        FlatBufferIntVectorToArray(sizeof(params->shape), new_shape,
                                   params->shape, error_reporter);
        params->num_dimensions = new_shape->Length();
      }
      builtin_data = reinterpret_cast<void*>(params);
      break;
    }
    case BuiltinOperator_SKIP_GRAM: {
      TfLiteSkipGramParams* params = MallocPOD<TfLiteSkipGramParams>();
      if (auto* skip_gram_params = op->builtin_options_as_SkipGramOptions()) {
        params->ngram_size = skip_gram_params->ngram_size();
        params->max_skip_size = skip_gram_params->max_skip_size();
        params->include_all_ngrams = skip_gram_params->include_all_ngrams();
      }
      builtin_data = reinterpret_cast<void*>(params);
      break;
    }
    case BuiltinOperator_SPACE_TO_DEPTH: {
      auto* params = MallocPOD<TfLiteSpaceToDepthParams>();
      if (auto* schema_params = op->builtin_options_as_SpaceToDepthOptions()) {
        params->block_size = schema_params->block_size();
      }
      builtin_data = reinterpret_cast<void*>(params);
      break;
    }
    case BuiltinOperator_GATHER: {
      TfLiteGatherParams* params = MallocPOD<TfLiteGatherParams>();
      params->axis = 0;
      if (auto* gather_params = op->builtin_options_as_GatherOptions()) {
        params->axis = gather_params->axis();
      }

      builtin_data = reinterpret_cast<void*>(params);
      break;
    }
    case BuiltinOperator_SPACE_TO_BATCH_ND: {
      break;
    }
    case BuiltinOperator_BATCH_TO_SPACE_ND: {
      break;
    }
    case BuiltinOperator_TRANSPOSE: {
      break;
    }
    case BuiltinOperator_MEAN: {
      auto* params = MallocPOD<TfLiteMeanParams>();
      if (auto* schema_params = op->builtin_options_as_MeanOptions()) {
        params->keep_dims = schema_params->keep_dims();
      }
      builtin_data = reinterpret_cast<void*>(params);
      break;
    }
    case BuiltinOperator_SQUEEZE: {
      auto* params = MallocPOD<TfLiteSqueezeParams>();
      if (auto* schema_params = op->builtin_options_as_SqueezeOptions()) {
        const auto& squeeze_dims = schema_params->squeeze_dims();
        FlatBufferIntVectorToArray(sizeof(params->squeeze_dims), squeeze_dims,
                                   params->squeeze_dims, error_reporter);
        params->num_squeeze_dims = squeeze_dims->Length();
      }
      builtin_data = reinterpret_cast<void*>(params);
      break;
    }
    case BuiltinOperator_STRIDED_SLICE: {
      auto* params = MallocPOD<TfLiteStridedSliceParams>();
      if (auto* schema_params = op->builtin_options_as_StridedSliceOptions()) {
        params->begin_mask = schema_params->begin_mask();
        params->end_mask = schema_params->end_mask();
        params->ellipsis_mask = schema_params->ellipsis_mask();
        params->new_axis_mask = schema_params->new_axis_mask();
        params->shrink_axis_mask = schema_params->shrink_axis_mask();
      }
      builtin_data = reinterpret_cast<void*>(params);
      break;
    }
  }
  return builtin_data;
}

}  // namespace

void initOpenCL() {
  //OpenCL init
  //note: andoird log
  // __android_log_print(ANDROID_LOG_INFO, "Ngising", "masuk initOpenCL sekali");

  cl_int err;

  err = clGetPlatformIDs(1, &cpPlatform, NULL);
 
  err = clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);

  context_cl = clCreateContext(0, 1, &device_id, NULL, NULL, &err);

  queueCL = clCreateCommandQueue(context_cl, device_id, 0, &err);

  program = clCreateProgramWithSource(context_cl, 1,
                          (const char **) & kernelSource, NULL, &err);

  clBuildProgram(program, 0, NULL, NULL, NULL, NULL);

  cl_kernel kernel_transpose = clCreateKernel(program, "transpose", &err);
  cl_kernel kernel_conv = clCreateKernel(program, "conv", &err);
  cl_kernel kernel_matrixVectorMul = clCreateKernel(program, "matrixVectorMul", &err);
  size_t prefWorkGroupSize1, prefWorkGroupSize2, prefWorkGroupSize3;
  size_t maxWorkGroupSize1, maxWorkGroupSize2, maxWorkGroupSize3;
  clGetKernelWorkGroupInfo(kernel_transpose,
    device_id,
    CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE,
    sizeof(size_t),
    &prefWorkGroupSize1,
    NULL);
  clGetKernelWorkGroupInfo(kernel_conv,
    device_id,
    CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE,
    sizeof(size_t),
    &prefWorkGroupSize2,
    NULL);
  clGetKernelWorkGroupInfo(kernel_matrixVectorMul,
    device_id,
    CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE,
    sizeof(size_t),
    &prefWorkGroupSize3,
    NULL);
  //note: andoird log
  __android_log_print(ANDROID_LOG_INFO, "Ngising", "Workgroup transpose: %d",prefWorkGroupSize1);
  __android_log_print(ANDROID_LOG_INFO, "Ngising", "Workgroup conv: %d",prefWorkGroupSize2);
  __android_log_print(ANDROID_LOG_INFO, "Ngising", "Workgroup matrixVectorMul: %d",prefWorkGroupSize3);
  clGetKernelWorkGroupInfo(kernel_transpose,
    device_id,
    CL_KERNEL_WORK_GROUP_SIZE,
    sizeof(size_t),
    &maxWorkGroupSize1,
    NULL);
  clGetKernelWorkGroupInfo(kernel_conv,
    device_id,
    CL_KERNEL_WORK_GROUP_SIZE,
    sizeof(size_t),
    &maxWorkGroupSize2,
    NULL);
  clGetKernelWorkGroupInfo(kernel_matrixVectorMul,
    device_id,
    CL_KERNEL_WORK_GROUP_SIZE,
    sizeof(size_t),
    &maxWorkGroupSize3,
    NULL);

  // if(d_input == NULL) {
  //   __android_log_print(ANDROID_LOG_INFO, "Convruntime", "runkernelmasuksekali");    
      
  // kernel = clCreateKernel(program, "convhalf", NULL);

  d_conv_input = clCreateBuffer(context_cl, CL_MEM_READ_ONLY, buffsizes[0]*sizeof(float), NULL, NULL);
  d_conv_filter = clCreateBuffer(context_cl, CL_MEM_READ_ONLY, buffsizes[1]*sizeof(float), NULL, NULL);
  d_conv_bias = clCreateBuffer(context_cl, CL_MEM_READ_ONLY, buffsizes[2]*sizeof(float), NULL, NULL);
  d_conv_output = clCreateBuffer(context_cl, CL_MEM_WRITE_ONLY, buffsizes[3]*sizeof(float), NULL, NULL);
  d_conv_dim_sizes = clCreateBuffer(context_cl, CL_MEM_READ_ONLY, 16*sizeof(int), NULL, NULL);
  d_conv_dim_strides = clCreateBuffer(context_cl, CL_MEM_READ_ONLY, 16*sizeof(int), NULL, NULL);

  // }
  //note: andoird log
  // __android_log_print(ANDROID_LOG_INFO, "Ngising", "MAX Workgroup transpose: %d",maxWorkGroupSize1);
  // __android_log_print(ANDROID_LOG_INFO, "Ngising", "MAX Workgroup conv: %d",maxWorkGroupSize2);
  // __android_log_print(ANDROID_LOG_INFO, "Ngising", "MAX Workgroup matrixVectorMul: %d",maxWorkGroupSize3);
}

//Vulkan
VkInstance instance;
VkPhysicalDevice physicalDevice;

VkDevice device;
VkPipeline pipelineConv;
VkPipeline pipelineMatmul;
VkPipelineLayout pipelineLayoutMatmul;
VkPipelineLayout pipelineLayoutConv;
VkShaderModule matmulShaderModule;
VkShaderModule convShaderModule;
VkDescriptorSetLayout descriptorSetLayoutMatmul;
VkDescriptorSetLayout descriptorSetLayoutConv;
VkQueue queue; 
uint32_t queueFamilyIndex = 0;

void createInstance() {
    VkApplicationInfo applicationInfo = {};
    applicationInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    applicationInfo.pApplicationName = "VulkanDeepLearning";
    applicationInfo.applicationVersion = 0;
    applicationInfo.pEngineName = "Naive";
    applicationInfo.engineVersion = 0;
    applicationInfo.apiVersion = VK_MAKE_VERSION(1, 0, 65);
    
    VkInstanceCreateInfo createInfo = {};
    createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    createInfo.flags = 0;
    createInfo.pApplicationInfo = &applicationInfo;

    VK_CHECK_RESULT(vkCreateInstance(
        &createInfo,
        NULL,
        &instance));
}

void findPhysicalDevice() {
    uint32_t deviceCount;
    vkEnumeratePhysicalDevices(instance, &deviceCount, NULL);
    if (deviceCount == 0) {
        throw std::runtime_error("could not find a device with vulkan support");
    }

    std::vector<VkPhysicalDevice> devices(deviceCount);
    vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());

    for (VkPhysicalDevice device : devices) {
        if (true) { // As above stated, we do no feature checks, so just accept.
            physicalDevice = device;
            break;
        }
    }
}

uint32_t getComputeQueueFamilyIndex() {
    uint32_t queueFamilyCount;

    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, NULL);

    std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, queueFamilies.data());

    uint32_t i = 0;
    for (; i < queueFamilies.size(); ++i) {
        VkQueueFamilyProperties props = queueFamilies[i];

        if (props.queueCount > 0 && (props.queueFlags & VK_QUEUE_COMPUTE_BIT)) {
            break;
        }
    }

    if (i == queueFamilies.size()) {
        throw std::runtime_error("could not find a queue family that supports operations");
    }

    return i;
}

void createDevice() {
    VkDeviceQueueCreateInfo queueCreateInfo = {};
    queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queueFamilyIndex = getComputeQueueFamilyIndex(); // find queue family with compute capability.
    queueCreateInfo.queueFamilyIndex = queueFamilyIndex;
    queueCreateInfo.queueCount = 1; // create one queue in this family. We don't need more.
    float queuePriorities = 1.0;  // we only have one queue, so this is not that imporant. 
    queueCreateInfo.pQueuePriorities = &queuePriorities;

    VkDeviceCreateInfo deviceCreateInfo = {};

    VkPhysicalDeviceFeatures deviceFeatures = {};

    deviceCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    deviceCreateInfo.pQueueCreateInfos = &queueCreateInfo; // when creating the logical device, we also specify what queues it has.
    deviceCreateInfo.queueCreateInfoCount = 1;
    deviceCreateInfo.pEnabledFeatures = &deviceFeatures;

    VK_CHECK_RESULT(vkCreateDevice(physicalDevice, &deviceCreateInfo, NULL, &device)); // create logical device.

    vkGetDeviceQueue(device, queueFamilyIndex, 0, &queue);
}

uint32_t findMemoryType(VkDeviceSize memorySize, VkMemoryPropertyFlags properties) {
    VkPhysicalDeviceMemoryProperties memoryProperties;

    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memoryProperties);

    for (uint32_t i = 0; i < memoryProperties.memoryTypeCount; ++i) {
        if ((memorySize < memoryProperties.memoryHeaps[memoryProperties.memoryTypes[i].heapIndex].size) &&
            ((memoryProperties.memoryTypes[i].propertyFlags & properties) == properties))
            return i;
    }
    return -1;
}

void createDescriptorSetLayoutMatmul() {
    VkDescriptorSetLayoutBinding descriptorSetLayoutBinding;

    descriptorSetLayoutBinding = {};
    descriptorSetLayoutBinding.binding = 0; // binding = 0
    descriptorSetLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptorSetLayoutBinding.descriptorCount = 1;
    descriptorSetLayoutBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCreateInfo = {};
    descriptorSetLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    descriptorSetLayoutCreateInfo.bindingCount = 1; // only a single binding in this descriptor set layout. 
    descriptorSetLayoutCreateInfo.pBindings = &descriptorSetLayoutBinding; 

    VK_CHECK_RESULT(vkCreateDescriptorSetLayout(device, &descriptorSetLayoutCreateInfo, NULL, &descriptorSetLayoutMatmul));
}

void createDescriptorSetLayoutConv() {
    VkDescriptorSetLayoutBinding descriptorSetLayoutBindings[2];

    descriptorSetLayoutBindings[0] = {};
    descriptorSetLayoutBindings[0].binding = 0; // binding = 0
    descriptorSetLayoutBindings[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptorSetLayoutBindings[0].descriptorCount = 1;
    descriptorSetLayoutBindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    descriptorSetLayoutBindings[1] = {};
    descriptorSetLayoutBindings[1].binding = 1; // binding = 1
    descriptorSetLayoutBindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptorSetLayoutBindings[1].descriptorCount = 1;
    descriptorSetLayoutBindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCreateInfo = {};
    descriptorSetLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    descriptorSetLayoutCreateInfo.bindingCount = 2; // only a single binding in this descriptor set layout. 
    descriptorSetLayoutCreateInfo.pBindings = descriptorSetLayoutBindings; 

    VK_CHECK_RESULT(vkCreateDescriptorSetLayout(device, &descriptorSetLayoutCreateInfo, NULL, &descriptorSetLayoutConv));
}

void createConvPipeline() {
    std::string source =
    "#version 450  \n" \
    "#extension GL_ARB_separate_shader_objects : enable  \n" \
    "layout(local_size_x = 8, local_size_y = 8, local_size_z = 8) in;  \n" \
    "layout(binding = 0) buffer floatBuffer {  \n" \
    "    float actMin;  \n" \
    "    float actMax;  \n" \
    "    float pad1;  \n" \
    "    float pad2;  \n" \
    "    float convFloatB[];  \n" \
    "};  \n" \
    "layout(binding = 1) readonly buffer intBuffer {  \n" \
    "    ivec4 stridePad;  \n" \
    "    ivec4 dimSizes[4];  \n" \
    "    ivec4 dimStrides[4];  \n" \
    "    ivec4 ifboSize;  \n" \
    "};  \n" \
    "void main() {  \n" \
    "    int out_channel = int(gl_GlobalInvocationID.x);   \n" \
    "    int out_y = int(gl_GlobalInvocationID.y);   \n" \
    "    int out_x = int(gl_GlobalInvocationID.z);   \n" \
    "    if((out_channel < dimSizes[1].w) && (out_x < dimSizes[3].y) && (out_y < dimSizes[3].z)) {  \n" \
    "      for(int batch = 0; batch < dimSizes[0].w; ++batch) { \n" \
    "        float total = 0.0;  \n" \
    "        for (int filter_y = 0; filter_y < dimSizes[1].z; ++filter_y) {  \n" \
    "          for (int filter_x = 0; filter_x < dimSizes[1].y; ++filter_x) {  \n" \
    "            for (int in_channel = 0; in_channel < dimSizes[0].x; ++in_channel) {  \n" \
    "              int in_x = (out_x * stridePad.x - stridePad.z) + filter_x;  \n" \
    "              int in_y = (out_y * stridePad.y - stridePad.w) + filter_y;  \n" \
    "              if ((in_x >= 0) && (in_x < dimSizes[0].y) && (in_y >= 0) &&  \n" \
    "                  (in_y < dimSizes[0].z)) {  \n" \
    "                total += (convFloatB[in_channel*dimStrides[0].x + in_x*dimStrides[0].y +in_y*dimStrides[0].z + batch*dimStrides[0].w] *   \n" \
    "                        convFloatB[ifboSize.x + in_channel*dimStrides[1].x + filter_x*dimStrides[1].y + filter_y*dimStrides[1].z + out_channel*dimStrides[1].w]);  \n" \
    "              }  \n" \
    "            }  \n" \
    "          }  \n" \
    "        }  \n" \
    "        float bias_value = 0.0;  \n" \
    "        if (ifboSize.z > 0) {  \n" \
    "          bias_value = convFloatB[ifboSize.x + ifboSize.y + (out_channel*dimStrides[2].x)];  \n" \
    "        }  \n" \
    "        convFloatB[ifboSize.x + ifboSize.y + ifboSize.z + out_channel*dimStrides[3].x + out_x*dimStrides[3].y + out_y*dimStrides[3].z + batch*dimStrides[3].w] = min(max(total + bias_value,actMin),actMax);  \n" \
    "      }  \n" \
    "    }  \n" \
    "}";

    //     std::string source =
    // "#version 450  \n" \
    // "#extension GL_ARB_separate_shader_objects : enable  \n" \
    // "layout(local_size_x = 8, local_size_y = 8, local_size_z = 8) in;  \n" \
    // "layout(binding = 0) buffer floatBuffer {  \n" \
    // "    float actMin;  \n" \
    // "    float actMax;  \n" \
    // "    vec4 convFloatB[];  \n" \
    // "};  \n" \
    // "layout(binding = 1) readonly buffer intBuffer {  \n" \
    // "    ivec4 stridePad;  \n" \
    // "    ivec4 dimSizes[4];  \n" \
    // "    ivec4 dimStrides[4];  \n" \
    // "    ivec4 ifboSize;  \n" \
    // "};  \n" \
    // "void main() {  \n" \
    // "    int out_channel = int(gl_GlobalInvocationID.x);   \n" \
    // "    int out_y = int(gl_GlobalInvocationID.y);   \n" \
    // "    int out_x = int(gl_GlobalInvocationID.z);   \n" \
    // "    if((out_channel < dimSizes[1].w) && (out_x < dimSizes[3].y) && (out_y < dimSizes[3].z)) {  \n" \
    // "      for(int batch = 0; batch < dimSizes[0].w; ++batch) { \n" \
    // "        float total = 0.0;  \n" \
    // "        for (int filter_y = 0; filter_y < dimSizes[1].z; ++filter_y) {  \n" \
    // "          for (int filter_x = 0; filter_x < dimSizes[1].y; ++filter_x) {  \n" \
    // "            for (int in_channel = 0; in_channel < dimSizes[0].x/4; ++in_channel) {  \n" \
    // "              int in_x = (out_x * stridePad.x - stridePad.z) + filter_x;  \n" \
    // "              int in_y = (out_y * stridePad.y - stridePad.w) + filter_y;  \n" \
    // "              if ((in_x >= 0) && (in_x < dimSizes[0].y) && (in_y >= 0) &&  \n" \
    // "                  (in_y < dimSizes[0].z)) {  \n" \
    // "                vec4 input_data = convFloatB[in_channel*dimStrides[0].x + in_x*dimStrides[0].y/4 +in_y*dimStrides[0].z/4 + batch*dimStrides[0].w/4]; \n" \
    // "                vec4 filter_data = convFloatB[ifboSize.x/4 + in_channel*dimStrides[1].x + filter_x*dimStrides[1].y/4 + filter_y*dimStrides[1].z/4 + out_channel*dimStrides[1].w/4]; \n" \
    // "                total += dot(input_data, filter_data); \n" \
    // "              }  \n" \
    // "            }  \n" \
    // "          }  \n" \
    // "        }  \n" \
    // "        float bias_value = 0.0;  \n" \
    // "        int tmp = (out_channel*dimStrides[2].x); \n" \
    // "        if (ifboSize.z > 0) {  \n" \
    // "          switch (int(mod(tmp,4))) {      \n" \
    // "              case 0: bias_value = convFloatB[ifboSize.x/4 + ifboSize.y/4 + tmp/4].x; break;      \n" \
    // "              case 1: bias_value = convFloatB[ifboSize.x/4 + ifboSize.y/4 + tmp/4].y; break;      \n" \
    // "              case 2: bias_value = convFloatB[ifboSize.x/4 + ifboSize.y/4 + tmp/4].z; break;      \n" \
    // "              case 3: bias_value = convFloatB[ifboSize.x/4 + ifboSize.y/4 + tmp/4].w; break;      \n" \
    // "          }      \n" \
    // "        }  \n" \
    // "        tmp = (out_channel*dimStrides[3].x + out_x*dimStrides[3].y + out_y*dimStrides[3].z + batch*dimStrides[3].w); \n" \
    // "        switch (int(mod(tmp,4))) {      \n" \
    // "            case 0: convFloatB[ifboSize.x/4 + ifboSize.y/4 + ifboSize.z/4 + tmp/4].x = min(max(total + bias_value,actMin),actMax); break;      \n" \
    // "            case 1: convFloatB[ifboSize.x/4 + ifboSize.y/4 + ifboSize.z/4 + tmp/4].y = min(max(total + bias_value,actMin),actMax); break;      \n" \
    // "            case 2: convFloatB[ifboSize.x/4 + ifboSize.y/4 + ifboSize.z/4 + tmp/4].z = min(max(total + bias_value,actMin),actMax); break;      \n" \
    // "            case 3: convFloatB[ifboSize.x/4 + ifboSize.y/4 + ifboSize.z/4 + tmp/4].w = min(max(total + bias_value,actMin),actMax); break;      \n" \
    // "        }      \n" \
    // "      }  \n" \
    // "    }  \n" \
    // "}";

    shaderc::Compiler compiler;
    shaderc::CompileOptions options;

    shaderc::SpvCompilationResult module = compiler.CompileGlslToSpv(
      source.c_str(), source.size(), shaderc_glsl_compute_shader, "conv.glsl", options);

    if (module.GetCompilationStatus() !=
        shaderc_compilation_status_success) {
    }

    std::vector<uint32_t> code(module.cbegin(), module.cend());

    VkShaderModuleCreateInfo createInfo = {};
    createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    createInfo.pCode = code.data();
    createInfo.codeSize = sizeof(uint32_t)*code.size();

    VK_CHECK_RESULT(vkCreateShaderModule(device, &createInfo, NULL, &convShaderModule));

    VkPipelineShaderStageCreateInfo shaderStageCreateInfo = {};
    shaderStageCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    shaderStageCreateInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    shaderStageCreateInfo.module = convShaderModule;
    shaderStageCreateInfo.pName = "main";

    VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = {};
    pipelineLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutCreateInfo.setLayoutCount = 1;
    pipelineLayoutCreateInfo.pSetLayouts = &descriptorSetLayoutConv; 
    
    VK_CHECK_RESULT(vkCreatePipelineLayout(device, &pipelineLayoutCreateInfo, NULL, &pipelineLayoutConv));

    VkComputePipelineCreateInfo pipelineCreateInfo = {};
    pipelineCreateInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipelineCreateInfo.stage = shaderStageCreateInfo;
    pipelineCreateInfo.layout = pipelineLayoutConv;

    VK_CHECK_RESULT(vkCreateComputePipelines(
        device, VK_NULL_HANDLE,
        1, &pipelineCreateInfo,
        NULL, &pipelineConv));
}

void createMatmulPipeline() {
    // std::string source =
    //   "#version 450 \n" \
    //   "#extension GL_ARB_separate_shader_objects : enable \n" \
    //   "layout(local_size_x = 32, local_size_y = 8, local_size_z = 1) in; \n" \
    //   "layout(binding = 0) buffer matrixA { \n" \
    //   "    vec4 aA[]; \n" \
    //   "}; \n" \
    //   "void main() { \n" \
    //   "    int row = int(gl_GlobalInvocationID.x)*4; \n" \
    //   "    int col = int(gl_GlobalInvocationID.y); \n" \
    //   "    vec4 mk = aA[0];  \n" \
    //   "    int mM = int(mk.x);  \n" \
    //   "    int kK = int(mk.y);  \n" \
    //   "    int nN = int(mk.z);  \n" \
    //   "    if ((row < mM) && (col < nN)) { \n" \
    //   "        vec4 sum = {0.0, 0.0, 0.0, 0.0}; \n" \
    //   "        for(int i = 1; i <= kK/4; i++){ \n" \
    //   "            vec4 currb = aA[(mM*kK/4) + i];\n" \
    //   "            sum.x += dot(aA[(row*kK/4) + i],currb);\n" \
    //   "            sum.y += dot(aA[((row+1)*kK/4) + i],currb); \n" \
    //   "            sum.z += dot(aA[((row+2)*kK/4) + i],currb);\n" \
    //   "            sum.w += dot(aA[((row+3)*kK/4) + i],currb);\n" \
    //   "        } \n" \
    //   "        aA[1 + (mM*kK/4) + (kK/4) + (row/4)] = sum; \n" \
    //   "    } \n" \
    //   "}";

    std::string source =
      "#version 450 \n" \
      "#extension GL_ARB_separate_shader_objects : enable \n" \
      "layout(local_size_x = 8, local_size_y = 32, local_size_z = 1) in; \n" \
      "layout(binding = 0) buffer matrixA { \n" \
      "    int mM; \n" \
      "    int kK; \n" \
      "    int nN; \n" \
      "    int tmp; \n" \
      "    vec4 aA[]; \n" \
      "}; \n" \
      "shared vec4 Acc[32]; \n" \
      "void main() { \n" \
      "    int row = int(gl_GlobalInvocationID.x)*4; \n" \
      "    int localidx = int(gl_LocalInvocationID.y); \n" \
      "    if (row < mM) { \n" \
      "        vec4 sum = {0.0, 0.0, 0.0, 0.0}; \n" \
      "        int starti = localidx*(kK/128); \n" \
      "        for(int i = starti; i < starti+(kK/128); i++) { \n" \
      "            vec4 currb = aA[(mM*kK/4) + i];\n" \
      "            sum.x += dot(aA[(row*kK/4) + i],currb);\n" \
      "            sum.y += dot(aA[((row+1)*kK/4) + i],currb); \n" \
      "            sum.z += dot(aA[((row+2)*kK/4) + i],currb);\n" \
      "            sum.w += dot(aA[((row+3)*kK/4) + i],currb);\n" \
      "        } \n" \
      "        Acc[localidx] = sum; \n" \
      "        barrier();      \n" \
      "        if(localidx == 0) { \n" \
      "            aA[(mM*kK/4) + (kK/4) + (row/4)] = Acc[0] + Acc[1] + Acc[2] + Acc[3] + Acc[4] + Acc[5] + Acc[6] + Acc[7]+ \n" \
      "            Acc[8] + Acc[9] + Acc[10] + Acc[11] + Acc[12] + Acc[13] + Acc[14] + Acc[15]+ \n" \
      "            Acc[16] + Acc[17] + Acc[18] + Acc[19] + Acc[20] + Acc[21] + Acc[22] + Acc[23]+ \n" \
      "            Acc[24] + Acc[25] + Acc[26] + Acc[27] + Acc[28] + Acc[29] + Acc[30] + Acc[31]; \n" \
      "        } \n" \
      "    } \n" \
      "}";

    shaderc::Compiler compiler;
    shaderc::CompileOptions options;

    shaderc::SpvCompilationResult module = compiler.CompileGlslToSpv(
      source.c_str(), source.size(), shaderc_glsl_compute_shader, "matmul.glsl", options);

    if (module.GetCompilationStatus() !=
        shaderc_compilation_status_success) {
    }

    std::vector<uint32_t> code(module.cbegin(), module.cend());

    VkShaderModuleCreateInfo createInfo = {};
    createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    createInfo.pCode = code.data();
    createInfo.codeSize = sizeof(uint32_t)*code.size();
    
    VK_CHECK_RESULT(vkCreateShaderModule(device, &createInfo, NULL, &matmulShaderModule));

    VkPipelineShaderStageCreateInfo shaderStageCreateInfo = {};
    shaderStageCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    shaderStageCreateInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    shaderStageCreateInfo.module = matmulShaderModule;
    shaderStageCreateInfo.pName = "main";

    VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = {};
    pipelineLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutCreateInfo.setLayoutCount = 1;
    pipelineLayoutCreateInfo.pSetLayouts = &descriptorSetLayoutMatmul; 
    
    VK_CHECK_RESULT(vkCreatePipelineLayout(device, &pipelineLayoutCreateInfo, NULL, &pipelineLayoutMatmul));

    VkComputePipelineCreateInfo pipelineCreateInfo = {};
    pipelineCreateInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipelineCreateInfo.stage = shaderStageCreateInfo;
    pipelineCreateInfo.layout = pipelineLayoutMatmul;

    VK_CHECK_RESULT(vkCreateComputePipelines(
        device, VK_NULL_HANDLE,
        1, &pipelineCreateInfo,
        NULL, &pipelineMatmul));
}

void createConvBuffer() {
    uint32_t matrixASize = (uint32_t) (sizeof(float) *buffsizes[0]);
    uint32_t matrixBSize = (uint32_t) (sizeof(float) *(buffsizes[1] + buffsizes[2] + 4));
    uint32_t matrixCSize = (uint32_t) (sizeof(float) * buffsizes[3]);
    uint32_t matrixSizesSize = sizeof(int) * 40;

    VkBufferCreateInfo matrixACreateInfo = {};
    matrixACreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    matrixACreateInfo.size = matrixASize+matrixBSize+matrixCSize; // buffer size in bytes. 
    matrixACreateInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT; // buffer is used as a storage buffer.
    matrixACreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE; // buffer is exclusive to a single queue family at a time. 

    VK_CHECK_RESULT(vkCreateBuffer(device, &matrixACreateInfo, NULL, &conv_matrixA)); // create buffer.

    VkBufferCreateInfo matrixSizesCreateInfo = {};
    matrixSizesCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    matrixSizesCreateInfo.size = matrixSizesSize; // buffer size in bytes. 
    matrixSizesCreateInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT; // buffer is used as a storage buffer.
    matrixSizesCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE; // buffer is exclusive to a single queue family at a time. 

    VK_CHECK_RESULT(vkCreateBuffer(device, &matrixSizesCreateInfo, NULL, &conv_matrixSizes)); // create buffer.
    
    VkMemoryRequirements memoryRequirementsmatrixA, memoryRequirementsmatrixSizes;
    vkGetBufferMemoryRequirements(device, conv_matrixA, &memoryRequirementsmatrixA);
    vkGetBufferMemoryRequirements(device, conv_matrixSizes, &memoryRequirementsmatrixSizes);

    const VkDeviceSize memorySize = memoryRequirementsmatrixA.size+memoryRequirementsmatrixSizes.size;

    VkMemoryAllocateInfo allocateInfo = {};
    allocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocateInfo.allocationSize = memorySize; // specify required memory.

    allocateInfo.memoryTypeIndex = findMemoryType(
        memorySize, VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);

    VK_CHECK_RESULT(vkAllocateMemory(device, &allocateInfo, NULL, &conv_bufferMemory));

    VK_CHECK_RESULT(vkBindBufferMemory(device, conv_matrixA, conv_bufferMemory, 0));
    VK_CHECK_RESULT(vkBindBufferMemory(device, conv_matrixSizes, conv_bufferMemory, matrixASize+matrixBSize+matrixCSize));

    VkCommandPoolCreateInfo commandPoolCreateInfo = {};
    commandPoolCreateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    commandPoolCreateInfo.flags = 0;
    commandPoolCreateInfo.queueFamilyIndex = queueFamilyIndex;
    VK_CHECK_RESULT(vkCreateCommandPool(device, &commandPoolCreateInfo, NULL, &conv_commandPool));

    VkCommandBufferAllocateInfo commandBufferAllocateInfo = {};
    commandBufferAllocateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    commandBufferAllocateInfo.commandPool = conv_commandPool;

    commandBufferAllocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    commandBufferAllocateInfo.commandBufferCount = 1; // allocate a single command buffer. 
    VK_CHECK_RESULT(vkAllocateCommandBuffers(device, &commandBufferAllocateInfo, &conv_commandBuffer));
}

void initVulkan() {
    createInstance();
    __android_log_print(ANDROID_LOG_INFO, "VulkanInit", "createInstance");
    findPhysicalDevice();
    __android_log_print(ANDROID_LOG_INFO, "VulkanInit", "findPhysicalDevice");
    createDevice();
    __android_log_print(ANDROID_LOG_INFO, "VulkanInit", "createDevice");
    createDescriptorSetLayoutMatmul();
    __android_log_print(ANDROID_LOG_INFO, "VulkanInit", "createDescriptorSetLayoutMatmul");
    createDescriptorSetLayoutConv();
    __android_log_print(ANDROID_LOG_INFO, "VulkanInit", "createDescriptorSetLayoutConv");
    createMatmulPipeline();
    __android_log_print(ANDROID_LOG_INFO, "VulkanInit", "createMatmulPipeline");
    createConvPipeline();
    __android_log_print(ANDROID_LOG_INFO, "VulkanInit", "createConvPipeline");
    createConvBuffer();
    __android_log_print(ANDROID_LOG_INFO, "VulkanInit", "createConvBuffer");
}

TfLiteStatus InterpreterBuilder::ParseNodes(
    const flatbuffers::Vector<flatbuffers::Offset<Operator>>* operators,
    Interpreter* interpreter) {
  TfLiteStatus status = kTfLiteOk;
  //note: andoird log
  // __android_log_print(ANDROID_LOG_INFO, "Ngising", "addnodewithparam");
  // initOpenCL();
  initVulkan();
  for (int i = 0; i < operators->Length(); ++i) {
    const auto* op = operators->Get(i);
    int index = op->opcode_index();
    if (index < 0 || index >= flatbuffer_op_index_to_registration_.size()) {
      error_reporter_->Report("Missing registration for opcode_index %d\n",
                              index);
      status = kTfLiteError;
      continue;
    }
    const TfLiteRegistration* reg =
        flatbuffer_op_index_to_registration_[op->opcode_index()];
    if (reg == nullptr) {
      error_reporter_->Report("Skipping op for opcode_index %d\n", index);
      status = kTfLiteError;
      continue;
    }

    auto op_type =
        flatbuffer_op_index_to_registration_types_[op->opcode_index()];
    if (op_type != BuiltinOperator_CUSTOM && op->custom_options()) {
      error_reporter_->Report(
          "Found builtin operator %s with custom options.\n",
          EnumNameBuiltinOperator(op_type));
    }

    // operators code: 3 convolution, 9 fully connected
    // __android_log_print(ANDROID_LOG_INFO, "Ngising", "operators %u", op_type);
    if((op_type == 9) || (op_type == 3)) {
      //note: andoird log
      // __android_log_print(ANDROID_LOG_INFO, "Ngising", "code 25 benar");
      cl_mem cl_mem_arr[6] = {d_conv_input,d_conv_filter,d_conv_bias,d_conv_output,d_conv_dim_sizes,d_conv_dim_strides};
      if (op->custom_options()) {
        // std::vector<int> vectmp = FlatBufferIntArrayToVector(op->inputs());
        // std::vector<int> vectmp2 = FlatBufferIntArrayToVector(op->outputs());
        // __android_log_print(ANDROID_LOG_INFO, "VectorSize", "InputSizeModel.cc: %d, %d, %d", vectmp[0], vectmp[1], vectmp[2]);
        // __android_log_print(ANDROID_LOG_INFO, "VectorSize", "OutputSizeModel.cc: %d", vectmp2[0]);
        interpreter->AddNodeWithParametersOpenCL(
            FlatBufferIntArrayToVector(op->inputs()),
            FlatBufferIntArrayToVector(op->outputs()),
            reinterpret_cast<const char*>(op->custom_options()->data()),
            op->custom_options()->size(), nullptr, reg,
            context_cl, queueCL, program, cl_mem_arr,
            physicalDevice, device, pipelineConv, pipelineMatmul, pipelineLayoutConv, pipelineLayoutMatmul, 
            descriptorSetLayoutConv, descriptorSetLayoutMatmul, queue, queueFamilyIndex,
            conv_commandPool, conv_commandBuffer, conv_matrixA, conv_matrixSizes, conv_bufferMemory);
      } else {
        //note: andoird log
        // __android_log_print(ANDROID_LOG_INFO, "Ngising", "addnodewithparam2");
        // std::vector<int> vectmp = FlatBufferIntArrayToVector(op->inputs());
        // std::vector<int> vectmp2 = FlatBufferIntArrayToVector(op->outputs());
        // __android_log_print(ANDROID_LOG_INFO, "VectorSize", "InputSizeModel.cc: %d, %d, %d", vectmp[0], vectmp[1], vectmp[2]);
        // __android_log_print(ANDROID_LOG_INFO, "VectorSize", "OutputSizeModel.cc: %d", vectmp[2]);
        interpreter->AddNodeWithParametersOpenCL(
            FlatBufferIntArrayToVector(op->inputs()),
            FlatBufferIntArrayToVector(op->outputs()), nullptr, 0,
            ParseOpData(op, op_type, error_reporter_), reg,
            context_cl, queueCL, program, cl_mem_arr,
            physicalDevice, device, pipelineConv, pipelineMatmul, pipelineLayoutConv, pipelineLayoutMatmul, 
            descriptorSetLayoutConv, descriptorSetLayoutMatmul, queue, queueFamilyIndex,
            conv_commandPool, conv_commandBuffer, conv_matrixA, conv_matrixSizes, conv_bufferMemory);
      }
    }
    else {
      if (op->custom_options()) {
        //note: andoird log
        // __android_log_print(ANDROID_LOG_INFO, "Ngising", "addnodewithparam1");
        interpreter->AddNodeWithParameters(
            FlatBufferIntArrayToVector(op->inputs()),
            FlatBufferIntArrayToVector(op->outputs()),
            reinterpret_cast<const char*>(op->custom_options()->data()),
            op->custom_options()->size(), nullptr, reg);
      } else {
        //note: andoird log
        // __android_log_print(ANDROID_LOG_INFO, "Ngising", "addnodewithparam2");
        interpreter->AddNodeWithParameters(
            FlatBufferIntArrayToVector(op->inputs()),
            FlatBufferIntArrayToVector(op->outputs()), nullptr, 0,
            ParseOpData(op, op_type, error_reporter_), reg);
      }
    }
  }

  return status;
}

TfLiteStatus InterpreterBuilder::ParseTensors(
    const flatbuffers::Vector<flatbuffers::Offset<Buffer>>* buffers,
    const flatbuffers::Vector<flatbuffers::Offset<Tensor>>* tensors,
    Interpreter* interpreter) {
  TfLiteStatus status = kTfLiteOk;

  // A little helper to get the names of inputs and outputs. Note that they
  // must outlive the interpreter.
  auto get_name = [](const tflite::Tensor* t) -> const char* {
    auto name = t->name();
    if (name) return name->c_str();
    return kEmptyTensorName;
  };

  for (int i = 0; i < tensors->Length(); ++i) {
    const auto* tensor = tensors->Get(i);
    std::vector<int> dims = FlatBufferIntArrayToVector(tensor->shape());

    // __android_log_print(ANDROID_LOG_INFO, "VectorSize", "Tensorname%d: %s", i, get_name(tensor));

    for(int j = 0; j < dims.size(); j++) {
      // __android_log_print(ANDROID_LOG_INFO, "VectorSize", "Tensordims%d: %d", i, dims[j]);
    }
    

    TfLiteQuantizationParams quantization;
    quantization.scale = 0;
    quantization.zero_point = 0;
    auto* q_params = tensor->quantization();
    if (q_params) {
      // Note that the schema could hold per-channel quantization parameters
      // but we really only support one value for the whole tensor.
      // TODO(aselle): This breaks as well if these are nullptr's.
      // TODO(aselle): This assumes non per-channel quantization.
      if (q_params->scale()) quantization.scale = q_params->scale()->Get(0);
      if (q_params->zero_point())
        quantization.zero_point = q_params->zero_point()->Get(0);
    }

    TfLiteType type;
    switch (tensor->type()) {
      case TensorType_FLOAT32:
        type = kTfLiteFloat32;
        break;
      case TensorType_INT32:
        type = kTfLiteInt32;
        break;
      case TensorType_UINT8:
        type = kTfLiteUInt8;
        break;
      case TensorType_INT64:
        type = kTfLiteInt64;
        break;
      case TensorType_STRING:
        type = kTfLiteString;
        break;
      default:
        // tensorType = ArrayType::NONE;
        error_reporter_->Report("Unimplemented data type %s (%d) in tensor\n",
                                EnumNameTensorType(tensor->type()),
                                tensor->type());
        status = kTfLiteError;
        continue;
    }
    auto get_readonly_data = [&](const char** buffer_data,
                                 size_t* buffer_size) {
      // TODO(aselle): Check what happens if we have an unspecified size
      // constant.
      *buffer_data = nullptr;
      if (tensor->buffer() == 0) return kTfLiteOk;
      if (tensor->buffer() >= buffers->size()) {
        error_reporter_->Report(
            "Tensor %d specifies out of range buffer %d (only %d buffers).\n",
            i, tensor->buffer(), buffers->size());
        return kTfLiteError;
      }
      if (auto* buffer = (*buffers)[tensor->buffer()]) {
        if (auto* array = buffer->data()) {
          if (size_t size = array->size()) {
            *buffer_size = size;
            *buffer_data = reinterpret_cast<const char*>(array->data());
            return kTfLiteOk;
          }
        }
      }
      return kTfLiteOk;
    };
    size_t buffer_size = 0;
    const char* buffer_ptr;
    TF_LITE_ENSURE_STATUS(get_readonly_data(&buffer_ptr, &buffer_size));

    if (buffer_ptr) {
      if (interpreter->SetTensorParametersReadOnly(
              i, type, get_name(tensor), dims, quantization, buffer_ptr,
              buffer_size, allocation_) != kTfLiteOk) {
        error_reporter_->Report("Tensor %d is invalidly specified in schema.\n",
                                i);
        status = kTfLiteError;
      }
    } else {
      if (interpreter->SetTensorParametersReadWrite(
              i, type, get_name(tensor), dims, quantization) != kTfLiteOk) {
        error_reporter_->Report("Tensor %d is invalidly specified in schema.\n",
                                i);
        status = kTfLiteError;
      }
    }
  }

  return status;
}

TfLiteStatus InterpreterBuilder::operator()(
    std::unique_ptr<Interpreter>* interpreter) {
  if (!interpreter) {
    error_reporter_->Report(
        "Null output pointer passed to InterpreterBuilder.");
    return kTfLiteError;
  }

  // Safe exit by deleting partially created interpreter, to reduce verbosity
  // on error conditions. Use by return cleanup_on_error();
  auto cleanup_and_error = [&interpreter]() {
    interpreter->reset();
    return kTfLiteError;
  };

  if (!model_) {
    error_reporter_->Report("Null pointer passed in as model.");
    return cleanup_and_error();
  }

  if (model_->version() != TFLITE_SCHEMA_VERSION) {
    error_reporter_->Report(
        "Model provided is schema version %d not equal "
        "to supported version %d.\n",
        model_->version(), TFLITE_SCHEMA_VERSION);
    return cleanup_and_error();
  }

  if (BuildLocalIndexToRegistrationMapping() != kTfLiteOk) {
    error_reporter_->Report("Registration failed.\n");
    return cleanup_and_error();
  }

  // Flatbuffer model schemas define a list of opcodes independent of the graph.
  // We first map those to registrations. This reduces string lookups for custom
  // ops since we only do it once per custom op rather than once per custom op
  // invocation in the model graph.
  // Construct interpreter with correct number of tensors and operators.
  auto* subgraphs = model_->subgraphs();
  auto* buffers = model_->buffers();
  if (subgraphs->size() != 1) {
    error_reporter_->Report("Only 1 subgraph is currently supported.\n");
    return cleanup_and_error();
  }
  const tflite::SubGraph* subgraph = (*subgraphs)[0];
  auto operators = subgraph->operators();
  auto tensors = subgraph->tensors();
  if (!operators || !tensors || !buffers) {
    error_reporter_->Report(
        "Did not get operators, tensors, or buffers in input flat buffer.\n");
    return cleanup_and_error();
  }
  interpreter->reset(new Interpreter(error_reporter_));
  if ((**interpreter).AddTensors(tensors->Length()) != kTfLiteOk) {
    return cleanup_and_error();
  }

  // Parse inputs/outputs
  (**interpreter).SetInputs(FlatBufferIntArrayToVector(subgraph->inputs()));
  (**interpreter).SetOutputs(FlatBufferIntArrayToVector(subgraph->outputs()));

  // Finally setup nodes and tensors
  if (ParseNodes(operators, interpreter->get()) != kTfLiteOk)
    return cleanup_and_error();
  if (ParseTensors(buffers, tensors, interpreter->get()) != kTfLiteOk)
    return cleanup_and_error();

  return kTfLiteOk;
}

}  // namespace tflite
