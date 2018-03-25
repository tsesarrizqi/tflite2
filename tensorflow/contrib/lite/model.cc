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

//note: android log
#include <android/log.h> 
#include <stdio.h> 

const char *kernelSource =           "\n" \
"#define TS 32      \n" \
"#define WIDTH 4      \n" \
"__kernel void conv(__constant float* input_data,   \n" \
"          __constant float* filter_data,   \n" \
"          __constant float* bias_data,   \n" \
"          __global float* output_data,  \n" \
"          int stride_width, int stride_height,   \n" \
"          int pad_width, int pad_height,   \n" \
"          __constant int* dim_sizes, __global int* dim_strides,  \n" \
"          float output_activation_min, float output_activation_max) {  \n" \
"  int gid0 = get_global_id(0);\n" \
"  int gid1 = get_global_id(1);  \n" \
"  const int batches = dim_sizes[3];  \n" \
"  const int input_depth = dim_sizes[0];  \n" \
"  const int output_depth = dim_sizes[7]; \n" \
"  const int output_height = dim_sizes[14];  \n" \
"  const int output_width = dim_sizes[13];   \n" \
"  \n" \
"  int batch = gid0/output_depth;  \n" \
"  int out_channel = gid0%output_depth;\n" \
"  int out_y = gid1/output_width;  \n" \
"  int out_x = gid1%output_width;  \n" \
"  \n" \
"  if((gid0 < batches*output_depth) && (gid1 < output_height*output_width)) {  \n" \
"    const int input_height = dim_sizes[2];  \n" \
"    const int input_width = dim_sizes[1];  \n" \
"    const int filter_height = dim_sizes[6];  \n" \
"    const int filter_width = dim_sizes[5];  \n" \
"    // for (int out_y = 0; out_y < output_height; ++out_y) {  \n" \
"    //   for (int out_x = 0; out_x < output_width; ++out_x) {  \n" \
"        const int in_x_origin = (out_x * stride_width) - pad_width;  \n" \
"        const int in_y_origin = (out_y * stride_height) - pad_height;  \n" \
"        float total = 0.f;  \n" \
"        for (int filter_y = 0; filter_y < filter_height; ++filter_y) {  \n" \
"          for (int filter_x = 0; filter_x < filter_width; ++filter_x) {  \n" \
"            for (int in_channel = 0; in_channel < input_depth; ++in_channel) {  \n" \
"              const int in_x = in_x_origin + filter_x;  \n" \
"              const int in_y = in_y_origin + filter_y;  \n" \
"              if ((in_x >= 0) && (in_x < input_width) && (in_y >= 0) &&  \n" \
"                  (in_y < input_height)) {  \n" \
"                float input_value = input_data[in_channel*dim_strides[0] + in_x*dim_strides[1] +   \n" \
"                                                in_y*dim_strides[2] + batch*dim_strides[3]];  \n" \
"                float filter_value =  \n" \
"                    filter_data[in_channel*dim_strides[4] + filter_x*dim_strides[5] +  \n" \
"                                       filter_y*dim_strides[6] + out_channel*dim_strides[7]];  \n" \
"                total += (input_value * filter_value);  \n" \
"              }  \n" \
"            }  \n" \
"          }  \n" \
"        }  \n" \
"        float bias_value = 0.0f;  \n" \
"        if (bias_data) {  \n" \
"          bias_value = bias_data[out_channel*dim_strides[8]];  \n" \
"        }  \n" \
"        float max = total+bias_value; \n" \
"        if(max < output_activation_min) max = output_activation_min; \n" \
"        float min = max; \n" \
"        if(min > output_activation_max) min = output_activation_max; \n" \
"        output_data[out_channel*dim_strides[12] + out_x*dim_strides[13] +   \n" \
"                     out_y*dim_strides[14] + batch*dim_strides[15]] = min; \n" \
"    //   }  \n" \
"    // }  \n" \
"  }  \n" \
"}\n" \
"  \n" \
"       \n" \
"__kernel void matrixVectorMul(__global float4* C,         \n" \
"                      const __global float4* A,         \n" \
"                      const __global float4* B,         \n" \
"                      int K, int M, int N) {         \n" \
"          \n" \
"    // Thread identifiers      \n" \
"    const int row = get_local_id(0); // Local row ID (max: TS/WIDTH)      \n" \
"    const int col = get_local_id(1); // Local col ID (max: TS)      \n" \
"    const int globalRow = (TS/WIDTH)*get_group_id(0) + row; // 0..M/WIDTH      \n" \
"    const int globalCol = TS*get_group_id(1) + col; // 0..N      \n" \
"       \n" \
"    // Local memory to fit a tile of TS*TS elements of A and B      \n" \
"    __local float4 Asub[TS][TS/WIDTH];      \n" \
"    __local float4 Bsub[TS][TS/WIDTH];      \n" \
"       \n" \
"    // Initialise the accumulation registers      \n" \
"    float4 acc = { 0.0f, 0.0f, 0.0f, 0.0f };      \n" \
"          \n" \
"    // Loop over all tiles      \n" \
"    const int numTiles = K/TS;      \n" \
"    for (int t=0; t<numTiles; t++) {      \n" \
"       \n" \
"        // Load one tile of A and B into local memory      \n" \
"        const int tiledRow = (TS/WIDTH)*t + row;      \n" \
"        const int tiledCol = TS*t + col;      \n" \
"        if(globalRow < (M/WIDTH)) {     \n" \
"          Asub[col][row] = A[tiledCol*(M/WIDTH) + globalRow];     \n" \
"        }      \n" \
"        else {     \n" \
"           float4 tmp = { 0.0f, 0.0f, 0.0f, 0.0f };  \n" \
"           Asub[col][row] = tmp;     \n" \
"        }     \n" \
"        if(globalCol < N) {     \n" \
"          Bsub[col][row] = B[globalCol*(K/WIDTH) + tiledRow];     \n" \
"        }      \n" \
"        else {     \n" \
"           float4 tmp = { 0.0f, 0.0f, 0.0f, 0.0f };  \n" \
"           Bsub[col][row] = tmp;     \n" \
"        }     \n" \
"             \n" \
"        // Synchronise to make sure the tile is loaded      \n" \
"        barrier(CLK_LOCAL_MEM_FENCE);      \n" \
"       \n" \
"        if(globalCol < N) {     \n" \
"          // Perform the computation for a single tile      \n" \
"          float4 vecA, vecB;      \n" \
"          float valB;      \n" \
"          for (int k=0; k<TS/WIDTH; k++) {      \n" \
"              vecB = Bsub[col][k];      \n" \
"              for (int w=0; w<WIDTH; w++) {      \n" \
"                  vecA = Asub[WIDTH*k + w][row];      \n" \
"                  switch (w) {      \n" \
"                      case 0: valB = vecB.x; break;      \n" \
"                      case 1: valB = vecB.y; break;      \n" \
"                      case 2: valB = vecB.z; break;      \n" \
"                      case 3: valB = vecB.w; break;      \n" \
"                  }      \n" \
"                  acc.x += vecA.x * valB;      \n" \
"                  acc.y += vecA.y * valB;      \n" \
"                  acc.z += vecA.z * valB;      \n" \
"                  acc.w += vecA.w * valB;      \n" \
"              }     \n" \
"          }      \n" \
"        }     \n" \
"      \n" \
"        // Synchronise before loading the next tile      \n" \
"        barrier(CLK_LOCAL_MEM_FENCE);      \n" \
"    }      \n" \
"        \n" \
"    if((globalCol < N) && (globalRow < (M/WIDTH))) {     \n" \
"      // Store the final results in C      \n" \
"      C[globalCol*(M/WIDTH) + globalRow] = acc;     \n" \
"    }      \n" \
"         \n" \
"}      \n" \
"     \n" \
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
"__kernel void matrixVectorMulNaive(__global float* resultVector,     \n" \
"    __global float* matrixA,     \n" \
"    __global float* vectorB,      \n" \
"    int width_A,     \n" \
"    int height_A,     \n" \
"    int width_B)     \n" \
"{     \n" \
"    int idx = get_global_id(0);\n" \
"    int idx2 = get_global_id(1);      \n" \
"      \n" \
"    if((idx < height_A) && (idx2 < width_B)) {  \n" \
"        float value = 0.0f;     \n" \
"        for (int k = 0; k < width_A; ++k) {     \n" \
"            value += matrixA[idx * width_A + k] * vectorB[idx2*width_A+k];     \n" \
"        }     \n" \
"        resultVector[idx2*height_A+idx] = value;     \n" \
"   }     \n" \   
"}  \n" \
"\n";

cl_platform_id cpPlatform = NULL;
cl_device_id device_id = NULL;    
cl_context context_cl = NULL;       
cl_command_queue queue = NULL;
cl_program program = NULL;

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

  queue = clCreateCommandQueue(context_cl, device_id, 0, &err);

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
  //note: andoird log
  __android_log_print(ANDROID_LOG_INFO, "Ngising", "MAX Workgroup transpose: %d",maxWorkGroupSize1);
  __android_log_print(ANDROID_LOG_INFO, "Ngising", "MAX Workgroup conv: %d",maxWorkGroupSize2);
  __android_log_print(ANDROID_LOG_INFO, "Ngising", "MAX Workgroup matrixVectorMul: %d",maxWorkGroupSize3);
}

TfLiteStatus InterpreterBuilder::ParseNodes(
    const flatbuffers::Vector<flatbuffers::Offset<Operator>>* operators,
    Interpreter* interpreter) {
  TfLiteStatus status = kTfLiteOk;
  //note: andoird log
  // __android_log_print(ANDROID_LOG_INFO, "Ngising", "addnodewithparam");
  // initOpenCL();
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
      if (op->custom_options()) {
        interpreter->AddNodeWithParametersOpenCL(
            FlatBufferIntArrayToVector(op->inputs()),
            FlatBufferIntArrayToVector(op->outputs()),
            reinterpret_cast<const char*>(op->custom_options()->data()),
            op->custom_options()->size(), nullptr, reg,
            context_cl, queue, program);
      } else {
        //note: andoird log
        // __android_log_print(ANDROID_LOG_INFO, "Ngising", "addnodewithparam2");
        interpreter->AddNodeWithParametersOpenCL(
            FlatBufferIntArrayToVector(op->inputs()),
            FlatBufferIntArrayToVector(op->outputs()), nullptr, 0,
            ParseOpData(op, op_type, error_reporter_), reg,
            context_cl, queue, program);
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
