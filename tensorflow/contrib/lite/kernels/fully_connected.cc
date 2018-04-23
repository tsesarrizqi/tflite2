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

#include <unistd.h>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <limits>

//note: android log
#include <android/log.h> 
#include <stdio.h> 
#include <stdlib.h> 
#include <math.h>

//note: android opencl
#include "CL/cl.h"

//note: vulkan
#include "vulkan/vulkan.h"
// #include "vulkan/vk_platform.h"

//note: shaderc
// #include "shaderc/shaderc.hpp"

#include "tensorflow/contrib/lite/builtin_op_data.h"
#include "tensorflow/contrib/lite/context.h"
#include "tensorflow/contrib/lite/kernels/activation_functor.h"
#include "tensorflow/contrib/lite/kernels/gemm_support.h"
#include "tensorflow/contrib/lite/kernels/internal/optimized/optimized_ops.h"
#include "tensorflow/contrib/lite/kernels/internal/quantization_util.h"
#include "tensorflow/contrib/lite/kernels/internal/reference/reference_ops.h"
#include "tensorflow/contrib/lite/kernels/internal/tensor.h"
#include "tensorflow/contrib/lite/kernels/internal/tensor_utils.h"
#include "tensorflow/contrib/lite/kernels/kernel_util.h"
#include "tensorflow/contrib/lite/kernels/op_macros.h"

namespace tflite {
namespace ops {
namespace builtin {
namespace fully_connected {

// This file has four implementations of FullyConnected
enum KernelType {
  kReference,
  kGenericOptimized,  // Neon-free
  kNeonOptimized,
  kPie,  // Used by the PIE team
};

struct OpData {
  // The scaling factor from input to output (aka the 'real multiplier') can
  // be represented as a fixed point multipler plus a left shift.
  int32_t output_multiplier;
  int output_shift;
  // The range of the fused activation layer. For example for kNone and
  // uint8_t these would be 0 and 255.
  int32_t output_activation_min;
  int32_t output_activation_max;
};

constexpr int kInputTensor = 0;
constexpr int kWeightsTensor = 1;
constexpr int kBiasTensor = 2;
constexpr int kOutputTensor = 0;
   
cl_context context_cl_global = NULL;       
cl_command_queue queue_global = NULL;
cl_program program_global = NULL;
cl_mem cl_mem_arr_global[6];
// cl_mem d_conv_input_global = NULL;
// cl_mem d_conv_filter_global = NULL;
// cl_mem d_conv_bias_global = NULL;
// cl_mem d_conv_output_global = NULL;
// cl_mem d_conv_dim_sizes_global = NULL;
// cl_mem d_conv_dim_strides_global = NULL;

VkPhysicalDevice physicalDevice_global = NULL;
VkDevice device_global = NULL;
VkPipeline pipelineConv_global = NULL;
VkPipeline pipelineMatmul_global = NULL;
VkPipelineLayout pipelineLayoutMatmul_global = NULL;
VkPipelineLayout pipelineLayoutConv_global = NULL;
VkDescriptorSetLayout descriptorSetLayoutMatmul_global = NULL;
VkDescriptorSetLayout descriptorSetLayoutConv_global = NULL;
VkQueue queueV_global = NULL; 
uint32_t queueFamilyIndex_global = 0;
VkCommandPool conv_commandPool_global = NULL;
VkCommandBuffer conv_commandBuffer_global = NULL;
VkBuffer conv_matrixA_global = NULL;
VkBuffer conv_matrixB_global = NULL;
VkBuffer conv_matrixC_global = NULL;
VkBuffer conv_matrixSizes_global = NULL;
VkDeviceMemory conv_bufferMemory_global = NULL;

// const char *kernelSource =           "\n" \
// "__kernel void matrixVectorMul(__global float* C,  \n" \
// "                      const __global float* A,  \n" \
// "                      const __global float* B,  \n" \
// "                      int K, int M, int N) {  \n" \
// "      \n" \
// "    const int row = get_local_id(0); // Local row ID (max: 32)  \n" \
// "    const int col = get_local_id(1); // Local col ID (max: 32)  \n" \
// "    const int globalRow = 32*get_group_id(0) + row; // Row ID of C (0..M)  \n" \
// "    const int globalCol = 32*get_group_id(1) + col; // Col ID of C (0..N)  \n" \
// "   \n" \
// "      __local float Asub[32][32];  \n" \
// "      __local float Bsub[32][32];  \n" \
// "     \n" \
// "      float acc = 0.0;  \n" \
// "        \n" \
// "      const int numTiles = ((K-1)/32)+1;  \n" \
// "      for (int t=0; t<numTiles; t++) {  \n" \
// "     \n" \
// "          const int tiledRow = 32*t + row;  \n" \
// "          const int tiledCol = 32*t + col;  \n" \
// "          if((tiledCol < K) && (globalRow < M)) { \n" \
// "            Asub[col][row] = A[globalRow*K + tiledCol];  \n" \
// "          }   \n" \
// "          else {    \n" \
// "            Asub[col][row] = 0.0;  \n" \
// "          }   \n" \
// "          if((tiledRow < K) && (globalCol < N)) { \n" \
// "            Bsub[col][row] = B[globalCol*K + tiledRow];  \n" \
// "          }   \n" \
// "          else {    \n" \
// "            Bsub[col][row] = 0.0;  \n" \
// "          }   \n" \
// "     \n" \
// "          barrier(CLK_LOCAL_MEM_FENCE);  \n" \
// "     \n" \
// "          for (int k=0; k<32; k++) {  \n" \
// "              acc += Asub[k][row] * Bsub[col][k];  \n" \
// "          }  \n" \
// "     \n" \
// "          barrier(CLK_LOCAL_MEM_FENCE);  \n" \
// "      }  \n" \
// "     \n" \
// "      if((globalRow < M) && (globalCol < N)) {  \n" \
// "          C[globalCol*M + globalRow] = acc;  \n" \
// "      } \n" \
// "} \n" \ 
// "\n";

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  // This is a builtin op, so we don't use the contents in 'buffer', if any.
  // Instead, we allocate a new object to carry information from Prepare() to
  // Eval().

  // cl_int err;

  // err = clGetPlatformIDs(1, &cpPlatform, NULL);
 
  // err = clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);

  // context_cl = clCreateContext(0, 1, &device_id, NULL, NULL, &err);

  // queue = clCreateCommandQueue(context_cl, device_id, 0, &err);

  // program = clCreateProgramWithSource(context_cl, 1,
  //                         (const char **) & kernelSource, NULL, &err);

  // clBuildProgram(program, 0, NULL, NULL, NULL, NULL);

  //note: andoird log
  // __android_log_print(ANDROID_LOG_INFO, "Ngising", "fcinitbiasa");

  gemm_support::IncrementUsageCounter(context);
  return new OpData;
}

void* InitOpenCL(TfLiteContext* context, const char* buffer, size_t length,
  cl_context context_cl, cl_command_queue queue, cl_program program, cl_mem cl_mem_arr[6],
  VkPhysicalDevice physicalDevice, VkDevice device, VkPipeline pipelineConv, VkPipeline pipelineMatmul, VkPipelineLayout pipelineLayoutConv, VkPipelineLayout pipelineLayoutMatmul, 
    VkDescriptorSetLayout descriptorSetLayoutConv, VkDescriptorSetLayout descriptorSetLayoutMatmul, VkQueue queueV, uint32_t queueFamilyIndex,
    VkCommandPool conv_commandPool, VkCommandBuffer conv_commandBuffer, VkBuffer conv_matrixA, VkBuffer conv_matrixB, VkBuffer conv_matrixC, VkBuffer conv_matrixSizes, VkDeviceMemory conv_bufferMemory) {
  // This is a builtin op, so we don't use the contents in 'buffer', if any.
  // Instead, we allocate a new object to carry information from Prepare() to
  // Eval().

  // cl_int err;

  // err = clGetPlatformIDs(1, &cpPlatform, NULL);
 
  // err = clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);

  // context_cl = clCreateContext(0, 1, &device_id, NULL, NULL, &err);

  // queue = clCreateCommandQueue(context_cl, device_id, 0, &err);

  // program = clCreateProgramWithSource(context_cl, 1,
  //                         (const char **) & kernelSource, NULL, &err);

  // clBuildProgram(program, 0, NULL, NULL, NULL, NULL);

  context_cl_global = context_cl;
  program_global = program;
  queue_global = queue;
  cl_mem_arr_global[0] = cl_mem_arr[0];
  cl_mem_arr_global[1] = cl_mem_arr[1];
  cl_mem_arr_global[2] = cl_mem_arr[2];
  cl_mem_arr_global[3] = cl_mem_arr[3];
  cl_mem_arr_global[4] = cl_mem_arr[4];
  cl_mem_arr_global[5] = cl_mem_arr[5];
  // d_conv_input_global = cl_mem_arr[0];
  // d_conv_filter_global = cl_mem_arr[1];
  // d_conv_bias_global = cl_mem_arr[2];
  // d_conv_output_global = cl_mem_arr[3];
  // d_conv_dim_sizes_global = cl_mem_arr[4];
  // d_conv_dim_strides_global = cl_mem_arr[5];

  physicalDevice_global = physicalDevice; 
  device_global = device;
  pipelineConv_global = pipelineConv;
  pipelineMatmul_global = pipelineMatmul;
  pipelineLayoutMatmul_global = pipelineLayoutMatmul;
  pipelineLayoutConv_global = pipelineLayoutConv;
  descriptorSetLayoutMatmul_global = descriptorSetLayoutMatmul;
  descriptorSetLayoutConv_global = descriptorSetLayoutConv;
  queueV_global = queueV; 
  queueFamilyIndex_global = queueFamilyIndex;
  conv_commandPool_global = conv_commandPool;
  conv_commandBuffer_global = conv_commandBuffer;
  conv_matrixA_global = conv_matrixA;
  conv_matrixB_global = conv_matrixB;
  conv_matrixC_global = conv_matrixC;
  conv_matrixSizes_global = conv_matrixSizes;
  conv_bufferMemory_global = conv_bufferMemory;

  //note: andoird log
  // __android_log_print(ANDROID_LOG_INFO, "Ngising", "fcinit");

  gemm_support::IncrementUsageCounter(context);
  return new OpData;
}

void Free(TfLiteContext* context, void* buffer) {
  gemm_support::DecrementUsageCounter(context);
  delete reinterpret_cast<OpData*>(buffer);
  // clReleaseProgram(program);
  // clReleaseCommandQueue(queue);
  // clReleaseContext(context_cl);
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  auto* params =
      reinterpret_cast<TfLiteFullyConnectedParams*>(node->builtin_data);
  OpData* data = reinterpret_cast<OpData*>(node->user_data);

  // Check we have all the inputs and outputs we need.
  TF_LITE_ENSURE_EQ(context, node->inputs->size, 3);
  TF_LITE_ENSURE_EQ(context, node->outputs->size, 1);

  TfLiteTensor* input = GetInput(context, node, kInputTensor);
  TfLiteTensor* filter = GetInput(context, node, kWeightsTensor);
  TfLiteTensor* bias = GetOptionalInputTensor(context, node, kBiasTensor);
  TfLiteTensor* output = GetOutput(context, node, kOutputTensor);

  // Check all the parameters of tensor match within themselves and match the
  // input configuration.
  int input_size = 1;
  for (int i = 0; i < input->dims->size; i++) {
    input_size *= input->dims->data[i];
  }

  const int batch_size = input_size / filter->dims->data[1];
  const int num_units = filter->dims->data[0];

  TF_LITE_ASSERT_EQ(input_size, batch_size * filter->dims->data[1]);
  if (bias) {
    TF_LITE_ASSERT_EQ(bias->dims->data[0], num_units);
  }

  TF_LITE_ENSURE_EQ(context, NumDimensions(filter), 2);
  TF_LITE_ENSURE_EQ(context, NumDimensions(bias), 1);

  // Note that quantized inference requires that all tensors have their
  // parameters set. This is usually done during quantized training.
  TfLiteType data_type = input->type;
  if (data_type != kTfLiteFloat32) {
    double real_multiplier = 0.0;
    TF_LITE_ENSURE_STATUS(GetQuantizedConvolutionMultipler(
        context, input, filter, bias, output, &real_multiplier));
    QuantizeMultiplierSmallerThanOne(real_multiplier, &data->output_multiplier,
                                     &data->output_shift);
    CalculateActivationRangeUint8(params->activation, output,
                                  &data->output_activation_min,
                                  &data->output_activation_max);
  }

  // Resize output.
  TfLiteIntArray* output_size_array = TfLiteIntArrayCreate(2);
  output_size_array->data[0] = batch_size;
  output_size_array->data[1] = num_units;
  TF_LITE_ENSURE_OK(context,
                    context->ResizeTensor(context, output, output_size_array));
  return kTfLiteOk;
}

TfLiteStatus EvalPie(TfLiteContext* context, TfLiteNode* node,
                     TfLiteFullyConnectedParams* params, OpData* data,
                     TfLiteTensor* input, TfLiteTensor* filter,
                     TfLiteTensor* bias, TfLiteTensor* output) {
  int total_input_size = 1;
  for (int i = 0; i < input->dims->size; i++) {
    total_input_size *= input->dims->data[i];
  }

  int input_size = filter->dims->data[1];
  const int batch_size = total_input_size / filter->dims->data[1];
  const int num_units = filter->dims->data[0];

  // Output = bias if bias tensor exists.
  if (bias) {
    tensor_utils::VectorBatchVectorAssign(bias->data.f, num_units, batch_size,
                                          output->data.f);
  } else {
    tensor_utils::ZeroVector(output->data.f, batch_size * num_units);
  }

  //note: andoird log
  // __android_log_print(ANDROID_LOG_INFO, "Ngisin", "fcmatmul");
  // Compute output += weight * input
  tensor_utils::MatrixBatchVectorMultiplyAccumulate(
      filter->data.f, num_units, input_size, input->data.f, batch_size,
      output->data.f, /*result_stride=*/1);

  // tensor_utils::MatrixBatchVectorMultiplyAccumulateOpenCL(
  //     filter->data.f, num_units, input_size, input->data.f, batch_size,
  //     output->data.f, /*result_stride=*/1, context_cl_global, queue_global, program_global, cl_mem_arr_global,
  //     physicalDevice_global, device_global, pipelineConv_global, pipelineMatmul_global, pipelineLayoutConv_global, 
  //       pipelineLayoutMatmul_global, descriptorSetLayoutConv_global, descriptorSetLayoutMatmul_global, queueV_global, queueFamilyIndex_global,
  //       conv_commandPool_global, conv_commandBuffer_global, conv_matrixA_global, conv_matrixB_global, conv_matrixC_global, conv_matrixSizes_global, conv_bufferMemory_global);

  // Apply activation function
  tensor_utils::ApplyActivationToVector(output->data.f, batch_size * num_units,
                                        params->activation, output->data.f);

  return kTfLiteOk;
}

#define TF_LITE_MACRO_DISPATCH(macro_name, params, target_namespace) \
  if (params->activation == kTfLiteActNone) {                        \
    macro_name(target_namespace, kNone);                             \
  }                                                                  \
  if (params->activation == kTfLiteActRelu) {                        \
    macro_name(target_namespace, kRelu);                             \
  }                                                                  \
  if (params->activation == kTfLiteActRelu6) {                       \
    macro_name(target_namespace, kRelu6);                            \
  }

template <KernelType kernel_type>
TfLiteStatus EvalQuantized(TfLiteContext* context, TfLiteNode* node,
                           TfLiteFullyConnectedParams* params, OpData* data,
                           TfLiteTensor* input, TfLiteTensor* filter,
                           TfLiteTensor* bias, TfLiteTensor* output) {
  gemmlowp::GemmContext* gemm_context = gemm_support::GetFromContext(context);

  int32_t input_offset = -input->params.zero_point;
  int32_t filter_offset = -filter->params.zero_point;
  int32_t output_offset = output->params.zero_point;

  //  //note: andoird log
  // __android_log_print(ANDROID_LOG_INFO, "Ngisin", "fcmatmul2");
#define TF_LITE_FULLY_CONNECTED(type)                                       \
  type::FullyConnected(                                                     \
      GetTensorData<uint8_t>(input), GetTensorDims(input), input_offset,    \
      GetTensorData<uint8_t>(filter), GetTensorDims(filter), filter_offset, \
      GetTensorData<int32_t>(bias), GetTensorDims(bias), output_offset,     \
      data->output_multiplier, data->output_shift,                          \
      data->output_activation_min, data->output_activation_max,             \
      GetTensorData<uint8_t>(output), GetTensorDims(output), gemm_context)
  if (kernel_type == kReference) {
    TF_LITE_FULLY_CONNECTED(reference_ops);
  } else if (kernel_type == kPie) {
    // TODO(ahentz): we don't have a quantized version of the PIE kernels, so
    // we just defer to the MINI ones.
    TF_LITE_FULLY_CONNECTED(optimized_ops);
  } else {
    TF_LITE_FULLY_CONNECTED(optimized_ops);
  }
#undef TF_LITE_FULLY_CONNECTED

  return kTfLiteOk;
}

template <KernelType kernel_type>
TfLiteStatus EvalFloat(TfLiteContext* context, TfLiteNode* node,
                       TfLiteFullyConnectedParams* params, OpData* data,
                       TfLiteTensor* input, TfLiteTensor* filter,
                       TfLiteTensor* bias, TfLiteTensor* output) {
  float output_activation_min, output_activation_max;
  CalculateActivationRangeFloat(params->activation, &output_activation_min,
                                &output_activation_max);
#define TF_LITE_FULLY_CONNECTED(type)                                       \
  type::FullyConnected(GetTensorData<float>(input), GetTensorDims(input),   \
                       GetTensorData<float>(filter), GetTensorDims(filter), \
                       GetTensorData<float>(bias), GetTensorDims(bias),     \
                       output_activation_min, output_activation_max,        \
                       GetTensorData<float>(output), GetTensorDims(output))
  if (kernel_type == kReference) {
    TF_LITE_FULLY_CONNECTED(reference_ops);
  } else if (kernel_type == kPie) {
    return EvalPie(context, node, params, data, input, filter, bias, output);
  } else {
    TF_LITE_FULLY_CONNECTED(optimized_ops);
  }
#undef TF_LITE_FULLY_CONNECTED

  return kTfLiteOk;
}

#undef TF_LITE_MACRO_DISPATCH

template <KernelType kernel_type>
TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  auto* params =
      reinterpret_cast<TfLiteFullyConnectedParams*>(node->builtin_data);
  OpData* data = reinterpret_cast<OpData*>(node->user_data);

  TfLiteTensor* input = GetInput(context, node, kInputTensor);
  TfLiteTensor* filter = GetInput(context, node, kWeightsTensor);
  TfLiteTensor* bias = GetOptionalInputTensor(context, node, kBiasTensor);
  TfLiteTensor* output = GetOutput(context, node, kOutputTensor);

  switch (input->type) {  // Already know in/out types are same.
    case kTfLiteFloat32:
      return EvalFloat<kernel_type>(context, node, params, data, input, filter,
                                    bias, output);
    case kTfLiteUInt8:
      return EvalQuantized<kernel_type>(context, node, params, data, input,
                                        filter, bias, output);
    default:
      context->ReportError(context, "Type not currently supported.");
      return kTfLiteError;
  }
  return kTfLiteOk;
}

}  // namespace fully_connected

TfLiteRegistration* Register_FULLY_CONNECTED_REF() {
  static TfLiteRegistration r = {
      fully_connected::Init, fully_connected::Free, fully_connected::Prepare,
      fully_connected::Eval<fully_connected::kReference>, fully_connected::InitOpenCL};
  return &r;
}

TfLiteRegistration* Register_FULLY_CONNECTED_NEON_OPT() {
  static TfLiteRegistration r = {
      fully_connected::Init, fully_connected::Free, fully_connected::Prepare,
      fully_connected::Eval<fully_connected::kNeonOptimized>, fully_connected::InitOpenCL};
  return &r;
}

TfLiteRegistration* Register_FULLY_CONNECTED_GENERIC_OPT() {
  static TfLiteRegistration r = {
      fully_connected::Init, fully_connected::Free, fully_connected::Prepare,
      fully_connected::Eval<fully_connected::kGenericOptimized>, fully_connected::InitOpenCL};
  return &r;
}

TfLiteRegistration* Register_FULLY_CONNECTED_PIE() {
  static TfLiteRegistration r = {fully_connected::Init, fully_connected::Free,
                                 fully_connected::Prepare,
                                 fully_connected::Eval<fully_connected::kPie>, fully_connected::InitOpenCL};
  return &r;
}

TfLiteRegistration* Register_FULLY_CONNECTED() {
  // TODO(ahentz): We don't have a dedicated quantized version of the PIE
  // kernel. For now, the quantized version just defer to the corresponding
  // optimized MINI kernel. At some point we will allow different libraries to
  // be built with different kernels, but for now we have to pick one here.
  return Register_FULLY_CONNECTED_PIE();
#ifdef USE_NEON
  return Register_FULLY_CONNECTED_NEON_OPT();
#else
  return Register_FULLY_CONNECTED_GENERIC_OPT();
#endif
}

}  // namespace builtin
}  // namespace ops
}  // namespace tflite
