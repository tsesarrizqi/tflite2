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
#include <string.h>

#include "tensorflow/contrib/lite/builtin_op_data.h"
#include "tensorflow/contrib/lite/kernels/activation_functor.h"
#include "tensorflow/contrib/lite/kernels/op_macros.h"

//note: android log
#include <android/log.h> 
#include <stdio.h> 
#include <stdlib.h> 
#include <math.h>

//note: android opencl
#include "../CL/cl.h"

//note: vulkan
#include "vulkan/vulkan.h"
#include "vulkan/vk_platform.h"

//note: shaderc
#include "shaderc/shaderc.hpp"

namespace tflite {
namespace tensor_utils {

float PortableClip(float f, float abs_limit) {
  float result = (abs_limit < f) ? abs_limit : f;
  result = (-abs_limit > result) ? -abs_limit : result;
  return result;
}

void PortableMatrixBatchVectorMultiplyAccumulate(const float* matrix,
                                                 int m_rows, int m_cols,
                                                 const float* vector,
                                                 int n_batch, float* result,
                                                 int result_stride) {
  // vector per kolom
  // matrix per baris
  // result per kolom
  float* result_in_batch = result;
  for (int b = 0; b < n_batch; b++) {
    const float* matrix_ptr = matrix;
    for (int r = 0; r < m_rows; r++) {
      const float* vector_in_batch = vector + b * m_cols;
      for (int c = 0; c < m_cols; c++) {
        *result_in_batch += *matrix_ptr++ * *vector_in_batch++;
      }
      result_in_batch += result_stride;
    }
  }

  // //note: andoird log
  // __android_log_print(ANDROID_LOG_INFO, "CobaLog", "1");
}

void PortableMatrixBatchVectorMultiplyAccumulateOpenCL(const float* matrix,
                                                 int m_rows, int m_cols,
                                                 const float* vector,
                                                 int n_batch, float* result,
                                                 int result_stride,
                                                 cl_context context_cl, cl_command_queue queue, cl_program program,
                                                 VkDevice device, VkPipeline pipelineConv, VkPipeline pipelineMatmul, VkPipelineLayout pipelineLayoutConv, VkPipelineLayout pipelineLayoutMatmul, 
    VkDescriptorSetLayout descriptorSetLayoutConv, VkDescriptorSetLayout descriptorSetLayoutMatmul, VkQueue queueV, uint32_t queueFamilyIndex) {
  // vector per kolom
  // matrix per baris
  // result per kolom
  float* result_in_batch = result;
  for (int b = 0; b < n_batch; b++) {
    const float* matrix_ptr = matrix;
    for (int r = 0; r < m_rows; r++) {
      const float* vector_in_batch = vector + b * m_cols;
      for (int c = 0; c < m_cols; c++) {
        *result_in_batch += *matrix_ptr++ * *vector_in_batch++;
      }
      result_in_batch += result_stride;
    }
  }

  // //note: andoird log
  // __android_log_print(ANDROID_LOG_INFO, "CobaLog", "1");
}

void PortableVectorVectorCwiseProduct(const float* vector1,
                                      const float* vector2, int v_size,
                                      float* result) {
  for (int v = 0; v < v_size; v++) {
    *result++ = *vector1++ * *vector2++;
  }
  //note: andoird log
  __android_log_print(ANDROID_LOG_INFO, "CobaLog", "Vector Cwise");
}

float PortableVectorVectorDotProduct(const float* vector1, const float* vector2,
                                     int v_size) {
  float result = 0.0;
  for (int v = 0; v < v_size; v++) {
    result += *vector1++ * *vector2++;
  }
  //note: andoird log
  __android_log_print(ANDROID_LOG_INFO, "CobaLog", "Vector Dot");
  return result;
}

void PortableBatchVectorBatchVectorDotProduct(const float* vector1,
                                              const float* vector2, int v_size,
                                              int n_batch, float* result,
                                              int result_stride) {
  float* result_ptr = result;
  const float* vector1_ptr = vector1;
  const float* vector2_ptr = vector2;
  for (int b = 0; b < n_batch; b++) {
    *result_ptr =
        PortableVectorVectorDotProduct(vector1_ptr, vector2_ptr, v_size);
    vector1_ptr += v_size;
    vector2_ptr += v_size;
    result_ptr += result_stride;
  }
  //note: andoird log
  __android_log_print(ANDROID_LOG_INFO, "CobaLog", "BatchBatch Dot");
}

void PortableVectorVectorCwiseProductAccumulate(const float* vector1,
                                                const float* vector2,
                                                int v_size, float* result) {
  for (int v = 0; v < v_size; v++) {
    *result++ += *vector1++ * *vector2++;
  }
  //note: andoird log
  // __android_log_print(ANDROID_LOG_INFO, "CobaLog", "5");
}

void PortableVectorBatchVectorCwiseProductAccumulate(const float* vector,
                                                     int v_size,
                                                     const float* batch_vector,
                                                     int n_batch,
                                                     float* result) {
  for (int b = 0; b < n_batch; b++) {
    for (int v = 0; v < v_size; v++) {
      *result++ += vector[v] * *batch_vector++;
    }
  }
  //note: andoird log
  // __android_log_print(ANDROID_LOG_INFO, "CobaLog", "6");
}

void PortableVectorBatchVectorAssign(const float* vector, int v_size,
                                     int n_batch, float* batch_vector) {
  for (int b = 0; b < n_batch; b++) {
    memcpy(batch_vector + b * v_size, vector, v_size * sizeof(float));
  }
  //note: andoird log
  // __android_log_print(ANDROID_LOG_INFO, "CobaLog", "7");
}

void PortableApplySigmoidToVector(const float* vector, int v_size,
                                  float* result) {
  auto sigmoid_func = ActivationFunctor(kTfLiteActSigmoid);
  for (int v = 0; v < v_size; v++) {
    *result++ = (sigmoid_func)(*vector++);
  }
  //note: andoird log
  // __android_log_print(ANDROID_LOG_INFO, "CobaLog", "8");
}

void PortableApplyActivationToVector(const float* vector, int v_size,
                                     TfLiteFusedActivation activation,
                                     float* result) {
  auto activation_func = ActivationFunctor(activation);
  for (int v = 0; v < v_size; v++) {
    *result++ = (activation_func)(*vector++);
  }
  //note: andoird log
  // __android_log_print(ANDROID_LOG_INFO, "CobaLog", "9");
}

void PortableCopyVector(const float* vector, int v_size, float* result) {
  memcpy(result, vector, v_size * sizeof(float));
  //note: andoird log
  // __android_log_print(ANDROID_LOG_INFO, "CobaLog", "10");
}

void PortableSub1Vector(const float* vector, int v_size, float* result) {
  for (int v = 0; v < v_size; v++) {
    *result++ = 1.0f - *vector++;
  }
  //note: andoird log
  // __android_log_print(ANDROID_LOG_INFO, "CobaLog", "11");
}

void PortableZeroVector(float* vector, int v_size) {
  memset(vector, 0, v_size * sizeof(float));
  //note: andoird log
  // __android_log_print(ANDROID_LOG_INFO, "CobaLog", "12");
}

void PortableClipVector(const float* vector, int v_size, float abs_limit,
                        float* result) {
  for (int v = 0; v < v_size; v++) {
    *result++ = PortableClip(*vector++, abs_limit);
  }
  //note: andoird log
  // __android_log_print(ANDROID_LOG_INFO, "CobaLog", "13");
}

void PortableVectorShiftLeft(float* vector, int v_size, float shift_value) {
  TF_LITE_ASSERT(v_size > 0);
  for (int i = 0; i < v_size - 1; i++) {
    vector[i] = vector[i + 1];
  }
  vector[v_size - 1] = shift_value;
  //note: andoird log
  // __android_log_print(ANDROID_LOG_INFO, "CobaLog", "14");
}

void PortableReductionSumVector(const float* input_vector, float* output_vector,
                                int output_size, int reduction_size) {
  const float* input_vector_ptr = input_vector;
  for (int o = 0; o < output_size; o++) {
    for (int r = 0; r < reduction_size; r++) {
      output_vector[o] += *input_vector_ptr++;
    }
  }
  //note: andoird log
  // __android_log_print(ANDROID_LOG_INFO, "CobaLog", "15");
}

}  // namespace tensor_utils
}  // namespace tflite
