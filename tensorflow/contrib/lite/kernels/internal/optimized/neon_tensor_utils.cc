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
#include "tensorflow/contrib/lite/kernels/internal/common.h"
#include "tensorflow/contrib/lite/kernels/activation_functor.h"
#include "tensorflow/contrib/lite/kernels/internal/optimized/tensor_utils_impl.h"

#include <string>
#include <cstring>

#include <fstream>
#include <iostream>
#include <vector>

#include <android/log.h> 
#include <stdio.h> 
#include <stdlib.h> 
#include <math.h>

#include "../CL/cl.h"
#include <sys/stat.h>

#include <vector>
#include <string.h>
#include <assert.h>
#include <stdexcept>
#include <cmath>

#include <time.h>
#include <sys/time.h>

#include "halftmp/half.hpp"

#ifdef USE_NEON

#define kFloatWeightsPerNeonLane 4

#include <string>
#include <sstream>

template <typename T>
std::string to_string(T value)
{
    std::ostringstream os ;
    os << value ;
    return os.str() ;
}

// AAssetManager *mgr;

// void Java_com_example_android_tflitecamerademo_CameraActivity_load(JNIEnv *env, jobject obj, jobject assetManager) {
//    mgr = AAssetManager_fromJava(env, assetManager);
// }

double get_wall_time(){
    struct timeval time;
    if (gettimeofday(&time,NULL)){
        //  Handle error
        return 0;
    }
    return (double)time.tv_sec + (double)time.tv_usec * .000001;
}

double get_cpu_time(){
    return (double)clock() / CLOCKS_PER_SEC;
}

void transpose_scalar_block(const float *A, float *B, const int n, const int m, const int block_size_row, const int block_size_col) {
    for(int i=0; i<block_size_row; i++) {
        for(int j=0; j<block_size_col; j++) {
            B[j*n + i] = A[i*m +j];
        }
    }
}

void transpose_block(const float *A, float *B, const int n, const int m, const int block_size) {
    for(int i=0; i<n; i+=block_size) {
        for(int j=0; j<m; j+=block_size) {
            transpose_scalar_block(&A[i*m +j], &B[j*n + i], n, m, fmin(block_size,n-i), fmin(block_size,m-j));
        }
    }
}

using half_float::half;
using half_float::half_cast;

size_t matmulWgHeight = 8;
size_t matmulWgWidth = 32;

void OpenCLPortableMatrixBatchVectorMultiplyAccumulate(const float* matrix,
                                                 int m_rows, int m_cols,
                                                 const float* vector,
                                                 int n_batch, float* result,
                                                 int result_stride,
                                                 cl_context context, cl_command_queue queue, cl_program program, cl_mem cl_mem_arr[6]) {

  cl_mem d_a = cl_mem_arr[0];
  cl_mem d_b = cl_mem_arr[1];
  cl_mem d_c = cl_mem_arr[3];

  int matrixsize = m_rows*m_cols*sizeof(float);
  int vectorsize = m_cols*n_batch*sizeof(float);
  int resultsize = m_rows*n_batch*sizeof(float);

  int d_n_batch = (((n_batch-1)/4+1)*4);

  cl_kernel kernel;
  cl_int err;

  kernel = clCreateKernel(program, "convmatmulblock", &err);

  int numchannel = m_cols;
  int addslot = (4-(numchannel%4))%4;
  numchannel = numchannel + addslot;
  int input_size = m_rows*numchannel;
  int filter_size = numchannel*n_batch;

  float *inputfloat = (float*)clEnqueueMapBuffer(
              queue,
              d_a,
              CL_TRUE,
              CL_MAP_WRITE,
              0,
              input_size*sizeof(float),
              0, NULL, NULL, &err);
  float *filterfloat = (float*)clEnqueueMapBuffer(
              queue,
              d_b,
              CL_TRUE,
              CL_MAP_WRITE,
              0,
              filter_size*sizeof(float),
              0, NULL, NULL, &err);


  for(int i = 0,i2 = 0; i < input_size; i+=numchannel,i2+=m_cols) {
    for(int j = 0; j < m_cols; j++) {
      inputfloat[i+j] = matrix[i2+j];
    }
    for(int j = m_cols; j < numchannel; j++) {
      inputfloat[i+j] = 0.0;
    }
  }
  for(int i = 0,i2 = 0; i < filter_size; i+=numchannel,i2+=m_cols) {
    for(int j = 0; j < m_cols; j++) {
      filterfloat[i+j] = vector[i2+j];
    }
    for(int j = m_cols; j < numchannel; j++) {
      filterfloat[i+j] = 0.0;
    }
  }

  clEnqueueUnmapMemObject(queue,d_a,(void *) inputfloat,0, NULL, NULL);
  clEnqueueUnmapMemObject(queue,d_b,(void *) filterfloat,0, NULL, NULL);

  err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_a);
  err  = clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_b);
  err  = clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_c);
  err  = clSetKernelArg(kernel, 3, sizeof(int), &m_rows);
  err  = clSetKernelArg(kernel, 4, sizeof(int), &m_cols);
  err  = clSetKernelArg(kernel, 5, sizeof(int), &n_batch);

  const size_t local[2] = { matmulWgHeight, matmulWgWidth };
  const size_t global[2] = { (size_t) ((d_n_batch/4-1)/matmulWgHeight+1)*matmulWgHeight, (size_t) ((m_rows-1)/matmulWgWidth+1)*matmulWgWidth };

  err = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global, local, 0, NULL, NULL);

  clFinish(queue);

  cl_float *host_result = (cl_float*)clEnqueueMapBuffer(
            queue,
            d_c,
            CL_TRUE,
            CL_MAP_READ,
            0,
            m_rows*d_n_batch*sizeof(float),
            0, NULL, NULL, NULL);

    for(int i = 0; i < m_rows; i++) {
      for(int j = 0; j < n_batch; j++) {
        result[j*m_rows + i] = host_result[i*d_n_batch + j];
      }
    }

  clEnqueueUnmapMemObject(queue,d_c,(void *) host_result,0, NULL, NULL);

  clReleaseKernel(kernel);

}

void PortableMatrixBatchVectorMultiplyAccumulate(const float* matrix,
                                                 int m_rows, int m_cols,
                                                 const float* vector,
                                                 int n_batch, float* result,
                                                 int result_stride) {

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
}

namespace tflite {
namespace tensor_utils {

void NeonMatrixBatchVectorMultiplyAccumulate(const float* matrix, int m_rows,
                                             int m_cols, const float* vector,
                                             int n_batch, float* result,
                                             int result_stride) {

  // If v_size is not divisible by kWeightsPerNeonLane, we cannot use the main
  // vectorized loop, and we need to process sequentially. postamble_start shows
  // the start index where this should happen.
  const int postamble_start =
      m_cols - (m_cols & (kFloatWeightsPerNeonLane - 1));

  // The arrays used to cache the vector.
  float32x4_t* vector_cache_float32x4 =
      new float32x4_t[(m_cols / kFloatWeightsPerNeonLane) *
                      sizeof(float32x4_t)];
  const int kUnrollSize = 2;
  for (int b = 0; b < n_batch; b++) {
    float* result_in_batch = result + b * m_rows * result_stride;
    const float* vector_in_batch = vector + b * m_cols;

    const float* matrix_ptr0 = matrix;
    // If there is only 1 row, we don't want to assign an illegal pointer.
    const float* matrix_ptr1 = nullptr;
    if (m_rows > 1) {
      matrix_ptr1 = matrix + m_cols;
    }

    // Cahce the vector.
    for (int c = 0; c < postamble_start; c += kFloatWeightsPerNeonLane) {
      vector_cache_float32x4[c >> 2] = vld1q_f32(vector_in_batch + c);
    }

    // Main matrix by vector multiplication loop, which handles two rows of
    // matrix by vector multiplication.
    for (int r = 0; r < (m_rows & ~(kUnrollSize - 1)); r += kUnrollSize) {
      float32x4_t acc0_32x4 = vmovq_n_f32(0.0);
      float32x4_t acc1_32x4 = vmovq_n_f32(0.0);
      for (int c = 0; c < postamble_start; c += kFloatWeightsPerNeonLane) {
        float32x4_t temp = vector_cache_float32x4[c >> 2];
        // Load 4 float values from vector1 and vector2 and accumulator.
        float32x4_t v0_f32x4 = vld1q_f32(matrix_ptr0 + c);
        float32x4_t v1_f32x4 = vld1q_f32(matrix_ptr1 + c);
        // Vector multiply-accumulate 4 float
        acc0_32x4 = vmlaq_f32(acc0_32x4, v0_f32x4, temp);
        acc1_32x4 = vmlaq_f32(acc1_32x4, v1_f32x4, temp);
      }
      // Add the 4 intermediate sum values to get the final dot-prod value for
      // this column.
      *result_in_batch +=
          (vgetq_lane_f32(acc0_32x4, 0) + vgetq_lane_f32(acc0_32x4, 1) +
           vgetq_lane_f32(acc0_32x4, 2) + vgetq_lane_f32(acc0_32x4, 3));
      *(result_in_batch + result_stride) +=
          (vgetq_lane_f32(acc1_32x4, 0) + vgetq_lane_f32(acc1_32x4, 1) +
           vgetq_lane_f32(acc1_32x4, 2) + vgetq_lane_f32(acc1_32x4, 3));
      for (int c = postamble_start; c < m_cols; c++) {
        *result_in_batch += matrix_ptr0[c] * vector_in_batch[c];
        *(result_in_batch + result_stride) +=
            matrix_ptr1[c] * vector_in_batch[c];
      }
      matrix_ptr0 += kUnrollSize * m_cols;
      matrix_ptr1 += kUnrollSize * m_cols;
      result_in_batch += kUnrollSize * result_stride;
    }
    for (int r = (m_rows & ~(kUnrollSize - 1)); r < m_rows; r++) {
      float32x4_t acc0_32x4 = vmovq_n_f32(0.0);
      for (int c = 0; c < postamble_start; c += kFloatWeightsPerNeonLane) {
        float32x4_t temp = vector_cache_float32x4[c >> 2];
        // Load 4 float values from vector1 and vector2 and accumulator.
        float32x4_t v0_f32x4 = vld1q_f32(matrix_ptr0 + c);
        // Vector multiply-accumulate 4 float
        acc0_32x4 = vmlaq_f32(acc0_32x4, v0_f32x4, temp);
      }
      // Add the 4 intermediate sum values to get the final dot-prod value for
      // this column.
      *result_in_batch +=
          (vgetq_lane_f32(acc0_32x4, 0) + vgetq_lane_f32(acc0_32x4, 1) +
           vgetq_lane_f32(acc0_32x4, 2) + vgetq_lane_f32(acc0_32x4, 3));
      for (int c = postamble_start; c < m_cols; c++) {
        *result_in_batch += matrix_ptr0[c] * vector_in_batch[c];
      }
      matrix_ptr0 += m_cols;
      result_in_batch += result_stride;
    }
  }
  delete[] vector_cache_float32x4;

}

void TransposeFloatTensor(const float* input, int rows, int cols, float* output) {
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      const float in_value = input[i * cols + j];
      output[j * rows + i] = in_value;
    }
  }
}

float ActivationFunctionWithMinMax(float x, float output_activation_min,
                                          float output_activation_max) {
  return std::min(std::max(x, output_activation_min), output_activation_max);
}

void ConvPort(const float* input_data, 
          const float* filter_data, 
          const float* bias_data, 
          float* output_data,
          int stride_width, int stride_height, 
          int pad_width, int pad_height, 
          const int* dim_sizes, const int* dim_strides,
          float output_activation_min, float output_activation_max) {
  const int batches = dim_sizes[3]; //MatchingArraySize(input_dims, 3, output_dims, 3);
  const int input_depth = dim_sizes[0]; //MatchingArraySize(input_dims, 0, filter_dims, 0);
  const int output_depth = dim_sizes[7]; //MatchingArraySize(filter_dims, 3, output_dims, 0);
  // if (bias_data) {
  //   TFLITE_DCHECK_EQ(ArraySize(filter_dims, 3), ArraySize(bias_dims, 0));
  // }
  const int input_height = dim_sizes[2]; //ArraySize(input_dims, 2);
  const int input_width = dim_sizes[1]; //ArraySize(input_dims, 1);
  const int filter_height = dim_sizes[6]; //ArraySize(filter_dims, 2);
  const int filter_width = dim_sizes[5]; //ArraySize(filter_dims, 1);
  const int output_height = dim_sizes[14]; //ArraySize(output_dims, 2);
  const int output_width = dim_sizes[13]; //ArraySize(output_dims, 1);
  for (int batch = 0; batch < batches; ++batch) {
    for (int out_y = 0; out_y < output_height; ++out_y) {
      for (int out_x = 0; out_x < output_width; ++out_x) {
        for (int out_channel = 0; out_channel < output_depth; ++out_channel) {
          const int in_x_origin = (out_x * stride_width) - pad_width;
          const int in_y_origin = (out_y * stride_height) - pad_height;
          float total = 0.f;
          for (int filter_y = 0; filter_y < filter_height; ++filter_y) {
            for (int filter_x = 0; filter_x < filter_width; ++filter_x) {
              for (int in_channel = 0; in_channel < input_depth; ++in_channel) {
                const int in_x = in_x_origin + filter_x;
                const int in_y = in_y_origin + filter_y;
                // If the location is outside the bounds of the input image,
                // use zero as a default value.
                if ((in_x >= 0) && (in_x < input_width) && (in_y >= 0) &&
                    (in_y < input_height)) {
                  float input_value = input_data[in_channel*dim_strides[0] + in_x*dim_strides[1] + 
                                                  in_y*dim_strides[2] + batch*dim_strides[3]];
                  float filter_value =
                      filter_data[in_channel*dim_strides[4] + filter_x*dim_strides[5] +
                                         filter_y*dim_strides[6] + out_channel*dim_strides[7]];
                  total += (input_value * filter_value);
                }
              }
            }
          }
          float bias_value = 0.0f;
          if (bias_data) {
            bias_value = bias_data[out_channel*dim_strides[8]];
          }
          output_data[out_channel*dim_strides[12] + out_x*dim_strides[13] + out_y*dim_strides[14] + batch*dim_strides[15]] 
            = std::min(std::max(total + bias_value,output_activation_min),output_activation_max);
        }
      }
    }
  }
}

void NeonMatrixBatchVectorMultiplyAccumulateOpenCL(const float* matrix, int m_rows,
                                             int m_cols, const float* vector,
                                             int n_batch, float* result,
                                             int result_stride,
                                             cl_context context_cl, cl_command_queue queue, cl_program program, cl_mem cl_mem_arr[6]) {
  

OpenCLPortableMatrixBatchVectorMultiplyAccumulate(matrix,m_rows,m_cols,vector,n_batch,result,1, context_cl, queue, program, cl_mem_arr);


}

void NeonVectorVectorCwiseProduct(const float* vector1, const float* vector2,
                                  int v_size, float* result) {
  // If v_size is not divisible by kWeightsPerNeonLane, we cannot use the main
  // vectorized loop, and we need to process sequentially. postamble_start shows
  // the start index where this should happen.
  const int postamble_start =
      v_size - (v_size & (kFloatWeightsPerNeonLane - 1));
  for (int v = 0; v < postamble_start; v += kFloatWeightsPerNeonLane) {
    // Load 4 float values from vector1 and vector2.
    float32x4_t v1_f32x4 = vld1q_f32(vector1 + v);
    float32x4_t v2_f32x4 = vld1q_f32(vector2 + v);
    // Vector multiply 4 float
    float32x4_t mul_32x4 = vmulq_f32(v1_f32x4, v2_f32x4);
    // Save to result array.
    vst1q_f32(&result[v], mul_32x4);
  }
  for (int v = postamble_start; v < v_size; v++) {
    result[v] = vector1[v] * vector2[v];
  }
}

void NeonVectorVectorCwiseProductAccumulate(const float* vector1,
                                            const float* vector2, int v_size,
                                            float* result) {
  // If v_size is not divisible by kWeightsPerNeonLane, we cannot use the main
  // vectorized loop, and we need to process sequentially. postamble_start shows
  // the start index where this should happen.
  const int postamble_start =
      v_size - (v_size & (kFloatWeightsPerNeonLane - 1));
  for (int v = 0; v < postamble_start; v += kFloatWeightsPerNeonLane) {
    // Load 4 float values from vector1 and vector2 and accumulator.
    float32x4_t v1_f32x4 = vld1q_f32(vector1 + v);
    float32x4_t v2_f32x4 = vld1q_f32(vector2 + v);
    float32x4_t acc_32x4 = vld1q_f32(result + v);
    // Vector multiply-accumulate 4 float
    acc_32x4 = vmlaq_f32(acc_32x4, v1_f32x4, v2_f32x4);
    // Save to result array.
    vst1q_f32(&result[v], acc_32x4);
  }
  for (int v = postamble_start; v < v_size; v++) {
    result[v] += vector1[v] * vector2[v];
  }
}

void NeonVectorBatchVectorCwiseProductAccumulate(const float* vector,
                                                 int v_size,
                                                 const float* batch_vector,
                                                 int n_batch, float* result) {

  // If v_size is not divisible by kWeightsPerNeonLane, we cannot use the main
  // vectorized loop, and we need to process sequentially. postamble_start shows
  // the start index where this should happen.
  const int postamble_start =
      v_size - (v_size & (kFloatWeightsPerNeonLane - 1));

  // The arrays used to cache the vector.
  float32x4_t* vector_cache_float32x4 =
      new float32x4_t[(v_size / kFloatWeightsPerNeonLane) *
                      sizeof(float32x4_t)];
  for (int v = 0; v < postamble_start; v += kFloatWeightsPerNeonLane) {
    vector_cache_float32x4[v >> 2] = vld1q_f32(vector + v);
  }

  float* result_ptr = result;
  const float* batch_vector_ptr = batch_vector;
  for (int b = 0; b < n_batch; b++) {
    for (int v = 0; v < postamble_start; v += kFloatWeightsPerNeonLane) {
      // Load from memory to vectors.
      float32x4_t result_f32x4 = vld1q_f32(result_ptr + v);
      float32x4_t batch_vector_f32x4 = vld1q_f32(batch_vector_ptr + v);
      // Multiply-accumulate.
      result_f32x4 = vmlaq_f32(result_f32x4, batch_vector_f32x4,
                               vector_cache_float32x4[v >> 2]);
      // Store.
      vst1q_f32(result_ptr + v, result_f32x4);
    }
    // Postamble loop
    for (int v = postamble_start; v < v_size; v++) {
      result_ptr[v] += vector[v] * batch_vector_ptr[v];
    }
    // Update the pointers.
    result_ptr += v_size;
    batch_vector_ptr += v_size;
  }
  delete[] vector_cache_float32x4;
}

void NeonSub1Vector(const float* vector, int v_size, float* result) {

  // If v_size is not divisible by kWeightsPerNeonLane, we cannot use the main
  // vectorized loop, and we need to process sequentially. postamble_start shows
  // the start index where this should happen.
  const int postamble_start =
      v_size - (v_size & (kFloatWeightsPerNeonLane - 1));

  float32x4_t one_f32x4 = vmovq_n_f32(1.0);
  for (int v = 0; v < postamble_start; v += kFloatWeightsPerNeonLane) {
    // Load 4 float values from the current pointers of the input column and
    // subtract from 1.
    float32x4_t v_f32x4 = vld1q_f32(vector + v);
    float32x4_t result_f32x4 = vsubq_f32(one_f32x4, v_f32x4);
    // Save to output.
    vst1q_f32(result + v, result_f32x4);
  }
  for (int v = postamble_start; v < v_size; v++) {
    result[v] = 1.0f - vector[v];
  }
}

void NeonClipVector(const float* vector, int v_size, float abs_limit,
                    float* result) {

  // If v_size is not divisible by kWeightsPerNeonLane, we cannot use the main
  // vectorized loop, and we need to process sequentially. postamble_start shows
  // the start index where this should happen.
  const int postamble_start =
      v_size - (v_size & (kFloatWeightsPerNeonLane - 1));

  // Replicate abs_limit and -abs_limit in two vectors.
  const float32x4_t abs_limit_f32x4 = vmovq_n_f32(abs_limit);
  const float32x4_t neg_abs_limit_f32x4 = vmovq_n_f32(-abs_limit);

  for (int v = 0; v < postamble_start; v += kFloatWeightsPerNeonLane) {
    // Load from memory to vector.
    float32x4_t v_f32x4 = vld1q_f32(vector + v);
    // Clip between abs_limit and -abs_limit.
    float32x4_t result_f32x4 = vminq_f32(abs_limit_f32x4, v_f32x4);
    result_f32x4 = vmaxq_f32(neg_abs_limit_f32x4, result_f32x4);
    // Save to output.
    vst1q_f32(result + v, result_f32x4);
  }
  // Postamble loop.
  for (int v = postamble_start; v < v_size; v++) {
    result[v] = (abs_limit < vector[v]) ? abs_limit : vector[v];
    result[v] = (-abs_limit > result[v]) ? -abs_limit : result[v];
  }
}

float NeonVectorVectorDotProduct(const float* vector1, const float* vector2,
                                 int v_size) {
  // If v_size is not divisible by kWeightsPerNeonLane, we cannot use the main
  // vectorized loop, and we need to process sequentially. postamble_start shows
  // the start index where this should happen.
  const int postamble_start =
      v_size - (v_size & (kFloatWeightsPerNeonLane - 1));
  float32x4_t acc_32x4 = vmovq_n_f32(0.0);
  for (int v = 0; v < postamble_start; v += kFloatWeightsPerNeonLane) {
    // Load 4 float values from vector1 and vector2 and accumulator.
    float32x4_t v1_f32x4 = vld1q_f32(vector1 + v);
    float32x4_t v2_f32x4 = vld1q_f32(vector2 + v);
    // Vector multiply-accumulate 4 float
    acc_32x4 = vmlaq_f32(acc_32x4, v1_f32x4, v2_f32x4);
  }

  float result = (vgetq_lane_f32(acc_32x4, 0) + vgetq_lane_f32(acc_32x4, 1) +
                  vgetq_lane_f32(acc_32x4, 2) + vgetq_lane_f32(acc_32x4, 3));
  // Postamble loop.
  for (int v = postamble_start; v < v_size; v++) {
    result += vector1[v] * vector2[v];
  }
  return result;
}

void NeonBatchVectorBatchVectorDotProduct(const float* vector1,
                                          const float* vector2, int v_size,
                                          int n_batch, float* result,
                                          int result_stride) {
  float* result_ptr = result;
  const float* vector1_ptr = vector1;
  const float* vector2_ptr = vector2;
  for (int b = 0; b < n_batch; b++) {
    *result_ptr = NeonVectorVectorDotProduct(vector1_ptr, vector2_ptr, v_size);
    vector1_ptr += v_size;
    vector2_ptr += v_size;
    result_ptr += result_stride;
  }
}

void NeonReductionSumVector(const float* input_vector, float* output_vector,
                            int output_size, int reduction_size) {
  const float* input_vector_ptr = input_vector;
  for (int o = 0; o < output_size; o++) {
    // If reduction_size is not divisible by kWeightsPerNeonLane, we cannot use
    // the main vectorized loop, and we need to process sequentially.
    // postamble_start shows the start index where this should happen.
    const int postamble_start =
        reduction_size - (reduction_size & (kFloatWeightsPerNeonLane - 1));
    float32x4_t sum_f32x4 = vmovq_n_f32(0.0);
    for (int r = 0; r < postamble_start; r += kFloatWeightsPerNeonLane) {
      float32x4_t v1_f32x4 = vld1q_f32(input_vector_ptr + r);
      sum_f32x4 = vaddq_f32(sum_f32x4, v1_f32x4);
    }
    output_vector[o] +=
        (vgetq_lane_f32(sum_f32x4, 0) + vgetq_lane_f32(sum_f32x4, 1) +
         vgetq_lane_f32(sum_f32x4, 2) + vgetq_lane_f32(sum_f32x4, 3));
    input_vector_ptr += postamble_start;

    // Postamble loop.
    for (int r = postamble_start; r < reduction_size; r++) {
      output_vector[o] += *input_vector_ptr++;
    }
  }
}

void NeonVectorShiftLeft(float* vector, int v_size, float shift_value) {

  // This variable keeps track of the next to the last index which is being
  // copied to make sure we are not out of the vector boundary.
  int last_index_copy = kFloatWeightsPerNeonLane;
  int current_index_copy = 0;
  while (last_index_copy < v_size) {
    float32x4_t v_f32x4 = vld1q_f32(vector + current_index_copy + 1);
    vst1q_f32(vector + current_index_copy, v_f32x4);
    current_index_copy += kFloatWeightsPerNeonLane;
    last_index_copy += kFloatWeightsPerNeonLane;
  }
  // Postamble loop.
  for (int i = current_index_copy; i < v_size - 1; i++) {
    vector[i] = vector[i + 1];
  }
  vector[v_size - 1] = shift_value;
}

}  // namespace tensor_utils
}  // namespace tflite

#endif  // USE_NEON
