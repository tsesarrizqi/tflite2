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

// clblast
#include <clblast.h>

#include <string>
#include <cstring>
// #include <unistd.h>

#include <fstream>
#include <iostream>
#include <vector>

//note: shaderc
// #include "shaderc/shaderc.hpp"

//note: android log
#include <android/log.h> 
#include <stdio.h> 
#include <stdlib.h> 
#include <math.h>

//note: android opencl
#include "../CL/cl.h"
#include <sys/stat.h>

//note: vulkan
#include "../vulkan/vulkan.h"
#include "../vulkan/vk_platform.h"
#include <vector>
#include <string.h>
#include <assert.h>
#include <stdexcept>
#include <cmath>

// note: timer
#include <time.h>
#include <sys/time.h>

#include "halftmp/half.hpp"

#ifdef USE_NEON

#define kFloatWeightsPerNeonLane 4

// Vulkan
// const int WIDTH = 3200; // Size of rendered mandelbrot set.
// const int HEIGHT = 2400; // Size of renderered mandelbrot set.
// const int WORKGROUP_SIZE = 32; // Workgroup size in compute shader.

// Used for validating return values of Vulkan API calls.
#define VK_CHECK_RESULT(f)                                        \
{                                                   \
    VkResult res = (f);                                         \
    if (res != VK_SUCCESS)                                        \
    {                                                 \
        __android_log_print(ANDROID_LOG_INFO, "Vulkanerror", "Fatal : VkResult is %d in %s at line %d\n", res,  __FILE__, __LINE__); \
        assert(res == VK_SUCCESS);                                    \
    }                                                 \
}

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
using namespace clblast;

// void OpenCLPortableMatrixBatchVectorMultiplyAccumulate(const float* matrix,
//                                                  int m_rows, int m_cols,
//                                                  const float* vector,
//                                                  int n_batch, float* result,
//                                                  int result_stride,
//                                                  cl_context context, cl_command_queue queue, cl_program program) {

//   int matrixsize = m_rows*m_cols*sizeof(cl_half);
//   int vectorsize = m_cols*n_batch*sizeof(cl_half);
//   int resultsize = m_rows*n_batch*sizeof(cl_half);

//   // Test half precision
//   cl_half* matrixHalf = (cl_half*) malloc(m_rows*m_cols*sizeof(cl_half));
//   for(int i = 0; i < m_rows*m_cols; i++) {
//     // half halfTmp(matrix[i]);
//     matrixHalf[i] = half_cast<half>(matrix[i]);
//   }
//   cl_half* vectorHalf = (cl_half*) malloc(m_cols*n_batch*sizeof(cl_half));
//   for(int i = 0; i < m_cols*n_batch; i++) {
//     // half halfTmp(vector[i]);
//     vectorHalf[i] = half_cast<half>(vector[i]);
//   }
//   cl_half* resultHalf = (cl_half*) malloc(m_rows*n_batch*sizeof(cl_half));

//   cl_mem d_a;
//   cl_mem d_at;
//   cl_mem d_b;
//   cl_mem d_c;
      
//   cl_kernel kernel, kernel2;

//   size_t localSizetmp;
//   cl_int err;

//   double wall0 = get_wall_time();
//   double cpu0  = get_cpu_time();

//   kernel = clCreateKernel(program, "matrixVectorMul", &err);
//   kernel2 = clCreateKernel(program, "transpose", &err);

//   double wall1 = get_wall_time();
//   double cpu1  = get_cpu_time();

//   double wall = wall1 - wall0;
//   double cpu = cpu1 - cpu0;
  
//   __android_log_print(ANDROID_LOG_INFO, "Matmulruntime", "createkernel: %lf", wall);

//   wall0 = get_wall_time();
//   cpu0  = get_cpu_time();

//   // | CL_MEM_ALLOC_HOST_PTR
//   d_a = clCreateBuffer(context, CL_MEM_READ_ONLY, matrixsize, NULL, NULL);
//   d_b = clCreateBuffer(context, CL_MEM_READ_ONLY, vectorsize, NULL, NULL);
//   d_c = clCreateBuffer(context, CL_MEM_WRITE_ONLY, resultsize, NULL, NULL);
//   d_at = clCreateBuffer(context, CL_MEM_READ_WRITE, matrixsize, NULL, NULL);

//   wall1 = get_wall_time();
//   cpu1  = get_cpu_time();

//   wall = wall1 - wall0;
//   cpu = cpu1 - cpu0;
  
//   __android_log_print(ANDROID_LOG_INFO, "Matmulruntime", "createbuffer: %lf", wall);


//   wall0 = get_wall_time();
//   cpu0  = get_cpu_time();

//   err = clEnqueueWriteBuffer(queue, d_a, CL_TRUE, 0,
//                                  m_rows*m_cols*sizeof(cl_half), matrixHalf, 0, NULL, NULL);
//   err = clEnqueueWriteBuffer(queue, d_b, CL_TRUE, 0,
//                                  m_cols*n_batch*sizeof(cl_half), vectorHalf, 0, NULL, NULL);

//   // cl_float *host_a = (cl_float*)clEnqueueMapBuffer(
//   //             queue,
//   //             d_a,
//   //             CL_TRUE,
//   //             CL_MAP_WRITE,
//   //             0,
//   //             matrixsize,
//   //             0, NULL, NULL, NULL);
//   // cl_float *host_b = (cl_float*)clEnqueueMapBuffer(
//   //             queue,
//   //             d_b,
//   //             CL_TRUE,
//   //             CL_MAP_WRITE,
//   //             0,
//   //             vectorsize,
//   //             0, NULL, NULL, NULL);

//   // std::memcpy(host_a, matrix, matrixsize);
//   // std::memcpy(host_b, vector, vectorsize);

//   // clEnqueueUnmapMemObject(queue,d_a,(void *) host_a,0, NULL, NULL);
//   // clEnqueueUnmapMemObject(queue,d_b,(void *) host_b,0, NULL, NULL);

//   clFinish(queue);

//   wall1 = get_wall_time();
//   cpu1  = get_cpu_time();

//   wall = wall1 - wall0;
//   cpu = cpu1 - cpu0;
  
//   __android_log_print(ANDROID_LOG_INFO, "Matmulruntime", "writebuffer: %lf", wall);

//   err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_c);
//   err  = clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_at);
//   err  = clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_b);
//   err  = clSetKernelArg(kernel, 3, sizeof(int), &m_cols);
//   err  = clSetKernelArg(kernel, 4, sizeof(int), &m_rows);
//   err  = clSetKernelArg(kernel, 5, sizeof(int), &n_batch);

//   err  = clSetKernelArg(kernel2, 0, sizeof(cl_mem), &d_a);
//   err  = clSetKernelArg(kernel2, 1, sizeof(cl_mem), &d_at);
//   err  = clSetKernelArg(kernel2, 2, sizeof(int), &m_rows);
//   err  = clSetKernelArg(kernel2, 3, sizeof(int), &m_cols);

//   const int TS = 32;
//   // const size_t localSize = (size_t)TS;
//   // const size_t globalSize0 = (size_t)(((m_rows-1)/TS+1)*TS);
//   // const size_t globalSize1 = (size_t)(((n_batch-1)/TS+1)*TS);

//   // __android_log_print(ANDROID_LOG_INFO, "Matmulruntime", "GlobalSize0: %d", globalSize0);
//   // __android_log_print(ANDROID_LOG_INFO, "Matmulruntime", "GlobalSize1: %d", globalSize1);
//   // __android_log_print(ANDROID_LOG_INFO, "Matmulruntime", "localSize: %d", localSize);

//   // const size_t local[2] = { localSize, localSize };
//   // const size_t global[2] = { (size_t)(((m_rows-1)/TS+1)*TS), (size_t)(((n_batch-1)/TS+1)*TS) };

//   // __android_log_print(ANDROID_LOG_INFO, "Matmulruntime", "matrixsize: %d %d %d", m_rows, m_cols, n_batch);

//   // const size_t local[2] = { (size_t) TS, (size_t) (TS/8) };
//   // const size_t global[2] = { (size_t) (((m_rows-1)/32+1)*32), (size_t) (((n_batch-1)/32+1)*4) };

//   const size_t local[2] = { 8, 32 };
//   const size_t global[2] = { (size_t) (((m_rows-1)/8+1)*8), 32 };
//   const size_t local2[2] = { 8, 32 };
//   const size_t global2[2] = { (size_t) (((m_rows-1)/8+1)*8), (size_t) (m_cols/4) };

//   // const size_t local2[2] = { 16, 16 };
//   // const size_t global2[2] = { (size_t) m_cols, (size_t) m_rows };

//   wall0 = get_wall_time();
//   cpu0  = get_cpu_time();

//   err = clEnqueueNDRangeKernel(queue, kernel2, 2, NULL, global2, local2, 0, NULL, NULL);

//   __android_log_print(ANDROID_LOG_INFO, "Matmulruntime", "Matmulerror2 %d", err);

//   err = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global, local, 0, NULL, NULL);

//   clFinish(queue);

//   __android_log_print(ANDROID_LOG_INFO, "Matmulruntime", "Matmulerror1: %d", err);

//   wall1 = get_wall_time();
//   cpu1  = get_cpu_time();

//   wall = wall1 - wall0;
//   cpu = cpu1 - cpu0;
  
//   __android_log_print(ANDROID_LOG_INFO, "Matmulruntime", "runkernelOclMatmul: %lf", wall);

//   wall0 = get_wall_time();
//   cpu0  = get_cpu_time();

//   clEnqueueReadBuffer(queue, d_c, CL_TRUE, 0, resultsize, resultHalf, 0, NULL, NULL );

//   clFinish(queue);

//   half halfTmp2 = half_cast<half>(matrix[0]);
//   for(int i = 0; i < m_rows*n_batch; i++) {
//     result[i] = (float) halfTmp2;
//   }

//   wall1 = get_wall_time();
//   cpu1  = get_cpu_time();

//   wall = wall1 - wall0;
//   cpu = cpu1 - cpu0;
  
//   __android_log_print(ANDROID_LOG_INFO, "Matmulruntime", "readbuffer: %lf", wall);

//   wall0 = get_wall_time();
//   cpu0  = get_cpu_time();

//   clReleaseMemObject(d_a);
//   clReleaseMemObject(d_b);
//   clReleaseMemObject(d_c);
//   clReleaseKernel(kernel);

//   wall1 = get_wall_time();
//   cpu1  = get_cpu_time();

//   wall = wall1 - wall0;
//   cpu = cpu1 - cpu0;
  
//   __android_log_print(ANDROID_LOG_INFO, "Matmulruntime", "cleaning: %lf", wall);

// }

void OpenCLPortableMatrixBatchVectorMultiplyAccumulate(float* matrix,
                                                 int m_rows, int m_cols,
                                                 float* vector,
                                                 int n_batch, float* result,
                                                 int result_stride,
                                                 cl_context context, cl_command_queue queue, cl_program program) {

  int matrixsize = m_rows*m_cols*sizeof(float);
  int vectorsize = m_cols*n_batch*sizeof(float);
  int resultsize = m_rows*n_batch*sizeof(float);

  cl_mem d_a;
  cl_mem d_b;
  cl_mem d_c;
  cl_kernel kernel;
  cl_int err;

  double wall0 = get_wall_time();
  double cpu0  = get_cpu_time();

  kernel = clCreateKernel(program, "matrixVectorMulF4", &err);

  double wall1 = get_wall_time();
  double cpu1  = get_cpu_time();

  double wall = wall1 - wall0;
  double cpu = cpu1 - cpu0;
  
  __android_log_print(ANDROID_LOG_INFO, "Matmulruntime", "createkernel: %lf", wall);

  wall0 = get_wall_time();
  cpu0  = get_cpu_time();

  // | CL_MEM_ALLOC_HOST_PTR
  d_a = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, matrixsize, matrix, NULL);
  d_b = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, vectorsize, vector, NULL);
  d_c = clCreateBuffer(context, CL_MEM_WRITE_ONLY, resultsize, NULL, NULL);

  wall1 = get_wall_time();
  cpu1  = get_cpu_time();

  wall = wall1 - wall0;
  cpu = cpu1 - cpu0;
  
  __android_log_print(ANDROID_LOG_INFO, "Matmulruntime", "createbuffer: %lf", wall);


  // wall0 = get_wall_time();
  // cpu0  = get_cpu_time();

  // err = clEnqueueWriteBuffer(queue, d_a, CL_TRUE, 0,
  //                                matrixsize, matrix, 0, NULL, NULL);
  // err = clEnqueueWriteBuffer(queue, d_b, CL_TRUE, 0,
  //                               vectorsize, vector, 0, NULL, NULL);

  // clFinish(queue);

  // wall1 = get_wall_time();
  // cpu1  = get_cpu_time();

  // wall = wall1 - wall0;
  // cpu = cpu1 - cpu0;
  
  // __android_log_print(ANDROID_LOG_INFO, "Matmulruntime", "writebuffer: %lf", wall);

  err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_c);
  err  = clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_a);
  err  = clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_b);
  err  = clSetKernelArg(kernel, 3, sizeof(int), &m_cols);
  err  = clSetKernelArg(kernel, 4, sizeof(int), &m_rows);
  err  = clSetKernelArg(kernel, 5, sizeof(int), &n_batch);

  const size_t local[2] = { 8, 32 };
  const size_t global[2] = { (size_t) (((m_rows/4-1)/8+1)*8), 32 };

  wall0 = get_wall_time();
  cpu0  = get_cpu_time();

  err = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global, local, 0, NULL, NULL);

  clFinish(queue);

  __android_log_print(ANDROID_LOG_INFO, "Matmulruntime", "Matmulerror1: %d", err);

  wall1 = get_wall_time();
  cpu1  = get_cpu_time();

  wall = wall1 - wall0;
  cpu = cpu1 - cpu0;
  
  __android_log_print(ANDROID_LOG_INFO, "Matmulruntime", "runkernelOclMatmul: %lf", wall);

  wall0 = get_wall_time();
  cpu0  = get_cpu_time();

  clEnqueueReadBuffer(queue, d_c, CL_TRUE, 0, resultsize, result, 0, NULL, NULL );

  clFinish(queue);

  wall1 = get_wall_time();
  cpu1  = get_cpu_time();

  wall = wall1 - wall0;
  cpu = cpu1 - cpu0;
  
  __android_log_print(ANDROID_LOG_INFO, "Matmulruntime", "readbuffer: %lf", wall);

  wall0 = get_wall_time();
  cpu0  = get_cpu_time();

  clReleaseMemObject(d_a);
  clReleaseMemObject(d_b);
  clReleaseMemObject(d_c);
  clReleaseKernel(kernel);

  wall1 = get_wall_time();
  cpu1  = get_cpu_time();

  wall = wall1 - wall0;
  cpu = cpu1 - cpu0;
  
  __android_log_print(ANDROID_LOG_INFO, "Matmulruntime", "cleaning: %lf", wall);

}

void PortableMatrixBatchVectorMultiplyAccumulate(const float* matrix,
                                                 int m_rows, int m_cols,
                                                 const float* vector,
                                                 int n_batch, float* result,
                                                 int result_stride) {
  // // Start Timers
  // double wall0 = get_wall_time();
  // double cpu0  = get_cpu_time();
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

  // // Stop timers
  // double wall1 = get_wall_time();
  // double cpu1  = get_cpu_time();

  // double wall = wall1 - wall0;
  // double cpu = cpu1 - cpu0;

  // // note: andoird log
  // __android_log_print(ANDROID_LOG_INFO, "Portablematmul", "runkernel: %lf", wall);
}

namespace tflite {
namespace tensor_utils {

void NeonMatrixBatchVectorMultiplyAccumulate(const float* matrix, int m_rows,
                                             int m_cols, const float* vector,
                                             int n_batch, float* result,
                                             int result_stride) {
  // Start Timers
  double wall0 = get_wall_time();
  double cpu0  = get_cpu_time();

  // PortableMatrixBatchVectorMultiplyAccumulate(matrix,m_rows,m_cols,vector,n_batch,result,1);
  // OpenCLPortableMatrixBatchVectorMultiplyAccumulate(matrix,m_rows,m_cols,vector,n_batch,result,1);

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

  // Stop timers
  double wall1 = get_wall_time();
  double cpu1  = get_cpu_time();

  double wall = wall1 - wall0;
  double cpu = cpu1 - cpu0;

  // note: andoird log
  __android_log_print(ANDROID_LOG_INFO, "Matmulruntime", "Walltime: %lf", wall);
}

void TransposeFloatTensor(const float* input, int rows, int cols, float* output) {
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      const float in_value = input[i * cols + j];
      output[j * rows + i] = in_value;
    }
  }
}

class ComputeApplication {

private:
    VkInstance instance;

    VkPhysicalDevice physicalDevice;

    VkDevice device;

    VkPipeline pipeline;
    VkPipelineLayout pipelineLayout;
    VkShaderModule computeShaderModule;

    VkCommandPool commandPool;
    VkCommandBuffer commandBuffer;

    VkDescriptorPool descriptorPool;
    VkDescriptorSet descriptorSet;
    VkDescriptorSetLayout descriptorSetLayout;

    VkBuffer matrixA, matrixB, matrixC, matrixSizes;
    VkDeviceMemory bufferMemory;
        
    uint32_t matrixASize, matrixBSize, matrixCSize, matrixSizesSize; // size of `buffer` in bytes.

    VkQueue queue; // a queue supporting compute operations.

    uint32_t queueFamilyIndex;
    int M,K,N;
    float* matA;
    float* matB;
    float* matC;

public:
    void run(const float* matA0, const float* matB0, float* matC0, int M0, int K0, int N0,
      VkPhysicalDevice physicalDevice0, VkDevice device0, VkPipeline pipelineMatmul0, VkPipelineLayout pipelineLayoutMatmul0, 
    VkDescriptorSetLayout descriptorSetLayoutMatmul0, VkQueue queueV0, uint32_t queueFamilyIndex0) {
        
        physicalDevice = physicalDevice0; 
        device = device0;
        pipeline = pipelineMatmul0;
        pipelineLayout = pipelineLayoutMatmul0;
        descriptorSetLayout = descriptorSetLayoutMatmul0;
        queue = queueV0;
        queueFamilyIndex = queueFamilyIndex0;


        // Buffer size of the storage buffer that will contain the rendered mandelbrot set.
        matrixASize = (uint32_t) (sizeof(float) * M0 * K0);
        matrixBSize = (uint32_t) (sizeof(float) * K0 * N0);
        matrixCSize = (uint32_t) (sizeof(float) * M0 * N0);
        matrixSizesSize = (uint32_t) (sizeof(int) * 4);
        // matrixSizesSize = (uint32_t) (sizeof(float) * 4);
        M = M0;
        K = K0;
        N = N0;
        matA = (float*) matA0;
        matB = (float*) matB0;
        matC = matC0;

        // // Start Timers
        double wall0 = get_wall_time();
        double cpu0  = get_cpu_time();
        // createInstance();
        // // Stop timers
        double wall1 = get_wall_time();
        double cpu1  = get_cpu_time();
        double wall = wall1 - wall0;
        double cpu = cpu1 - cpu0;
        // __android_log_print(ANDROID_LOG_INFO, "VulkanConvDetail", "createInstance: %lf", wall);
        
        // wall0 = get_wall_time();
        // cpu0  = get_cpu_time();
        // findPhysicalDevice();
        // wall1 = get_wall_time();
        // cpu1  = get_cpu_time();
        // wall = wall1 - wall0;
        // cpu = cpu1 - cpu0;
        // __android_log_print(ANDROID_LOG_INFO, "VulkanConvDetail", "findPhysicalDevice: %lf", wall);
        
        // wall0 = get_wall_time();
        // cpu0  = get_cpu_time();
        // createDevice();
        // wall1 = get_wall_time();
        // cpu1  = get_cpu_time();
        // wall = wall1 - wall0;
        // cpu = cpu1 - cpu0;
        // __android_log_print(ANDROID_LOG_INFO, "VulkanConvDetail", "createDevice: %lf", wall);
        
        wall0 = get_wall_time();
        cpu0  = get_cpu_time();
        createBuffer();
        wall1 = get_wall_time();
        cpu1  = get_cpu_time();
        wall = wall1 - wall0;
        cpu = cpu1 - cpu0;
        __android_log_print(ANDROID_LOG_INFO, "VulkanConvDetail", "createBuffer: %lf", wall);
        
        // wall0 = get_wall_time();
        // cpu0  = get_cpu_time();
        // createDescriptorSetLayout();
        // wall1 = get_wall_time();
        // cpu1  = get_cpu_time();
        // wall = wall1 - wall0;
        // cpu = cpu1 - cpu0;
        // __android_log_print(ANDROID_LOG_INFO, "VulkanConvDetail", "createDescriptorSetLayout: %lf", wall);
        
        wall0 = get_wall_time();
        cpu0  = get_cpu_time();
        createDescriptorSet();
        wall1 = get_wall_time();
        cpu1  = get_cpu_time();
        wall = wall1 - wall0;
        cpu = cpu1 - cpu0;
        __android_log_print(ANDROID_LOG_INFO, "VulkanConvDetail", "createDescriptorSet: %lf", wall);
        
        // wall0 = get_wall_time();
        // cpu0  = get_cpu_time();
        // createComputePipeline();
        // wall1 = get_wall_time();
        // cpu1  = get_cpu_time();
        // wall = wall1 - wall0;
        // cpu = cpu1 - cpu0;
        // __android_log_print(ANDROID_LOG_INFO, "VulkanConvDetail", "createComputePipeline: %lf", wall);
        
        wall0 = get_wall_time();
        cpu0  = get_cpu_time();
        createCommandBuffer();
        wall1 = get_wall_time();
        cpu1  = get_cpu_time();
        wall = wall1 - wall0;
        cpu = cpu1 - cpu0;
        __android_log_print(ANDROID_LOG_INFO, "VulkanConvDetail", "createCommandBuffer: %lf", wall);
        
        wall0 = get_wall_time();
        cpu0  = get_cpu_time();
        runCommandBuffer();
        wall1 = get_wall_time();
        cpu1  = get_cpu_time();
        wall = wall1 - wall0;
        cpu = cpu1 - cpu0;
        __android_log_print(ANDROID_LOG_INFO, "VulkanConvDetail", "runCommandBuffer: %lf", wall);
        
        wall0 = get_wall_time();
        cpu0  = get_cpu_time();
        getresult();
        wall1 = get_wall_time();
        cpu1  = get_cpu_time();
        wall = wall1 - wall0;
        cpu = cpu1 - cpu0;
        __android_log_print(ANDROID_LOG_INFO, "VulkanConvDetail", "getResult: %lf", wall);
        
        wall0 = get_wall_time();
        cpu0  = get_cpu_time();
        cleanup();
        wall1 = get_wall_time();
        cpu1  = get_cpu_time();
        wall = wall1 - wall0;
        cpu = cpu1 - cpu0;
        __android_log_print(ANDROID_LOG_INFO, "VulkanConvDetail", "cleanup: %lf", wall);
    }

    // void createInstance() {
    //     VkApplicationInfo applicationInfo = {};
    //     applicationInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    //     applicationInfo.pApplicationName = "Matrix Multiplication";
    //     applicationInfo.applicationVersion = 0;
    //     applicationInfo.pEngineName = "Naive";
    //     applicationInfo.engineVersion = 0;
    //     applicationInfo.apiVersion = VK_MAKE_VERSION(1, 0, 31);
        
    //     VkInstanceCreateInfo createInfo = {};
    //     createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    //     createInfo.flags = 0;
    //     createInfo.pApplicationInfo = &applicationInfo;

    //     VK_CHECK_RESULT(vkCreateInstance(
    //         &createInfo,
    //         NULL,
    //         &instance));
    // }

    // void findPhysicalDevice() {
    //     uint32_t deviceCount;
    //     vkEnumeratePhysicalDevices(instance, &deviceCount, NULL);
    //     if (deviceCount == 0) {
    //         throw std::runtime_error("could not find a device with vulkan support");
    //     }

    //     std::vector<VkPhysicalDevice> devices(deviceCount);
    //     vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());

    //     for (VkPhysicalDevice device : devices) {
    //         if (true) { // As above stated, we do no feature checks, so just accept.
    //             physicalDevice = device;
    //             break;
    //         }
    //     }
    // }

    // // Returns the index of a queue family that supports compute operations. 
    // uint32_t getComputeQueueFamilyIndex() {
    //     uint32_t queueFamilyCount;

    //     vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, NULL);

    //     std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
    //     vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, queueFamilies.data());

    //     uint32_t i = 0;
    //     for (; i < queueFamilies.size(); ++i) {
    //         VkQueueFamilyProperties props = queueFamilies[i];

    //         if (props.queueCount > 0 && (props.queueFlags & VK_QUEUE_COMPUTE_BIT)) {
    //             break;
    //         }
    //     }

    //     if (i == queueFamilies.size()) {
    //         throw std::runtime_error("could not find a queue family that supports operations");
    //     }

    //     return i;
    // }

    // void createDevice() {
    //     VkDeviceQueueCreateInfo queueCreateInfo = {};
    //     queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    //     queueFamilyIndex = getComputeQueueFamilyIndex(); // find queue family with compute capability.
    //     queueCreateInfo.queueFamilyIndex = queueFamilyIndex;
    //     queueCreateInfo.queueCount = 1; // create one queue in this family. We don't need more.
    //     float queuePriorities = 1.0;  // we only have one queue, so this is not that imporant. 
    //     queueCreateInfo.pQueuePriorities = &queuePriorities;

    //     VkDeviceCreateInfo deviceCreateInfo = {};

    //     VkPhysicalDeviceFeatures deviceFeatures = {};

    //     deviceCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    //     deviceCreateInfo.pQueueCreateInfos = &queueCreateInfo; // when creating the logical device, we also specify what queues it has.
    //     deviceCreateInfo.queueCreateInfoCount = 1;
    //     deviceCreateInfo.pEnabledFeatures = &deviceFeatures;

    //     VK_CHECK_RESULT(vkCreateDevice(physicalDevice, &deviceCreateInfo, NULL, &device)); // create logical device.

    //     vkGetDeviceQueue(device, queueFamilyIndex, 0, &queue);
    // }

    // find memory type with desired properties.
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

    void createBuffer() {

        VkBufferCreateInfo matrixACreateInfo = {};
        matrixACreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        matrixACreateInfo.size = matrixASize+matrixBSize+matrixCSize+matrixSizesSize; // buffer size in bytes. 
        matrixACreateInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT; // buffer is used as a storage buffer.
        matrixACreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE; // buffer is exclusive to a single queue family at a time. 

        VK_CHECK_RESULT(vkCreateBuffer(device, &matrixACreateInfo, NULL, &matrixA)); // create buffer.

        VkMemoryRequirements memoryRequirementsmatrixA;
        vkGetBufferMemoryRequirements(device, matrixA, &memoryRequirementsmatrixA);
        
        const VkDeviceSize memorySize = memoryRequirementsmatrixA.size;

        VkMemoryAllocateInfo allocateInfo = {};
        allocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocateInfo.allocationSize = memorySize; // specify required memory.

        allocateInfo.memoryTypeIndex = findMemoryType(
            memorySize, VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);

        VK_CHECK_RESULT(vkAllocateMemory(device, &allocateInfo, NULL, &bufferMemory));

        int* matSizetmp;
        VK_CHECK_RESULT(vkMapMemory(device, bufferMemory, 0, matrixSizesSize, 0, (void **) &matSizetmp));

        matSizetmp[0] = M;
        matSizetmp[1] = K;
        matSizetmp[2] = N;

        vkUnmapMemory(device, bufferMemory);

        float* matBuffertmp;
        VK_CHECK_RESULT(vkMapMemory(device, bufferMemory, matrixSizesSize, matrixASize+matrixBSize+matrixCSize, 0, (void **) &matBuffertmp));
        // VK_CHECK_RESULT(vkMapMemory(device, bufferMemory, 0, matrixASize+matrixBSize+matrixCSize+matrixSizesSize, 0, (void **) &matBuffertmp));

        // matBuffertmp[0] = M;
        // matBuffertmp[1] = K;
        // matBuffertmp[2] = N;

        int offset = 0;
        // for(int i = 0; i < M*K; i++) {
        //     matBuffertmp[offset+i] = matA[i]; 
        // }
        TransposeFloatTensor(matA, M, K, matBuffertmp);
        offset = M*K;
        for(int i = 0; i < N*K; i++) {
            matBuffertmp[offset+i] = matB[i]; 
        }

        vkUnmapMemory(device, bufferMemory);

        VK_CHECK_RESULT(vkBindBufferMemory(device, matrixA, bufferMemory, 0));
    }

    // void createDescriptorSetLayout() {
    //     VkDescriptorSetLayoutBinding descriptorSetLayoutBinding;

    //     descriptorSetLayoutBinding = {};
    //     descriptorSetLayoutBinding.binding = 0; // binding = 0
    //     descriptorSetLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    //     descriptorSetLayoutBinding.descriptorCount = 1;
    //     descriptorSetLayoutBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    //     VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCreateInfo = {};
    //     descriptorSetLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    //     descriptorSetLayoutCreateInfo.bindingCount = 1; // only a single binding in this descriptor set layout. 
    //     descriptorSetLayoutCreateInfo.pBindings = &descriptorSetLayoutBinding; 

    //     VK_CHECK_RESULT(vkCreateDescriptorSetLayout(device, &descriptorSetLayoutCreateInfo, NULL, &descriptorSetLayout));
    // }

    void createDescriptorSet() {

        VkDescriptorPoolSize descriptorPoolSize;

        descriptorPoolSize = {};
        descriptorPoolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        descriptorPoolSize.descriptorCount = 1;

        VkDescriptorPoolCreateInfo descriptorPoolCreateInfo = {};
        descriptorPoolCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        descriptorPoolCreateInfo.maxSets = 1; // we only need to allocate one descriptor set from the pool.
        descriptorPoolCreateInfo.poolSizeCount = 1;
        descriptorPoolCreateInfo.pPoolSizes = &descriptorPoolSize;

        VK_CHECK_RESULT(vkCreateDescriptorPool(device, &descriptorPoolCreateInfo, NULL, &descriptorPool));

        VkDescriptorSetAllocateInfo descriptorSetAllocateInfo = {};
        descriptorSetAllocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO; 
        descriptorSetAllocateInfo.descriptorPool = descriptorPool; // pool to allocate from.
        descriptorSetAllocateInfo.descriptorSetCount = 1; // allocate a single descriptor set.
        descriptorSetAllocateInfo.pSetLayouts = &descriptorSetLayout;

        VK_CHECK_RESULT(vkAllocateDescriptorSets(device, &descriptorSetAllocateInfo, &descriptorSet));

        VkDescriptorBufferInfo descriptorBufferInfoMatA = {};
        descriptorBufferInfoMatA.buffer = matrixA;
        descriptorBufferInfoMatA.offset = 0;
        descriptorBufferInfoMatA.range = VK_WHOLE_SIZE;

        VkWriteDescriptorSet writeDescriptorSet;

        writeDescriptorSet = {};
        writeDescriptorSet.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writeDescriptorSet.dstSet = descriptorSet; // write to this descriptor set.
        writeDescriptorSet.dstBinding = 0; // write to the first, and only binding.
        writeDescriptorSet.descriptorCount = 1; // update a single descriptor.
        writeDescriptorSet.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; // storage buffer.
        writeDescriptorSet.pBufferInfo = &descriptorBufferInfoMatA;

        vkUpdateDescriptorSets(device, 1, &writeDescriptorSet, 0, NULL);
    }

    // void createComputePipeline() {
    //     std::string source =
    //       "#version 450 \n" \
    //       "#extension GL_ARB_separate_shader_objects : enable \n" \
    //       "layout(local_size_x = 32, local_size_y = 8, local_size_z = 1) in; \n" \
    //       "layout(binding = 0) buffer matrixA { \n" \
    //       "    vec4 aA[]; \n" \
    //       "}; \n" \
    //       "void main() { \n" \
    //       "    int row = int(gl_GlobalInvocationID.x)*4; \n" \
    //       "    int col = int(gl_GlobalInvocationID.y); \n" \
    //       "    vec4 mk = aA[0];  \n" \
    //       "    int mM = int(mk.x);  \n" \
    //       "    int kK = int(mk.y);  \n" \
    //       "    if ((row < mM) && (col < 1)) { \n" \
    //       "        vec4 sum = {0.0, 0.0, 0.0, 0.0}; \n" \
    //       "        for(int i = 1; i <= kK/4; i++){ \n" \
    //       "            vec4 currb = aA[(mM*kK/4) + i];\n" \
    //       "            sum.x += dot(aA[(row*kK/4) + i],currb);\n" \
    //       "            sum.y += dot(aA[((row+1)*kK/4) + i],currb); \n" \
    //       "            sum.z += dot(aA[((row+2)*kK/4) + i],currb);\n" \
    //       "            sum.w += dot(aA[((row+3)*kK/4) + i],currb);\n" \
    //       "        } \n" \
    //       "        aA[1 + (mM*kK/4) + (kK/4) + (row/4)] = sum; \n" \
    //       "    } \n" \
    //       "}";

    //     shaderc::Compiler compiler;
    //     shaderc::CompileOptions options;

    //     shaderc::SpvCompilationResult module = compiler.CompileGlslToSpv(
    //       source.c_str(), source.size(), shaderc_glsl_compute_shader, "matmul.glsl", options);

    //     if (module.GetCompilationStatus() !=
    //         shaderc_compilation_status_success) {
    //     }

    //     std::vector<uint32_t> code(module.cbegin(), module.cend());

    //     VkShaderModuleCreateInfo createInfo = {};
    //     createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    //     createInfo.pCode = code.data();
    //     createInfo.codeSize = sizeof(uint32_t)*code.size();
        
    //     VK_CHECK_RESULT(vkCreateShaderModule(device, &createInfo, NULL, &computeShaderModule));

    //     VkPipelineShaderStageCreateInfo shaderStageCreateInfo = {};
    //     shaderStageCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    //     shaderStageCreateInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    //     shaderStageCreateInfo.module = computeShaderModule;
    //     shaderStageCreateInfo.pName = "main";

    //     VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = {};
    //     pipelineLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    //     pipelineLayoutCreateInfo.setLayoutCount = 1;
    //     pipelineLayoutCreateInfo.pSetLayouts = &descriptorSetLayout; 
        
    //     VK_CHECK_RESULT(vkCreatePipelineLayout(device, &pipelineLayoutCreateInfo, NULL, &pipelineLayout));

    //     VkComputePipelineCreateInfo pipelineCreateInfo = {};
    //     pipelineCreateInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    //     pipelineCreateInfo.stage = shaderStageCreateInfo;
    //     pipelineCreateInfo.layout = pipelineLayout;

    //     VK_CHECK_RESULT(vkCreateComputePipelines(
    //         device, VK_NULL_HANDLE,
    //         1, &pipelineCreateInfo,
    //         NULL, &pipeline));
    // }

    void createCommandBuffer() {
        VkCommandPoolCreateInfo commandPoolCreateInfo = {};
        commandPoolCreateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        commandPoolCreateInfo.flags = 0;
        commandPoolCreateInfo.queueFamilyIndex = queueFamilyIndex;
        VK_CHECK_RESULT(vkCreateCommandPool(device, &commandPoolCreateInfo, NULL, &commandPool));

        VkCommandBufferAllocateInfo commandBufferAllocateInfo = {};
        commandBufferAllocateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        commandBufferAllocateInfo.commandPool = commandPool;

        commandBufferAllocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        commandBufferAllocateInfo.commandBufferCount = 1; // allocate a single command buffer. 
        VK_CHECK_RESULT(vkAllocateCommandBuffers(device, &commandBufferAllocateInfo, &commandBuffer)); // allocate command buffer.

        VkCommandBufferBeginInfo beginInfo = {};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT; // the buffer is only submitted and used once in this application.
        VK_CHECK_RESULT(vkBeginCommandBuffer(commandBuffer, &beginInfo)); // start recording commands.

        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1, &descriptorSet, 0, NULL);

        // const size_t local[2] = { 8, 32 };
        // const size_t global[2] = { (size_t) (((m_rows-1)/8+1)*8), 32 };

        vkCmdDispatch(commandBuffer, ((M/4)-1)/8+1, 1, 1);
        // vkCmdDispatch(commandBuffer, (M-1)/8+1, 1, 1);

        VK_CHECK_RESULT(vkEndCommandBuffer(commandBuffer)); // end recording commands.
    }

    void runCommandBuffer() {
        VkSubmitInfo submitInfo = {};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.commandBufferCount = 1; // submit a single command buffer
        submitInfo.pCommandBuffers = &commandBuffer; // the command buffer to submit.

        // Start Timers
        double wall0 = get_wall_time();
        double cpu0  = get_cpu_time();

        VK_CHECK_RESULT(vkQueueSubmit(queue, 1, &submitInfo, 0));
        VK_CHECK_RESULT(vkQueueWaitIdle(queue));

        // Stop timers
        double wall1 = get_wall_time();
        double cpu1  = get_cpu_time();

        double wall = wall1 - wall0;
        double cpu = cpu1 - cpu0;

        __android_log_print(ANDROID_LOG_INFO, "MatmulHalfResult", "runkernelVMatmul: %lf", wall);
    }

    void getresult() {
        float *matCtmp;
        VK_CHECK_RESULT(vkMapMemory(device, bufferMemory, matrixASize+matrixBSize+matrixSizesSize, matrixCSize, 0, (void **)&matCtmp));
      
        // float sumC = 0.0;
        for (int k = 0; k < M; k++) {
          matC[k] = matCtmp[k];
          if(k < 100) {
              __android_log_print(ANDROID_LOG_INFO, "VulkanTest", "Matmul %d: %lf", k, matCtmp[k]);
          }
        }

        // std::memcpy(matC, matCtmp, matrixCSize);

        // __android_log_print(ANDROID_LOG_INFO, "VulkanTest", "Matmul sumC: %lf", sumC);

        vkUnmapMemory(device, bufferMemory); 
    }

    void cleanup() {
        vkFreeMemory(device, bufferMemory, NULL);
        vkDestroyBuffer(device, matrixA, NULL);
        //vkDestroyShaderModule(device, computeShaderModule, NULL);
        vkDestroyDescriptorPool(device, descriptorPool, NULL);
        //vkDestroyDescriptorSetLayout(device, descriptorSetLayout, NULL);
        //vkDestroyPipelineLayout(device, pipelineLayout, NULL);
        //vkDestroyPipeline(device, pipeline, NULL);
        vkDestroyCommandPool(device, commandPool, NULL);  
        //vkDestroyDevice(device, NULL);
        //vkDestroyInstance(instance, NULL);    
    }
};

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

void vulkanTest(const float* matrixA, const float* matrixB, float* matrixC, int M, int K, int N,
  VkPhysicalDevice physicalDevice, VkDevice device, VkPipeline pipelineMatmul, VkPipelineLayout pipelineLayoutMatmul, 
    VkDescriptorSetLayout descriptorSetLayoutMatmul, VkQueue queueV, uint32_t queueFamilyIndex) {
    ComputeApplication app;
    app.run(matrixA, matrixB, matrixC, M, K, N,
      physicalDevice, device, pipelineMatmul, pipelineLayoutMatmul, descriptorSetLayoutMatmul, queueV, queueFamilyIndex);
}

void clBlastOpenCL() {

}

void NeonMatrixBatchVectorMultiplyAccumulateOpenCL(const float* matrix, int m_rows,
                                             int m_cols, const float* vector,
                                             int n_batch, float* result,
                                             int result_stride,
                                             cl_context context_cl, cl_command_queue queue, cl_program program,
                                             VkPhysicalDevice physicalDevice, VkDevice device, VkPipeline pipelineConv, VkPipeline pipelineMatmul, VkPipelineLayout pipelineLayoutConv, VkPipelineLayout pipelineLayoutMatmul, 
    VkDescriptorSetLayout descriptorSetLayoutConv, VkDescriptorSetLayout descriptorSetLayoutMatmul, VkQueue queueV, uint32_t queueFamilyIndex) {
  

  // int* sizes;
  // int* strides;

  // sizes = (int*)malloc(16*sizeof(int));
  // strides = (int*)malloc(16*sizeof(int));

  // // Dims<4> input_dims,filter_dims,bias_dims,output_dims;
  
  // //input
  // sizes[0] = 8;
  // sizes[1] = 12;
  // sizes[2] = 16;
  // sizes[3] = 1;
  // strides[0] = 1;
  // strides[1] = 1;
  // strides[2] = 1;
  // strides[3] = 1;

  // //filter
  // sizes[4] = 8;
  // sizes[5] = 8;
  // sizes[6] = 8;
  // sizes[7] = 12;
  // strides[4] = 1;
  // strides[5] = 1;
  // strides[6] = 1;
  // strides[7] = 1;

  // //bias
  // sizes[8] = 12;
  // sizes[9] = 1;
  // sizes[10] = 1;
  // sizes[11] = 1;
  // strides[8] = 1;
  // strides[9] = 1;
  // strides[10] = 1;
  // strides[11] = 1;

  // //output
  // sizes[12] = 12;
  // sizes[13] = 8;
  // sizes[14] = 12;
  // sizes[15] = 1;
  // strides[12] = 1;
  // strides[13] = 1;
  // strides[14] = 1;
  // strides[15] = 1;

  // int input_size = sizes[0]*sizes[1]*sizes[2]*sizes[3];
  // int filter_size = sizes[4]*sizes[5]*sizes[6]*sizes[7];
  // int bias_size = sizes[8]*sizes[9]*sizes[10]*sizes[11];
  // int output_size = sizes[12]*sizes[13]*sizes[14]*sizes[15];

  // float* input;
  // float* output;
  // float* filter;
  // float* bias;

  // input = (float*)malloc(input_size*sizeof(float));
  // filter = (float*)malloc(filter_size*sizeof(float));
  // bias = (float*)malloc(bias_size*sizeof(float));
  // output = (float*)malloc(output_size*sizeof(float));

  // for(int i = 0; i < input_size; i++) {
  //   input[i] = (i+1)*1.0/10;
  // }
  // for(int i = 0; i < filter_size; i++) {
  //   filter[i] = (i+1)*1.0/10;
  // }
  // for(int i = 0; i < bias_size; i++) {
  //   bias[i] = (i+1)*1.0/10;
  // }

  // // vulkanTestConv(input, input_size, filter, filter_size, bias, bias_size, output, output_size, 
  // //     1, 1, 0, 0, sizes, strides, 0.0, 1000.0);

  // // Start Timers
  // double wall0 = get_wall_time();
  // double cpu0  = get_cpu_time();

  // ConvPort(input, filter, bias, output, 1, 1, 0, 0, sizes, strides, 0.0, 1000.0);

  // // Stop timers
  // double wall1 = get_wall_time();
  // double cpu1  = get_cpu_time();


  // double wall = wall1 - wall0;
  // double cpu = cpu1 - cpu0;

  // double sumC = 0.0;
  // for (int k = 0; k < output_size; k++) {
  //   sumC += output[k];
  //   if(k < 100) {
  //       __android_log_print(ANDROID_LOG_INFO, "VulkanConv", "Conv %d: %lf", k, output[k]);
  //   }
  // }

  // // note: andoird log
  // __android_log_print(ANDROID_LOG_INFO, "PortableConv", "runkernel: %lf", wall);

  // // for(int k = 0; k < 10; k++) {
  // //   __android_log_print(ANDROID_LOG_INFO, "VulkanConv", "Conv %d: %lf", k, output[k]);
  // // }

  // delete input;
  // delete output;
  // delete filter;
  // delete sizes;
  // delete strides;

// PortableMatrixBatchVectorMultiplyAccumulate(matrix,m_rows,m_cols,vector,n_batch,result,1);

m_cols = 2048;
m_rows = 1008;
float* matA = (float*) malloc(m_cols*m_rows*sizeof(float));
float* matB = (float*) malloc(m_cols*sizeof(float));
float* matC = (float*) malloc(m_rows*sizeof(float));

for(int i = 0; i < m_cols*m_rows; i++) {
  matA[i] = 1;
}
for(int i = 0; i < m_cols; i++) {
  matB[i] = 1;
}

// // Start Timers
// double wall0 = get_wall_time();
// double cpu0  = get_cpu_time();

// PortableMatrixBatchVectorMultiplyAccumulate(matrix,m_rows,m_cols,vector,n_batch,result,1);
// PortableMatrixBatchVectorMultiplyAccumulate(matA,m_rows,m_cols,matB,n_batch,matC,1);
// OpenCLPortableMatrixBatchVectorMultiplyAccumulate(matrix,m_rows,m_cols,vector,n_batch,result,1, context_cl, queue, program);
// OpenCLPortableMatrixBatchVectorMultiplyAccumulate(matA,m_rows,m_cols,matB,n_batch,matC,1, context_cl, queue, program);

// //clblast
//     //  Start Timers
//   double wall01 = get_wall_time();
//   double cpu01  = get_cpu_time();

//   cl_int err;
//   cl_mem d_a;
//   cl_mem d_b;
//   cl_mem d_c;

//   d_a = clCreateBuffer(context_cl, CL_MEM_READ_ONLY, m_rows*m_cols*sizeof(float), NULL, NULL);
//   d_b = clCreateBuffer(context_cl, CL_MEM_READ_ONLY, m_cols*n_batch*sizeof(float), NULL, NULL);
//   d_c = clCreateBuffer(context_cl, CL_MEM_WRITE_ONLY, m_rows*n_batch*sizeof(float), NULL, NULL);

//   err = clEnqueueWriteBuffer(queue, d_a, CL_TRUE, 0,
//                                  m_rows*m_cols*sizeof(float), matA, 0, NULL, NULL);
//   err = clEnqueueWriteBuffer(queue, d_b, CL_TRUE, 0,
//                                  m_cols*n_batch*sizeof(float), matB, 0, NULL, NULL);

//   clFinish(queue);

//   const size_t m = m_rows;
//   const size_t n = m_cols;
//   const float alpha = 1.0f;
//   const float beta = 0.0f;
//   const auto a_ld = m_cols;
//   const size_t x_st = 1;
//   const size_t y_st = 1;

//     double wall0 = get_wall_time();
//     double cpu0  = get_cpu_time();

//     Gemv(Layout::kRowMajor, Transpose::kNo,
//                 m, n,
//                 alpha,
//                 d_a, 0, a_ld,
//                 d_b, 0, x_st,
//                 beta,
//                 d_c, 0, y_st,
//                 &queue, NULL);
//     clFinish(queue);

//       // Stop timers
//     double wall1 = get_wall_time();
//     double cpu1  = get_cpu_time();


//     double wall = wall1 - wall0;
//     double cpu = cpu1 - cpu0;

//     // note: andoird log
//     __android_log_print(ANDROID_LOG_INFO, "MatmulHalfResult", "runtimeGemVOnly: %lf", wall);

//     clEnqueueReadBuffer(queue, d_c, CL_TRUE, 0, m_rows*n_batch*sizeof(float), matC, 0, NULL, NULL );

//     clFinish(queue);

//     clReleaseMemObject(d_a);
//     clReleaseMemObject(d_b);
//     clReleaseMemObject(d_c);

//     // Stop timers
//     wall1 = get_wall_time();
//     cpu1  = get_cpu_time();


//     wall = wall1 - wall01;
//     cpu = cpu1 - cpu01;

//     // note: andoird log
//     __android_log_print(ANDROID_LOG_INFO, "MatmulHalfResult", "runtime: %lf", wall);

// double sum = 0.0;
// for(int i = 0; i < m_rows*n_batch; i++) {
//     sum += matC[i];
//     // __android_log_print(ANDROID_LOG_INFO, "MatmulHalfResult", "%d: %f", i, matC[i]);
// }
// __android_log_print(ANDROID_LOG_INFO, "MatmulHalfResult", "MatmulSumResult: %lf", sum);

// PortableMatrixBatchVectorMultiplyAccumulate(matA,m_rows,m_cols,matB,n_batch,matC,1);

// // Stop timers
// double wall1 = get_wall_time();
// double cpu1  = get_cpu_time();


// double wall = wall1 - wall0;
// double cpu = cpu1 - cpu0;

// note: andoird log
// __android_log_print(ANDROID_LOG_INFO, "Portablematmul", "runkernel: %lf", wall);
// double sum = 0.0;
// for(int i = 0; i < m_rows; i++) {
//   sum += matC[i];  
// }
// __android_log_print(ANDROID_LOG_INFO, "MatmulSize", "FLoat size: %d", sizeof(float));
// __android_log_print(ANDROID_LOG_INFO, "MatmulSize", "Half size: %d", sizeof(cl_half));

// // Start Timers
// double wall0 = get_wall_time();
// double cpu0  = get_cpu_time();

float* resultport = (float*) malloc(m_rows*n_batch*sizeof(float));
float* resultocl = (float*) malloc(m_rows*n_batch*sizeof(float));

PortableMatrixBatchVectorMultiplyAccumulate(matA,m_rows,m_cols,matB,n_batch,resultport,1);

sleep(1);

// Start Timers
double wall0 = get_wall_time();
double cpu0  = get_cpu_time();

OpenCLPortableMatrixBatchVectorMultiplyAccumulate(matA,m_rows,m_cols,matB,n_batch,resultocl,1, context_cl, queue, program);

// vulkanTest(matA, matB, resultocl, m_rows, m_cols, n_batch,
//   physicalDevice, device, pipelineMatmul, pipelineLayoutMatmul, 
//   descriptorSetLayoutMatmul, queueV, queueFamilyIndex);

double wall1 = get_wall_time();
double cpu1  = get_cpu_time();

double wall = wall1 - wall0;
double cpu = cpu1 - cpu0;

// note: andoird log
__android_log_print(ANDROID_LOG_INFO, "MatmulHalfResult", "runkerneltotal: %lf", wall);

double sum = 0.0;
for(int i = 0; i < m_rows*n_batch; i++) {
    sum += resultport[i];
    // __android_log_print(ANDROID_LOG_INFO, "MatmulHalfResult", "%d: %f", i, matC[i]);
}
__android_log_print(ANDROID_LOG_INFO, "MatmulResult", "Resultmatmulport: %lf", sum);

sum = 0.0;
for(int i = 0; i < m_rows*n_batch; i++) {
    sum += resultocl[i];
    // __android_log_print(ANDROID_LOG_INFO, "MatmulHalfResult", "%d: %f", i, matC[i]);
}

__android_log_print(ANDROID_LOG_INFO, "MatmulResult", "Resultmatmulocl: %lf", sum);

// // Stop timers
// double wall1 = get_wall_time();
// double cpu1  = get_cpu_time();

// double wall = wall1 - wall0;
// double cpu = cpu1 - cpu0;

// double sum = 0.0;
// for(int i = 0; i < m_rows*n_batch; i++) {
//     sum += matC[i];
//     // __android_log_print(ANDROID_LOG_INFO, "MatmulHalfResult", "%d: %f", i, matC[i]);
// }
// __android_log_print(ANDROID_LOG_INFO, "MatmulHalfResult", "MatmulSumResult: %lf", sum);

// // note: andoird log
// __android_log_print(ANDROID_LOG_INFO, "MatmulHalfResult", "total: %lf", wall);

// sum = 0.0;
// for(int i = 0; i < m_rows; i++) {
//   sum += matC[i];  
// }
// __android_log_print(ANDROID_LOG_INFO, "Matmulruntime", "Vulkan result: %lf", sum);
  // OpenCLPortableMatrixBatchVectorMultiplyAccumulate(matA,m_rows,m_cols,matB,n_batch,matC,1, context_cl, queue, program);
  // sum = 0.0;
  // for(int i = 0; i < m_rows; i++) {
  //   sum += matC[i];  
  // }
  // __android_log_print(ANDROID_LOG_INFO, "Matmulruntime", "OpenCL result: %lf", sum);
  // Stop timers
  // double wall1 = get_wall_time();
  // double cpu1  = get_cpu_time();


  // double wall = wall1 - wall0;
  // double cpu = cpu1 - cpu0;

  // // note: andoird log
  // __android_log_print(ANDROID_LOG_INFO, "Matmulruntime", "Walltime: %lf", wall);
  // __android_log_print(ANDROID_LOG_INFO, "Matmulruntime", "Sum: %lf", sum);
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
  //note: andoird log
  // __android_log_print(ANDROID_LOG_INFO, "CobaLog", "neon1");
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
  //note: andoird log
  // __android_log_print(ANDROID_LOG_INFO, "CobaLog", "neon1");
}

void NeonVectorBatchVectorCwiseProductAccumulate(const float* vector,
                                                 int v_size,
                                                 const float* batch_vector,
                                                 int n_batch, float* result) {
  //note: andoird log
  // __android_log_print(ANDROID_LOG_INFO, "CobaLog", "neon1");

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
  //note: andoird log
  // __android_log_print(ANDROID_LOG_INFO, "CobaLog", "neon1");

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
  //note: andoird log
  // __android_log_print(ANDROID_LOG_INFO, "CobaLog", "neon1");

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
  //note: andoird log
  // __android_log_print(ANDROID_LOG_INFO, "CobaLog", "neondot");
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
  //note: andoird log
  // __android_log_print(ANDROID_LOG_INFO, "CobaLog", "neon1");
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
  //note: andoird log
  // __android_log_print(ANDROID_LOG_INFO, "CobaLog", "neon1");
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
  //note: andoird log
  // __android_log_print(ANDROID_LOG_INFO, "CobaLog", "neon1");

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

// // Branch-free implementation of half-precision (16 bit) floating point
// // Copyright 2006 Mike Acton <macton@gmail.com>
// // 
// // Permission is hereby granted, free of charge, to any person obtaining a 
// // copy of this software and associated documentation files (the "Software"),
// // to deal in the Software without restriction, including without limitation
// // the rights to use, copy, modify, merge, publish, distribute, sublicense, 
// // and/or sell copies of the Software, and to permit persons to whom the 
// // Software is furnished to do so, subject to the following conditions:
// // 
// // The above copyright notice and this permission notice shall be included 
// // in all copies or substantial portions of the Software.
// // 
// // THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// // IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
// // FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// // AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
// // LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// // OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// // THE SOFTWARE
// //
// // Half-precision floating point format
// // ------------------------------------
// //
// //   | Field    | Last | First | Note
// //   |----------|------|-------|----------
// //   | Sign     | 15   | 15    |
// //   | Exponent | 14   | 10    | Bias = 15
// //   | Mantissa | 9    | 0     |
// //
// // Compiling
// // ---------
// //
// //  Preferred compile flags for GCC: 
// //     -O3 -fstrict-aliasing -std=c99 -pedantic -Wall -Wstrict-aliasing
// //
// //     This file is a C99 source file, intended to be compiled with a C99 
// //     compliant compiler. However, for the moment it remains combatible
// //     with C++98. Therefore if you are using a compiler that poorly implements
// //     C standards (e.g. MSVC), it may be compiled as C++. This is not
// //     guaranteed for future versions. 
// //

// // Load immediate
// static inline uint32_t _uint32_li( uint32_t a )
// {
//   return (a);
// }

// // Decrement
// static inline uint32_t _uint32_dec( uint32_t a )
// {
//   return (a - 1);
// }

// // Increment
// static inline uint32_t _uint32_inc( uint32_t a )
// {
//   return (a + 1);
// }

// // Complement
// static inline uint32_t _uint32_not( uint32_t a )
// {
//   return (~a);
// }

// // Negate
// static inline uint32_t _uint32_neg( uint32_t a )
// {
//   return (-a);
// }

// // Extend sign
// static inline uint32_t _uint32_ext( uint32_t a )
// {
//   return (((int32_t)a)>>31);
// }

// // And
// static inline uint32_t _uint32_and( uint32_t a, uint32_t b )
// {
//   return (a & b);
// }

// // Exclusive Or
// static inline uint32_t _uint32_xor( uint32_t a, uint32_t b )
// {
//   return (a ^ b);
// }

// // And with Complement
// static inline uint32_t _uint32_andc( uint32_t a, uint32_t b )
// {
//   return (a & ~b);
// }

// // Or
// static inline uint32_t _uint32_or( uint32_t a, uint32_t b )
// {
//   return (a | b);
// }

// // Shift Right Logical
// static inline uint32_t _uint32_srl( uint32_t a, int sa )
// {
//   return (a >> sa);
// }

// // Shift Left Logical
// static inline uint32_t _uint32_sll( uint32_t a, int sa )
// {
//   return (a << sa);
// }

// // Add
// static inline uint32_t _uint32_add( uint32_t a, uint32_t b )
// {
//   return (a + b);
// }

// // Subtract
// static inline uint32_t _uint32_sub( uint32_t a, uint32_t b )
// {
//   return (a - b);
// }

// // Multiply
// static inline uint32_t _uint32_mul( uint32_t a, uint32_t b )
// {
//   return (a * b);
// }

// // Select on Sign bit
// static inline uint32_t _uint32_sels( uint32_t test, uint32_t a, uint32_t b )
// {
//   const uint32_t mask   = _uint32_ext( test );
//   const uint32_t sel_a  = _uint32_and(  a,     mask  );
//   const uint32_t sel_b  = _uint32_andc( b,     mask  );
//   const uint32_t result = _uint32_or(   sel_a, sel_b );

//   return (result);
// }

// // Select Bits on mask
// static inline uint32_t _uint32_selb( uint32_t mask, uint32_t a, uint32_t b )
// {
//   const uint32_t sel_a  = _uint32_and(  a,     mask  );
//   const uint32_t sel_b  = _uint32_andc( b,     mask  );
//   const uint32_t result = _uint32_or(   sel_a, sel_b );

//   return (result);
// }

// // Load Immediate
// static inline uint16_t _uint16_li( uint16_t a )
// {
//   return (a);
// }

// // Extend sign
// static inline uint16_t _uint16_ext( uint16_t a )
// {
//   return (((int16_t)a)>>15);
// }

// // Negate
// static inline uint16_t _uint16_neg( uint16_t a )
// {
//   return (-a);
// }

// // Complement
// static inline uint16_t _uint16_not( uint16_t a )
// {
//   return (~a);
// }

// // Decrement
// static inline uint16_t _uint16_dec( uint16_t a )
// {
//   return (a - 1);
// }

// // Shift Left Logical
// static inline uint16_t _uint16_sll( uint16_t a, int sa )
// {
//   return (a << sa);
// }

// // Shift Right Logical
// static inline uint16_t _uint16_srl( uint16_t a, int sa )
// {
//   return (a >> sa);
// }

// // Add
// static inline uint16_t _uint16_add( uint16_t a, uint16_t b )
// {
//   return (a + b);
// }

// // Subtract
// static inline uint16_t _uint16_sub( uint16_t a, uint16_t b )
// {
//   return (a - b);
// }

// // And
// static inline uint16_t _uint16_and( uint16_t a, uint16_t b )
// {
//   return (a & b);
// }

// // Or
// static inline uint16_t _uint16_or( uint16_t a, uint16_t b )
// {
//   return (a | b);
// }

// // Exclusive Or
// static inline uint16_t _uint16_xor( uint16_t a, uint16_t b )
// {
//   return (a ^ b);
// }

// // And with Complement
// static inline uint16_t _uint16_andc( uint16_t a, uint16_t b )
// {
//   return (a & ~b);
// }

// // And then Shift Right Logical
// static inline uint16_t _uint16_andsrl( uint16_t a, uint16_t b, int sa )
// {
//   return ((a & b) >> sa);
// }

// // Shift Right Logical then Mask
// static inline uint16_t _uint16_srlm( uint16_t a, int sa, uint16_t mask )
// {
//   return ((a >> sa) & mask);
// }

// // Add then Mask
// static inline uint16_t _uint16_addm( uint16_t a, uint16_t b, uint16_t mask )
// {
//   return ((a + b) & mask);
// }


// // Select on Sign bit
// static inline uint16_t _uint16_sels( uint16_t test, uint16_t a, uint16_t b )
// {
//   const uint16_t mask   = _uint16_ext( test );
//   const uint16_t sel_a  = _uint16_and(  a,     mask  );
//   const uint16_t sel_b  = _uint16_andc( b,     mask  );
//   const uint16_t result = _uint16_or(   sel_a, sel_b );

//   return (result);
// }

// // Count Leading Zeros
// static inline uint32_t _uint32_cntlz( uint32_t x )
// {
// #ifdef __GNUC__
//   /* NOTE: __builtin_clz is undefined for x == 0 */
//   /* On PowerPC, this will map to insn: cntlzw   */
//   /* On Pentium, this will map to insn: clz      */
//   uint32_t is_x_nez_msb = _uint32_neg( x );
//   uint32_t nlz          = __builtin_clz( x );
//   uint32_t result       = _uint32_sels( is_x_nez_msb, nlz, 0x00000020 );
//   return (result);
// #else
//   const uint32_t x0  = _uint32_srl(  x,  1 );
//   const uint32_t x1  = _uint32_or(   x,  x0 );
//   const uint32_t x2  = _uint32_srl(  x1, 2 );
//   const uint32_t x3  = _uint32_or(   x1, x2 );
//   const uint32_t x4  = _uint32_srl(  x3, 4 );
//   const uint32_t x5  = _uint32_or(   x3, x4 );
//   const uint32_t x6  = _uint32_srl(  x5, 8 );
//   const uint32_t x7  = _uint32_or(   x5, x6 );
//   const uint32_t x8  = _uint32_srl(  x7, 16 );
//   const uint32_t x9  = _uint32_or(   x7, x8 );
//   const uint32_t xA  = _uint32_not(  x9 );
//   const uint32_t xB  = _uint32_srl(  xA, 1 );
//   const uint32_t xC  = _uint32_and(  xB, 0x55555555 );
//   const uint32_t xD  = _uint32_sub(  xA, xC );
//   const uint32_t xE  = _uint32_and(  xD, 0x33333333 );
//   const uint32_t xF  = _uint32_srl(  xD, 2 );
//   const uint32_t x10 = _uint32_and(  xF, 0x33333333 );
//   const uint32_t x11 = _uint32_add(  xE, x10 );
//   const uint32_t x12 = _uint32_srl(  x11, 4 );
//   const uint32_t x13 = _uint32_add(  x11, x12 );
//   const uint32_t x14 = _uint32_and(  x13, 0x0f0f0f0f );
//   const uint32_t x15 = _uint32_srl(  x14, 8 );
//   const uint32_t x16 = _uint32_add(  x14, x15 );
//   const uint32_t x17 = _uint32_srl(  x16, 16 );
//   const uint32_t x18 = _uint32_add(  x16, x17 );
//   const uint32_t x19 = _uint32_and(  x18, 0x0000003f );
//   return ( x19 );
// #endif
// }

// // Count Leading Zeros
// static inline uint16_t _uint16_cntlz( uint16_t x )
// {
// #ifdef __GNUC__
//   uint16_t nlz32 = (uint16_t)_uint32_cntlz( (uint32_t)x );
//   uint32_t nlz   = _uint32_sub( nlz32, 16 );
//   return (nlz);
// #else
//   const uint16_t x0  = _uint16_srl(  x,  1 );
//   const uint16_t x1  = _uint16_or(   x,  x0 );
//   const uint16_t x2  = _uint16_srl(  x1, 2 );
//   const uint16_t x3  = _uint16_or(   x1, x2 );
//   const uint16_t x4  = _uint16_srl(  x3, 4 );
//   const uint16_t x5  = _uint16_or(   x3, x4 );
//   const uint16_t x6  = _uint16_srl(  x5, 8 );
//   const uint16_t x7  = _uint16_or(   x5, x6 );
//   const uint16_t x8  = _uint16_not(  x7 );
//   const uint16_t x9  = _uint16_srlm( x8, 1, 0x5555 );
//   const uint16_t xA  = _uint16_sub(  x8, x9 );
//   const uint16_t xB  = _uint16_and(  xA, 0x3333 );
//   const uint16_t xC  = _uint16_srlm( xA, 2, 0x3333 );
//   const uint16_t xD  = _uint16_add(  xB, xC );
//   const uint16_t xE  = _uint16_srl(  xD, 4 );
//   const uint16_t xF  = _uint16_addm( xD, xE, 0x0f0f );
//   const uint16_t x10 = _uint16_srl(  xF, 8 );
//   const uint16_t x11 = _uint16_addm( xF, x10, 0x001f );
//   return ( x11 );
// #endif
// }

// uint16_t
// half_from_float( uint32_t f )
// {
//   const uint32_t one                        = _uint32_li( 0x00000001 );
//   const uint32_t f_s_mask                   = _uint32_li( 0x80000000 );
//   const uint32_t f_e_mask                   = _uint32_li( 0x7f800000 );
//   const uint32_t f_m_mask                   = _uint32_li( 0x007fffff );
//   const uint32_t f_m_hidden_bit             = _uint32_li( 0x00800000 );
//   const uint32_t f_m_round_bit              = _uint32_li( 0x00001000 );
//   const uint32_t f_snan_mask                = _uint32_li( 0x7fc00000 );
//   const uint32_t f_e_pos                    = _uint32_li( 0x00000017 );
//   const uint32_t h_e_pos                    = _uint32_li( 0x0000000a );
//   const uint32_t h_e_mask                   = _uint32_li( 0x00007c00 );
//   const uint32_t h_snan_mask                = _uint32_li( 0x00007e00 );
//   const uint32_t h_e_mask_value             = _uint32_li( 0x0000001f );
//   const uint32_t f_h_s_pos_offset           = _uint32_li( 0x00000010 );
//   const uint32_t f_h_bias_offset            = _uint32_li( 0x00000070 );
//   const uint32_t f_h_m_pos_offset           = _uint32_li( 0x0000000d );
//   const uint32_t h_nan_min                  = _uint32_li( 0x00007c01 );
//   const uint32_t f_h_e_biased_flag          = _uint32_li( 0x0000008f );
//   const uint32_t f_s                        = _uint32_and( f,               f_s_mask         );
//   const uint32_t f_e                        = _uint32_and( f,               f_e_mask         );
//   const uint16_t h_s                        = _uint32_srl( f_s,             f_h_s_pos_offset );
//   const uint32_t f_m                        = _uint32_and( f,               f_m_mask         );
//   const uint16_t f_e_amount                 = _uint32_srl( f_e,             f_e_pos          );
//   const uint32_t f_e_half_bias              = _uint32_sub( f_e_amount,      f_h_bias_offset  );
//   const uint32_t f_snan                     = _uint32_and( f,               f_snan_mask      );
//   const uint32_t f_m_round_mask             = _uint32_and( f_m,             f_m_round_bit    );
//   const uint32_t f_m_round_offset           = _uint32_sll( f_m_round_mask,  one              );
//   const uint32_t f_m_rounded                = _uint32_add( f_m,             f_m_round_offset );
//   const uint32_t f_m_denorm_sa              = _uint32_sub( one,             f_e_half_bias    );
//   const uint32_t f_m_with_hidden            = _uint32_or(  f_m_rounded,     f_m_hidden_bit   );
//   const uint32_t f_m_denorm                 = _uint32_srl( f_m_with_hidden, f_m_denorm_sa    );
//   const uint32_t h_m_denorm                 = _uint32_srl( f_m_denorm,      f_h_m_pos_offset );
//   const uint32_t f_m_rounded_overflow       = _uint32_and( f_m_rounded,     f_m_hidden_bit   );
//   const uint32_t m_nan                      = _uint32_srl( f_m,             f_h_m_pos_offset );
//   const uint32_t h_em_nan                   = _uint32_or(  h_e_mask,        m_nan            );
//   const uint32_t h_e_norm_overflow_offset   = _uint32_inc( f_e_half_bias );
//   const uint32_t h_e_norm_overflow          = _uint32_sll( h_e_norm_overflow_offset, h_e_pos          );
//   const uint32_t h_e_norm                   = _uint32_sll( f_e_half_bias,            h_e_pos          );
//   const uint32_t h_m_norm                   = _uint32_srl( f_m_rounded,              f_h_m_pos_offset );
//   const uint32_t h_em_norm                  = _uint32_or(  h_e_norm,                 h_m_norm         );
//   const uint32_t is_h_ndenorm_msb           = _uint32_sub( f_h_bias_offset,   f_e_amount    );
//   const uint32_t is_f_e_flagged_msb         = _uint32_sub( f_h_e_biased_flag, f_e_half_bias );
//   const uint32_t is_h_denorm_msb            = _uint32_not( is_h_ndenorm_msb );
//   const uint32_t is_f_m_eqz_msb             = _uint32_dec( f_m   );
//   const uint32_t is_h_nan_eqz_msb           = _uint32_dec( m_nan );
//   const uint32_t is_f_inf_msb               = _uint32_and( is_f_e_flagged_msb, is_f_m_eqz_msb   );
//   const uint32_t is_f_nan_underflow_msb     = _uint32_and( is_f_e_flagged_msb, is_h_nan_eqz_msb );
//   const uint32_t is_e_overflow_msb          = _uint32_sub( h_e_mask_value,     f_e_half_bias    );
//   const uint32_t is_h_inf_msb               = _uint32_or(  is_e_overflow_msb,  is_f_inf_msb     );
//   const uint32_t is_f_nsnan_msb             = _uint32_sub( f_snan,             f_snan_mask      );
//   const uint32_t is_m_norm_overflow_msb     = _uint32_neg( f_m_rounded_overflow );
//   const uint32_t is_f_snan_msb              = _uint32_not( is_f_nsnan_msb );
//   const uint32_t h_em_overflow_result       = _uint32_sels( is_m_norm_overflow_msb, h_e_norm_overflow, h_em_norm                 );
//   const uint32_t h_em_nan_result            = _uint32_sels( is_f_e_flagged_msb,     h_em_nan,          h_em_overflow_result      );
//   const uint32_t h_em_nan_underflow_result  = _uint32_sels( is_f_nan_underflow_msb, h_nan_min,         h_em_nan_result           );
//   const uint32_t h_em_inf_result            = _uint32_sels( is_h_inf_msb,           h_e_mask,          h_em_nan_underflow_result );
//   const uint32_t h_em_denorm_result         = _uint32_sels( is_h_denorm_msb,        h_m_denorm,        h_em_inf_result           );
//   const uint32_t h_em_snan_result           = _uint32_sels( is_f_snan_msb,          h_snan_mask,       h_em_denorm_result        );
//   const uint32_t h_result                   = _uint32_or( h_s, h_em_snan_result );

//   return (uint16_t)(h_result);
// }

// uint32_t 
// half_to_float( uint16_t h )
// {
//   const uint32_t h_e_mask              = _uint32_li( 0x00007c00 );
//   const uint32_t h_m_mask              = _uint32_li( 0x000003ff );
//   const uint32_t h_s_mask              = _uint32_li( 0x00008000 );
//   const uint32_t h_f_s_pos_offset      = _uint32_li( 0x00000010 );
//   const uint32_t h_f_e_pos_offset      = _uint32_li( 0x0000000d );
//   const uint32_t h_f_bias_offset       = _uint32_li( 0x0001c000 );
//   const uint32_t f_e_mask              = _uint32_li( 0x7f800000 );
//   const uint32_t f_m_mask              = _uint32_li( 0x007fffff );
//   const uint32_t h_f_e_denorm_bias     = _uint32_li( 0x0000007e );
//   const uint32_t h_f_m_denorm_sa_bias  = _uint32_li( 0x00000008 );
//   const uint32_t f_e_pos               = _uint32_li( 0x00000017 );
//   const uint32_t h_e_mask_minus_one    = _uint32_li( 0x00007bff );
//   const uint32_t h_e                   = _uint32_and( h, h_e_mask );
//   const uint32_t h_m                   = _uint32_and( h, h_m_mask );
//   const uint32_t h_s                   = _uint32_and( h, h_s_mask );
//   const uint32_t h_e_f_bias            = _uint32_add( h_e, h_f_bias_offset );
//   const uint32_t h_m_nlz               = _uint32_cntlz( h_m );
//   const uint32_t f_s                   = _uint32_sll( h_s,        h_f_s_pos_offset );
//   const uint32_t f_e                   = _uint32_sll( h_e_f_bias, h_f_e_pos_offset );
//   const uint32_t f_m                   = _uint32_sll( h_m,        h_f_e_pos_offset );
//   const uint32_t f_em                  = _uint32_or(  f_e,        f_m              );
//   const uint32_t h_f_m_sa              = _uint32_sub( h_m_nlz,             h_f_m_denorm_sa_bias );
//   const uint32_t f_e_denorm_unpacked   = _uint32_sub( h_f_e_denorm_bias,   h_f_m_sa             );
//   const uint32_t h_f_m                 = _uint32_sll( h_m,                 h_f_m_sa             );
//   const uint32_t f_m_denorm            = _uint32_and( h_f_m,               f_m_mask             );
//   const uint32_t f_e_denorm            = _uint32_sll( f_e_denorm_unpacked, f_e_pos              );
//   const uint32_t f_em_denorm           = _uint32_or(  f_e_denorm,          f_m_denorm           );
//   const uint32_t f_em_nan              = _uint32_or(  f_e_mask,            f_m                  );
//   const uint32_t is_e_eqz_msb          = _uint32_dec(  h_e );
//   const uint32_t is_m_nez_msb          = _uint32_neg(  h_m );
//   const uint32_t is_e_flagged_msb      = _uint32_sub(  h_e_mask_minus_one, h_e );
//   const uint32_t is_zero_msb           = _uint32_andc( is_e_eqz_msb,       is_m_nez_msb );
//   const uint32_t is_inf_msb            = _uint32_andc( is_e_flagged_msb,   is_m_nez_msb );
//   const uint32_t is_denorm_msb         = _uint32_and(  is_m_nez_msb,       is_e_eqz_msb );
//   const uint32_t is_nan_msb            = _uint32_and(  is_e_flagged_msb,   is_m_nez_msb ); 
//   const uint32_t is_zero               = _uint32_ext(  is_zero_msb );
//   const uint32_t f_zero_result         = _uint32_andc( f_em, is_zero );
//   const uint32_t f_denorm_result       = _uint32_sels( is_denorm_msb, f_em_denorm, f_zero_result );
//   const uint32_t f_inf_result          = _uint32_sels( is_inf_msb,    f_e_mask,    f_denorm_result );
//   const uint32_t f_nan_result          = _uint32_sels( is_nan_msb,    f_em_nan,    f_inf_result    );
//   const uint32_t f_result              = _uint32_or( f_s, f_nan_result );
 
//   return (f_result);
// }

// // half_add
// // --------
// //
// //  (SUM)        uint16_t z = half_add( x, y );
// //  (DIFFERENCE) uint16_t z = half_add( x, -y );
// //
// //  * Difference of ZEROs is always +ZERO
// //  * Sum round with guard + round + sticky bit (grs)
// //  * QNaN + <x>  = QNaN
// //  * <x>  + +INF = +INF
// //  * <x>  - -INF = -INF
// //  * INF  - INF  = SNaN
// //
// //  Will have exactly (0 ulps difference) the same result as:
// //  (Round up)
// //
// //     union FLOAT_32
// //     {
// //       float    f32;
// //       uint32_t u32;
// //     };
// //
// //     union FLOAT_32 fx = { .u32 = half_to_float( x ) };
// //     union FLOAT_32 fy = { .u32 = half_to_float( y ) };
// //     union FLOAT_32 fz = { .f32 = fx.f32 + fy.f32    };
// //     uint16_t       z  = float_to_half( fz );
// //
// uint16_t
// half_add( uint16_t x, uint16_t y )
// {
//   const uint16_t one                       = _uint16_li( 0x0001 );
//   const uint16_t msb_to_lsb_sa             = _uint16_li( 0x000f );
//   const uint16_t h_s_mask                  = _uint16_li( 0x8000 );
//   const uint16_t h_e_mask                  = _uint16_li( 0x7c00 );
//   const uint16_t h_m_mask                  = _uint16_li( 0x03ff );
//   const uint16_t h_m_msb_mask              = _uint16_li( 0x2000 );
//   const uint16_t h_m_msb_sa                = _uint16_li( 0x000d );
//   const uint16_t h_m_hidden                = _uint16_li( 0x0400 );
//   const uint16_t h_e_pos                   = _uint16_li( 0x000a );
//   const uint16_t h_e_bias_minus_one        = _uint16_li( 0x000e );
//   const uint16_t h_m_grs_carry             = _uint16_li( 0x4000 );
//   const uint16_t h_m_grs_carry_pos         = _uint16_li( 0x000e );
//   const uint16_t h_grs_size                = _uint16_li( 0x0003 );
//   const uint16_t h_snan                    = _uint16_li( 0xfe00 );
//   const uint16_t h_e_mask_minus_one        = _uint16_li( 0x7bff );
//   const uint16_t h_grs_round_carry         = _uint16_sll( one, h_grs_size );
//   const uint16_t h_grs_round_mask          = _uint16_sub( h_grs_round_carry, one );
//   const uint16_t x_e                       = _uint16_and( x, h_e_mask );
//   const uint16_t y_e                       = _uint16_and( y, h_e_mask );
//   const uint16_t is_y_e_larger_msb         = _uint16_sub( x_e, y_e );
//   const uint16_t a                         = _uint16_sels( is_y_e_larger_msb, y, x);
//   const uint16_t a_s                       = _uint16_and( a, h_s_mask );
//   const uint16_t a_e                       = _uint16_and( a, h_e_mask );
//   const uint16_t a_m_no_hidden_bit         = _uint16_and( a, h_m_mask );
//   const uint16_t a_em_no_hidden_bit        = _uint16_or( a_e, a_m_no_hidden_bit );
//   const uint16_t b                         = _uint16_sels( is_y_e_larger_msb, x, y);
//   const uint16_t b_s                       = _uint16_and( b, h_s_mask );
//   const uint16_t b_e                       = _uint16_and( b, h_e_mask );
//   const uint16_t b_m_no_hidden_bit         = _uint16_and( b, h_m_mask );
//   const uint16_t b_em_no_hidden_bit        = _uint16_or( b_e, b_m_no_hidden_bit );
//   const uint16_t is_diff_sign_msb          = _uint16_xor( a_s, b_s );
//   const uint16_t is_a_inf_msb              = _uint16_sub( h_e_mask_minus_one, a_em_no_hidden_bit );
//   const uint16_t is_b_inf_msb              = _uint16_sub( h_e_mask_minus_one, b_em_no_hidden_bit );
//   const uint16_t is_undenorm_msb           = _uint16_dec( a_e );
//   const uint16_t is_undenorm               = _uint16_ext( is_undenorm_msb );
//   const uint16_t is_both_inf_msb           = _uint16_and( is_a_inf_msb, is_b_inf_msb );
//   const uint16_t is_invalid_inf_op_msb     = _uint16_and( is_both_inf_msb, b_s );
//   const uint16_t is_a_e_nez_msb            = _uint16_neg( a_e );
//   const uint16_t is_b_e_nez_msb            = _uint16_neg( b_e );
//   const uint16_t is_a_e_nez                = _uint16_ext( is_a_e_nez_msb );
//   const uint16_t is_b_e_nez                = _uint16_ext( is_b_e_nez_msb );
//   const uint16_t a_m_hidden_bit            = _uint16_and( is_a_e_nez, h_m_hidden );
//   const uint16_t b_m_hidden_bit            = _uint16_and( is_b_e_nez, h_m_hidden );
//   const uint16_t a_m_no_grs                = _uint16_or( a_m_no_hidden_bit, a_m_hidden_bit );
//   const uint16_t b_m_no_grs                = _uint16_or( b_m_no_hidden_bit, b_m_hidden_bit );
//   const uint16_t diff_e                    = _uint16_sub( a_e,        b_e );
//   const uint16_t a_e_unbias                = _uint16_sub( a_e,        h_e_bias_minus_one );
//   const uint16_t a_m                       = _uint16_sll( a_m_no_grs, h_grs_size );
//   const uint16_t a_e_biased                = _uint16_srl( a_e,        h_e_pos );
//   const uint16_t m_sa_unbias               = _uint16_srl( a_e_unbias, h_e_pos );
//   const uint16_t m_sa_default              = _uint16_srl( diff_e,     h_e_pos );
//   const uint16_t m_sa_unbias_mask          = _uint16_andc( is_a_e_nez_msb,   is_b_e_nez_msb );
//   const uint16_t m_sa                      = _uint16_sels( m_sa_unbias_mask, m_sa_unbias, m_sa_default );
//   const uint16_t b_m_no_sticky             = _uint16_sll( b_m_no_grs,        h_grs_size );
//   const uint16_t sh_m                      = _uint16_srl( b_m_no_sticky,     m_sa );
//   const uint16_t sticky_overflow           = _uint16_sll( one,               m_sa );
//   const uint16_t sticky_mask               = _uint16_dec( sticky_overflow );
//   const uint16_t sticky_collect            = _uint16_and( b_m_no_sticky, sticky_mask );
//   const uint16_t is_sticky_set_msb         = _uint16_neg( sticky_collect );
//   const uint16_t sticky                    = _uint16_srl( is_sticky_set_msb, msb_to_lsb_sa);
//   const uint16_t b_m                       = _uint16_or( sh_m, sticky );
//   const uint16_t is_c_m_ab_pos_msb         = _uint16_sub( b_m, a_m );
//   const uint16_t c_inf                     = _uint16_or( a_s, h_e_mask );
//   const uint16_t c_m_sum                   = _uint16_add( a_m, b_m );
//   const uint16_t c_m_diff_ab               = _uint16_sub( a_m, b_m );
//   const uint16_t c_m_diff_ba               = _uint16_sub( b_m, a_m );
//   const uint16_t c_m_smag_diff             = _uint16_sels( is_c_m_ab_pos_msb, c_m_diff_ab, c_m_diff_ba );
//   const uint16_t c_s_diff                  = _uint16_sels( is_c_m_ab_pos_msb, a_s,         b_s         );
//   const uint16_t c_s                       = _uint16_sels( is_diff_sign_msb,  c_s_diff,    a_s         );
//   const uint16_t c_m_smag_diff_nlz         = _uint16_cntlz( c_m_smag_diff );
//   const uint16_t diff_norm_sa              = _uint16_sub( c_m_smag_diff_nlz, one );
//   const uint16_t is_diff_denorm_msb        = _uint16_sub( a_e_biased, diff_norm_sa );
//   const uint16_t is_diff_denorm            = _uint16_ext( is_diff_denorm_msb );
//   const uint16_t is_a_or_b_norm_msb        = _uint16_neg( a_e_biased );
//   const uint16_t diff_denorm_sa            = _uint16_dec( a_e_biased );
//   const uint16_t c_m_diff_denorm           = _uint16_sll( c_m_smag_diff, diff_denorm_sa );
//   const uint16_t c_m_diff_norm             = _uint16_sll( c_m_smag_diff, diff_norm_sa );
//   const uint16_t c_e_diff_norm             = _uint16_sub( a_e_biased,  diff_norm_sa );
//   const uint16_t c_m_diff_ab_norm          = _uint16_sels( is_diff_denorm_msb, c_m_diff_denorm, c_m_diff_norm );
//   const uint16_t c_e_diff_ab_norm          = _uint16_andc( c_e_diff_norm, is_diff_denorm );
//   const uint16_t c_m_diff                  = _uint16_sels( is_a_or_b_norm_msb, c_m_diff_ab_norm, c_m_smag_diff );
//   const uint16_t c_e_diff                  = _uint16_sels( is_a_or_b_norm_msb, c_e_diff_ab_norm, a_e_biased    );
//   const uint16_t is_diff_eqz_msb           = _uint16_dec( c_m_diff );
//   const uint16_t is_diff_exactly_zero_msb  = _uint16_and( is_diff_sign_msb, is_diff_eqz_msb );
//   const uint16_t is_diff_exactly_zero      = _uint16_ext( is_diff_exactly_zero_msb );
//   const uint16_t c_m_added                 = _uint16_sels( is_diff_sign_msb, c_m_diff, c_m_sum );
//   const uint16_t c_e_added                 = _uint16_sels( is_diff_sign_msb, c_e_diff, a_e_biased );
//   const uint16_t c_m_carry                 = _uint16_and( c_m_added, h_m_grs_carry );
//   const uint16_t is_c_m_carry_msb          = _uint16_neg( c_m_carry );
//   const uint16_t c_e_hidden_offset         = _uint16_andsrl( c_m_added, h_m_grs_carry, h_m_grs_carry_pos );
//   const uint16_t c_m_sub_hidden            = _uint16_srl( c_m_added, one );
//   const uint16_t c_m_no_hidden             = _uint16_sels( is_c_m_carry_msb, c_m_sub_hidden, c_m_added );
//   const uint16_t c_e_no_hidden             = _uint16_add( c_e_added,         c_e_hidden_offset  );
//   const uint16_t c_m_no_hidden_msb         = _uint16_and( c_m_no_hidden,     h_m_msb_mask       );
//   const uint16_t undenorm_m_msb_odd        = _uint16_srl( c_m_no_hidden_msb, h_m_msb_sa         );
//   const uint16_t undenorm_fix_e            = _uint16_and( is_undenorm,       undenorm_m_msb_odd );
//   const uint16_t c_e_fixed                 = _uint16_add( c_e_no_hidden,     undenorm_fix_e     );
//   const uint16_t c_m_round_amount          = _uint16_and( c_m_no_hidden,     h_grs_round_mask   );
//   const uint16_t c_m_rounded               = _uint16_add( c_m_no_hidden,     c_m_round_amount   );
//   const uint16_t c_m_round_overflow        = _uint16_andsrl( c_m_rounded, h_m_grs_carry, h_m_grs_carry_pos );
//   const uint16_t c_e_rounded               = _uint16_add( c_e_fixed, c_m_round_overflow );
//   const uint16_t c_m_no_grs                = _uint16_srlm( c_m_rounded, h_grs_size,  h_m_mask );
//   const uint16_t c_e                       = _uint16_sll( c_e_rounded, h_e_pos );
//   const uint16_t c_em                      = _uint16_or( c_e, c_m_no_grs );
//   const uint16_t c_normal                  = _uint16_or( c_s, c_em );
//   const uint16_t c_inf_result              = _uint16_sels( is_a_inf_msb, c_inf, c_normal );
//   const uint16_t c_zero_result             = _uint16_andc( c_inf_result, is_diff_exactly_zero );
//   const uint16_t c_result                  = _uint16_sels( is_invalid_inf_op_msb, h_snan, c_zero_result );

//   return (c_result);
// }

// // half_mul
// // --------
// //
// //  May have 0 or 1 ulp difference from the following result:
// //  (Round to nearest) 
// //  NOTE: Rounding mode differs between conversion and multiply
// //
// //     union FLOAT_32
// //     {
// //       float    f32;
// //       uint32_t u32;
// //     };
// //
// //     union FLOAT_32 fx = { .u32 = half_to_float( x ) };
// //     union FLOAT_32 fy = { .u32 = half_to_float( y ) };
// //     union FLOAT_32 fz = { .f32 = fx.f32 * fy.f32    };
// //     uint16_t       z  = float_to_half( fz );
// //
// uint16_t
// half_mul( uint16_t x, uint16_t y )
// {
//   const uint32_t one                                = _uint32_li( 0x00000001 );
//   const uint32_t h_s_mask                           = _uint32_li( 0x00008000 );
//   const uint32_t h_e_mask                           = _uint32_li( 0x00007c00 );
//   const uint32_t h_m_mask                           = _uint32_li( 0x000003ff );
//   const uint32_t h_m_hidden                         = _uint32_li( 0x00000400 );
//   const uint32_t h_e_pos                            = _uint32_li( 0x0000000a );
//   const uint32_t h_e_bias                           = _uint32_li( 0x0000000f );
//   const uint32_t h_m_bit_count                      = _uint32_li( 0x0000000a );
//   const uint32_t h_m_bit_half_count                 = _uint32_li( 0x00000005 );
//   const uint32_t h_nan_min                          = _uint32_li( 0x00007c01 );
//   const uint32_t h_e_mask_minus_one                 = _uint32_li( 0x00007bff );
//   const uint32_t h_snan                             = _uint32_li( 0x0000fe00 );
//   const uint32_t m_round_overflow_bit               = _uint32_li( 0x00000020 );
//   const uint32_t m_hidden_bit                       = _uint32_li( 0x00100000 );
//   const uint32_t a_s                                = _uint32_and(  x,   h_s_mask );
//   const uint32_t b_s                                = _uint32_and(  y,   h_s_mask );
//   const uint32_t c_s                                = _uint32_xor(  a_s, b_s      );
//   const uint32_t x_e                                = _uint32_and(  x,   h_e_mask );
//   const uint32_t x_e_eqz_msb                        = _uint32_dec(  x_e );
//   const uint32_t a                                  = _uint32_sels( x_e_eqz_msb, y, x );
//   const uint32_t b                                  = _uint32_sels( x_e_eqz_msb, x, y );
//   const uint32_t a_e                                = _uint32_and(  a,   h_e_mask );
//   const uint32_t b_e                                = _uint32_and(  b,   h_e_mask );
//   const uint32_t a_m                                = _uint32_and(  a,   h_m_mask );
//   const uint32_t b_m                                = _uint32_and(  b,   h_m_mask );
//   const uint32_t a_e_amount                         = _uint32_srl(  a_e,                 h_e_pos                 );
//   const uint32_t b_e_amount                         = _uint32_srl(  b_e,                 h_e_pos                 );
//   const uint32_t a_m_with_hidden                    = _uint32_or(   a_m,                 h_m_hidden              );
//   const uint32_t b_m_with_hidden                    = _uint32_or(   b_m,                 h_m_hidden              );
//   const uint32_t c_m_normal                         = _uint32_mul(  a_m_with_hidden,     b_m_with_hidden         );
//   const uint32_t c_m_denorm_biased                  = _uint32_mul(  a_m_with_hidden,     b_m                     );
//   const uint32_t c_e_denorm_unbias_e                = _uint32_sub(  h_e_bias,            a_e_amount              );
//   const uint32_t c_m_denorm_round_amount            = _uint32_and(  c_m_denorm_biased,   h_m_mask                );
//   const uint32_t c_m_denorm_rounded                 = _uint32_add(  c_m_denorm_biased,   c_m_denorm_round_amount );
//   const uint32_t c_m_denorm_inplace                 = _uint32_srl(  c_m_denorm_rounded,  h_m_bit_count           );
//   const uint32_t c_m_denorm_unbiased                = _uint32_srl(  c_m_denorm_inplace,  c_e_denorm_unbias_e     );
//   const uint32_t c_m_denorm                         = _uint32_and(  c_m_denorm_unbiased, h_m_mask                );
//   const uint32_t c_e_amount_biased                  = _uint32_add(  a_e_amount,          b_e_amount              );
//   const uint32_t c_e_amount_unbiased                = _uint32_sub(  c_e_amount_biased,   h_e_bias                );
//   const uint32_t is_c_e_unbiased_underflow          = _uint32_ext(  c_e_amount_unbiased );
//   const uint32_t c_e_underflow_half_sa              = _uint32_neg(  c_e_amount_unbiased );
//   const uint32_t c_e_underflow_sa                   = _uint32_sll(  c_e_underflow_half_sa,     one );
//   const uint32_t c_m_underflow                      = _uint32_srl(  c_m_normal,                c_e_underflow_sa );
//   const uint32_t c_e_underflow_added                = _uint32_andc( c_e_amount_unbiased,       is_c_e_unbiased_underflow );
//   const uint32_t c_m_underflow_added                = _uint32_selb( is_c_e_unbiased_underflow, c_m_underflow, c_m_normal );
//   const uint32_t is_mul_overflow_test               = _uint32_and(  c_e_underflow_added, m_round_overflow_bit );
//   const uint32_t is_mul_overflow_msb                = _uint32_neg(  is_mul_overflow_test );
//   const uint32_t c_e_norm_radix_corrected           = _uint32_inc(  c_e_underflow_added );
//   const uint32_t c_m_norm_radix_corrected           = _uint32_srl(  c_m_underflow_added, one );
//   const uint32_t c_m_norm_hidden_bit                = _uint32_and(  c_m_norm_radix_corrected,  m_hidden_bit );
//   const uint32_t is_c_m_norm_no_hidden_msb          = _uint32_dec(  c_m_norm_hidden_bit );
//   const uint32_t c_m_norm_lo                        = _uint32_srl(  c_m_norm_radix_corrected, h_m_bit_half_count );
//   const uint32_t c_m_norm_lo_nlz                    = _uint16_cntlz( c_m_norm_lo );
//   const uint32_t is_c_m_hidden_nunderflow_msb       = _uint32_sub(  c_m_norm_lo_nlz, c_e_norm_radix_corrected );
//   const uint32_t is_c_m_hidden_underflow_msb        = _uint32_not(  is_c_m_hidden_nunderflow_msb );
//   const uint32_t is_c_m_hidden_underflow            = _uint32_ext(  is_c_m_hidden_underflow_msb  );
//   const uint32_t c_m_hidden_underflow_normalized_sa = _uint32_srl(  c_m_norm_lo_nlz, one );
//   const uint32_t c_m_hidden_underflow_normalized    = _uint32_sll(  c_m_norm_radix_corrected, c_m_hidden_underflow_normalized_sa );
//   const uint32_t c_m_hidden_normalized              = _uint32_sll(  c_m_norm_radix_corrected, c_m_norm_lo_nlz );
//   const uint32_t c_e_hidden_normalized              = _uint32_sub(  c_e_norm_radix_corrected, c_m_norm_lo_nlz );
//   const uint32_t c_e_hidden                         = _uint32_andc( c_e_hidden_normalized, is_c_m_hidden_underflow );
//   const uint32_t c_m_hidden                         = _uint32_sels( is_c_m_hidden_underflow_msb, c_m_hidden_underflow_normalized, c_m_hidden_normalized );
//   const uint32_t c_m_normalized                     = _uint32_sels( is_c_m_norm_no_hidden_msb, c_m_hidden, c_m_norm_radix_corrected );
//   const uint32_t c_e_normalized                     = _uint32_sels( is_c_m_norm_no_hidden_msb, c_e_hidden, c_e_norm_radix_corrected );
//   const uint32_t c_m_norm_round_amount              = _uint32_and(  c_m_normalized, h_m_mask );
//   const uint32_t c_m_norm_rounded                   = _uint32_add(  c_m_normalized, c_m_norm_round_amount );
//   const uint32_t is_round_overflow_test             = _uint32_and(  c_e_normalized, m_round_overflow_bit  );
//   const uint32_t is_round_overflow_msb              = _uint32_neg(  is_round_overflow_test );
//   const uint32_t c_m_norm_inplace                   = _uint32_srl(  c_m_norm_rounded,    h_m_bit_count );
//   const uint32_t c_m                                = _uint32_and(  c_m_norm_inplace,    h_m_mask      );
//   const uint32_t c_e_norm_inplace                   = _uint32_sll(  c_e_normalized, h_e_pos       );
//   const uint32_t c_e                                = _uint32_and(  c_e_norm_inplace,    h_e_mask      );
//   const uint32_t c_em_nan                           = _uint32_or(   h_e_mask,  a_m        );
//   const uint32_t c_nan                              = _uint32_or(   a_s,       c_em_nan   );
//   const uint32_t c_denorm                           = _uint32_or(   c_s,       c_m_denorm );
//   const uint32_t c_inf                              = _uint32_or(   c_s,       h_e_mask   );
//   const uint32_t c_em_norm                          = _uint32_or(   c_e,       c_m        );
//   const uint32_t is_a_e_flagged_msb                 = _uint32_sub(  h_e_mask_minus_one, a_e );
//   const uint32_t is_b_e_flagged_msb                 = _uint32_sub(  h_e_mask_minus_one, b_e );
//   const uint32_t is_a_e_eqz_msb                     = _uint32_dec(  a_e );
//   const uint32_t is_a_m_eqz_msb                     = _uint32_dec(  a_m );
//   const uint32_t is_b_e_eqz_msb                     = _uint32_dec(  b_e );
//   const uint32_t is_b_m_eqz_msb                     = _uint32_dec(  b_m );
//   const uint32_t is_b_eqz_msb                       = _uint32_and(  is_b_e_eqz_msb,          is_b_m_eqz_msb         );
//   const uint32_t is_a_eqz_msb                       = _uint32_and(  is_a_e_eqz_msb,          is_a_m_eqz_msb         );
//   const uint32_t is_c_nan_via_a_msb                 = _uint32_andc( is_a_e_flagged_msb,      is_b_e_flagged_msb     );
//   const uint32_t is_c_nan_via_b_msb                 = _uint32_andc( is_b_e_flagged_msb,      is_b_m_eqz_msb         );
//   const uint32_t is_c_nan_msb                       = _uint32_or(   is_c_nan_via_a_msb,      is_c_nan_via_b_msb     );
//   const uint32_t is_c_denorm_msb                    = _uint32_andc( is_b_e_eqz_msb,          is_a_e_flagged_msb     );
//   const uint32_t is_a_inf_msb                       = _uint32_and(  is_a_e_flagged_msb,      is_a_m_eqz_msb         );
//   const uint32_t is_c_snan_msb                      = _uint32_and(  is_a_inf_msb,            is_b_eqz_msb           );
//   const uint32_t is_c_nan_min_via_a_msb             = _uint32_and(  is_a_e_flagged_msb,      is_b_eqz_msb           );
//   const uint32_t is_c_nan_min_via_b_msb             = _uint32_and(  is_b_e_flagged_msb,      is_a_eqz_msb           );
//   const uint32_t is_c_nan_min_msb                   = _uint32_or(   is_c_nan_min_via_a_msb,  is_c_nan_min_via_b_msb );
//   const uint32_t is_c_inf_msb                       = _uint32_or(   is_a_e_flagged_msb,      is_b_e_flagged_msb     );
//   const uint32_t is_overflow_msb                    = _uint32_or(   is_round_overflow_msb,   is_mul_overflow_msb    );
//   const uint32_t c_em_overflow_result               = _uint32_sels( is_overflow_msb, h_e_mask, c_em_norm );
//   const uint32_t c_common_result                    = _uint32_or(   c_s, c_em_overflow_result );
//   const uint32_t c_zero_result                      = _uint32_sels( is_b_eqz_msb,     c_s,       c_common_result  );
//   const uint32_t c_nan_result                       = _uint32_sels( is_c_nan_msb,     c_nan,     c_zero_result );
//   const uint32_t c_nan_min_result                   = _uint32_sels( is_c_nan_min_msb, h_nan_min, c_nan_result     );
//   const uint32_t c_inf_result                       = _uint32_sels( is_c_inf_msb,     c_inf,     c_nan_min_result   );
//   const uint32_t c_denorm_result                    = _uint32_sels( is_c_denorm_msb,  c_denorm,  c_inf_result);
//   const uint32_t c_result                           = _uint32_sels( is_c_snan_msb,    h_snan,    c_denorm_result );

//   return (uint16_t)(c_result);
// }

#endif  // USE_NEON
