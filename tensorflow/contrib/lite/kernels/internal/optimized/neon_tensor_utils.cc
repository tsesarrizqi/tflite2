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

// // clblast
// #include <clblast.h>

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
// using namespace clblast;

// void OpenCLPortableMatrixBatchVectorMultiplyAccumulate(const float* matrix,
//                                                  int m_rows, int m_cols,
//                                                  const float* vector,
//                                                  int n_batch, float* result,
//                                                  int result_stride,
//                                                  cl_context context, cl_command_queue queue, cl_program program) {

//   int matrixsize = m_rows*m_cols*sizeof(half);
//   int vectorsize = m_cols*n_batch*sizeof(half);
//   int resultsize = m_rows*n_batch*sizeof(half);

//   // Test half precision
//   float tessum = 0;
//   half* matrixHalf = (half*) malloc(matrixsize);
//   for(int i = 0; i < m_rows*m_cols; i++) {
//     // half halfTmp(matrix[i]);
//     matrixHalf[i] = half_cast<half,std::round_toward_zero>(matrix[i]);
//     tessum += (float)matrixHalf[i];
//   }
//   half coba = half_cast<half,std::round_toward_zero>(741771.187500);
//   __android_log_print(ANDROID_LOG_INFO, "Matmulruntime", "tessumcoba: %lf", (float) coba);
//   __android_log_print(ANDROID_LOG_INFO, "Matmulruntime", "tessummatrix: %lf", (float) tessum);
//   tessum = 0;
//   float tessumfloat = 0.0;
//   half* vectorHalf = (half*) malloc(vectorsize);
//   for(int i = 0; i < m_cols*n_batch; i++) {
//     // half halfTmp(vector[i]);
//     vectorHalf[i] = half_cast<half,std::round_toward_zero>(vector[i]);
//     if(isinf((float)vectorHalf[i])) {
//       __android_log_print(ANDROID_LOG_INFO, "Matmulruntime", "tessumadaelemeninf: %lf", (float) vectorHalf[i]);
//       __android_log_print(ANDROID_LOG_INFO, "Matmulruntime", "tessumadaelemeninfaslinya: %lf", (float) vector[i]);
//     }
//     tessum += (float)vectorHalf[i];
//     tessumfloat += vector[i];
//   }
//   __android_log_print(ANDROID_LOG_INFO, "Matmulruntime", "tessumvector: %lf", (float) tessum);
//   __android_log_print(ANDROID_LOG_INFO, "Matmulruntime", "tessumvectorfloat: %lf", tessumfloat);
//   half* resultHalf = (half*) malloc(resultsize);

//   cl_mem d_a;
//   // cl_mem d_at;
//   cl_mem d_b;
//   cl_mem d_c;
      
//   cl_kernel kernel;

//   // size_t localSizetmp;
//   cl_int err;

//   double wall0 = get_wall_time();
//   double cpu0  = get_cpu_time();

//   kernel = clCreateKernel(program, "matrixVectorMulF4", &err);
//   // kernel2 = clCreateKernel(program, "transpose", &err);

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
//   // d_at = clCreateBuffer(context, CL_MEM_READ_WRITE, matrixsize, NULL, NULL);

//   wall1 = get_wall_time();
//   cpu1  = get_cpu_time();

//   wall = wall1 - wall0;
//   cpu = cpu1 - cpu0;
  
//   __android_log_print(ANDROID_LOG_INFO, "Matmulruntime", "createbuffer: %lf", wall);


//   wall0 = get_wall_time();
//   cpu0  = get_cpu_time();

//   err = clEnqueueWriteBuffer(queue, d_a, CL_TRUE, 0,
//                                  matrixsize, matrixHalf, 0, NULL, NULL);
//   err = clEnqueueWriteBuffer(queue, d_b, CL_TRUE, 0,
//                                  vectorsize, vectorHalf, 0, NULL, NULL);

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
//   err  = clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_a);
//   err  = clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_b);
//   err  = clSetKernelArg(kernel, 3, sizeof(int), &m_cols);
//   err  = clSetKernelArg(kernel, 4, sizeof(int), &m_rows);
//   err  = clSetKernelArg(kernel, 5, sizeof(int), &n_batch);

//   // err  = clSetKernelArg(kernel2, 0, sizeof(cl_mem), &d_a);
//   // err  = clSetKernelArg(kernel2, 1, sizeof(cl_mem), &d_at);
//   // err  = clSetKernelArg(kernel2, 2, sizeof(int), &m_rows);
//   // err  = clSetKernelArg(kernel2, 3, sizeof(int), &m_cols);

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

//   // const size_t local[2] = { 8, 32 };
//   // const size_t global[2] = { (size_t) (((m_rows-1)/8+1)*8), 32 };
//   // const size_t local2[2] = { 8, 32 };
//   // const size_t global2[2] = { (size_t) (((m_rows-1)/8+1)*8), (size_t) (m_cols/4) };

//   const size_t local[2] = { 8, 32 };
//   const size_t global[2] = { (size_t) (((m_rows/4-1)/8+1)*8), 32 };

//   // const size_t local2[2] = { 16, 16 };
//   // const size_t global2[2] = { (size_t) m_cols, (size_t) m_rows };

//   wall0 = get_wall_time();
//   cpu0  = get_cpu_time();

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

//   // half halfTmp2 = half_cast<half>(matrix[0]);
//   for(int i = 0; i < m_rows*n_batch; i++) {
//     result[i] = (float) resultHalf[i];
//   }

//   wall1 = get_wall_time();
//   cpu1  = get_cpu_time();

//   wall = wall1 - wall0;
//   cpu = cpu1 - cpu0;
  
//   __android_log_print(ANDROID_LOG_INFO, "Matmulruntime", "readbuffer: %lf", wall);

//   wall0 = get_wall_time();
//   cpu0  = get_cpu_time();

//   free(matrixHalf);
//   free(vectorHalf);
//   free(resultHalf);
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

void OpenCLPortableMatrixBatchVectorMultiplyAccumulate(const float* matrix,
                                                 int m_rows, int m_cols,
                                                 const float* vector,
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

  kernel = clCreateKernel(program, "matrixVectorMulF4float", &err);

  double wall1 = get_wall_time();
  double cpu1  = get_cpu_time();

  double wall = wall1 - wall0;
  double cpu = cpu1 - cpu0;
  
  // __android_log_print(ANDROID_LOG_INFO, "Matmulruntime", "createkernel: %lf", wall);

  // wall0 = get_wall_time();
  // cpu0  = get_cpu_time();

  // | CL_MEM_ALLOC_HOST_PTR
  d_a = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, matrixsize, NULL, NULL);
  d_b = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, vectorsize, NULL, NULL);
  d_c = clCreateBuffer(context, CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR, resultsize, NULL, NULL);

  // wall1 = get_wall_time();
  // cpu1  = get_cpu_time();

  // wall = wall1 - wall0;
  // cpu = cpu1 - cpu0;
  
  // __android_log_print(ANDROID_LOG_INFO, "Matmulruntime", "createbuffer: %lf", wall);


  wall0 = get_wall_time();
  cpu0  = get_cpu_time();

  // err = clEnqueueWriteBuffer(queue, d_a, CL_TRUE, 0,
  //                                matrixsize, matrix, 0, NULL, NULL);
  // err = clEnqueueWriteBuffer(queue, d_b, CL_TRUE, 0,
  //                               vectorsize, vector, 0, NULL, NULL);

  cl_float *host_a = (cl_float*)clEnqueueMapBuffer(
              queue,
              d_a,
              CL_TRUE,
              CL_MAP_WRITE,
              0,
              matrixsize,
              0, NULL, NULL, NULL);
  cl_float *host_b = (cl_float*)clEnqueueMapBuffer(
              queue,
              d_b,
              CL_TRUE,
              CL_MAP_WRITE,
              0,
              vectorsize,
              0, NULL, NULL, NULL);

  // clFinish(queue);

  std::memcpy(host_a, matrix, matrixsize);
  std::memcpy(host_b, vector, vectorsize);

  clEnqueueUnmapMemObject(queue,d_a,(void *) host_a,0, NULL, NULL);
  clEnqueueUnmapMemObject(queue,d_b,(void *) host_b,0, NULL, NULL);

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

  // wall0 = get_wall_time();
  // cpu0  = get_cpu_time();

  err = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global, local, 0, NULL, NULL);

  // clFinish(queue);

  // __android_log_print(ANDROID_LOG_INFO, "Matmulruntime", "runkernelMatmulerror1: %d", err);

  // wall1 = get_wall_time();
  // cpu1  = get_cpu_time();

  // wall = wall1 - wall0;
  // cpu = cpu1 - cpu0;
  
  // __android_log_print(ANDROID_LOG_INFO, "Matmulruntime", "runkernelOclMatmul: %lf", wall);

  // wall0 = get_wall_time();
  // cpu0  = get_cpu_time();

  // clEnqueueReadBuffer(queue, d_c, CL_TRUE, 0, resultsize, result, 0, NULL, NULL );

  cl_float *host_c = (cl_float*)clEnqueueMapBuffer(
              queue,
              d_c,
              CL_TRUE,
              CL_MAP_READ,
              0,
              resultsize,
              0, NULL, NULL, NULL);

  // clFinish(queue);

  std::memcpy(result, host_c, resultsize);

  clEnqueueUnmapMemObject(queue,d_c,(void *) host_c,0, NULL, NULL);

  // clFinish(queue);

  wall1 = get_wall_time();
  cpu1  = get_cpu_time();

  wall = wall1 - wall0;
  cpu = cpu1 - cpu0;
  
  // __android_log_print(ANDROID_LOG_INFO, "Matmulruntime", "readbuffer: %lf", wall);

  __android_log_print(ANDROID_LOG_INFO, "Matmulruntime", "runkernelOclTotal: %lf", wall);

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
  // Start Timers
  double wall0 = get_wall_time();
  double cpu0  = get_cpu_time();
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

  // Stop timers
  double wall1 = get_wall_time();
  double cpu1  = get_cpu_time();

  double wall = wall1 - wall0;
  double cpu = cpu1 - cpu0;

  // note: andoird log
  __android_log_print(ANDROID_LOG_INFO, "Matmulruntime", "runkernelPortable: %lf", wall);
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
  __android_log_print(ANDROID_LOG_INFO, "Matmulruntime", "runkernelNEON: %lf", wall);
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

        double totalwall = 0.0;

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
        
        // wall0 = get_wall_time();
        // cpu0  = get_cpu_time();
        createBuffer();
        // wall1 = get_wall_time();
        // cpu1  = get_cpu_time();
        // wall = wall1 - wall0;
        // cpu = cpu1 - cpu0;
        // __android_log_print(ANDROID_LOG_INFO, "VulkanConvDetail", "createBuffer: %lf", wall);

        wall0 = get_wall_time();
        cpu0  = get_cpu_time();
        copyInputData();
        wall1 = get_wall_time();
        cpu1  = get_cpu_time();
        totalwall += wall1 - wall0;
        // cpu = cpu1 - cpu0;
        // __android_log_print(ANDROID_LOG_INFO, "VulkanConvDetail", "createBuffer: %lf", wall);
        
        // wall0 = get_wall_time();
        // cpu0  = get_cpu_time();
        // createDescriptorSetLayout();
        // wall1 = get_wall_time();
        // cpu1  = get_cpu_time();
        // wall = wall1 - wall0;
        // cpu = cpu1 - cpu0;
        // __android_log_print(ANDROID_LOG_INFO, "VulkanConvDetail", "createDescriptorSetLayout: %lf", wall);
        
        // wall0 = get_wall_time();
        // cpu0  = get_cpu_time();
        createDescriptorSet();
        // wall1 = get_wall_time();
        // cpu1  = get_cpu_time();
        // wall = wall1 - wall0;
        // cpu = cpu1 - cpu0;
        // __android_log_print(ANDROID_LOG_INFO, "VulkanConvDetail", "createDescriptorSet: %lf", wall);
        
        // wall0 = get_wall_time();
        // cpu0  = get_cpu_time();
        // createComputePipeline();
        // wall1 = get_wall_time();
        // cpu1  = get_cpu_time();
        // wall = wall1 - wall0;
        // cpu = cpu1 - cpu0;
        // __android_log_print(ANDROID_LOG_INFO, "VulkanConvDetail", "createComputePipeline: %lf", wall);
        
        // wall0 = get_wall_time();
        // cpu0  = get_cpu_time();
        createCommandBuffer();
        // wall1 = get_wall_time();
        // cpu1  = get_cpu_time();
        // wall = wall1 - wall0;
        // cpu = cpu1 - cpu0;
        // __android_log_print(ANDROID_LOG_INFO, "VulkanConvDetail", "createCommandBuffer: %lf", wall);

        wall0 = get_wall_time();
        cpu0  = get_cpu_time();
        beginCommandBuffer();
        wall1 = get_wall_time();
        cpu1  = get_cpu_time();
        totalwall += wall1 - wall0;
        // cpu = cpu1 - cpu0;
        // __android_log_print(ANDROID_LOG_INFO, "VulkanConvDetail", "createCommandBuffer: %lf", wall);
        
        wall0 = get_wall_time();
        cpu0  = get_cpu_time();
        runCommandBuffer();
        wall1 = get_wall_time();
        cpu1  = get_cpu_time();
        totalwall += wall1 - wall0;
        // cpu = cpu1 - cpu0;
        // __android_log_print(ANDROID_LOG_INFO, "VulkanConvDetail", "runCommandBuffer: %lf", wall);
        
        wall0 = get_wall_time();
        cpu0  = get_cpu_time();
        getresult();
        wall1 = get_wall_time();
        cpu1  = get_cpu_time();
        totalwall += wall1 - wall0;
        // cpu = cpu1 - cpu0;
        // __android_log_print(ANDROID_LOG_INFO, "VulkanConvDetail", "getResult: %lf", wall);
        __android_log_print(ANDROID_LOG_INFO, "VulkanConvDetail", "runkernelTotal: %lf", totalwall);
        
        // wall0 = get_wall_time();
        // cpu0  = get_cpu_time();
        cleanup();
        // wall1 = get_wall_time();
        // cpu1  = get_cpu_time();
        // wall = wall1 - wall0;
        // cpu = cpu1 - cpu0;
        // __android_log_print(ANDROID_LOG_INFO, "VulkanConvDetail", "cleanup: %lf", wall);
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
        matrixACreateInfo.size = matrixASize; // buffer size in bytes. 
        matrixACreateInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT; // buffer is used as a storage buffer.
        matrixACreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE; // buffer is exclusive to a single queue family at a time. 

        VK_CHECK_RESULT(vkCreateBuffer(device, &matrixACreateInfo, NULL, &matrixA)); // create buffer.

        VkBufferCreateInfo matrixBCreateInfo = {};
        matrixBCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        matrixBCreateInfo.size = matrixBSize; // buffer size in bytes. 
        matrixBCreateInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT; // buffer is used as a storage buffer.
        matrixBCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE; // buffer is exclusive to a single queue family at a time. 

        VK_CHECK_RESULT(vkCreateBuffer(device, &matrixBCreateInfo, NULL, &matrixB)); // create buffer.

        VkBufferCreateInfo matrixCCreateInfo = {};
        matrixCCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        matrixCCreateInfo.size = matrixCSize; // buffer size in bytes. 
        matrixCCreateInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT; // buffer is used as a storage buffer.
        matrixCCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE; // buffer is exclusive to a single queue family at a time. 

        VK_CHECK_RESULT(vkCreateBuffer(device, &matrixCCreateInfo, NULL, &matrixC)); // create buffer.

        VkBufferCreateInfo matrixSizesCreateInfo = {};
        matrixSizesCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        matrixSizesCreateInfo.size = matrixSizesSize; // buffer size in bytes. 
        matrixSizesCreateInfo.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT; // buffer is used as a storage buffer.
        matrixSizesCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE; // buffer is exclusive to a single queue family at a time. 

        VK_CHECK_RESULT(vkCreateBuffer(device, &matrixSizesCreateInfo, NULL, &matrixSizes)); // create buffer.

        VkMemoryRequirements memoryRequirementsmatrixA, memoryRequirementsmatrixB, memoryRequirementsmatrixC, memoryRequirementsmatrixSizes;
        vkGetBufferMemoryRequirements(device, matrixA, &memoryRequirementsmatrixA);
        vkGetBufferMemoryRequirements(device, matrixB, &memoryRequirementsmatrixB);
        vkGetBufferMemoryRequirements(device, matrixC, &memoryRequirementsmatrixC);
        vkGetBufferMemoryRequirements(device, matrixSizes, &memoryRequirementsmatrixSizes);
        
        const VkDeviceSize memorySize = memoryRequirementsmatrixA.size+memoryRequirementsmatrixB.size+memoryRequirementsmatrixC.size+memoryRequirementsmatrixSizes.size;

        VkMemoryAllocateInfo allocateInfo = {};
        allocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocateInfo.allocationSize = memorySize; // specify required memory.

        allocateInfo.memoryTypeIndex = findMemoryType(
            memorySize, VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);

        VK_CHECK_RESULT(vkAllocateMemory(device, &allocateInfo, NULL, &bufferMemory));

        VK_CHECK_RESULT(vkBindBufferMemory(device, matrixA, bufferMemory, 0));
        VK_CHECK_RESULT(vkBindBufferMemory(device, matrixB, bufferMemory, matrixASize));
        VK_CHECK_RESULT(vkBindBufferMemory(device, matrixC, bufferMemory, matrixASize+matrixBSize));
        VK_CHECK_RESULT(vkBindBufferMemory(device, matrixSizes, bufferMemory, matrixASize+matrixBSize+matrixCSize));
      }

      void copyInputData() {

        float* matBuffertmp;
        VK_CHECK_RESULT(vkMapMemory(device, bufferMemory, 0, matrixASize+matrixBSize+matrixCSize, 0, (void **) &matBuffertmp));
        // VK_CHECK_RESULT(vkMapMemory(device, bufferMemory, 0, matrixASize+matrixBSize+matrixCSize+matrixSizesSize, 0, (void **) &matBuffertmp));

        // matBuffertmp[0] = M;
        // matBuffertmp[1] = K;
        // matBuffertmp[2] = N;

        int offset = 0;
        for(int i = 0; i < M*K; i++) {
            matBuffertmp[offset+i] = matA[i]; 
        }
        // TransposeFloatTensor(matA, M, K, matBuffertmp);
        offset = M*K;
        for(int i = 0; i < N*K; i++) {
            matBuffertmp[offset+i] = matB[i]; 
        }

        vkUnmapMemory(device, bufferMemory);

        int* matSizetmp;
        VK_CHECK_RESULT(vkMapMemory(device, bufferMemory, matrixASize+matrixBSize+matrixCSize, matrixSizesSize, 0, (void **) &matSizetmp));

        matSizetmp[0] = M;
        matSizetmp[1] = K;
        matSizetmp[2] = N;
        matSizetmp[3] = 0;

        vkUnmapMemory(device, bufferMemory);
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

        VkDescriptorPoolSize descriptorPoolSize[2];

        descriptorPoolSize[0] = {};
        descriptorPoolSize[0].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        descriptorPoolSize[0].descriptorCount = 2;

        descriptorPoolSize[1] = {};
        descriptorPoolSize[1].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        descriptorPoolSize[1].descriptorCount = 2;

        VkDescriptorPoolCreateInfo descriptorPoolCreateInfo = {};
        descriptorPoolCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        descriptorPoolCreateInfo.maxSets = 1; // we only need to allocate one descriptor set from the pool.
        descriptorPoolCreateInfo.poolSizeCount = 2;
        descriptorPoolCreateInfo.pPoolSizes = descriptorPoolSize;

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

        VkDescriptorBufferInfo descriptorBufferInfoMatB = {};
        descriptorBufferInfoMatB.buffer = matrixB;
        descriptorBufferInfoMatB.offset = 0;
        descriptorBufferInfoMatB.range = VK_WHOLE_SIZE;

        VkDescriptorBufferInfo descriptorBufferInfoMatC = {};
        descriptorBufferInfoMatC.buffer = matrixC;
        descriptorBufferInfoMatC.offset = 0;
        descriptorBufferInfoMatC.range = VK_WHOLE_SIZE;

        VkDescriptorBufferInfo descriptorBufferInfoMatSizes = {};
        descriptorBufferInfoMatSizes.buffer = matrixSizes;
        descriptorBufferInfoMatSizes.offset = 0;
        descriptorBufferInfoMatSizes.range = VK_WHOLE_SIZE;

        VkWriteDescriptorSet writeDescriptorSets[4];

        writeDescriptorSets[0] = {};
        writeDescriptorSets[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writeDescriptorSets[0].dstSet = descriptorSet; // write to this descriptor set.
        writeDescriptorSets[0].dstBinding = 0; // write to the first, and only binding.
        writeDescriptorSets[0].descriptorCount = 1; // update a single descriptor.
        writeDescriptorSets[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; // storage buffer.
        writeDescriptorSets[0].pBufferInfo = &descriptorBufferInfoMatA;

        writeDescriptorSets[1] = {};
        writeDescriptorSets[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writeDescriptorSets[1].dstSet = descriptorSet; // write to this descriptor set.
        writeDescriptorSets[1].dstBinding = 1; // write to the first, and only binding.
        writeDescriptorSets[1].descriptorCount = 1; // update a single descriptor.
        writeDescriptorSets[1].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER; // storage buffer.
        writeDescriptorSets[1].pBufferInfo = &descriptorBufferInfoMatB;

        writeDescriptorSets[2] = {};
        writeDescriptorSets[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writeDescriptorSets[2].dstSet = descriptorSet; // write to this descriptor set.
        writeDescriptorSets[2].dstBinding = 2; // write to the first, and only binding.
        writeDescriptorSets[2].descriptorCount = 1; // update a single descriptor.
        writeDescriptorSets[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; // storage buffer.
        writeDescriptorSets[2].pBufferInfo = &descriptorBufferInfoMatC;

        writeDescriptorSets[3] = {};
        writeDescriptorSets[3].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writeDescriptorSets[3].dstSet = descriptorSet; // write to this descriptor set.
        writeDescriptorSets[3].dstBinding = 3; // write to the first, and only binding.
        writeDescriptorSets[3].descriptorCount = 1; // update a single descriptor.
        writeDescriptorSets[3].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER; // storage buffer.
        writeDescriptorSets[3].pBufferInfo = &descriptorBufferInfoMatSizes;

        vkUpdateDescriptorSets(device, 4, writeDescriptorSets, 0, NULL);
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
    }

    void beginCommandBuffer() {
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
        // double wall0 = get_wall_time();
        // double cpu0  = get_cpu_time();

        VK_CHECK_RESULT(vkQueueSubmit(queue, 1, &submitInfo, 0));
        VK_CHECK_RESULT(vkQueueWaitIdle(queue));

        // Stop timers
        // double wall1 = get_wall_time();
        // double cpu1  = get_cpu_time();

        // double wall = wall1 - wall0;
        // double cpu = cpu1 - cpu0;

        // __android_log_print(ANDROID_LOG_INFO, "MatmulHalfResult", "runkernelVMatmul: %lf", wall);
    }

    void getresult() {
        float *matCtmp;
        VK_CHECK_RESULT(vkMapMemory(device, bufferMemory, matrixASize+matrixBSize, matrixCSize, 0, (void **)&matCtmp));
      
        // float sumC = 0.0;
        for (int k = 0; k < M; k++) {
          matC[k] = matCtmp[k];
          // if(k < 100) {
          //     __android_log_print(ANDROID_LOG_INFO, "VulkanTest", "Matmul %d: %lf", k, matCtmp[k]);
          // }
        }

        // std::memcpy(matC, matCtmp, matrixCSize);

        // __android_log_print(ANDROID_LOG_INFO, "VulkanTest", "Matmul sumC: %lf", sumC);

        vkUnmapMemory(device, bufferMemory); 
    }

    void cleanup() {
        vkFreeMemory(device, bufferMemory, NULL);
        vkDestroyBuffer(device, matrixA, NULL);
        vkDestroyBuffer(device, matrixB, NULL);
        vkDestroyBuffer(device, matrixC, NULL);
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

// void clBlastOpenCL() {

// }

void NeonMatrixBatchVectorMultiplyAccumulateOpenCL(const float* matrix, int m_rows,
                                             int m_cols, const float* vector,
                                             int n_batch, float* result,
                                             int result_stride,
                                             cl_context context_cl, cl_command_queue queue, cl_program program, cl_mem cl_mem_arr[6],
                                             VkPhysicalDevice physicalDevice, VkDevice device, VkPipeline pipelineConv, VkPipeline pipelineMatmul, VkPipelineLayout pipelineLayoutConv, VkPipelineLayout pipelineLayoutMatmul, 
    VkDescriptorSetLayout descriptorSetLayoutConv, VkDescriptorSetLayout descriptorSetLayoutMatmul, VkQueue queueV, uint32_t queueFamilyIndex,
    VkCommandPool conv_commandPool, VkCommandBuffer conv_commandBuffer, VkBuffer conv_matrixA, VkBuffer conv_matrixB, VkBuffer conv_matrixC, VkBuffer conv_matrixSizes, VkDeviceMemory conv_bufferMemory) {
  

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

m_cols = 8192;
m_rows = 8192;
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

// float* resultport = (float*) malloc(m_rows*n_batch*sizeof(float));
// float* resultocl = (float*) malloc(m_rows*n_batch*sizeof(float));

// // Start Timers
// double wall0 = get_wall_time();
// double cpu0  = get_cpu_time();

// PortableMatrixBatchVectorMultiplyAccumulate(matrix,m_rows,m_cols,vector,n_batch,resultport,1);

// double wall1 = get_wall_time();
// double cpu1  = get_cpu_time();

// double wall = wall1 - wall0;
// double cpu = cpu1 - cpu0;

// // note: andoird log
// __android_log_print(ANDROID_LOG_INFO, "MatmulResult", "runkernelProtable: %lf", wall);

// sleep(1);

// Start Timers
// double wall0 = get_wall_time();
// double cpu0  = get_cpu_time();

// m_cols = 2048;
// m_rows = 1008;

// OpenCLPortableMatrixBatchVectorMultiplyAccumulate(matrix,m_rows,m_cols,vector,n_batch,result,1, context_cl, queue, program);

// m_cols = 32;
// m_rows = 32;
// __android_log_print(ANDROID_LOG_INFO, "MatmulResult", "runkernel32x32");
// // NeonMatrixBatchVectorMultiplyAccumulate(matA,m_rows,m_cols,matB,n_batch,matC,1);
// // OpenCLPortableMatrixBatchVectorMultiplyAccumulate(matA,m_rows,m_cols,matB,n_batch,matC,1, context_cl, queue, program);
// vulkanTest(matA, matB, matC, m_rows, m_cols, n_batch,
//   physicalDevice, device, pipelineMatmul, pipelineLayoutMatmul, 
//   descriptorSetLayoutMatmul, queueV, queueFamilyIndex);
// // double wall0 = get_wall_time();
// // double cpu0  = get_cpu_time();
// // PortableMatrixBatchVectorMultiplyAccumulate(matA,m_rows,m_cols,matB,n_batch,matC,1);
// // double wall1 = get_wall_time();
// // double cpu1  = get_cpu_time();
// // double wall = wall1 - wall0;
// // double cpu = cpu1 - cpu0;
// // __android_log_print(ANDROID_LOG_INFO, "MatmulResult", "runkernelProtable: %lf", wall);

// sleep(2);

// m_cols = 64;
// m_rows = 64;
// __android_log_print(ANDROID_LOG_INFO, "MatmulResult", "runkernel64x64");
// // NeonMatrixBatchVectorMultiplyAccumulate(matA,m_rows,m_cols,matB,n_batch,matC,1);
// // OpenCLPortableMatrixBatchVectorMultiplyAccumulate(matA,m_rows,m_cols,matB,n_batch,matC,1, context_cl, queue, program);
// vulkanTest(matA, matB, matC, m_rows, m_cols, n_batch,
//   physicalDevice, device, pipelineMatmul, pipelineLayoutMatmul, 
//   descriptorSetLayoutMatmul, queueV, queueFamilyIndex);
// // wall0 = get_wall_time();
// // cpu0  = get_cpu_time();
// // PortableMatrixBatchVectorMultiplyAccumulate(matA,m_rows,m_cols,matB,n_batch,matC,1);
// // wall1 = get_wall_time();
// // cpu1  = get_cpu_time();
// // wall = wall1 - wall0;
// // cpu = cpu1 - cpu0;
// // __android_log_print(ANDROID_LOG_INFO, "MatmulResult", "runkernelProtable: %lf", wall);

// sleep(2);

m_cols = 128;
m_rows = 128;
__android_log_print(ANDROID_LOG_INFO, "MatmulResult", "runkernel128x128");
// NeonMatrixBatchVectorMultiplyAccumulate(matA,m_rows,m_cols,matB,n_batch,matC,1);
// OpenCLPortableMatrixBatchVectorMultiplyAccumulate(matA,m_rows,m_cols,matB,n_batch,matC,1, context_cl, queue, program);
vulkanTest(matA, matB, matC, m_rows, m_cols, n_batch,
  physicalDevice, device, pipelineMatmul, pipelineLayoutMatmul, 
  descriptorSetLayoutMatmul, queueV, queueFamilyIndex);
// wall0 = get_wall_time();
// cpu0  = get_cpu_time();
// PortableMatrixBatchVectorMultiplyAccumulate(matA,m_rows,m_cols,matB,n_batch,matC,1);
// wall1 = get_wall_time();
// cpu1  = get_cpu_time();
// wall = wall1 - wall0;
// cpu = cpu1 - cpu0;
// __android_log_print(ANDROID_LOG_INFO, "MatmulResult", "runkernelProtable: %lf", wall);

sleep(2);

m_cols = 256;
m_rows = 256;
__android_log_print(ANDROID_LOG_INFO, "MatmulResult", "runkernel256x256");
// NeonMatrixBatchVectorMultiplyAccumulate(matA,m_rows,m_cols,matB,n_batch,matC,1);
// OpenCLPortableMatrixBatchVectorMultiplyAccumulate(matA,m_rows,m_cols,matB,n_batch,matC,1, context_cl, queue, program);
vulkanTest(matA, matB, matC, m_rows, m_cols, n_batch,
  physicalDevice, device, pipelineMatmul, pipelineLayoutMatmul, 
  descriptorSetLayoutMatmul, queueV, queueFamilyIndex);
// wall0 = get_wall_time();
// cpu0  = get_cpu_time();
// PortableMatrixBatchVectorMultiplyAccumulate(matA,m_rows,m_cols,matB,n_batch,matC,1);
// wall1 = get_wall_time();
// cpu1  = get_cpu_time();
// wall = wall1 - wall0;
// cpu = cpu1 - cpu0;
// __android_log_print(ANDROID_LOG_INFO, "MatmulResult", "runkernelProtable: %lf", wall);
sleep(2);

m_cols = 512;
m_rows = 512;
__android_log_print(ANDROID_LOG_INFO, "MatmulResult", "runkernel512x512");
// NeonMatrixBatchVectorMultiplyAccumulate(matA,m_rows,m_cols,matB,n_batch,matC,1);
// OpenCLPortableMatrixBatchVectorMultiplyAccumulate(matA,m_rows,m_cols,matB,n_batch,matC,1, context_cl, queue, program);
vulkanTest(matA, matB, matC, m_rows, m_cols, n_batch,
  physicalDevice, device, pipelineMatmul, pipelineLayoutMatmul, 
  descriptorSetLayoutMatmul, queueV, queueFamilyIndex);
// wall0 = get_wall_time();
// cpu0  = get_cpu_time();
// PortableMatrixBatchVectorMultiplyAccumulate(matA,m_rows,m_cols,matB,n_batch,matC,1);
// wall1 = get_wall_time();
// cpu1  = get_cpu_time();
// wall = wall1 - wall0;
// cpu = cpu1 - cpu0;
// __android_log_print(ANDROID_LOG_INFO, "MatmulResult", "runkernelProtable: %lf", wall);

sleep(2);

m_cols = 1024;
m_rows = 1024;
__android_log_print(ANDROID_LOG_INFO, "MatmulResult", "runkernel1024x1024");
// NeonMatrixBatchVectorMultiplyAccumulate(matA,m_rows,m_cols,matB,n_batch,matC,1);
// OpenCLPortableMatrixBatchVectorMultiplyAccumulate(matA,m_rows,m_cols,matB,n_batch,matC,1, context_cl, queue, program);
vulkanTest(matA, matB, matC, m_rows, m_cols, n_batch,
  physicalDevice, device, pipelineMatmul, pipelineLayoutMatmul, 
  descriptorSetLayoutMatmul, queueV, queueFamilyIndex);
// wall0 = get_wall_time();
// cpu0  = get_cpu_time();
// PortableMatrixBatchVectorMultiplyAccumulate(matA,m_rows,m_cols,matB,n_batch,matC,1);
// wall1 = get_wall_time();
// cpu1  = get_cpu_time();
// wall = wall1 - wall0;
// cpu = cpu1 - cpu0;
// __android_log_print(ANDROID_LOG_INFO, "MatmulResult", "runkernelProtable: %lf", wall);

sleep(2);

m_cols = 2048;
m_rows = 2048;
__android_log_print(ANDROID_LOG_INFO, "MatmulResult", "runkernel2048x2048");
// NeonMatrixBatchVectorMultiplyAccumulate(matA,m_rows,m_cols,matB,n_batch,matC,1);
// OpenCLPortableMatrixBatchVectorMultiplyAccumulate(matA,m_rows,m_cols,matB,n_batch,matC,1, context_cl, queue, program);
vulkanTest(matA, matB, matC, m_rows, m_cols, n_batch,
  physicalDevice, device, pipelineMatmul, pipelineLayoutMatmul, 
  descriptorSetLayoutMatmul, queueV, queueFamilyIndex);
// wall0 = get_wall_time();
// cpu0  = get_cpu_time();
// PortableMatrixBatchVectorMultiplyAccumulate(matA,m_rows,m_cols,matB,n_batch,matC,1);
// wall1 = get_wall_time();
// cpu1  = get_cpu_time();
// wall = wall1 - wall0;
// cpu = cpu1 - cpu0;
// __android_log_print(ANDROID_LOG_INFO, "MatmulResult", "runkernelProtable: %lf", wall);

sleep(2);

m_cols = 4096;
m_rows = 4096;
__android_log_print(ANDROID_LOG_INFO, "MatmulResult", "runkernel4096x4096");
// NeonMatrixBatchVectorMultiplyAccumulate(matA,m_rows,m_cols,matB,n_batch,matC,1);
// OpenCLPortableMatrixBatchVectorMultiplyAccumulate(matA,m_rows,m_cols,matB,n_batch,matC,1, context_cl, queue, program);
vulkanTest(matA, matB, matC, m_rows, m_cols, n_batch,
  physicalDevice, device, pipelineMatmul, pipelineLayoutMatmul, 
  descriptorSetLayoutMatmul, queueV, queueFamilyIndex);
// wall0 = get_wall_time();
// cpu0  = get_cpu_time();
// PortableMatrixBatchVectorMultiplyAccumulate(matA,m_rows,m_cols,matB,n_batch,matC,1);
// wall1 = get_wall_time();
// cpu1  = get_cpu_time();
// wall = wall1 - wall0;
// cpu = cpu1 - cpu0;
// __android_log_print(ANDROID_LOG_INFO, "MatmulResult", "runkernelProtable: %lf", wall);

sleep(2);

m_cols = 8192;
m_rows = 8192;
__android_log_print(ANDROID_LOG_INFO, "MatmulResult", "runkernel8192x8192");
// NeonMatrixBatchVectorMultiplyAccumulate(matA,m_rows,m_cols,matB,n_batch,matC,1);
// OpenCLPortableMatrixBatchVectorMultiplyAccumulate(matA,m_rows,m_cols,matB,n_batch,matC,1, context_cl, queue, program);
vulkanTest(matA, matB, matC, m_rows, m_cols, n_batch,
  physicalDevice, device, pipelineMatmul, pipelineLayoutMatmul, 
  descriptorSetLayoutMatmul, queueV, queueFamilyIndex);
// wall0 = get_wall_time();
// cpu0  = get_cpu_time();
// PortableMatrixBatchVectorMultiplyAccumulate(matA,m_rows,m_cols,matB,n_batch,matC,1);
// wall1 = get_wall_time();
// cpu1  = get_cpu_time();
// wall = wall1 - wall0;
// cpu = cpu1 - cpu0;
// __android_log_print(ANDROID_LOG_INFO, "MatmulResult", "runkernelProtable: %lf", wall);

sleep(2);

free(matA);
free(matB);
free(matC);

// vulkanTest(matrix, vector, result, m_rows, m_cols, n_batch,
//   physicalDevice, device, pipelineMatmul, pipelineLayoutMatmul, 
//   descriptorSetLayoutMatmul, queueV, queueFamilyIndex);

// double wall1 = get_wall_time();
// double cpu1  = get_cpu_time();

// double wall = wall1 - wall0;
// double cpu = cpu1 - cpu0;

// // note: andoird log
// __android_log_print(ANDROID_LOG_INFO, "MatmulResult", "runkerneltotal: %lf", wall);

// double sum = 0.0;
// for(int i = 0; i < m_rows*n_batch; i++) {
//     sum += resultport[i];
//     // __android_log_print(ANDROID_LOG_INFO, "MatmulHalfResult", "%d: %f", i, matC[i]);
// }
// __android_log_print(ANDROID_LOG_INFO, "MatmulResult", "Resultmatmulport: %lf", sum);

// sum = 0.0;
// for(int i = 0; i < m_rows*n_batch; i++) {
//     sum += resultocl[i];
//     // if(i < 200)
//     // __android_log_print(ANDROID_LOG_INFO, "MatmulResult", "%d: %lf", i, resultocl[i]);
// }

// __android_log_print(ANDROID_LOG_INFO, "MatmulResult", "Resultmatmulocl: %lf", sum);

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

#endif  // USE_NEON
