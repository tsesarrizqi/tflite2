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
// #include <unistd.h>

#include <fstream>
#include <iostream>
#include <vector>

//note: shaderc
#include <shaderc/shaderc.hpp>

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
  cl_mem d_at;
  cl_mem d_b;
  cl_mem d_c;
      
  cl_kernel kernel, kernel2;

  size_t localSizetmp;
  cl_int err;

  double wall0 = get_wall_time();
  double cpu0  = get_cpu_time();

  kernel = clCreateKernel(program, "matrixVectorMulNaive", &err);
  kernel2 = clCreateKernel(program, "transpose", &err);

  double wall1 = get_wall_time();
  double cpu1  = get_cpu_time();

  double wall = wall1 - wall0;
  double cpu = cpu1 - cpu0;
  
  __android_log_print(ANDROID_LOG_INFO, "Matmulruntime", "createkernel: %lf", wall);

  wall0 = get_wall_time();
  cpu0  = get_cpu_time();

  // | CL_MEM_ALLOC_HOST_PTR
  d_a = clCreateBuffer(context, CL_MEM_READ_ONLY, matrixsize, NULL, NULL);
  d_b = clCreateBuffer(context, CL_MEM_READ_ONLY, vectorsize, NULL, NULL);
  d_c = clCreateBuffer(context, CL_MEM_WRITE_ONLY, resultsize, NULL, NULL);
  d_at = clCreateBuffer(context, CL_MEM_READ_WRITE, matrixsize, NULL, NULL);

  wall1 = get_wall_time();
  cpu1  = get_cpu_time();

  wall = wall1 - wall0;
  cpu = cpu1 - cpu0;
  
  __android_log_print(ANDROID_LOG_INFO, "Matmulruntime", "createbuffer: %lf", wall);


  wall0 = get_wall_time();
  cpu0  = get_cpu_time();

  err = clEnqueueWriteBuffer(queue, d_a, CL_TRUE, 0,
                                 m_rows*m_cols*sizeof(float), matrix, 0, NULL, NULL);
  err = clEnqueueWriteBuffer(queue, d_b, CL_TRUE, 0,
                                 m_cols*n_batch*sizeof(float), vector, 0, NULL, NULL);

  // cl_float *host_a = (cl_float*)clEnqueueMapBuffer(
  //             queue,
  //             d_a,
  //             CL_TRUE,
  //             CL_MAP_WRITE,
  //             0,
  //             matrixsize,
  //             0, NULL, NULL, NULL);
  // cl_float *host_b = (cl_float*)clEnqueueMapBuffer(
  //             queue,
  //             d_b,
  //             CL_TRUE,
  //             CL_MAP_WRITE,
  //             0,
  //             vectorsize,
  //             0, NULL, NULL, NULL);

  // std::memcpy(host_a, matrix, matrixsize);
  // std::memcpy(host_b, vector, vectorsize);

  // clEnqueueUnmapMemObject(queue,d_a,(void *) host_a,0, NULL, NULL);
  // clEnqueueUnmapMemObject(queue,d_b,(void *) host_b,0, NULL, NULL);

  clFinish(queue);

  wall1 = get_wall_time();
  cpu1  = get_cpu_time();

  wall = wall1 - wall0;
  cpu = cpu1 - cpu0;
  
  __android_log_print(ANDROID_LOG_INFO, "Matmulruntime", "writebuffer: %lf", wall);

  err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_c);
  err  = clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_a);
  err  = clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_b);
  err  = clSetKernelArg(kernel, 3, sizeof(int), &m_cols);
  err  = clSetKernelArg(kernel, 4, sizeof(int), &m_rows);
  err  = clSetKernelArg(kernel, 5, sizeof(int), &n_batch);

  // err  = clSetKernelArg(kernel2, 0, sizeof(cl_mem), &d_a);
  // err  = clSetKernelArg(kernel2, 1, sizeof(cl_mem), &d_at);
  // err  = clSetKernelArg(kernel2, 2, sizeof(int), &m_rows);
  // err  = clSetKernelArg(kernel2, 3, sizeof(int), &m_cols);

  const int TS = 32;
  // const size_t localSize = (size_t)TS;
  // const size_t globalSize0 = (size_t)(((m_rows-1)/TS+1)*TS);
  // const size_t globalSize1 = (size_t)(((n_batch-1)/TS+1)*TS);

  // __android_log_print(ANDROID_LOG_INFO, "Matmulruntime", "GlobalSize0: %d", globalSize0);
  // __android_log_print(ANDROID_LOG_INFO, "Matmulruntime", "GlobalSize1: %d", globalSize1);
  // __android_log_print(ANDROID_LOG_INFO, "Matmulruntime", "localSize: %d", localSize);

  // const size_t local[2] = { localSize, localSize };
  // const size_t global[2] = { (size_t)(((m_rows-1)/TS+1)*TS), (size_t)(((n_batch-1)/TS+1)*TS) };

  // __android_log_print(ANDROID_LOG_INFO, "Matmulruntime", "matrixsize: %d %d %d", m_rows, m_cols, n_batch);

  // const size_t local[2] = { (size_t) TS, (size_t) (TS/8) };
  // const size_t global[2] = { (size_t) (((m_rows-1)/32+1)*32), (size_t) (((n_batch-1)/32+1)*4) };

  const size_t local[2] = { 32, 8 };
  const size_t global[2] = { (size_t) (((m_rows-1)/32+1)*32), 8 };
  // const size_t local2[2] = { 8, 32 };
  // const size_t global2[2] = { (size_t) (((m_rows-1)/8+1)*8), (size_t) (m_cols/4) };

  // const size_t local2[2] = { 16, 16 };
  // const size_t global2[2] = { (size_t) m_cols, (size_t) m_rows };

  wall0 = get_wall_time();
  cpu0  = get_cpu_time();

  // err = clEnqueueNDRangeKernel(queue, kernel2, 2, NULL, global2, local2, 0, NULL, NULL);

  err = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global, local, 0, NULL, NULL);
  clFinish(queue);

  wall1 = get_wall_time();
  cpu1  = get_cpu_time();

  wall = wall1 - wall0;
  cpu = cpu1 - cpu0;
  
  __android_log_print(ANDROID_LOG_INFO, "Matmulruntime", "runkernel: %lf", wall);

  __android_log_print(ANDROID_LOG_INFO, "Matmulruntime", "Matmulerror: %d", err);

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


/*
The application launches a compute shader that renders the mandelbrot set,
by rendering it into a storage buffer.
The storage buffer is then read from the GPU, and saved as .png. 
*/
class ComputeApplication {


private:
    // // The pixels of the rendered mandelbrot set are in this format:
    // struct UniformBufferObject {
    //     int M;
    //     int K;
    //     int N;
    // };
    // UniformBufferObject matrixSizesStruct;
    
    /*
    In order to use Vulkan, you must create an instance. 
    */
    VkInstance instance;

    // VkDebugReportCallbackEXT debugReportCallback;
    /*
    The physical device is some device on the system that supports usage of Vulkan.
    Often, it is simply a graphics card that supports Vulkan. 
    */
    VkPhysicalDevice physicalDevice;
    /*
    Then we have the logical device VkDevice, which basically allows 
    us to interact with the physical device. 
    */
    VkDevice device;

    /*
    The pipeline specifies the pipeline that all graphics and compute commands pass though in Vulkan.

    We will be creating a simple compute pipeline in this application. 
    */
    VkPipeline pipeline;
    VkPipelineLayout pipelineLayout;
    VkShaderModule computeShaderModule;

    /*
    The command buffer is used to record commands, that will be submitted to a queue.

    To allocate such command buffers, we use a command pool.
    */
    VkCommandPool commandPool;
    VkCommandBuffer commandBuffer;

    /*

    Descriptors represent resources in shaders. They allow us to use things like
    uniform buffers, storage buffers and images in GLSL. 

    A single descriptor represents a single resource, and several descriptors are organized
    into descriptor sets, which are basically just collections of descriptors.
    */
    VkDescriptorPool descriptorPool;
    VkDescriptorSet descriptorSet;
    VkDescriptorSetLayout descriptorSetLayout;

    /*
    The mandelbrot set will be rendered to this buffer.

    The memory that backs the buffer is bufferMemory. 
    */
    VkBuffer matrixA, matrixB, matrixC, matrixSizes;
    // VkDeviceMemory bufferMemorymatA, bufferMemorymatB, bufferMemorymatC, bufferMemoryS;
    VkDeviceMemory bufferMemory;
        
    uint32_t matrixASize, matrixBSize, matrixCSize, matrixSizesSize; // size of `buffer` in bytes.

    // std::vector<const char *> enabledLayers;

    /*
    In order to execute commands on a device(GPU), the commands must be submitted
    to a queue. The commands are stored in a command buffer, and this command buffer
    is given to the queue. 

    There will be different kinds of queues on the device. Not all queues support
    graphics operations, for instance. For this application, we at least want a queue
    that supports compute operations. 
    */
    VkQueue queue; // a queue supporting compute operations.

    /*
    Groups of queues that have the same capabilities(for instance, they all supports graphics and computer operations),
    are grouped into queue families. 
    
    When submitting a command buffer, you must specify to which queue in the family you are submitting to. 
    This variable keeps track of the index of that queue in its family. 
    */
    uint32_t queueFamilyIndex;
    int M,K,N;
    float* matA;
    float* matB;
    float* matC;

public:
    void run(const float* matA0, const float* matB0, float* matC0, int M0, int K0, int N0) {
        // Buffer size of the storage buffer that will contain the rendered mandelbrot set.
        matrixASize = (uint32_t) (sizeof(float) * M0 * K0);
        matrixBSize = (uint32_t) (sizeof(float) * K0 * N0);
        matrixCSize = (uint32_t) (sizeof(float) * M0 * N0);
        matrixSizesSize = sizeof(float) * 2;
        M = M0;
        K = K0;
        N = N0;
        matA = (float*) matA0;
        matB = (float*) matB0;
        matC = matC0;
        // matrixSizesStruct = {};
        // matrixSizesStruct.M = 4;
        // matrixSizesStruct.K = 3;
        // matrixSizesStruct.N = 2;

        // Initialize vulkan:
        createInstance();
        // __android_log_print(ANDROID_LOG_INFO, "VulkanTest", "createInstance Sukses");
        findPhysicalDevice();
        // __android_log_print(ANDROID_LOG_INFO, "VulkanTest", "findPhysicalDevice Sukses");
        createDevice();
        // __android_log_print(ANDROID_LOG_INFO, "VulkanTest", "createDevice Sukses");
        createBuffer();
        // __android_log_print(ANDROID_LOG_INFO, "VulkanTest", "createBuffer Sukses");
        createDescriptorSetLayout();
        // __android_log_print(ANDROID_LOG_INFO, "VulkanTest", "createDescriptorSetLayout Sukses");
        createDescriptorSet();
        // __android_log_print(ANDROID_LOG_INFO, "VulkanTest", "createDescriptorSet Sukses");
        createComputePipeline();
        // __android_log_print(ANDROID_LOG_INFO, "VulkanTest", "createComputePipeline Sukses");
        createCommandBuffer();
        // __android_log_print(ANDROID_LOG_INFO, "VulkanTest", "createCommandBuffer Sukses");

        // Finally, run the recorded command buffer.
        runCommandBuffer();
        // __android_log_print(ANDROID_LOG_INFO, "VulkanTest", "runCommandBuffer Sukses");

        getresult();
        // __android_log_print(ANDROID_LOG_INFO, "VulkanTest", "getresult Sukses");

        // Clean up all vulkan resources.
        cleanup();
        // __android_log_print(ANDROID_LOG_INFO, "VulkanTest", "cleanup Sukses");
    }

    void createInstance() {

        /*
        Next, we actually create the instance.
        
        */
        
        /*
        Contains application info. This is actually not that important.
        The only real important field is apiVersion.
        */
        VkApplicationInfo applicationInfo = {};
        applicationInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        applicationInfo.pApplicationName = "Matrix Multiplication";
        applicationInfo.applicationVersion = 0;
        applicationInfo.pEngineName = "Naive";
        applicationInfo.engineVersion = 0;
        applicationInfo.apiVersion = VK_MAKE_VERSION(1, 0, 31);
        
        VkInstanceCreateInfo createInfo = {};
        createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        createInfo.flags = 0;
        createInfo.pApplicationInfo = &applicationInfo;
  
        /*
        Actually create the instance.
        Having created the instance, we can actually start using vulkan.
        */
        VK_CHECK_RESULT(vkCreateInstance(
            &createInfo,
            NULL,
            &instance));
    }

    void findPhysicalDevice() {
        /*
        In this function, we find a physical device that can be used with Vulkan.
        */

        /*
        So, first we will list all physical devices on the system with vkEnumeratePhysicalDevices .
        */
        uint32_t deviceCount;
        vkEnumeratePhysicalDevices(instance, &deviceCount, NULL);
        if (deviceCount == 0) {
            throw std::runtime_error("could not find a device with vulkan support");
        }

        // __android_log_print(ANDROID_LOG_INFO, "VulkanTest", "Device count: %d", deviceCount);

        std::vector<VkPhysicalDevice> devices(deviceCount);
        vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());

        /*
        Next, we choose a device that can be used for our purposes. 

        With VkPhysicalDeviceFeatures(), we can retrieve a fine-grained list of physical features supported by the device.
        However, in this demo, we are simply launching a simple compute shader, and there are no 
        special physical features demanded for this task.

        With VkPhysicalDeviceProperties(), we can obtain a list of physical device properties. Most importantly,
        we obtain a list of physical device limitations. For this application, we launch a compute shader,
        and the maximum size of the workgroups and total number of compute shader invocations is limited by the physical device,
        and we should ensure that the limitations named maxComputeWorkGroupCount, maxComputeWorkGroupInvocations and 
        maxComputeWorkGroupSize are not exceeded by our application.  Moreover, we are using a storage buffer in the compute shader,
        and we should ensure that it is not larger than the device can handle, by checking the limitation maxStorageBufferRange. 

        However, in our application, the workgroup size and total number of shader invocations is relatively small, and the storage buffer is
        not that large, and thus a vast majority of devices will be able to handle it. This can be verified by looking at some devices at_
        http://vulkan.gpuinfo.org/

        Therefore, to keep things simple and clean, we will not perform any such checks here, and just pick the first physical
        device in the list. But in a real and serious application, those limitations should certainly be taken into account.

        */
        for (VkPhysicalDevice device : devices) {
            if (true) { // As above stated, we do no feature checks, so just accept.
                physicalDevice = device;
                break;
            }
        }
    }

    // Returns the index of a queue family that supports compute operations. 
    uint32_t getComputeQueueFamilyIndex() {
        uint32_t queueFamilyCount;

        vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, NULL);

        // Retrieve all queue families.
        std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
        vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, queueFamilies.data());

        // Now find a family that supports compute.
        uint32_t i = 0;
        for (; i < queueFamilies.size(); ++i) {
            VkQueueFamilyProperties props = queueFamilies[i];

            if (props.queueCount > 0 && (props.queueFlags & VK_QUEUE_COMPUTE_BIT)) {
                // found a queue with compute. We're done!
                break;
            }
        }

        if (i == queueFamilies.size()) {
            throw std::runtime_error("could not find a queue family that supports operations");
        }

        return i;
    }

    void createDevice() {
        /*
        We create the logical device in this function.
        */

        /*
        When creating the device, we also specify what queues it has.
        */
        VkDeviceQueueCreateInfo queueCreateInfo = {};
        queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        queueFamilyIndex = getComputeQueueFamilyIndex(); // find queue family with compute capability.
        queueCreateInfo.queueFamilyIndex = queueFamilyIndex;
        queueCreateInfo.queueCount = 1; // create one queue in this family. We don't need more.
        float queuePriorities = 1.0;  // we only have one queue, so this is not that imporant. 
        queueCreateInfo.pQueuePriorities = &queuePriorities;

        /*
        Now we create the logical device. The logical device allows us to interact with the physical
        device.
        */
        VkDeviceCreateInfo deviceCreateInfo = {};

        // Specify any desired device features here. We do not need any for this application, though.
        VkPhysicalDeviceFeatures deviceFeatures = {};

        deviceCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        deviceCreateInfo.pQueueCreateInfos = &queueCreateInfo; // when creating the logical device, we also specify what queues it has.
        deviceCreateInfo.queueCreateInfoCount = 1;
        deviceCreateInfo.pEnabledFeatures = &deviceFeatures;

        VK_CHECK_RESULT(vkCreateDevice(physicalDevice, &deviceCreateInfo, NULL, &device)); // create logical device.

        // Get a handle to the only member of the queue family.
        vkGetDeviceQueue(device, queueFamilyIndex, 0, &queue);
    }

    // find memory type with desired properties.
    uint32_t findMemoryType(VkDeviceSize memorySize, VkMemoryPropertyFlags properties) {
        VkPhysicalDeviceMemoryProperties memoryProperties;

        vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memoryProperties);

        /*
        How does this search work?
        See the documentation of VkPhysicalDeviceMemoryProperties for a detailed description. 
        */
        for (uint32_t i = 0; i < memoryProperties.memoryTypeCount; ++i) {
            if ((memorySize < memoryProperties.memoryHeaps[memoryProperties.memoryTypes[i].heapIndex].size) &&
                ((memoryProperties.memoryTypes[i].propertyFlags & properties) == properties))
                return i;
        }
        return -1;
    }

    void createBuffer() {
        /*
        We will now create a buffer. We will render the mandelbrot set into this buffer
        in a computer shade later. 
        */
        
        VkBufferCreateInfo matrixACreateInfo = {};
        matrixACreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        matrixACreateInfo.size = matrixASize+matrixBSize+matrixCSize+(2*sizeof(float)); // buffer size in bytes. 
        matrixACreateInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT; // buffer is used as a storage buffer.
        matrixACreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE; // buffer is exclusive to a single queue family at a time. 

        VK_CHECK_RESULT(vkCreateBuffer(device, &matrixACreateInfo, NULL, &matrixA)); // create buffer.

        // VkBufferCreateInfo matrixACreateInfo = {};
        // matrixACreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        // matrixACreateInfo.size = matrixASize; // buffer size in bytes. 
        // matrixACreateInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT; // buffer is used as a storage buffer.
        // matrixACreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE; // buffer is exclusive to a single queue family at a time. 

        // VK_CHECK_RESULT(vkCreateBuffer(device, &matrixACreateInfo, NULL, &matrixA)); // create buffer.

        // VkBufferCreateInfo matrixBCreateInfo = {};
        // matrixACreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        // matrixACreateInfo.size = matrixBSize; // buffer size in bytes. 
        // matrixACreateInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT; // buffer is used as a storage buffer.
        // matrixACreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE; // buffer is exclusive to a single queue family at a time. 

        // VK_CHECK_RESULT(vkCreateBuffer(device, &matrixBCreateInfo, NULL, &matrixB)); // create buffer.

        // VkBufferCreateInfo matrixSizesCreateInfo = {};
        // matrixACreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        // matrixACreateInfo.size = matrixSizesSize; // buffer size in bytes. 
        // matrixACreateInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT; // buffer is used as a uniform buffer.
        // matrixACreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE; // buffer is exclusive to a single queue family at a time. 

        // VK_CHECK_RESULT(vkCreateBuffer(device, &matrixSizesCreateInfo, NULL, &matrixSizes)); // create buffer.

        // VkBufferCreateInfo matrixCCreateInfo = {};
        // matrixACreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        // matrixACreateInfo.size = matrixCSize; // buffer size in bytes. 
        // matrixACreateInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT; // buffer is used as a storage buffer.
        // matrixACreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE; // buffer is exclusive to a single queue family at a time. 

        // VK_CHECK_RESULT(vkCreateBuffer(device, &matrixCCreateInfo, NULL, &matrixC)); // create buffer.

                /*
        But the buffer doesn't allocate memory for itself, so we must do that manually.
        */
    
        /*
        First, we find the memory requirements for the buffer.
        */
        VkMemoryRequirements memoryRequirementsmatrixA;
        vkGetBufferMemoryRequirements(device, matrixA, &memoryRequirementsmatrixA);
        // VkMemoryRequirements memoryRequirementsmatrixA, memoryRequirementsmatrixB, 
        //         memoryRequirementsmatrixC, memoryRequirementsmatrixSizes;
        // vkGetBufferMemoryRequirements(device, matrixA, &memoryRequirementsmatrixA);
        // vkGetBufferMemoryRequirements(device, matrixB, &memoryRequirementsmatrixB);
        // vkGetBufferMemoryRequirements(device, matrixSizes, &memoryRequirementsmatrixSizes);
        // vkGetBufferMemoryRequirements(device, matrixC, &memoryRequirementsmatrixC);
        
        
        const VkDeviceSize memorySize = memoryRequirementsmatrixA.size;
        // const VkDeviceSize memorySizematA = memoryRequirementsmatrixA.size;
        // const VkDeviceSize memorySizematB = memoryRequirementsmatrixB.size;
        // const VkDeviceSize memorySizematC = memoryRequirementsmatrixC.size; 
        // const VkDeviceSize memorySizeS = memoryRequirementsmatrixSizes.size;
        // const VkDeviceSize memorySize = memoryRequirementsmatrixA.size+memoryRequirementsmatrixSizes.size
        //           +memoryRequirementsmatrixB.size+memoryRequirementsmatrixC.size;

        /*
        Now use obtained memory requirements info to allocate the memory for the buffer.
        // */
        // VkMemoryAllocateInfo allocateInfomatA = {};
        // allocateInfomatA.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        // allocateInfomatA.allocationSize = memorySizematA; // specify required memory.

        // VkMemoryAllocateInfo allocateInfomatB = {};
        // allocateInfomatB.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        // allocateInfomatB.allocationSize = memorySizematB; // specify required memory.

        // VkMemoryAllocateInfo allocateInfomatC = {};
        // allocateInfomatC.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        // allocateInfomatC.allocationSize = memorySizematC; // specify required memory.

        // VkMemoryAllocateInfo allocateInfoS = {};
        // allocateInfoS.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        // allocateInfoS.allocationSize = memorySizeS; // specify required memory.

        VkMemoryAllocateInfo allocateInfo = {};
        allocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocateInfo.allocationSize = memorySize; // specify required memory.

        // allocateInfomatA.memoryTypeIndex = findMemoryType(
        //     memorySizematA, VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
        // allocateInfomatB.memoryTypeIndex = findMemoryType(
        //     memorySizematB, VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
        // allocateInfomatC.memoryTypeIndex = findMemoryType(
        //     memorySizematC, VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
        // allocateInfoS.memoryTypeIndex = findMemoryType(
        //     memorySizeS, VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
        allocateInfo.memoryTypeIndex = findMemoryType(
            memorySize, VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);


        // VK_CHECK_RESULT(vkAllocateMemory(device, &allocateInfomatC, NULL, &bufferMemorymatC));
        // VK_CHECK_RESULT(vkAllocateMemory(device, &allocateInfomatB, NULL, &bufferMemorymatB));
        // VK_CHECK_RESULT(vkAllocateMemory(device, &allocateInfomatA, NULL, &bufferMemorymatA)); // allocate memory on device.
        // VK_CHECK_RESULT(vkAllocateMemory(device, &allocateInfomatB, NULL, &bufferMemorymatB));
        // VK_CHECK_RESULT(vkAllocateMemory(device, &allocateInfomatC, NULL, &bufferMemorymatC));
        // VK_CHECK_RESULT(vkAllocateMemory(device, &allocateInfoS, NULL, &bufferMemoryS));
        

        VK_CHECK_RESULT(vkAllocateMemory(device, &allocateInfo, NULL, &bufferMemory));

        float* matBuffertmp;
        VK_CHECK_RESULT(vkMapMemory(device, bufferMemory, 0, (2*sizeof(float))+matrixASize+matrixBSize+matrixCSize, 0, (void **) &matBuffertmp));

        matBuffertmp[0] = M*1.0;
        matBuffertmp[1] = K*1.0;
        int offset = 2;
        for(int i = 0; i < M*K; i++) {
            matBuffertmp[offset+i] = matA[i]; 
        }
        offset = 2 + (M*K);
        for(int i = 0; i < K; i++) {
            matBuffertmp[offset+i] = matB[i]; 
        }

        vkUnmapMemory(device, bufferMemory);

        // float* matAtmp;
        // VK_CHECK_RESULT(vkMapMemory(device, bufferMemory, 0, matrixASize, 0, (void **) &matAtmp));
        
        // std::memcpy(matAtmp, matA, matrixASize);

        // vkUnmapMemory(device, bufferMemory);

        // float* matBtmp;
        // VK_CHECK_RESULT(vkMapMemory(device, bufferMemory, matrixASize, matrixBSize, 0, (void **) &matBtmp));
        
        // std::memcpy(matBtmp, matB, matrixBSize);

        // vkUnmapMemory(device, bufferMemory);

        // int* matSizestmp;
        // VK_CHECK_RESULT(vkMapMemory(device, bufferMemory, matrixASize+matrixBSize, matrixSizesSize, 0, (void **) &matSizestmp));

        // matSizestmp[0] = M;
        // matSizestmp[1] = K;

        // vkUnmapMemory(device, bufferMemory);

        VK_CHECK_RESULT(vkBindBufferMemory(device, matrixA, bufferMemory, 0));
        // VK_CHECK_RESULT(vkBindBufferMemory(device, matrixB, bufferMemory, matrixASize));
        // VK_CHECK_RESULT(vkBindBufferMemory(device, matrixSizes, bufferMemory, matrixASize+matrixBSize));
        // VK_CHECK_RESULT(vkBindBufferMemory(device, matrixC, bufferMemory, matrixASize+matrixBSize+matrixSizesSize));
    }

    void createDescriptorSetLayout() {
        VkDescriptorSetLayoutBinding descriptorSetLayoutBinding;

        descriptorSetLayoutBinding = {};
        descriptorSetLayoutBinding.binding = 0; // binding = 0
        descriptorSetLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        descriptorSetLayoutBinding.descriptorCount = 1;
        descriptorSetLayoutBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        // VkDescriptorSetLayoutBinding descriptorSetLayoutBindings[4];

        // descriptorSetLayoutBindings[0] = {};
        // descriptorSetLayoutBindings[0].binding = 0; // binding = 0
        // descriptorSetLayoutBindings[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        // descriptorSetLayoutBindings[0].descriptorCount = 1;
        // descriptorSetLayoutBindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        // descriptorSetLayoutBindings[1] = {};
        // descriptorSetLayoutBindings[1].binding = 1; // binding = 1
        // descriptorSetLayoutBindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        // descriptorSetLayoutBindings[1].descriptorCount = 1;
        // descriptorSetLayoutBindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        // descriptorSetLayoutBindings[2] = {};
        // descriptorSetLayoutBindings[2].binding = 2; // binding = 3
        // descriptorSetLayoutBindings[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        // descriptorSetLayoutBindings[2].descriptorCount = 1;
        // descriptorSetLayoutBindings[2].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        // descriptorSetLayoutBindings[3] = {};
        // descriptorSetLayoutBindings[3].binding = 3; // binding = 2
        // descriptorSetLayoutBindings[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        // descriptorSetLayoutBindings[3].descriptorCount = 1;
        // descriptorSetLayoutBindings[3].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCreateInfo = {};
        descriptorSetLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        descriptorSetLayoutCreateInfo.bindingCount = 1; // only a single binding in this descriptor set layout. 
        descriptorSetLayoutCreateInfo.pBindings = &descriptorSetLayoutBinding; 

        VK_CHECK_RESULT(vkCreateDescriptorSetLayout(device, &descriptorSetLayoutCreateInfo, NULL, &descriptorSetLayout));
    }

    void createDescriptorSet() {
        // VkDescriptorPoolSize descriptorPoolSizes[4];

        // descriptorPoolSizes[0] = {};
        // descriptorPoolSizes[0].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        // descriptorPoolSizes[0].descriptorCount = 1;

        // descriptorPoolSizes[1] = {};
        // descriptorPoolSizes[1].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        // descriptorPoolSizes[1].descriptorCount = 1;

        // descriptorPoolSizes[2] = {};
        // descriptorPoolSizes[2].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        // descriptorPoolSizes[2].descriptorCount = 1;

        // descriptorPoolSizes[3] = {};
        // descriptorPoolSizes[3].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        // descriptorPoolSizes[3].descriptorCount = 1;

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

        // VkDescriptorBufferInfo descriptorBufferInfoMatB = {};
        // descriptorBufferInfoMatB.buffer = matrixB;
        // descriptorBufferInfoMatB.offset = 0;
        // descriptorBufferInfoMatB.range = VK_WHOLE_SIZE;

        // VkDescriptorBufferInfo descriptorBufferInfoMatSizes = {};
        // descriptorBufferInfoMatSizes.buffer = matrixSizes;
        // descriptorBufferInfoMatSizes.offset = 0;
        // descriptorBufferInfoMatSizes.range = VK_WHOLE_SIZE;

        // VkDescriptorBufferInfo descriptorBufferInfoMatC = {};
        // descriptorBufferInfoMatC.buffer = matrixC;
        // descriptorBufferInfoMatC.offset = 0;
        // descriptorBufferInfoMatC.range = VK_WHOLE_SIZE;

        VkWriteDescriptorSet writeDescriptorSet;

        writeDescriptorSet = {};
        writeDescriptorSet.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writeDescriptorSet.dstSet = descriptorSet; // write to this descriptor set.
        writeDescriptorSet.dstBinding = 0; // write to the first, and only binding.
        writeDescriptorSet.descriptorCount = 1; // update a single descriptor.
        writeDescriptorSet.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; // storage buffer.
        writeDescriptorSet.pBufferInfo = &descriptorBufferInfoMatA;

        // VkWriteDescriptorSet writeDescriptorSets[4];

        // writeDescriptorSets[0] = {};
        // writeDescriptorSets[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        // writeDescriptorSets[0].dstSet = descriptorSet; // write to this descriptor set.
        // writeDescriptorSets[0].dstBinding = 0; // write to the first, and only binding.
        // writeDescriptorSets[0].descriptorCount = 1; // update a single descriptor.
        // writeDescriptorSets[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; // storage buffer.
        // writeDescriptorSets[0].pBufferInfo = &descriptorBufferInfoMatA;

        // writeDescriptorSets[1] = {};
        // writeDescriptorSets[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        // writeDescriptorSets[1].dstSet = descriptorSet; // write to this descriptor set.
        // writeDescriptorSets[1].dstBinding = 1; // write to the first, and only binding.
        // writeDescriptorSets[1].descriptorCount = 1; // update a single descriptor.
        // writeDescriptorSets[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; // storage buffer.
        // writeDescriptorSets[1].pBufferInfo = &descriptorBufferInfoMatB;

        // writeDescriptorSets[2] = {};
        // writeDescriptorSets[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        // writeDescriptorSets[2].dstSet = descriptorSet; // write to this descriptor set.
        // writeDescriptorSets[2].dstBinding = 2; // write to the first, and only binding.
        // writeDescriptorSets[2].descriptorCount = 1; // update a single descriptor.
        // writeDescriptorSets[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; // storage buffer.
        // writeDescriptorSets[2].pBufferInfo = &descriptorBufferInfoMatSizes;

        // writeDescriptorSets[3] = {};
        // writeDescriptorSets[3].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        // writeDescriptorSets[3].dstSet = descriptorSet; // write to this descriptor set.
        // writeDescriptorSets[3].dstBinding = 3; // write to the first, and only binding.
        // writeDescriptorSets[3].descriptorCount = 1; // update a single descriptor.
        // writeDescriptorSets[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; // storage buffer.
        // writeDescriptorSets[3].pBufferInfo = &descriptorBufferInfoMatC;

        vkUpdateDescriptorSets(device, 1, &writeDescriptorSet, 0, NULL);
    }

    void createComputePipeline() {
        VkPhysicalDeviceProperties devprops;
        vkGetPhysicalDeviceProperties(physicalDevice, &devprops);
        __android_log_print(ANDROID_LOG_INFO, "VulkanLimit", "maxComputeSharedMemorySize: %d", devprops.limits.maxComputeSharedMemorySize);
        __android_log_print(ANDROID_LOG_INFO, "VulkanLimit", "maxComputeWorkGroupCount[3]: %d %d %d", devprops.limits.maxComputeWorkGroupCount[0], devprops.limits.maxComputeWorkGroupCount[1], devprops.limits.maxComputeWorkGroupCount[2]);
        __android_log_print(ANDROID_LOG_INFO, "VulkanLimit", "maxComputeWorkGroupInvocations: %d", devprops.limits.maxComputeWorkGroupInvocations);
        __android_log_print(ANDROID_LOG_INFO, "VulkanLimit", "maxComputeWorkGroupSize[3]: %d %d %d", devprops.limits.maxComputeWorkGroupSize[0], devprops.limits.maxComputeWorkGroupSize[1], devprops.limits.maxComputeWorkGroupSize[2]);
        __android_log_print(ANDROID_LOG_INFO, "VulkanLimit", "maxDescriptorSetStorageBuffers: %d", devprops.limits.maxDescriptorSetStorageBuffers);
        __android_log_print(ANDROID_LOG_INFO, "VulkanLimit", "maxPerStageDescriptorStorageBuffers: %d", devprops.limits.maxPerStageDescriptorStorageBuffers);
        __android_log_print(ANDROID_LOG_INFO, "VulkanLimit", "maxPerStageResources: %d", devprops.limits.maxPerStageResources);
        __android_log_print(ANDROID_LOG_INFO, "VulkanLimit", "maxStorageBufferRange: %d", devprops.limits.maxStorageBufferRange);

        // set = 0
        // std::string source =
        //   "#version 450 \n" \
        //   "#extension GL_ARB_separate_shader_objects : enable \n" \
        //   "layout(local_size_x = 8, local_size_y = 1, local_size_z = 1) in; \n" \
        //   "layout(binding = 0) buffer matrixA { \n" \
        //   "    float A[]; \n" \
        //   "}; \n" \
        //   "layout(binding = 1) buffer matrixB { \n" \
        //   "    float B[]; \n" \
        //   "}; \n" \
        //   "layout (binding = 2) buffer matrixSizes { \n" \
        //   "    int S[]; \n" \
        //   "}; \n" \
        //   "layout(binding = 3) buffer matrixC { \n" \
        //   "    float C[]; \n" \
        //   "}; \n" \
        //   "void main() { \n" \
        //   "    int row = int(gl_GlobalInvocationID.x);"
        //   "    if (row < 1008) { \n" \
        //   "        C[row] = 1.0; \n" \
        //   "    } \n" \
        //   "}";
        std::string source =
          "#version 450 \n" \
          "#extension GL_ARB_separate_shader_objects : enable \n" \
          "layout(local_size_x = 512, local_size_y = 1, local_size_z = 1) in; \n" \
          "layout(binding = 0) buffer matrixA { \n" \
          "    float aA[]; \n" \
          "}; \n" \
          "void main() { \n" \
          "    int row = int(gl_GlobalInvocationID.x); \n" \
          "    int mM = int(aA[0]);  \n" \
          "    int kK = int(aA[1]);  \n" \
          "    if (row < mM) { \n" \
          "        float sum = 0.0; \n" \
          "        for(int i = 0; i < kK; i++){ \n" \
          "            sum += (aA[2 + (row*kK) + i]*aA[2 + (mM*kK) + i]); \n" \
          "        } \n" \
          "        aA[2 + (mM*kK) + kK + row] = sum; \n" \
          "    } \n" \
          "}";

        shaderc::Compiler compiler;
        shaderc::CompileOptions options;

        // // options.AddMacroDefinition("MY_DEFINE", "1");

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
        
        VK_CHECK_RESULT(vkCreateShaderModule(device, &createInfo, NULL, &computeShaderModule));

        VkPipelineShaderStageCreateInfo shaderStageCreateInfo = {};
        shaderStageCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        shaderStageCreateInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        shaderStageCreateInfo.module = computeShaderModule;
        shaderStageCreateInfo.pName = "main";

        VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = {};
        pipelineLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipelineLayoutCreateInfo.setLayoutCount = 1;
        pipelineLayoutCreateInfo.pSetLayouts = &descriptorSetLayout; 
        
        VK_CHECK_RESULT(vkCreatePipelineLayout(device, &pipelineLayoutCreateInfo, NULL, &pipelineLayout));

        VkComputePipelineCreateInfo pipelineCreateInfo = {};
        pipelineCreateInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        pipelineCreateInfo.stage = shaderStageCreateInfo;
        pipelineCreateInfo.layout = pipelineLayout;

        VK_CHECK_RESULT(vkCreateComputePipelines(
            device, VK_NULL_HANDLE,
            1, &pipelineCreateInfo,
            NULL, &pipeline));
    }

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

        vkCmdDispatch(commandBuffer, (M-1)/512+1, 1, 1);

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

        __android_log_print(ANDROID_LOG_INFO, "Vulkantest", "runkernel: %lf", wall);
    }

    void getresult() {
        float *matCtmp;
        VK_CHECK_RESULT(vkMapMemory(device, bufferMemory, matrixASize+matrixBSize+(2*sizeof(float)), matrixCSize, 0, (void **)&matCtmp));
      
        float sumC = 0.0;
        for (int k = 0; k < M; k++) {
          sumC += matCtmp[k];
          if(k < 100) {
              __android_log_print(ANDROID_LOG_INFO, "VulkanTest", "Matmul %d: %lf", k, matCtmp[k]);
          }
        }

        std::memcpy(matC, matCtmp, matrixCSize);

        __android_log_print(ANDROID_LOG_INFO, "VulkanTest", "Matmul sumC: %lf", sumC);

        vkUnmapMemory(device, bufferMemory);

        // float *matAtmp;
        // VK_CHECK_RESULT(vkMapMemory(device, bufferMemory, 2*sizeof(float), matrixASize, 0, (void **) &matAtmp));
        
        // float sumA = 0.0;
        // for (int k = 0; k < matrixASize / sizeof(float); k++) {
        //   sumA += matAtmp[k];
        // }

        // vkUnmapMemory(device, bufferMemory);

        // __android_log_print(ANDROID_LOG_INFO, "VulkanTest", "Matmul sumA: %lf", sumA);

        // float *matBtmp;
        // VK_CHECK_RESULT(vkMapMemory(device, bufferMemory, 2*sizeof(float)+matrixASize, matrixBSize, 0, (void **) &matBtmp));
        
        // float sumB = 0.0;
        // for (int k = 0; k < matrixBSize / sizeof(float); k++) {
        //   sumB += matBtmp[k];
        // }

        // vkUnmapMemory(device, bufferMemory);

        // __android_log_print(ANDROID_LOG_INFO, "VulkanTest", "Matmul sumB: %lf", sumB);

        // int *matSizestmp;
        // VK_CHECK_RESULT(vkMapMemory(device, bufferMemory, 0, 2*sizeof(float), 0, (void **) &matSizestmp));

        // __android_log_print(ANDROID_LOG_INFO, "VulkanTest", "Matmul matsizeM: %d %d 1", (int) matSizestmp[0], (int) matSizestmp[1]);

        // vkUnmapMemory(device, bufferMemory);    
    }

    void cleanup() {
        // vkFreeMemory(device, bufferMemorymatA, NULL);
        // vkFreeMemory(device, bufferMemorymatB, NULL);
        // vkFreeMemory(device, bufferMemorymatC, NULL);
        // vkFreeMemory(device, bufferMemoryS, NULL);
        vkFreeMemory(device, bufferMemory, NULL);
        vkDestroyBuffer(device, matrixA, NULL);
        // vkDestroyBuffer(device, matrixB, NULL);
        // vkDestroyBuffer(device, matrixC, NULL);
        // vkDestroyBuffer(device, matrixSizes, NULL);
        vkDestroyShaderModule(device, computeShaderModule, NULL);
        vkDestroyDescriptorPool(device, descriptorPool, NULL);
        vkDestroyDescriptorSetLayout(device, descriptorSetLayout, NULL);
        vkDestroyPipelineLayout(device, pipelineLayout, NULL);
        vkDestroyPipeline(device, pipeline, NULL);
        vkDestroyCommandPool(device, commandPool, NULL);  
        vkDestroyDevice(device, NULL);
        vkDestroyInstance(instance, NULL);    
    }
};

class VulkanConvolution {
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
    // VkDeviceMemory bufferMemorymatA, bufferMemorymatB, bufferMemorymatC, bufferMemoryS;
    VkDeviceMemory bufferMemory;      
    uint32_t matrixASize, matrixBSize, matrixCSize, matrixSizesSize, inputSize, filterSize, biasSize, outputSize;
    VkQueue queue; 
    uint32_t queueFamilyIndex;
    float* inputData;
    float* filterData;
    float* biasData;
    float* outputData;
    int strideWidth = 0;
    int strideHeight = 0;
    int padWidth = 0;
    int padHeight = 0;
    float* dimSizes;
    float* dimStrides;
    float outputActivationMin = 0;
    float outputActivationMax = 0;
    
public:
    void run(const float* input_data, const int input_size,
          const float* filter_data, const int filter_size,
          const float* bias_data, const int bias_size,
          float* output_data, const int output_size,
          int stride_width, int stride_height, 
          int pad_width, int pad_height, 
          const int* dim_sizes, const int* dim_strides,
          float output_activation_min, float output_activation_max) {
        
        matrixASize = (uint32_t) (sizeof(float) *dim_sizes[0]*dim_sizes[1]*dim_sizes[2]*dim_sizes[3] );
        matrixBSize = (uint32_t) (sizeof(float) *((dim_sizes[4]*dim_sizes[5]*dim_sizes[6]*dim_sizes[7]) +
            (dim_sizes[8]*dim_sizes[9]*dim_sizes[10]*dim_sizes[11]) + 2));
        matrixCSize = (uint32_t) (sizeof(float) * dim_sizes[12]*dim_sizes[13]*dim_sizes[14]*dim_sizes[15]);
        matrixSizesSize = sizeof(int) * 36;
        
        inputSize = (uint32_t) (sizeof(float)*input_size);
        filterSize = (uint32_t) (sizeof(float)*filter_size);
        biasSize = (uint32_t) (sizeof(float)*bias_size);
        outputSize = (uint32_t) (sizeof(float)*output_size);

        inputData = (float*) input_data;
        filterData = (float*) filter_data;
        biasData = (float*) bias_data;
        outputData = output_data;
        strideWidth = stride_width;
        strideHeight = stride_height;
        padWidth = pad_width;
        padHeight = pad_height;
        dimSizes = (float*) dim_sizes;
        dimStrides = (float*) dim_strides;
        outputActivationMin = output_activation_min;
        outputActivationMax = output_activation_max;

        createInstance();
        findPhysicalDevice();
        createDevice();
        createBuffer();
        createDescriptorSetLayout();
        createDescriptorSet();
        createComputePipeline();
        createCommandBuffer();
        runCommandBuffer();
        getresult();
        cleanup();
    }

    void createInstance() {
        VkApplicationInfo applicationInfo = {};
        applicationInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        applicationInfo.pApplicationName = "Matrix Convolution";
        applicationInfo.applicationVersion = 0;
        applicationInfo.pEngineName = "Naive";
        applicationInfo.engineVersion = 0;
        applicationInfo.apiVersion = VK_MAKE_VERSION(1, 0, 31);
        
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

    void createBuffer() {
        VkBufferCreateInfo matrixACreateInfo = {};
        matrixACreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        matrixACreateInfo.size = matrixASize; // buffer size in bytes. 
        matrixACreateInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT; // buffer is used as a storage buffer.
        matrixACreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE; // buffer is exclusive to a single queue family at a time. 

        VK_CHECK_RESULT(vkCreateBuffer(device, &matrixACreateInfo, NULL, &matrixA)); // create buffer.

        VkBufferCreateInfo matrixBCreateInfo = {};
        matrixACreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        matrixACreateInfo.size = matrixBSize; // buffer size in bytes. 
        matrixACreateInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT; // buffer is used as a storage buffer.
        matrixACreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE; // buffer is exclusive to a single queue family at a time. 

        VK_CHECK_RESULT(vkCreateBuffer(device, &matrixBCreateInfo, NULL, &matrixB)); // create buffer.

        VkBufferCreateInfo matrixSizesCreateInfo = {};
        matrixACreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        matrixACreateInfo.size = matrixSizesSize; // buffer size in bytes. 
        matrixACreateInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT; // buffer is used as a uniform buffer.
        matrixACreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE; // buffer is exclusive to a single queue family at a time. 

        VK_CHECK_RESULT(vkCreateBuffer(device, &matrixSizesCreateInfo, NULL, &matrixSizes)); // create buffer.

        VkBufferCreateInfo matrixCCreateInfo = {};
        matrixACreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        matrixACreateInfo.size = matrixCSize; // buffer size in bytes. 
        matrixACreateInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT; // buffer is used as a storage buffer.
        matrixACreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE; // buffer is exclusive to a single queue family at a time. 

        VK_CHECK_RESULT(vkCreateBuffer(device, &matrixCCreateInfo, NULL, &matrixC)); // create buffer.
        
        VkMemoryRequirements memoryRequirementsmatrixA, memoryRequirementsmatrixB, 
                memoryRequirementsmatrixC, memoryRequirementsmatrixSizes;
        vkGetBufferMemoryRequirements(device, matrixA, &memoryRequirementsmatrixA);
        vkGetBufferMemoryRequirements(device, matrixB, &memoryRequirementsmatrixB);
        vkGetBufferMemoryRequirements(device, matrixSizes, &memoryRequirementsmatrixSizes);
        vkGetBufferMemoryRequirements(device, matrixC, &memoryRequirementsmatrixC);
        
        
        // const VkDeviceSize memorySizematA = memoryRequirementsmatrixA.size;
        // const VkDeviceSize memorySizematB = memoryRequirementsmatrixB.size;
        // const VkDeviceSize memorySizematC = memoryRequirementsmatrixC.size; 
        // const VkDeviceSize memorySizeS = memoryRequirementsmatrixSizes.size;
        const VkDeviceSize memorySize = memoryRequirementsmatrixA.size+memoryRequirementsmatrixSizes.size
                  +memoryRequirementsmatrixB.size+memoryRequirementsmatrixC.size;

        // VkMemoryAllocateInfo allocateInfomatA = {};
        // allocateInfomatA.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        // allocateInfomatA.allocationSize = memorySizematA; // specify required memory.

        // VkMemoryAllocateInfo allocateInfomatB = {};
        // allocateInfomatB.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        // allocateInfomatB.allocationSize = memorySizematB; // specify required memory.

        // VkMemoryAllocateInfo allocateInfomatC = {};
        // allocateInfomatC.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        // allocateInfomatC.allocationSize = memorySizematC; // specify required memory.

        // VkMemoryAllocateInfo allocateInfoS = {};
        // allocateInfoS.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        // allocateInfoS.allocationSize = memorySizeS; // specify required memory.

        VkMemoryAllocateInfo allocateInfo = {};
        allocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocateInfo.allocationSize = memorySize; // specify required memory.

        // allocateInfomatA.memoryTypeIndex = findMemoryType(
        //     memorySizematA, VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
        // allocateInfomatB.memoryTypeIndex = findMemoryType(
        //     memorySizematB, VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
        // allocateInfomatC.memoryTypeIndex = findMemoryType(
        //     memorySizematC, VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
        // allocateInfoS.memoryTypeIndex = findMemoryType(
        //     memorySizeS, VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
        allocateInfo.memoryTypeIndex = findMemoryType(
            memorySize, VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);

        // VK_CHECK_RESULT(vkAllocateMemory(device, &allocateInfomatA, NULL, &bufferMemorymatA)); // allocate memory on device.
        // VK_CHECK_RESULT(vkAllocateMemory(device, &allocateInfomatB, NULL, &bufferMemorymatB));
        // VK_CHECK_RESULT(vkAllocateMemory(device, &allocateInfomatC, NULL, &bufferMemorymatC));
        // VK_CHECK_RESULT(vkAllocateMemory(device, &allocateInfoS, NULL, &bufferMemoryS));
        VK_CHECK_RESULT(vkAllocateMemory(device, &allocateInfo, NULL, &bufferMemory));

        float* iDatatmp;
        VK_CHECK_RESULT(vkMapMemory(device, bufferMemory, 0, inputSize, 0, (void **) &iDatatmp));
        
        std::memcpy(iDatatmp, inputData, inputSize);

        vkUnmapMemory(device, bufferMemory);

        float* oActMinMaxtmp;
        VK_CHECK_RESULT(vkMapMemory(device, bufferMemory, inputSize, sizeof(float)*2, 0, (void **) &oActMinMaxtmp));
        
        oActMinMaxtmp[0] = outputActivationMin;
        oActMinMaxtmp[1] = outputActivationMax;

        vkUnmapMemory(device, bufferMemory);

        float* fDatatmp;
        VK_CHECK_RESULT(vkMapMemory(device, bufferMemory, inputSize+(sizeof(float)*2), filterSize, 0, (void **) &fDatatmp));
        
        std::memcpy(fDatatmp, filterData, filterSize);

        vkUnmapMemory(device, bufferMemory);

        float* bDatatmp;
        VK_CHECK_RESULT(vkMapMemory(device, bufferMemory, inputSize+(sizeof(float)*2)+filterSize, biasSize, 0, (void **) &bDatatmp));
        
        std::memcpy(bDatatmp, biasData, biasSize);

        vkUnmapMemory(device, bufferMemory);

        int* stridepaddimstmp;
        VK_CHECK_RESULT(vkMapMemory(device, bufferMemory, inputSize+(sizeof(float)*2)+filterSize+biasSize, sizeof(int)*36, 0, (void **) &stridepaddimstmp));
        
        stridepaddimstmp[0] = strideWidth;
        stridepaddimstmp[1] = strideHeight;
        stridepaddimstmp[2] = padWidth;
        stridepaddimstmp[3] = padHeight;
        for(int i = 0; i < 16; i++) {
            stridepaddimstmp[i+4] = dimSizes[i];
            stridepaddimstmp[i+20] = dimStrides[i];
        }

        vkUnmapMemory(device, bufferMemory);

        VK_CHECK_RESULT(vkBindBufferMemory(device, matrixA, bufferMemory, 0));
        VK_CHECK_RESULT(vkBindBufferMemory(device, matrixB, bufferMemory, matrixASize));
        VK_CHECK_RESULT(vkBindBufferMemory(device, matrixSizes, bufferMemory, matrixASize+matrixBSize));
        VK_CHECK_RESULT(vkBindBufferMemory(device, matrixC, bufferMemory, matrixASize+matrixBSize+matrixSizesSize));
    }

    void createDescriptorSetLayout() {
        VkDescriptorSetLayoutBinding descriptorSetLayoutBindings[4];

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

        descriptorSetLayoutBindings[2] = {};
        descriptorSetLayoutBindings[2].binding = 2; // binding = 3
        descriptorSetLayoutBindings[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        descriptorSetLayoutBindings[2].descriptorCount = 1;
        descriptorSetLayoutBindings[2].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        descriptorSetLayoutBindings[3] = {};
        descriptorSetLayoutBindings[3].binding = 3; // binding = 2
        descriptorSetLayoutBindings[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        descriptorSetLayoutBindings[3].descriptorCount = 1;
        descriptorSetLayoutBindings[3].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCreateInfo = {};
        descriptorSetLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        descriptorSetLayoutCreateInfo.bindingCount = 4; // only a single binding in this descriptor set layout. 
        descriptorSetLayoutCreateInfo.pBindings = descriptorSetLayoutBindings; 

        VK_CHECK_RESULT(vkCreateDescriptorSetLayout(device, &descriptorSetLayoutCreateInfo, NULL, &descriptorSetLayout));
    }

    void createDescriptorSet() {
        // VkDescriptorPoolSize descriptorPoolSizes[4];

        // descriptorPoolSizes[0] = {};
        // descriptorPoolSizes[0].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        // descriptorPoolSizes[0].descriptorCount = 1;

        // descriptorPoolSizes[1] = {};
        // descriptorPoolSizes[1].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        // descriptorPoolSizes[1].descriptorCount = 1;

        // descriptorPoolSizes[2] = {};
        // descriptorPoolSizes[2].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        // descriptorPoolSizes[2].descriptorCount = 1;

        // descriptorPoolSizes[3] = {};
        // descriptorPoolSizes[3].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        // descriptorPoolSizes[3].descriptorCount = 1;

        VkDescriptorPoolSize descriptorPoolSize;

        descriptorPoolSize = {};
        descriptorPoolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        descriptorPoolSize.descriptorCount = 4;

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

        VkDescriptorBufferInfo descriptorBufferInfoMatB = {};
        descriptorBufferInfoMatB.buffer = matrixB;
        descriptorBufferInfoMatB.offset = 0;
        descriptorBufferInfoMatB.range = VK_WHOLE_SIZE;

        VkDescriptorBufferInfo descriptorBufferInfoMatSizes = {};
        descriptorBufferInfoMatSizes.buffer = matrixSizes;
        descriptorBufferInfoMatSizes.offset = 0;
        descriptorBufferInfoMatSizes.range = VK_WHOLE_SIZE;

        VkDescriptorBufferInfo descriptorBufferInfoMatC = {};
        descriptorBufferInfoMatC.buffer = matrixC;
        descriptorBufferInfoMatC.offset = 0;
        descriptorBufferInfoMatC.range = VK_WHOLE_SIZE;

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
        writeDescriptorSets[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; // storage buffer.
        writeDescriptorSets[1].pBufferInfo = &descriptorBufferInfoMatB;

        writeDescriptorSets[2] = {};
        writeDescriptorSets[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writeDescriptorSets[2].dstSet = descriptorSet; // write to this descriptor set.
        writeDescriptorSets[2].dstBinding = 2; // write to the first, and only binding.
        writeDescriptorSets[2].descriptorCount = 1; // update a single descriptor.
        writeDescriptorSets[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; // storage buffer.
        writeDescriptorSets[2].pBufferInfo = &descriptorBufferInfoMatSizes;

        writeDescriptorSets[3] = {};
        writeDescriptorSets[3].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writeDescriptorSets[3].dstSet = descriptorSet; // write to this descriptor set.
        writeDescriptorSets[3].dstBinding = 3; // write to the first, and only binding.
        writeDescriptorSets[3].descriptorCount = 1; // update a single descriptor.
        writeDescriptorSets[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; // storage buffer.
        writeDescriptorSets[3].pBufferInfo = &descriptorBufferInfoMatC;

        vkUpdateDescriptorSets(device, 4, writeDescriptorSets, 0, NULL);
    }

    void createComputePipeline() {
        VkPhysicalDeviceProperties devprops;
        vkGetPhysicalDeviceProperties(physicalDevice, &devprops);
        __android_log_print(ANDROID_LOG_INFO, "VulkanLimit", "maxComputeSharedMemorySize: %d", devprops.limits.maxComputeSharedMemorySize);
        __android_log_print(ANDROID_LOG_INFO, "VulkanLimit", "maxComputeWorkGroupCount[3]: %d %d %d", devprops.limits.maxComputeWorkGroupCount[0], devprops.limits.maxComputeWorkGroupCount[1], devprops.limits.maxComputeWorkGroupCount[2]);
        __android_log_print(ANDROID_LOG_INFO, "VulkanLimit", "maxComputeWorkGroupInvocations: %d", devprops.limits.maxComputeWorkGroupInvocations);
        __android_log_print(ANDROID_LOG_INFO, "VulkanLimit", "maxComputeWorkGroupSize[3]: %d %d %d", devprops.limits.maxComputeWorkGroupSize[0], devprops.limits.maxComputeWorkGroupSize[1], devprops.limits.maxComputeWorkGroupSize[2]);
        __android_log_print(ANDROID_LOG_INFO, "VulkanLimit", "maxDescriptorSetStorageBuffers: %d", devprops.limits.maxDescriptorSetStorageBuffers);
        __android_log_print(ANDROID_LOG_INFO, "VulkanLimit", "maxPerStageDescriptorStorageBuffers: %d", devprops.limits.maxPerStageDescriptorStorageBuffers);
        __android_log_print(ANDROID_LOG_INFO, "VulkanLimit", "maxPerStageResources: %d", devprops.limits.maxPerStageResources);
        __android_log_print(ANDROID_LOG_INFO, "VulkanLimit", "maxStorageBufferRange: %d", devprops.limits.maxStorageBufferRange);

        std::string source =
            "#version 450 \n" \
            "#extension GL_ARB_separate_shader_objects : enable \n" \
            "layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in; \n" \
            "layout(binding = 0) readonly buffer inputData { \n" \
            "    float iData[]; \n" \
            "}; \n" \
            "layout(binding = 1) readonly buffer filterBiasActivationMinMax { \n" \
            "    float oActMin; \n" \
            "    float oActMax; \n" \
            "    float fData[]; \n" \
            "}; \n" \
            "layout(binding = 2) readonly buffer dimSizeStrideStridePad { \n" \
            "    int strideWidth; \n" \
            "    int strideHeight; \n" \
            "    int padWidth; \n" \
            "    int padHeight; \n" \
            "    int dimSizes[16]; \n" \
            "    int dimStrides[16]; \n" \
            "}; \n" \
            "layout(binding = 3) buffer outputData { \n" \
            "    float oData[]; \n" \
            "}; \n" \
            "void main() { \n" \
            "  int gid0 = int(gl_GlobalInvocationID.x); \n" \
            "  int gid1 = int(gl_GlobalInvocationID.y); \n" \
            "  int output_depth = dimSizes[7]; \n" \
            "  int output_width = dimSizes[13]; \n" \
            "  if((gid0 < dimSizes[3]*output_depth) && (gid1 < dimSizes[14]*output_width)) { \n" \
            "        float total = 0.0; \n" \
            "        for (int filter_y = 0; filter_y < dimSizes[6]; ++filter_y) { \n" \
            "          for (int filter_x = 0; filter_x < dimSizes[5]; ++filter_x) { \n" \
            "            for (int in_channel = 0; in_channel < dimSizes[0]; ++in_channel) { \n" \
            "              int in_x = ((int(mod(gid1,output_width)) * strideWidth) - padWidth) + filter_x; \n" \
            "              int in_y = (((gid1/output_width) * strideHeight) - padHeight) + filter_y; \n" \
            "              if ((in_x >= 0) && (in_x < dimSizes[1]) && (in_y >= 0) && \n" \
            "                  (in_y < dimSizes[2])) { \n" \
            "                total += (iData[in_channel*dimStrides[0] + in_x*dimStrides[1] +in_y*dimStrides[2] + (gid0/output_depth)*dimStrides[3]] *  \n" \
            "                        fData[in_channel*dimStrides[4] + filter_x*dimStrides[5] + filter_y*dimStrides[6] + int(mod(gid0,output_depth))*dimStrides[7]]); \n" \
            "              } \n" \
            "            } \n" \
            "          } \n" \
            "        } \n" \
            "        float bias_value = 0.0; \n" \
            "        if (dimSizes[8]*dimSizes[9]*dimSizes[10]*dimSizes[11] > 0) { \n" \
            "          bias_value = fData[(dimSizes[4]*dimSizes[5]*dimSizes[6]*dimSizes[7])+(int(mod(gid0,output_depth))*dimStrides[8])]; \n" \
            "        } \n" \
            "        float max = total+bias_value; \n" \
            "        if(max < oActMin) max = oActMin; \n" \
            "        float min = max; \n" \
            "        if(min > oActMax) min = oActMax; \n" \
            "        oData[int(mod(gid0,output_depth))*dimStrides[12] + int(mod(gid1,output_width))*dimStrides[13] + \n" \
            "                     (gid1/output_width)*dimStrides[14] + (gid0/output_depth)*dimStrides[15]] = min; \n" \
            "  } \n" \
            "}";


        shaderc::Compiler compiler;
        shaderc::CompileOptions options;

        // // options.AddMacroDefinition("MY_DEFINE", "1");

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
        
        VK_CHECK_RESULT(vkCreateShaderModule(device, &createInfo, NULL, &computeShaderModule));

        VkPipelineShaderStageCreateInfo shaderStageCreateInfo = {};
        shaderStageCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        shaderStageCreateInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        shaderStageCreateInfo.module = computeShaderModule;
        shaderStageCreateInfo.pName = "main";

        VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = {};
        pipelineLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipelineLayoutCreateInfo.setLayoutCount = 1;
        pipelineLayoutCreateInfo.pSetLayouts = &descriptorSetLayout; 
        
        VK_CHECK_RESULT(vkCreatePipelineLayout(device, &pipelineLayoutCreateInfo, NULL, &pipelineLayout));

        VkComputePipelineCreateInfo pipelineCreateInfo = {};
        pipelineCreateInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        pipelineCreateInfo.stage = shaderStageCreateInfo;
        pipelineCreateInfo.layout = pipelineLayout;

        VK_CHECK_RESULT(vkCreateComputePipelines(
            device, VK_NULL_HANDLE,
            1, &pipelineCreateInfo,
            NULL, &pipeline));
    }

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

        int batches = dimSizes[3];
        int output_depth = dimSizes[7];
        int output_height = dimSizes[14];  
        int output_width = dimSizes[13];
        vkCmdDispatch(commandBuffer, batches*output_depth, output_height*output_width, 1);

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

        __android_log_print(ANDROID_LOG_INFO, "Vulkantest", "runkernel: %lf", wall);
    }

    void getresult() {
        float *matCtmp;
        VK_CHECK_RESULT(vkMapMemory(device, bufferMemory, 
            inputSize+(sizeof(float)*2)+filterSize+biasSize+(sizeof(int)*36), matrixCSize, 0, (void **)&matCtmp));
      
        double sumC = 0.0;
        for (int k = 0; k < matrixCSize/sizeof(float); k++) {
          sumC += matCtmp[k];
          if(k < 10) {
              __android_log_print(ANDROID_LOG_INFO, "VulkanTest", "Matmul %d: %lf", k, matCtmp[k]);
          }
        }

        std::memcpy(outputData, matCtmp, matrixCSize);

        __android_log_print(ANDROID_LOG_INFO, "VulkanTest", "Matmul sumC: %lf", sumC);

        vkUnmapMemory(device, bufferMemory);  
    }

    void cleanup() {
        // vkFreeMemory(device, bufferMemorymatA, NULL);
        // vkFreeMemory(device, bufferMemorymatB, NULL);
        // vkFreeMemory(device, bufferMemorymatC, NULL);
        // vkFreeMemory(device, bufferMemoryS, NULL);
        vkFreeMemory(device, bufferMemory, NULL);
        vkDestroyBuffer(device, matrixA, NULL);
        vkDestroyBuffer(device, matrixB, NULL);
        vkDestroyBuffer(device, matrixC, NULL);
        vkDestroyBuffer(device, matrixSizes, NULL);
        vkDestroyShaderModule(device, computeShaderModule, NULL);
        vkDestroyDescriptorPool(device, descriptorPool, NULL);
        vkDestroyDescriptorSetLayout(device, descriptorSetLayout, NULL);
        vkDestroyPipelineLayout(device, pipelineLayout, NULL);
        vkDestroyPipeline(device, pipeline, NULL);
        vkDestroyCommandPool(device, commandPool, NULL);  
        vkDestroyDevice(device, NULL);
        vkDestroyInstance(instance, NULL);    
    }
};

void vulkanTest(const float* matrixA, const float* matrixB, float* matrixC, int M, int K, int N) {
    ComputeApplication app;
    app.run(matrixA, matrixB, matrixC, M, K, N);
}

void vulkanTestConv(const float* input_data, const int input_size,
          const float* filter_data, const int filter_size,
          const float* bias_data, const int bias_size,
          float* output_data, const int output_size,
          int stride_width, int stride_height, 
          int pad_width, int pad_height, 
          const int* dim_sizes, const int* dim_strides,
          float output_activation_min, float output_activation_max) {
    VulkanConvolution app;
    app.run(input_data,input_size,
          filter_data,filter_size,
          bias_data,bias_size,
          output_data,output_size,
          stride_width, stride_height, 
          pad_width, pad_height, 
          dim_sizes, dim_strides,
          output_activation_min, output_activation_max);
}

void NeonMatrixBatchVectorMultiplyAccumulateOpenCL(const float* matrix, int m_rows,
                                             int m_cols, const float* vector,
                                             int n_batch, float* result,
                                             int result_stride,
                                             cl_context context_cl, cl_command_queue queue, cl_program program) {
  // float* input;
  // float* output;
  // float* filter;
  // float* bias;

  // input = (float*)malloc(24000*sizeof(float));
  // filter = (float*)malloc(240000*sizeof(float));
  // bias = (float*)malloc(30*sizeof(float));
  // output = (float*)malloc(18000*sizeof(float));

  // for(int i = 0; i < 24000; i++) {
  //   input[i] = (i+1)*1.0/100;
  // }
  // for(int i = 0; i < 240000; i++) {
  //   filter[i] = (i+1)*1.0/100;
  // }
  // for(int i = 0; i < 3; i++) {
  //   bias[i] = (i+1)*1.0/100;
  // }

  // int* sizes;
  // int* strides;

  // sizes = (int*)malloc(16*sizeof(int));
  // strides = (int*)malloc(16*sizeof(int));

  // // Dims<4> input_dims,filter_dims,bias_dims,output_dims;
  
  // //input
  // sizes[0] = 20;
  // sizes[1] = 30;
  // sizes[2] = 40;
  // sizes[3] = 1;
  // strides[0] = 1;
  // strides[1] = 1;
  // strides[2] = 1;
  // strides[3] = 1;

  // //filter
  // sizes[4] = 20;
  // sizes[5] = 20;
  // sizes[6] = 20;
  // sizes[7] = 30;
  // strides[4] = 1;
  // strides[5] = 1;
  // strides[6] = 1;
  // strides[7] = 1;

  // //bias
  // sizes[8] = 30;
  // sizes[9] = 1;
  // sizes[10] = 1;
  // sizes[11] = 1;
  // strides[8] = 1;
  // strides[9] = 1;
  // strides[10] = 1;
  // strides[11] = 1;

  // //output
  // sizes[12] = 30;
  // sizes[13] = 20;
  // sizes[14] = 30;
  // sizes[15] = 1;
  // strides[12] = 1;
  // strides[13] = 1;
  // strides[14] = 1;
  // strides[15] = 1;

  // int input_size = 24000;
  // int filter_size = 240000;
  // int bias_size = 30;
  // int output_size = 18000;

  // vulkanTestConv(input, input_size, filter, filter_size, bias, bias_size, output, output_size, 
  //     1, 1, 0, 0, sizes, strides, 0.0, 1000.0);

  // delete input;
  // delete output;
  // delete filter;
  // delete sizes;
  // delete strides;

m_cols = 2048;
m_rows = 1008;
float* matA = (float*) malloc(m_cols*m_rows*sizeof(float));
float* matB = (float*) malloc(m_cols*sizeof(float));
float* matC = (float*) malloc(m_rows*sizeof(float));

for(int i = 0; i < m_cols*m_rows; i++) {
  matA[i] = 1.0;
}
for(int i = 0; i < m_cols; i++) {
  matB[i] = 1.0;
}

// // Start Timers
// double wall0 = get_wall_time();
// double cpu0  = get_cpu_time();

// // PortableMatrixBatchVectorMultiplyAccumulate(matrix,m_rows,m_cols,vector,n_batch,result,1);
// // OpenCLPortableMatrixBatchVectorMultiplyAccumulate(matrix,m_rows,m_cols,vector,n_batch,result,1, context_cl, queue, program);

// PortableMatrixBatchVectorMultiplyAccumulate(matA,m_rows,m_cols,matB,n_batch,matC,1);

// // Stop timers
// double wall1 = get_wall_time();
// double cpu1  = get_cpu_time();


// double wall = wall1 - wall0;
// double cpu = cpu1 - cpu0;

// note: andoird log
// __android_log_print(ANDROID_LOG_INFO, "Portablematmul", "runkernel: %lf", wall);
double sum = 0.0;
// for(int i = 0; i < m_rows; i++) {
//   sum += matC[i];  
// }
// __android_log_print(ANDROID_LOG_INFO, "Matmulruntime", "Portable result: %lf", sum);
vulkanTest(matA, matB, matC, m_rows, m_cols, n_batch);
// sum = 0.0;
for(int i = 0; i < m_rows; i++) {
  sum += matC[i];  
}
__android_log_print(ANDROID_LOG_INFO, "Matmulruntime", "Vulkan result: %lf", sum);
  // OpenCLPortableMatrixBatchVectorMultiplyAccumulate(matA,m_rows,m_cols,matB,n_batch,matC,1, context_cl, queue, program);
  // sum = 0.0;
  // for(int i = 0; i < m_rows; i++) {
  //   sum += matC[i];  
  // }
  // __android_log_print(ANDROID_LOG_INFO, "Matmulruntime", "OpenCL result: %lf", sum);
  // // Stop timers
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
