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
        __android_log_print(ANDROID_LOG_INFO, "VulkanTest", "Fatal : VkResult is %d in %s at line %d\n", res,  __FILE__, __LINE__); \
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
    const float* matA = NULL;
    const float* matB = NULL;
    float* matC = NULL;

public:
    void run(const float* matA0, const float* matB0, float* matC0, int M0, int K0, int N0) {
        // Buffer size of the storage buffer that will contain the rendered mandelbrot set.
        matrixASize = (uint32_t) (sizeof(float) * M0 * K0);
        matrixBSize = (uint32_t) (sizeof(float) * K0 * N0);
        matrixCSize = (uint32_t) (sizeof(float) * M0 * N0);
        matrixSizesSize = sizeof(int) * 3;
        M = M0;
        K = K0;
        N = N0;
        matA = matA0;
        matB = matB0;
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

        VkBufferCreateInfo matrixCCreateInfo = {};
        matrixACreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        matrixACreateInfo.size = matrixCSize; // buffer size in bytes. 
        matrixACreateInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT; // buffer is used as a storage buffer.
        matrixACreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE; // buffer is exclusive to a single queue family at a time. 

        VK_CHECK_RESULT(vkCreateBuffer(device, &matrixCCreateInfo, NULL, &matrixC)); // create buffer.

        VkBufferCreateInfo matrixSizesCreateInfo = {};
        matrixACreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        matrixACreateInfo.size = matrixSizesSize; // buffer size in bytes. 
        matrixACreateInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT; // buffer is used as a uniform buffer.
        matrixACreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE; // buffer is exclusive to a single queue family at a time. 

        VK_CHECK_RESULT(vkCreateBuffer(device, &matrixSizesCreateInfo, NULL, &matrixSizes)); // create buffer.

        /*
        But the buffer doesn't allocate memory for itself, so we must do that manually.
        */
    
        /*
        First, we find the memory requirements for the buffer.
        */
        VkMemoryRequirements memoryRequirementsmatrixA, memoryRequirementsmatrixB, 
                memoryRequirementsmatrixC, memoryRequirementsmatrixSizes;
        vkGetBufferMemoryRequirements(device, matrixA, &memoryRequirementsmatrixA);
        vkGetBufferMemoryRequirements(device, matrixB, &memoryRequirementsmatrixB);
        vkGetBufferMemoryRequirements(device, matrixC, &memoryRequirementsmatrixC);
        vkGetBufferMemoryRequirements(device, matrixSizes, &memoryRequirementsmatrixSizes);
        
        const VkDeviceSize memorySize = 
        memoryRequirementsmatrixSizes.size+memoryRequirementsmatrixA.size+memoryRequirementsmatrixB.size+memoryRequirementsmatrixC.size; 

        /*
        Now use obtained memory requirements info to allocate the memory for the buffer.
        */
        VkMemoryAllocateInfo allocateInfo = {};
        allocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocateInfo.allocationSize = memorySize; // specify required memory.
        /*
        There are several types of memory that can be allocated, and we must choose a memory type that:

        1) Satisfies the memory requirements(memoryRequirements.memoryTypeBits). 
        2) Satifies our own usage requirements. We want to be able to read the buffer memory from the GPU to the CPU
           with vkMapMemory, so we set VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT. 
        Also, by setting VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, memory written by the device(GPU) will be easily 
        visible to the host(CPU), without having to call any extra flushing commands. So mainly for convenience, we set
        this flag.
        */
        allocateInfo.memoryTypeIndex = findMemoryType(
            memorySize, VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);

        VK_CHECK_RESULT(vkAllocateMemory(device, &allocateInfo, NULL, &bufferMemory)); // allocate memory on device.
        
         // fill the buffers
        float *matAtmp;
        VK_CHECK_RESULT(vkMapMemory(device, bufferMemory, 0, matrixASize, 0, (void **) &matAtmp));
        
        memcpy(matAtmp, matA, matrixASize);

        vkUnmapMemory(device, bufferMemory);

        float *matBtmp;
        VK_CHECK_RESULT(vkMapMemory(device, bufferMemory, matrixASize, matrixBSize, 0, (void **) &matBtmp));
        
        memcpy(matBtmp, matB, matrixBSize);

        vkUnmapMemory(device, bufferMemory);

        int *matSizestmp;
        VK_CHECK_RESULT(vkMapMemory(device, bufferMemory, matrixASize+matrixBSize+matrixCSize, matrixSizesSize, 0, (void **) &matSizestmp));

        matSizestmp[0] = M;
        matSizestmp[1] = K;
        matSizestmp[2] = N;

        vkUnmapMemory(device, bufferMemory);

        // Now associate that allocated memory with the buffer. With that, the buffer is backed by actual memory. 
        VK_CHECK_RESULT(vkBindBufferMemory(device, matrixA, bufferMemory, 0));
        VK_CHECK_RESULT(vkBindBufferMemory(device, matrixB, bufferMemory, matrixASize));
        VK_CHECK_RESULT(vkBindBufferMemory(device, matrixC, bufferMemory, matrixASize+matrixBSize));
        VK_CHECK_RESULT(vkBindBufferMemory(device, matrixSizes, bufferMemory, matrixASize+matrixBSize+matrixCSize));
    }

    void createDescriptorSetLayout() {
        /*
        Here we specify a descriptor set layout. This allows us to bind our descriptors to 
        resources in the shader. 

        */

        /*
        Here we specify a binding of type VK_DESCRIPTOR_TYPE_STORAGE_BUFFER to the binding point
        0. This binds to 

          layout(std140, binding = 0) buffer buf

        in the compute shader.
        */
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
        descriptorSetLayoutBindings[2].binding = 2; // binding = 2
        descriptorSetLayoutBindings[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        descriptorSetLayoutBindings[2].descriptorCount = 1;
        descriptorSetLayoutBindings[2].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        descriptorSetLayoutBindings[3] = {};
        descriptorSetLayoutBindings[3].binding = 3; // binding = 3
        descriptorSetLayoutBindings[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        descriptorSetLayoutBindings[3].descriptorCount = 1;
        descriptorSetLayoutBindings[3].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCreateInfo = {};
        descriptorSetLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        descriptorSetLayoutCreateInfo.bindingCount = 4; // only a single binding in this descriptor set layout. 
        descriptorSetLayoutCreateInfo.pBindings = descriptorSetLayoutBindings; 

        // Create the descriptor set layout. 
        VK_CHECK_RESULT(vkCreateDescriptorSetLayout(device, &descriptorSetLayoutCreateInfo, NULL, &descriptorSetLayout));
    }

    void createDescriptorSet() {
        /*
        So we will allocate a descriptor set here.
        But we need to first create a descriptor pool to do that. 
        */

        /*
        Our descriptor pool can only allocate a single storage buffer.
        */
        VkDescriptorPoolSize descriptorPoolSizes[4];

        descriptorPoolSizes[0] = {};
        descriptorPoolSizes[0].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        descriptorPoolSizes[0].descriptorCount = 1;

        descriptorPoolSizes[1] = {};
        descriptorPoolSizes[1].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        descriptorPoolSizes[1].descriptorCount = 1;

        descriptorPoolSizes[2] = {};
        descriptorPoolSizes[2].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        descriptorPoolSizes[2].descriptorCount = 1;

        descriptorPoolSizes[3] = {};
        descriptorPoolSizes[3].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        descriptorPoolSizes[3].descriptorCount = 1;


        VkDescriptorPoolCreateInfo descriptorPoolCreateInfo = {};
        descriptorPoolCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        descriptorPoolCreateInfo.maxSets = 1; // we only need to allocate one descriptor set from the pool.
        descriptorPoolCreateInfo.poolSizeCount = 4;
        descriptorPoolCreateInfo.pPoolSizes = descriptorPoolSizes;

        // create descriptor pool.
        VK_CHECK_RESULT(vkCreateDescriptorPool(device, &descriptorPoolCreateInfo, NULL, &descriptorPool));

        /*
        With the pool allocated, we can now allocate the descriptor set. 
        */
        VkDescriptorSetAllocateInfo descriptorSetAllocateInfo = {};
        descriptorSetAllocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO; 
        descriptorSetAllocateInfo.descriptorPool = descriptorPool; // pool to allocate from.
        descriptorSetAllocateInfo.descriptorSetCount = 1; // allocate a single descriptor set.
        descriptorSetAllocateInfo.pSetLayouts = &descriptorSetLayout;

        // allocate descriptor set.
        VK_CHECK_RESULT(vkAllocateDescriptorSets(device, &descriptorSetAllocateInfo, &descriptorSet));

        /*
        Next, we need to connect our actual storage buffer with the descrptor. 
        We use vkUpdateDescriptorSets() to update the descriptor set.
        */

        // Specify the buffer to bind to the descriptor.
        VkDescriptorBufferInfo descriptorBufferInfoMatA = {};
        descriptorBufferInfoMatA.buffer = matrixA;
        descriptorBufferInfoMatA.offset = 0;
        descriptorBufferInfoMatA.range = matrixASize;

        VkDescriptorBufferInfo descriptorBufferInfoMatB = {};
        descriptorBufferInfoMatB.buffer = matrixB;
        descriptorBufferInfoMatB.offset = 0;
        descriptorBufferInfoMatB.range = matrixBSize;

        VkDescriptorBufferInfo descriptorBufferInfoMatC = {};
        descriptorBufferInfoMatC.buffer = matrixC;
        descriptorBufferInfoMatC.offset = 0;
        descriptorBufferInfoMatC.range = matrixCSize;

        VkDescriptorBufferInfo descriptorBufferInfoMatSizes = {};
        descriptorBufferInfoMatSizes.buffer = matrixSizes;
        descriptorBufferInfoMatSizes.offset = 0;
        descriptorBufferInfoMatSizes.range = matrixSizesSize;

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
        writeDescriptorSets[2].pBufferInfo = &descriptorBufferInfoMatC;

        writeDescriptorSets[3] = {};
        writeDescriptorSets[3].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writeDescriptorSets[3].dstSet = descriptorSet; // write to this descriptor set.
        writeDescriptorSets[3].dstBinding = 3; // write to the first, and only binding.
        writeDescriptorSets[3].descriptorCount = 1; // update a single descriptor.
        writeDescriptorSets[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; // storage buffer.
        writeDescriptorSets[3].pBufferInfo = &descriptorBufferInfoMatSizes;

        // perform the update of the descriptor set.
        vkUpdateDescriptorSets(device, 4, writeDescriptorSets, 0, NULL);
    }

    void createComputePipeline() {
        /*
        We create a compute pipeline here. 
        */

        /*
        Create a shader module. A shader module basically just encapsulates some shader code.
        */
        // __android_log_print(ANDROID_LOG_INFO, "VulkanTest", "Before Readfile");
        
        std::string source =
          "#version 450 \n" \
          "#extension GL_ARB_separate_shader_objects : enable \n" \
          "layout(local_size_x = 32, local_size_y = 16) in; \n" \
          "layout(binding = 0) buffer matrixA { \n" \
          "    float A[]; \n" \
          "}; \n" \
          "layout(binding = 1) buffer matrixB { \n" \
          "    float B[]; \n" \
          "}; \n" \
          "layout(binding = 2) buffer matrixC { \n" \
          "    float C[]; \n" \
          "}; \n" \
          "layout (binding = 3) buffer matrixSizes { \n" \
          "    int S[]; \n" \
          "}; \n" \
          "void main() { \n" \
          "    int row = int(gl_GlobalInvocationID.x*16 + gl_GlobalInvocationID.y);  \n" \
          "    float currr = C[0];  \n" \
          "    float nextt = currr+1;  \n" \
          "    int M = S[0];  \n" \
          "    if (row < M) { \n" \
          "        C[0] = nextt; \n" \
          "    } \n" \
          "}";

        shaderc::Compiler compiler;
        shaderc::CompileOptions options;

        // // options.AddMacroDefinition("MY_DEFINE", "1");

        shaderc::SpvCompilationResult module = compiler.CompileGlslToSpv(
          source.c_str(), source.size(), shaderc_glsl_compute_shader, "matmul.glsl", options);

        if (module.GetCompilationStatus() !=
            shaderc_compilation_status_success) {
          // __android_log_print(ANDROID_LOG_INFO, "VulkanTest", "Compile Shader Error"); 
        }

        std::vector<uint32_t> code(module.cbegin(), module.cend());
        
        // __android_log_print(ANDROID_LOG_INFO, "VulkanTest", "After Readfile");  

        VkShaderModuleCreateInfo createInfo = {};
        createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        createInfo.pCode = code.data();
        createInfo.codeSize = sizeof(uint32_t)*code.size();
        
        // __android_log_print(ANDROID_LOG_INFO, "VulkanTest", "Before CreateShader");
        VK_CHECK_RESULT(vkCreateShaderModule(device, &createInfo, NULL, &computeShaderModule));
        // __android_log_print(ANDROID_LOG_INFO, "VulkanTest", "After CreateShader");

        /*
        Now let us actually create the compute pipeline.
        A compute pipeline is very simple compared to a graphics pipeline.
        It only consists of a single stage with a compute shader. 

        So first we specify the compute shader stage, and it's entry point(main).
        */
        // __android_log_print(ANDROID_LOG_INFO, "VulkanTest", "1");
        VkPipelineShaderStageCreateInfo shaderStageCreateInfo = {};
        shaderStageCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        shaderStageCreateInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        shaderStageCreateInfo.module = computeShaderModule;
        shaderStageCreateInfo.pName = "main";

        /*
        The pipeline layout allows the pipeline to access descriptor sets. 
        So we just specify the descriptor set layout we created earlier.
        */
        // __android_log_print(ANDROID_LOG_INFO, "VulkanTest", "2");
        VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = {};
        pipelineLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipelineLayoutCreateInfo.setLayoutCount = 1;
        pipelineLayoutCreateInfo.pSetLayouts = &descriptorSetLayout; 
        
        // __android_log_print(ANDROID_LOG_INFO, "VulkanTest", "3");
        VK_CHECK_RESULT(vkCreatePipelineLayout(device, &pipelineLayoutCreateInfo, NULL, &pipelineLayout));

        // __android_log_print(ANDROID_LOG_INFO, "VulkanTest", "4");
        VkComputePipelineCreateInfo pipelineCreateInfo = {};
        pipelineCreateInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        pipelineCreateInfo.stage = shaderStageCreateInfo;
        pipelineCreateInfo.layout = pipelineLayout;

        // __android_log_print(ANDROID_LOG_INFO, "VulkanTest", "5");
        /*
        Now, we finally create the compute pipeline. 
        */
        VK_CHECK_RESULT(vkCreateComputePipelines(
            device, VK_NULL_HANDLE,
            1, &pipelineCreateInfo,
            NULL, &pipeline));
    }

    void createCommandBuffer() {
        // __android_log_print(ANDROID_LOG_INFO, "VulkanTest", "6");
        /*
        We are getting closer to the end. In order to send commands to the device(GPU),
        we must first record commands into a command buffer.
        To allocate a command buffer, we must first create a command pool. So let us do that.
        */
        VkCommandPoolCreateInfo commandPoolCreateInfo = {};
        commandPoolCreateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        commandPoolCreateInfo.flags = 0;
        // the queue family of this command pool. All command buffers allocated from this command pool,
        // must be submitted to queues of this family ONLY. 
        commandPoolCreateInfo.queueFamilyIndex = queueFamilyIndex;
        VK_CHECK_RESULT(vkCreateCommandPool(device, &commandPoolCreateInfo, NULL, &commandPool));

        // __android_log_print(ANDROID_LOG_INFO, "VulkanTest", "7");

        /*
        Now allocate a command buffer from the command pool. 
        */
        VkCommandBufferAllocateInfo commandBufferAllocateInfo = {};
        commandBufferAllocateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        commandBufferAllocateInfo.commandPool = commandPool; // specify the command pool to allocate from. 
        // if the command buffer is primary, it can be directly submitted to queues. 
        // A secondary buffer has to be called from some primary command buffer, and cannot be directly 
        // submitted to a queue. To keep things simple, we use a primary command buffer. 
        commandBufferAllocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        commandBufferAllocateInfo.commandBufferCount = 1; // allocate a single command buffer. 
        VK_CHECK_RESULT(vkAllocateCommandBuffers(device, &commandBufferAllocateInfo, &commandBuffer)); // allocate command buffer.

        // __android_log_print(ANDROID_LOG_INFO, "VulkanTest", "8");

        /*
        Now we shall start recording commands into the newly allocated command buffer. 
        */
        VkCommandBufferBeginInfo beginInfo = {};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT; // the buffer is only submitted and used once in this application.
        VK_CHECK_RESULT(vkBeginCommandBuffer(commandBuffer, &beginInfo)); // start recording commands.

        // __android_log_print(ANDROID_LOG_INFO, "VulkanTest", "9");

        /*
        We need to bind a pipeline, AND a descriptor set before we dispatch.

        The validation layer will NOT give warnings if you forget these, so be very careful not to forget them.
        */
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1, &descriptorSet, 0, NULL);

        // __android_log_print(ANDROID_LOG_INFO, "VulkanTest", "10");

        /*
        Calling vkCmdDispatch basically starts the compute pipeline, and executes the compute shader.
        The number of workgroups is specified in the arguments.
        If you are already familiar with compute shaders from OpenGL, this should be nothing new to you.
        */
        vkCmdDispatch(commandBuffer, (M/16-1)/32+1, 1, 1);

        // __android_log_print(ANDROID_LOG_INFO, "VulkanTest", "11");

        VK_CHECK_RESULT(vkEndCommandBuffer(commandBuffer)); // end recording commands.
    }

    void runCommandBuffer() {
        /*
        Now we shall finally submit the recorded command buffer to a queue.
        */

        VkSubmitInfo submitInfo = {};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.commandBufferCount = 1; // submit a single command buffer
        submitInfo.pCommandBuffers = &commandBuffer; // the command buffer to submit.

        /*
          We create a fence.
        */
        // VkFence fence;
        // VkFenceCreateInfo fenceCreateInfo = {};
        // fenceCreateInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        // fenceCreateInfo.flags = 0;
        // VK_CHECK_RESULT(vkCreateFence(device, &fenceCreateInfo, NULL, &fence));

        // Start Timers
        double wall0 = get_wall_time();
        double cpu0  = get_cpu_time();

        /*
        We submit the command buffer on the queue, at the same time giving a fence.
        */
        VK_CHECK_RESULT(vkQueueSubmit(queue, 1, &submitInfo, 0));
        /*
        The command will not have finished executing until the fence is signalled.
        So we wait here.
        We will directly after this read our buffer from the GPU,
        and we will not be sure that the command has finished executing unless we wait for the fence.
        Hence, we use a fence here.
        */
        // VK_CHECK_RESULT(vkWaitForFences(device, 1, &fence, VK_TRUE, 100000000000));

        VK_CHECK_RESULT(vkQueueWaitIdle(queue));

        // Stop timers
        double wall1 = get_wall_time();
        double cpu1  = get_cpu_time();

        double wall = wall1 - wall0;
        double cpu = cpu1 - cpu0;

        __android_log_print(ANDROID_LOG_INFO, "Vulkantest", "runkernel: %lf", wall);

        // vkDestroyFence(device, fence, NULL);
    }

    void getresult() {
        float *matCtmp;
        VK_CHECK_RESULT(vkMapMemory(device, bufferMemory, matrixASize+matrixBSize, matrixCSize, 0, (void **)&matCtmp));
        
        memcpy(matC, matCtmp, matrixCSize);

        float sum = 0.0;
        for (int k = 0; k < 1008; k++) {
          sum += matCtmp[k];
          __android_log_print(ANDROID_LOG_INFO, "VulkanTest", "Matmul %d: %lf", k, matCtmp[k]);
        }

        __android_log_print(ANDROID_LOG_INFO, "VulkanTest", "Matmul sum: %lf", sum);

        vkUnmapMemory(device, bufferMemory);


        // //////////////////////////////
        float *matAtmp;
        VK_CHECK_RESULT(vkMapMemory(device, bufferMemory, 0, matrixASize, 0, (void **) &matAtmp));
        
        float sumA = 0.0;
        for (int k = 0; k < matrixASize / sizeof(float); k++) {
          sumA += matAtmp[k];
        }

        vkUnmapMemory(device, bufferMemory);

        __android_log_print(ANDROID_LOG_INFO, "VulkanTest", "Matmul sumA: %lf", sumA);

        float *matBtmp;
        VK_CHECK_RESULT(vkMapMemory(device, bufferMemory, matrixASize, matrixBSize, 0, (void **) &matBtmp));
        
        float sumB = 0.0;
        for (int k = 0; k < matrixBSize / sizeof(float); k++) {
          sumB += matBtmp[k];
        }

        vkUnmapMemory(device, bufferMemory);

        __android_log_print(ANDROID_LOG_INFO, "VulkanTest", "Matmul sumB: %lf", sumB);

        int *matSizestmp;
        VK_CHECK_RESULT(vkMapMemory(device, bufferMemory, matrixASize+matrixBSize+matrixCSize, matrixSizesSize, 0, (void **) &matSizestmp));

        __android_log_print(ANDROID_LOG_INFO, "VulkanTest", "Matmul matsizeM: %d %d %d", matSizestmp[0], matSizestmp[1], matSizestmp[2]);

        vkUnmapMemory(device, bufferMemory);

        
    }

    void cleanup() {
        /*
        Clean up all Vulkan Resources. 
        */
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

    // try {
    // uint32_t filelength;
    // uint32_t* code = app.readFile(filelength, "matmul.spv");
        app.run(matrixA, matrixB, matrixC, M, K, N);
    // }
    // catch (const std::runtime_error& e) {
    //     printf("%s\n", e.what());
    //     return EXIT_FAILURE;
    // }
    
    // return EXIT_SUCCESS;
}



void NeonMatrixBatchVectorMultiplyAccumulateOpenCL(const float* matrix, int m_rows,
                                             int m_cols, const float* vector,
                                             int n_batch, float* result,
                                             int result_stride,
                                             cl_context context_cl, cl_command_queue queue, cl_program program) {
  
  // vulkanTest();
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
  OpenCLPortableMatrixBatchVectorMultiplyAccumulate(matA,m_rows,m_cols,matB,n_batch,matC,1, context_cl, queue, program);
  sum = 0.0;
  for(int i = 0; i < m_rows; i++) {
    sum += matC[i];  
  }
  __android_log_print(ANDROID_LOG_INFO, "Matmulruntime", "OpenCL result: %lf", sum);
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
