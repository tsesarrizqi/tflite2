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

#ifndef TENSORFLOW_CONTRIB_LITE_KERNELS_INTERNAL_MULTITHREAD_CONV
#define TENSORFLOW_CONTRIB_LITE_KERNELS_INTERNAL_MULTITHREAD_CONV

#include <assert.h>
#include <stdint.h>
#include <sys/types.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <tuple>
#include <type_traits>

//note: vulkan
#include "vulkan/vulkan.h"
#include "vulkan/vk_platform.h"
#include <vector>
#include <string.h>
#include <assert.h>
#include <stdexcept>
#include <cmath>

//note: android opencl
#include "../CL/cl.h"

//note: android log
#include <android/log.h> 
#include <stdio.h> 
#include <stdlib.h> 
#include <math.h>

//note: string
#include <string>
#include <iostream>

#include <string>
// #include <unistd.h>

#include <fstream>
#include <iostream>
#include <vector>

//note: shaderc
// #include "shaderc/shaderc.hpp"

// note: timer
#include <time.h>
#include <sys/time.h>

#include "halftmp/half.hpp"

#include "tensorflow/contrib/lite/builtin_op_data.h"
#include "tensorflow/contrib/lite/kernels/internal/common.h"
#include "tensorflow/contrib/lite/kernels/internal/optimized/eigen_spatial_convolutions.h"
#include "tensorflow/contrib/lite/kernels/internal/optimized/optimized_ops.h"
#include "tensorflow/contrib/lite/kernels/internal/types.h"

#define VK_CHECK_RESULT(f)                                        \
{                                                   \
    VkResult res = (f);                                         \
    if (res != VK_SUCCESS)                                        \
    {                                                 \
        __android_log_print(ANDROID_LOG_INFO, "VulkanTest", "Fatal : VkResult is %d in %s at line %d\n", res,  __FILE__, __LINE__); \
        assert(res == VK_SUCCESS);                                    \
    }                                                 \
}

namespace tflite {
namespace multithreaded_ops {


using half_float::half;
using half_float::half_cast;
// const char *kernelSource =           "\n" \
// "__kernel void conv(__global float* input_data,   \n" \
// "          __global float* filter_data,   \n" \
// "          __global float* bias_data,   \n" \
// "          __global float* output_data,  \n" \
// "          int stride_width, int stride_height,   \n" \
// "          int pad_width, int pad_height,   \n" \
// "          __global int* dim_sizes, __global int* dim_strides,  \n" \
// "          float output_activation_min, float output_activation_max) {  \n" \
// "  int gid = get_global_id(0);  \n" \
// "  const int batches = dim_sizes[3];  \n" \
// "  const int input_depth = dim_sizes[0];  \n" \
// "  const int output_depth = dim_sizes[7];  \n" \
// "  int batch = gid/output_depth;  \n" \
// "  int out_channel = gid%output_depth;  \n" \
// "  if(gid < batches*output_depth) {  \n" \
// "    const int input_height = dim_sizes[2];  \n" \
// "    const int input_width = dim_sizes[1];  \n" \
// "    const int filter_height = dim_sizes[6];  \n" \
// "    const int filter_width = dim_sizes[5];  \n" \
// "    const int output_height = dim_sizes[14];  \n" \
// "    const int output_width = dim_sizes[13];  \n" \
// "    for (int out_y = 0; out_y < output_height; ++out_y) {  \n" \
// "      for (int out_x = 0; out_x < output_width; ++out_x) {  \n" \
// "        const int in_x_origin = (out_x * stride_width) - pad_width;  \n" \
// "        const int in_y_origin = (out_y * stride_height) - pad_height;  \n" \
// "        float total = 0.f;  \n" \
// "        for (int filter_y = 0; filter_y < filter_height; ++filter_y) {  \n" \
// "          for (int filter_x = 0; filter_x < filter_width; ++filter_x) {  \n" \
// "            for (int in_channel = 0; in_channel < input_depth; ++in_channel) {  \n" \
// "              const int in_x = in_x_origin + filter_x;  \n" \
// "              const int in_y = in_y_origin + filter_y;  \n" \
// "              if ((in_x >= 0) && (in_x < input_width) && (in_y >= 0) &&  \n" \
// "                  (in_y < input_height)) {  \n" \
// "                float input_value = input_data[in_channel*dim_strides[0] + in_x*dim_strides[1] +   \n" \
// "                                                in_y*dim_strides[2] + batch*dim_strides[3]];  \n" \
// "                float filter_value =  \n" \
// "                    filter_data[in_channel*dim_strides[4] + filter_x*dim_strides[5] +  \n" \
// "                                       filter_y*dim_strides[6] + out_channel*dim_strides[7]];  \n" \
// "                total += (input_value * filter_value);  \n" \
// "              }  \n" \
// "            }  \n" \
// "          }  \n" \
// "        }  \n" \
// "        float bias_value = 0.0f;  \n" \
// "        if (bias_data) {  \n" \
// "          bias_value = bias_data[out_channel*dim_strides[8]];  \n" \
// "        }  \n" \
// "        float max = total+bias_value; \n" \
// "        if(max < output_activation_min) max = output_activation_min; \n" \
// "        float min = max; \n" \
// "        if(min > output_activation_max) min = output_activation_max; \n" \
// "        output_data[out_channel*dim_strides[12] + out_x*dim_strides[13] +   \n" \
// "                     out_y*dim_strides[14] + batch*dim_strides[15]] = min; \n" \
// "      }  \n" \
// "    }  \n" \
// "  }  \n" \
// "}  \n" \
// "\n";


class EigenThreadPoolWrapper : public Eigen::ThreadPoolInterface {
 public:
  explicit EigenThreadPoolWrapper(Eigen::ThreadPool* pool) : pool_(pool) {}
  ~EigenThreadPoolWrapper() override {}

  void Schedule(std::function<void()> fn) override {
    pool_->Schedule(std::move(fn));
  }
  int NumThreads() const override { return pool_->NumThreads(); }
  int CurrentThreadId() const override { return pool_->CurrentThreadId(); }

 private:
  Eigen::ThreadPool* pool_ = nullptr;
};

// We have a single global threadpool for all convolution operations. This means
// that inferences started from different threads may block each other, but
// since the underlying resource of CPU cores should be consumed by the
// operations anyway, it shouldn't affect overall performance.
const Eigen::ThreadPoolDevice& GetThreadPoolDevice() {
  const int thread_count = 4;
  static Eigen::ThreadPool* tp = new Eigen::ThreadPool(thread_count);
  static EigenThreadPoolWrapper* thread_pool_wrapper =
      new EigenThreadPoolWrapper(tp);
  static Eigen::ThreadPoolDevice* device =
      new Eigen::ThreadPoolDevice(thread_pool_wrapper, thread_count);
  return *device;
}

// Shorthands for the types we need when interfacing with the EigenTensor
// library.
typedef Eigen::TensorMap<
    Eigen::Tensor<float, 2, Eigen::RowMajor, Eigen::DenseIndex>, Eigen::Aligned>
    EigenMatrix;
typedef Eigen::TensorMap<
    Eigen::Tensor<const float, 2, Eigen::RowMajor, Eigen::DenseIndex>,
    Eigen::Aligned>
    ConstEigenMatrix;

typedef Eigen::TensorMap<
    Eigen::Tensor<float, 4, Eigen::RowMajor, Eigen::DenseIndex>, Eigen::Aligned>
    EigenTensor;
typedef Eigen::TensorMap<
    Eigen::Tensor<const float, 4, Eigen::RowMajor, Eigen::DenseIndex>,
    Eigen::Aligned>
    ConstEigenTensor;

// Utility functions we need for the EigenTensor API.
template <typename Device, typename T>
struct MatMulConvFunctor {
  // Computes on device "d": out = in0 * in1, where * is matrix
  // multiplication.
  void operator()(
      const Device& d, EigenMatrix out, ConstEigenMatrix in0,
      ConstEigenMatrix in1,
      const Eigen::array<Eigen::IndexPair<Eigen::DenseIndex>, 1>& dim_pair) {
    out.device(d) = in0.contract(in1, dim_pair);
  }
};

template <class T>
class EigenTensorConvFunctor {
 private:
  Eigen::PaddingType TfLitePadding2EigenPadding(TfLitePadding padding) {
    switch (padding) {
      case kTfLitePaddingValid:
        return Eigen::PADDING_VALID;
      case kTfLitePaddingSame:
        return Eigen::PADDING_SAME;
      case kTfLitePaddingUnknown:
        assert(false);  // should never get here.
        return Eigen::PADDING_VALID;
    }
    return Eigen::PADDING_SAME;  // Prevent compiler warning about missing
                                 // return
  }

 public:
  void operator()(const T* input_data, T* im2col_buffer, int input_batches,
                  int input_height, int input_width, int input_depth,
                  const T* filter_data, int filter_height, int filter_width,
                  int filter_count, int stride_rows, int stride_cols,
                  int pad_width, int pad_height, TfLitePadding padding,
                  T* output_data, int output_height, int output_width) {
    // char* type_input_data = ;
    // char* type_filter_data = typeid(filter_data).name();
    // char* type_output_data = typeid(output_data).name();

    //note: andoird log
    // __android_log_print(ANDROID_LOG_INFO, "multithread_conv_var1", "input_data: %s\nfilter_data: %s\noutput_data: %s\n",
    //   typeid(input_data).name(),typeid(filter_data).name(),typeid(output_data).name());
    // __android_log_print(ANDROID_LOG_INFO, "multithread_conv_var2", "pad1: %d\npad2: %d\n",pad_width,pad_height);

    const Eigen::ThreadPoolDevice& device = GetThreadPoolDevice();

    const bool is_1x1_kernel = (filter_height == 1 && filter_width == 1 &&
                                stride_rows == 1 && stride_cols == 1);
    if (is_1x1_kernel) {
      // For 1x1 kernel, the 2D convolution is reduced to matrix
      // multiplication.
      const int conv_width = output_height * output_width;
      Eigen::array<Eigen::IndexPair<Eigen::DenseIndex>, 1> dim_pair;
      dim_pair[0] = Eigen::IndexPair<Eigen::DenseIndex>(1, 0);
      EigenMatrix output(output_data, conv_width, filter_count);
      ConstEigenMatrix input(input_data, conv_width, input_depth);
      ConstEigenMatrix filter(filter_data, input_depth, filter_count);
      //note: andoird log
      // __android_log_print(ANDROID_LOG_INFO, "multithread_conv", "1x1 kernel");
      MatMulConvFunctor<Eigen::ThreadPoolDevice, T>()(device, output, input,
                                                      filter, dim_pair);
    } else if (filter_height == input_height && filter_width == input_width &&
               pad_width == 0 && pad_height == 0) {
      // If the input data and filter have the same height/width,
      // the 2D convolution is reduced to matrix multiplication.
      const int k =  // Length of reduction dimension.
          filter_width * filter_height * input_depth;
      Eigen::array<Eigen::IndexPair<Eigen::DenseIndex>, 1> dim_pair;
      dim_pair[0] = Eigen::IndexPair<Eigen::DenseIndex>(1, 0);
      EigenMatrix output(output_data, 1, filter_count);
      ConstEigenMatrix input(input_data, 1, k);
      ConstEigenMatrix filter(filter_data, k, filter_count);
      //note: andoird log
      // __android_log_print(ANDROID_LOG_INFO, "multithread_conv", "input=filter height width");
      MatMulConvFunctor<Eigen::ThreadPoolDevice, T>()(device, output, input,
                                                      filter, dim_pair);
    } else {
      EigenTensor output(output_data, input_batches, output_height,
                         output_width, filter_count);
      ConstEigenTensor input(input_data, input_batches, input_height,
                             input_width, input_depth);
      ConstEigenTensor filter(filter_data, filter_height, filter_width,
                              input_depth, filter_count);
      //note: andoird log
      // __android_log_print(ANDROID_LOG_INFO, "multithread_conv", "spatial conv");
      output.device(device) =
          Eigen::SpatialConvolution(input, filter, stride_cols, stride_rows,
                                    TfLitePadding2EigenPadding(padding));
    }
  }
};

inline double get_wall_time(){
    struct timeval time;
    if (gettimeofday(&time,NULL)){
        //  Handle error
        return 0;
    }
    return (double)time.tv_sec + (double)time.tv_usec * .000001;
}

inline double get_cpu_time(){
    return (double)clock() / CLOCKS_PER_SEC;
}

// static cl_mem d_all = NULL;
static cl_kernel kernel = NULL;
// static cl_mem d_input = NULL;
// static cl_mem d_filter = NULL;
// static cl_mem d_bias = NULL;
// static cl_mem d_output = NULL;
// static cl_mem d_dim_sizes = NULL;
// static cl_mem d_dim_strides = NULL;
// static VkCommandPool commandPool = NULL;
// static VkCommandBuffer commandBuffer = NULL;
// static VkBuffer matrixA = NULL;
// static VkBuffer matrixSizes = NULL;
// static VkDeviceMemory bufferMemory = NULL;

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
    uint32_t matrixASize, matrixBSize, matrixCSize, matrixSizesSize, inputSize, filterSize, biasSize, outputSize, 
      inputSizeAll, filterSizeAll, biasSizeAll, outputSizeAll;
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
    int* dimSizes;
    int* dimStrides;
    float outputActivationMin = 0;
    float outputActivationMax = 0;
    VkDebugReportCallbackEXT debugReportCallback;
    std::vector<const char *> enabledLayers;
    bool enableValidationLayers = true;
    
public:
    void run(int buffsizes[4], const float* input_data, const int input_size,
          const float* filter_data, const int filter_size,
          const float* bias_data, const int bias_size,
          float* output_data, const int output_size,
          int stride_width, int stride_height, 
          int pad_width, int pad_height, 
          const int* dim_sizes, const int* dim_strides,
          float output_activation_min, float output_activation_max,
          VkPhysicalDevice physicalDevice0, VkDevice device0, VkPipeline pipelineConv0, VkPipelineLayout pipelineLayoutConv0, 
    VkDescriptorSetLayout descriptorSetLayoutConv0, VkQueue queueV0, uint32_t queueFamilyIndex0,
    VkCommandPool commandPool0, VkCommandBuffer commandBuffer0, VkBuffer matrixA0, VkBuffer matrixSizes0, VkDeviceMemory bufferMemory0) {
        
        physicalDevice = physicalDevice0; 
        device = device0;
        pipeline = pipelineConv0;
        pipelineLayout = pipelineLayoutConv0;
        descriptorSetLayout = descriptorSetLayoutConv0;
        queue = queueV0;
        queueFamilyIndex = queueFamilyIndex0;
        commandPool = commandPool0;
        commandBuffer = commandBuffer0;
        matrixA = matrixA0;
        matrixSizes = matrixSizes0;
        bufferMemory = bufferMemory0;

        matrixASize = (uint32_t) (sizeof(float) *buffsizes[0]);
        matrixBSize = (uint32_t) (sizeof(float) *(buffsizes[1] + buffsizes[2] + 4));
        matrixCSize = (uint32_t) (sizeof(float) * buffsizes[3]);
        matrixSizesSize = sizeof(int) * 40;
        
        inputSize = (uint32_t) (sizeof(float)*input_size);
        filterSize = (uint32_t) (sizeof(float)*filter_size);
        biasSize = (uint32_t) (sizeof(float)*bias_size);
        outputSize = (uint32_t) (sizeof(float)*output_size);

        inputSizeAll = (uint32_t) (sizeof(float) *buffsizes[0]);
        filterSizeAll = (uint32_t)(sizeof(float) *buffsizes[1]);
        biasSizeAll = (uint32_t) (sizeof(float) *buffsizes[2]);
        outputSizeAll = (uint32_t)(sizeof(float) *buffsizes[3]);
        
        inputData = (float*) input_data;
        filterData = (float*) filter_data;
        biasData = (float*) bias_data;
        outputData = output_data;
        strideWidth = stride_width;
        strideHeight = stride_height;
        padWidth = pad_width;
        padHeight = pad_height;
        dimSizes = (int*) dim_sizes;
        dimStrides = (int*) dim_strides;
        outputActivationMin = output_activation_min;
        outputActivationMax = output_activation_max;

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
    //     applicationInfo.pApplicationName = "Matrix Convolution";
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

    // uint32_t findMemoryType(VkDeviceSize memorySize, VkMemoryPropertyFlags properties) {
    //     VkPhysicalDeviceMemoryProperties memoryProperties;

    //     vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memoryProperties);

    //     for (uint32_t i = 0; i < memoryProperties.memoryTypeCount; ++i) {
    //         if ((memorySize < memoryProperties.memoryHeaps[memoryProperties.memoryTypes[i].heapIndex].size) &&
    //             ((memoryProperties.memoryTypes[i].propertyFlags & properties) == properties))
    //             return i;
    //     }
    //     return -1;
    // }

    void createBuffer() {
        // VkBufferCreateInfo matrixACreateInfo = {};
        // matrixACreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        // matrixACreateInfo.size = matrixASize+matrixBSize+matrixCSize; // buffer size in bytes. 
        // matrixACreateInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT; // buffer is used as a storage buffer.
        // matrixACreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE; // buffer is exclusive to a single queue family at a time. 

        // VK_CHECK_RESULT(vkCreateBuffer(device, &matrixACreateInfo, NULL, &matrixA)); // create buffer.

        // VkBufferCreateInfo matrixSizesCreateInfo = {};
        // matrixSizesCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        // matrixSizesCreateInfo.size = matrixSizesSize; // buffer size in bytes. 
        // matrixSizesCreateInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT; // buffer is used as a storage buffer.
        // matrixSizesCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE; // buffer is exclusive to a single queue family at a time. 

        // VK_CHECK_RESULT(vkCreateBuffer(device, &matrixSizesCreateInfo, NULL, &matrixSizes)); // create buffer.
        
        // VkMemoryRequirements memoryRequirementsmatrixA, memoryRequirementsmatrixSizes;
        // vkGetBufferMemoryRequirements(device, matrixA, &memoryRequirementsmatrixA);
        // vkGetBufferMemoryRequirements(device, matrixSizes, &memoryRequirementsmatrixSizes);

        // const VkDeviceSize memorySize = memoryRequirementsmatrixA.size+memoryRequirementsmatrixSizes.size;

        // VkMemoryAllocateInfo allocateInfo = {};
        // allocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        // allocateInfo.allocationSize = memorySize; // specify required memory.

        // allocateInfo.memoryTypeIndex = findMemoryType(
        //     memorySize, VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);

        // VK_CHECK_RESULT(vkAllocateMemory(device, &allocateInfo, NULL, &bufferMemory));

        float* oActMinMaxtmp;
        VK_CHECK_RESULT(vkMapMemory(device, bufferMemory, 0, sizeof(float)*4, 0, (void **) &oActMinMaxtmp));
        
        oActMinMaxtmp[0] = outputActivationMin;
        oActMinMaxtmp[1] = outputActivationMax;

        vkUnmapMemory(device, bufferMemory);

        int numchannel = dimSizes[0];
        int addslot = (4-(numchannel%4))%4;
        numchannel = numchannel + addslot;
        inputSize = (uint32_t) (sizeof(float)*dimSizes[1]*dimSizes[2]*dimSizes[3]*numchannel);
        filterSize = (uint32_t) (sizeof(float)*dimSizes[5]*dimSizes[6]*dimSizes[7]*numchannel);

        float* iDatatmp;
        VK_CHECK_RESULT(vkMapMemory(device, bufferMemory, sizeof(float)*4, inputSizeAll, 0, (void **) &iDatatmp));
        
        for(int i = 0,i2=0; i < inputSize/sizeof(float); i+=numchannel,i2+=dimSizes[0]) {
          for(int j = 0; j < dimSizes[0]; j++) {
            iDatatmp[i+j] = inputData[i2+j];
          }
          for(int j = dimSizes[0]; j < numchannel; j++) {
            iDatatmp[i+j] = 0.0;
          }
        }
      
        vkUnmapMemory(device, bufferMemory);

        float* fDatatmp;
        VK_CHECK_RESULT(vkMapMemory(device, bufferMemory, inputSizeAll+(sizeof(float)*4), filterSizeAll, 0, (void **) &fDatatmp));
        
        for(int i = 0,i2=0; i < filterSize/sizeof(float); i+=numchannel,i2+=dimSizes[0]) {
          for(int j = 0; j < dimSizes[0]; j++) {
            fDatatmp[i+j] = filterData[i2+j];
          }
          for(int j = dimSizes[0]; j < numchannel; j++) {
            fDatatmp[i+j] = 0.0;
          }
        }

        vkUnmapMemory(device, bufferMemory);


        float* bDatatmp;
        VK_CHECK_RESULT(vkMapMemory(device, bufferMemory, inputSizeAll+(sizeof(float)*4)+filterSizeAll, biasSizeAll, 0, (void **) &bDatatmp));
        
        std::memcpy(bDatatmp, biasData, biasSize);

        vkUnmapMemory(device, bufferMemory);

        dimStrides[1] = (dimStrides[1]/dimSizes[0])*numchannel;
        dimStrides[2] = (dimStrides[2]/dimSizes[0])*numchannel;
        dimStrides[3] = (dimStrides[3]/dimSizes[0])*numchannel;
        dimStrides[5] = (dimStrides[5]/dimSizes[0])*numchannel;
        dimStrides[6] = (dimStrides[6]/dimSizes[0])*numchannel;
        dimStrides[7] = (dimStrides[7]/dimSizes[0])*numchannel;
        dimSizes[0] = numchannel;
        dimSizes[4] = numchannel;

        int* stridepaddimstmp;
        VK_CHECK_RESULT(vkMapMemory(device, bufferMemory, matrixASize+matrixBSize+matrixCSize, matrixSizesSize, 0, (void **) &stridepaddimstmp));
        
        stridepaddimstmp[0] = strideWidth;
        stridepaddimstmp[1] = strideHeight;
        stridepaddimstmp[2] = padWidth;
        stridepaddimstmp[3] = padHeight;
        stridepaddimstmp[36] = inputSizeAll/sizeof(float);
        stridepaddimstmp[37] = filterSizeAll/sizeof(float);
        stridepaddimstmp[38] = biasSizeAll/sizeof(float);
        stridepaddimstmp[39] = outputSizeAll/sizeof(float);
        for(int i = 0; i < 16; i++) {
            stridepaddimstmp[i+4] = dimSizes[i];
            stridepaddimstmp[i+20] = dimStrides[i];
        }

        vkUnmapMemory(device, bufferMemory);

        __android_log_print(ANDROID_LOG_INFO, "VulkanConvDetail", "runkernelSizeInput: %d", inputSizeAll/sizeof(float));
        __android_log_print(ANDROID_LOG_INFO, "VulkanConvDetail", "runkernelSizeFilter: %d", filterSizeAll/sizeof(float));

        // VK_CHECK_RESULT(vkBindBufferMemory(device, matrixA, bufferMemory, 0));
        // VK_CHECK_RESULT(vkBindBufferMemory(device, matrixSizes, bufferMemory, matrixASize+matrixBSize+matrixCSize));
    }

    // void createDescriptorSetLayout() {
    //     VkDescriptorSetLayoutBinding descriptorSetLayoutBindings[4];

    //     descriptorSetLayoutBindings[0] = {};
    //     descriptorSetLayoutBindings[0].binding = 0; // binding = 0
    //     descriptorSetLayoutBindings[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    //     descriptorSetLayoutBindings[0].descriptorCount = 1;
    //     descriptorSetLayoutBindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    //     descriptorSetLayoutBindings[1] = {};
    //     descriptorSetLayoutBindings[1].binding = 1; // binding = 1
    //     descriptorSetLayoutBindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    //     descriptorSetLayoutBindings[1].descriptorCount = 1;
    //     descriptorSetLayoutBindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    //     VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCreateInfo = {};
    //     descriptorSetLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    //     descriptorSetLayoutCreateInfo.bindingCount = 2; // only a single binding in this descriptor set layout. 
    //     descriptorSetLayoutCreateInfo.pBindings = descriptorSetLayoutBindings; 

    //     VK_CHECK_RESULT(vkCreateDescriptorSetLayout(device, &descriptorSetLayoutCreateInfo, NULL, &descriptorSetLayout));
    // }

    void createDescriptorSet() {
        VkDescriptorPoolSize descriptorPoolSize;

        descriptorPoolSize = {};
        descriptorPoolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        descriptorPoolSize.descriptorCount = 2;

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

        VkDescriptorBufferInfo descriptorBufferInfoMatSizes = {};
        descriptorBufferInfoMatSizes.buffer = matrixSizes;
        descriptorBufferInfoMatSizes.offset = 0;
        descriptorBufferInfoMatSizes.range = VK_WHOLE_SIZE;

        VkWriteDescriptorSet writeDescriptorSets[2];

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
        writeDescriptorSets[1].pBufferInfo = &descriptorBufferInfoMatSizes;

        vkUpdateDescriptorSets(device, 2, writeDescriptorSets, 0, NULL);
    }

    // void createComputePipeline() {
    //     // VkPhysicalDeviceProperties devprops;
    //     // vkGetPhysicalDeviceProperties(physicalDevice, &devprops);
    //     // __android_log_print(ANDROID_LOG_INFO, "VulkanLimit", "maxComputeSharedMemorySize: %d", devprops.limits.maxComputeSharedMemorySize);
    //     // __android_log_print(ANDROID_LOG_INFO, "VulkanLimit", "maxComputeWorkGroupCount[3]: %d %d %d", devprops.limits.maxComputeWorkGroupCount[0], devprops.limits.maxComputeWorkGroupCount[1], devprops.limits.maxComputeWorkGroupCount[2]);
    //     // __android_log_print(ANDROID_LOG_INFO, "VulkanLimit", "maxComputeWorkGroupInvocations: %d", devprops.limits.maxComputeWorkGroupInvocations);
    //     // __android_log_print(ANDROID_LOG_INFO, "VulkanLimit", "maxComputeWorkGroupSize[3]: %d %d %d", devprops.limits.maxComputeWorkGroupSize[0], devprops.limits.maxComputeWorkGroupSize[1], devprops.limits.maxComputeWorkGroupSize[2]);
    //     // __android_log_print(ANDROID_LOG_INFO, "VulkanLimit", "maxDescriptorSetStorageBuffers: %d", devprops.limits.maxDescriptorSetStorageBuffers);
    //     // __android_log_print(ANDROID_LOG_INFO, "VulkanLimit", "maxPerStageDescriptorStorageBuffers: %d", devprops.limits.maxPerStageDescriptorStorageBuffers);
    //     // __android_log_print(ANDROID_LOG_INFO, "VulkanLimit", "maxPerStageResources: %d", devprops.limits.maxPerStageResources);
    //     // __android_log_print(ANDROID_LOG_INFO, "VulkanLimit", "maxStorageBufferRange: %d", devprops.limits.maxStorageBufferRange);

    //     // Start Timers
    //     double wall0 = get_wall_time();
    //     double cpu0  = get_cpu_time();

    //     std::string source =
    //     "#version 450 \n" \
    //     "#extension GL_ARB_separate_shader_objects : enable \n" \
    //     "layout(local_size_x = 8, local_size_y = 8, local_size_z = 8) in; \n" \
    //     "layout(binding = 0) buffer floatBuffer { \n" \
    //     "    float convFloatB[]; \n" \
    //     "}; \n" \
    //     "layout(binding = 1) readonly buffer intBuffer { \n" \
    //     "    int convIntB[]; \n" \
    //     "}; \n" \
    //     "void main() { \n" \
    //     "  int out_channel = int(gl_GlobalInvocationID.x); \n" \
    //     "  int out_y = int(gl_GlobalInvocationID.y); \n" \
    //     "  int out_x = int(gl_GlobalInvocationID.z); \n" \
    //     "  if((out_channel < convIntB[11]) && (out_x < convIntB[17]) && (out_y < convIntB[18])) { \n" \
    //     "      for (int batch = 0; batch < convIntB[7]; ++batch) { \n" \
    //     "        float total = 0.0; \n" \
    //     "        for (int filter_y = 0; filter_y < convIntB[10]; ++filter_y) { \n" \
    //     "          for (int filter_x = 0; filter_x < convIntB[9]; ++filter_x) { \n" \
    //     "            for (int in_channel = 0; in_channel < convIntB[4]; ++in_channel) { \n" \
    //     "              int in_x = (out_x * convIntB[0] - convIntB[2]) + filter_x; \n" \
    //     "              int in_y = (out_y * convIntB[1] - convIntB[3]) + filter_y; \n" \
    //     "              if ((in_x >= 0) && (in_x < convIntB[5]) && (in_y >= 0) && \n" \
    //     "                  (in_y < convIntB[6])) { \n" \
    //     "                total += (convFloatB[2 + in_channel*convIntB[20] + in_x*convIntB[21] +in_y*convIntB[22] + batch*convIntB[23]] *  \n" \
    //     "                        convFloatB[convIntB[36] + 2 + in_channel*convIntB[24] + filter_x*convIntB[25] + filter_y*convIntB[26] + out_channel*convIntB[27]]); \n" \
    //     "              } \n" \
    //     "            } \n" \
    //     "          } \n" \
    //     "        } \n" \
    //     "        float bias_value = 0.0; \n" \
    //     "        if (convIntB[38] > 0) { \n" \
    //     "          bias_value = convFloatB[convIntB[36] + 2 + convIntB[37]+(out_channel*convIntB[28])]; \n" \
    //     "        } \n" \
    //     "        convFloatB[2 + convIntB[36] + convIntB[37] + convIntB[38] + out_channel*convIntB[32] + out_x*convIntB[33] + out_y*convIntB[34] + batch*convIntB[35]] = min(max(total + bias_value,convFloatB[0]),convFloatB[1]); \n" \
    //     "      } \n" \
    //     "  } \n" \
    //     "}";

    //     shaderc::Compiler compiler;
    //     shaderc::CompileOptions options;

    //     shaderc::SpvCompilationResult module = compiler.CompileGlslToSpv(
    //       source.c_str(), source.size(), shaderc_glsl_compute_shader, "conv.glsl", options);

    //     if (module.GetCompilationStatus() !=
    //         shaderc_compilation_status_success) {
    //     }

    //     std::vector<uint32_t> code(module.cbegin(), module.cend());

    //     // Stop timers
    //     double wall1 = get_wall_time();
    //     double cpu1  = get_cpu_time();
    //     double wall = wall1 - wall0;
    //     double cpu = cpu1 - cpu0;
    //     __android_log_print(ANDROID_LOG_INFO, "VulkanConvDetail", "compileShader: %lf", wall);

    //     VkShaderModuleCreateInfo createInfo = {};
    //     createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    //     createInfo.pCode = code.data();
    //     createInfo.codeSize = sizeof(uint32_t)*code.size();
        
    //     // __android_log_print(ANDROID_LOG_INFO, "VulkanConv", "codeSize : %d", createInfo.codeSize);

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
        vkCmdDispatch(commandBuffer, (output_depth-1)/8+1, (output_height-1)/8+1, (output_width-1)/8+1);

        VK_CHECK_RESULT(vkEndCommandBuffer(commandBuffer)); // end recording commands.
    }

    void runCommandBuffer() {
        VkSubmitInfo submitInfo = {};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.commandBufferCount = 1; // submit a single command buffer
        submitInfo.pCommandBuffers = &commandBuffer; // the command buffer to submit.

        VkFence fence;
        VkFenceCreateInfo fenceCreateInfo = {};
        fenceCreateInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        fenceCreateInfo.flags = 0;
        VK_CHECK_RESULT(vkCreateFence(device, &fenceCreateInfo, NULL, &fence));

        // // Start Timers
        double wall0 = get_wall_time();
        double cpu0  = get_cpu_time();

        VK_CHECK_RESULT(vkQueueSubmit(queue, 1, &submitInfo, fence)); //Error -3
        // VK_CHECK_RESULT(vkQueueWaitIdle(queue)); //Error -4

        VK_CHECK_RESULT(vkWaitForFences(device, 1, &fence, VK_TRUE, 10000000000000000000)); //error 2

        vkDestroyFence(device, fence, NULL);

        // // Stop timers
        double wall1 = get_wall_time();
        double cpu1  = get_cpu_time();

        double wall = wall1 - wall0;
        double cpu = cpu1 - cpu0;

        __android_log_print(ANDROID_LOG_INFO, "VulkanConv", "runkernelV: %lf", wall);
    }

    void getresult() {
        float *matCtmp;
        VK_CHECK_RESULT(vkMapMemory(device, bufferMemory, 
            inputSizeAll+(sizeof(float)*4)+filterSizeAll+biasSizeAll, outputSizeAll, 0, (void **)&matCtmp));

        std::memcpy(outputData, matCtmp, outputSize);

        vkUnmapMemory(device, bufferMemory);  
    }

    void cleanup() {
        // vkFreeMemory(device, bufferMemory, NULL);
        // vkDestroyBuffer(device, matrixA, NULL);
        // vkDestroyBuffer(device, matrixSizes, NULL);
        //vkDestroyShaderModule(device, computeShaderModule, NULL);
        vkDestroyDescriptorPool(device, descriptorPool, NULL);
        //vkDestroyDescriptorSetLayout(device, descriptorSetLayout, NULL);
        //vkDestroyPipelineLayout(device, pipelineLayout, NULL);
        //vkDestroyPipeline(device, pipeline, NULL);
        // vkDestroyCommandPool(device, commandPool, NULL);  
        //vkDestroyDevice(device, NULL);
        //vkDestroyInstance(instance, NULL);    
    }
};

void vulkanTestConv(int buffsizes[4], const float* input_data, const int input_size,
          const float* filter_data, const int filter_size,
          const float* bias_data, const int bias_size,
          float* output_data, const int output_size,
          int stride_width, int stride_height, 
          int pad_width, int pad_height, 
          const int* dim_sizes, const int* dim_strides,
          float output_activation_min, float output_activation_max,
          VkPhysicalDevice physicalDevice, VkDevice device, VkPipeline pipelineConv, VkPipelineLayout pipelineLayoutConv, 
    VkDescriptorSetLayout descriptorSetLayoutConv, VkQueue queueV, uint32_t queueFamilyIndex,
    VkCommandPool conv_commandPool, VkCommandBuffer conv_commandBuffer, VkBuffer conv_matrixA, VkBuffer conv_matrixSizes, VkDeviceMemory conv_bufferMemory) {

    VulkanConvolution app;
    app.run(buffsizes, input_data,input_size,
          filter_data,filter_size,
          bias_data,bias_size,
          output_data,output_size,
          stride_width, stride_height, 
          pad_width, pad_height, 
          dim_sizes, dim_strides,
          output_activation_min, output_activation_max,
          physicalDevice, device, pipelineConv, pipelineLayoutConv, descriptorSetLayoutConv, queueV, queueFamilyIndex,
          conv_commandPool, conv_commandBuffer, conv_matrixA, conv_matrixSizes, conv_bufferMemory);
}

// inline void OpenCLConv(const float* input_data, int input_size,
//           const float* filter_data, int filter_size,
//           const float* bias_data, const int bias_size,
//           float* output_data, const int output_size,
//           int stride_width, int stride_height, 
//           int pad_width, int pad_height, 
//           int* dim_sizes, int* dim_strides,
//           half output_activation_min, half output_activation_max,
//           cl_context context, cl_command_queue queue, cl_program program, cl_mem cl_mem_arr[6]) {
//   cl_mem d_input = cl_mem_arr[0];
//   cl_mem d_filter = cl_mem_arr[1];
//   cl_mem d_bias = cl_mem_arr[2];
//   cl_mem d_output = cl_mem_arr[3];
//   cl_mem d_dim_sizes = cl_mem_arr[4];
//   cl_mem d_dim_strides = cl_mem_arr[5];

//   //add cl_event
//   cl_event event_runkernel;
//   cl_event event_mapinput;
//   cl_event event_mapfilter;
//   cl_event event_mapbias;
//   cl_event event_mapoutput;
//   cl_event event_unmapinput;
//   cl_event event_unmapfilter;
//   cl_event event_unmapbias;
//   cl_event event_unmapoutput;
//   cl_event event_writedimsizes;
//   cl_event event_writedimstrides;

//   cl_int err;
//   // cl_kernel kernel;
  
//   int batches = dim_sizes[3];
//   int output_depth = dim_sizes[7];
//   int output_height = dim_sizes[14];  
//   int output_width = dim_sizes[13];

//   const size_t local[2] = { 8, 32 };
//   const size_t global[2] = { (size_t) (((output_depth*batches-1)/8+1)*8), (size_t) (((output_height*output_width-1)/32+1)*32) };

  

//   // // Start Timers
//   // double wall0 = get_wall_time();
//   // double cpu0  = get_cpu_time();

  

//   // // Stop timers
//   // double wall1 = get_wall_time();
//   // double cpu1  = get_cpu_time();

//   // double wall = wall1 - wall0;
//   // double cpu = cpu1 - cpu0;

//   // // note: andoird log
//   // __android_log_print(ANDROID_LOG_INFO, "Convruntime", "Walltime createkernel: %lf", wall);

//   // wall0 = get_wall_time();
//   // cpu0  = get_cpu_time();

//   // // d_input = clCreateBuffer(context, CL_MEM_READ_ONLY, input_size*sizeof(half), NULL, NULL);
//   // // d_filter = clCreateBuffer(context, CL_MEM_READ_ONLY, filter_size*sizeof(half), NULL, NULL);
//   // // d_bias = clCreateBuffer(context, CL_MEM_READ_ONLY, bias_size*sizeof(half), NULL, NULL);
//   // // d_output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, output_size*sizeof(half), NULL, NULL);
//   // // d_dim_sizes = clCreateBuffer(context, CL_MEM_READ_ONLY, 16*sizeof(int), NULL, NULL);
//   // // d_dim_strides = clCreateBuffer(context, CL_MEM_READ_ONLY, 16*sizeof(int), NULL, NULL);

//   // // Stop timers
//   // wall1 = get_wall_time();
//   // cpu1  = get_cpu_time();

//   // wall = wall1 - wall0;
//   // cpu = cpu1 - cpu0;

//   // //note: andoird log
//   // __android_log_print(ANDROID_LOG_INFO, "Convruntime", "Walltime createbuffer: %lf", wall);

//   // wall0 = get_wall_time();
//   // cpu0  = get_cpu_time();

//   // // err = clEnqueueWriteBuffer(queue, d_input, CL_TRUE, 0,
//   // //                                input_size*sizeof(half), input_data, 0, NULL, NULL);
//   // // err = clEnqueueWriteBuffer(queue, d_filter, CL_TRUE, 0,
//   // //                                filter_size*sizeof(half), filter_data, 0, NULL, NULL);
//   // // err = clEnqueueWriteBuffer(queue, d_bias, CL_TRUE, 0,
//   // //                                bias_size*sizeof(half), bias_data, 0, NULL, NULL);

//   int numchannel = dim_sizes[0];
//   int addslot = (4-(numchannel%4))%4;
//   numchannel = numchannel + addslot;
//   input_size = (input_size/dim_sizes[0])*numchannel;
//   filter_size = (filter_size/dim_sizes[0])*numchannel;

//   half *inputHalf = (half*)clEnqueueMapBuffer(
//               queue,
//               d_input,
//               CL_TRUE,
//               CL_MAP_WRITE,
//               0,
//               input_size*sizeof(half),
//               0, NULL, NULL, &err);
//   half *filterHalf = (half*)clEnqueueMapBuffer(
//               queue,
//               d_filter,
//               CL_TRUE,
//               CL_MAP_WRITE,
//               0,
//               filter_size*sizeof(half),
//               0, NULL, NULL, &err);
//   half *biasHalf = (half*)clEnqueueMapBuffer(
//               queue,
//               d_bias,
//               CL_TRUE,
//               CL_MAP_WRITE,
//               0,
//               bias_size*sizeof(half),
//               0, NULL, NULL, &err);

//   for(int i = 0,i2 = 0; i < input_size; i+=numchannel,i2+=dim_sizes[0]) {
//     for(int j = 0; j < dim_sizes[0]; j++) {
//       inputHalf[i+j] = half_cast<half>(input_data[i2+j]);
//     }
//     for(int j = dim_sizes[0]; j < numchannel; j++) {
//       inputHalf[i+j] = half_cast<half>(0.0);
//     }
//   }
//   for(int i = 0,i2 = 0; i < filter_size; i+=numchannel,i2+=dim_sizes[0]) {
//     for(int j = 0; j < dim_sizes[0]; j++) {
//       filterHalf[i+j] = half_cast<half>(filter_data[i2+j]);
//     }
//     for(int j = dim_sizes[0]; j < numchannel; j++) {
//       filterHalf[i+j] = half_cast<half>(0.0);
//     }
//   }
//   for(int i = 0; i < bias_size; i++) {
//     biasHalf[i] = half_cast<half>(bias_data[i]);
//   }

//   clEnqueueUnmapMemObject(queue,d_input,(void *) inputHalf,0, NULL, &event_unmapinput);
//   clEnqueueUnmapMemObject(queue,d_filter,(void *) filterHalf,0, NULL, &event_unmapfilter);
//   clEnqueueUnmapMemObject(queue,d_bias,(void *) biasHalf,0, NULL, &event_unmapbias);

//   dim_strides[1] = (dim_strides[1]/dim_sizes[0])*numchannel;
//   dim_strides[2] = (dim_strides[2]/dim_sizes[0])*numchannel;
//   dim_strides[3] = (dim_strides[3]/dim_sizes[0])*numchannel;
//   dim_strides[5] = (dim_strides[5]/dim_sizes[0])*numchannel;
//   dim_strides[6] = (dim_strides[6]/dim_sizes[0])*numchannel;
//   dim_strides[7] = (dim_strides[7]/dim_sizes[0])*numchannel;
//   dim_sizes[0] = numchannel;
//   dim_sizes[4] = numchannel;

//   err = clEnqueueWriteBuffer(queue, d_dim_sizes, CL_TRUE, 0,
//                                  16*sizeof(int), dim_sizes, 0, NULL, &event_writedimsizes);
//   err = clEnqueueWriteBuffer(queue, d_dim_strides, CL_TRUE, 0,
//                                  16*sizeof(int), dim_strides, 0, NULL, &event_writedimstrides);
//   // clFinish(queue);

//   // // Stop timers
//   // wall1 = get_wall_time();
//   // cpu1  = get_cpu_time();

//   // wall = wall1 - wall0;
//   // cpu = cpu1 - cpu0;

//   // // note: andoird log
//   // __android_log_print(ANDROID_LOG_INFO, "Convruntime", "Walltime writebuffer: %lf", wall);

//   // wall0 = get_wall_time();
//   // cpu0  = get_cpu_time();

//   err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_input);
//   err  = clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_filter);
//   err  = clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_bias);
//   err  = clSetKernelArg(kernel, 3, sizeof(cl_mem), &d_output);
//   err  = clSetKernelArg(kernel, 4, sizeof(int), &stride_width);
//   err  = clSetKernelArg(kernel, 5, sizeof(int), &stride_height);
//   err  = clSetKernelArg(kernel, 6, sizeof(int), &pad_width);
//   err  = clSetKernelArg(kernel, 7, sizeof(int), &pad_height);
//   err  = clSetKernelArg(kernel, 8, sizeof(cl_mem), &d_dim_sizes);
//   err  = clSetKernelArg(kernel, 9, sizeof(cl_mem), &d_dim_strides);
//   err  = clSetKernelArg(kernel, 10, sizeof(half), &output_activation_min);
//   err  = clSetKernelArg(kernel, 11, sizeof(half), &output_activation_max);

//   // // Stop timers
//   // wall1 = get_wall_time();
//   // cpu1  = get_cpu_time();

//   // wall = wall1 - wall0;
//   // cpu = cpu1 - cpu0;

//   // // note: andoird log
//   // __android_log_print(ANDROID_LOG_INFO, "Convruntime", "Walltime setkernelargs: %lf", wall);

//   cl_event eventwaitlist[5] = {event_unmapinput,event_unmapfilter,event_unmapbias,event_writedimsizes,event_writedimstrides};

//   double wall0 = get_wall_time();
//   double cpu0  = get_cpu_time();

//   err = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global, local, 5, eventwaitlist, &event_runkernel);

//   clFinish(queue);

//   // // Stop timers
//   double wall1 = get_wall_time();
//   double cpu1  = get_cpu_time();

//   double wall = wall1 - wall0;
//   double cpu = cpu1 - cpu0;

//   // note: andoird log
//   __android_log_print(ANDROID_LOG_INFO, "Convruntime", "runkernelOclConv: %lf", wall);

//   // __android_log_print(ANDROID_LOG_INFO, "Convruntime", "runkernelConverror: %d", err);

//   // wall0 = get_wall_time();
//   // cpu0  = get_cpu_time();

//   // clEnqueueReadBuffer(queue, d_output, CL_TRUE, 0, output_size*sizeof(half), output_data, 0, NULL, NULL );
  
//   half *outputHalf = (half*)clEnqueueMapBuffer(
//               queue,
//               d_output,
//               CL_TRUE,
//               CL_MAP_READ,
//               0,
//               output_size*sizeof(half),
//               1, &event_runkernel, NULL, NULL);

//   for(int i = 0; i < output_size; i++) {
//     // half halfTmp(matrix[i]);
//     output_data[i] = (float) outputHalf[i];
//   }

//   clEnqueueUnmapMemObject(queue,d_output,(void *) outputHalf,0, NULL, NULL);

//   clFinish(queue);

//   // // Stop timers
//   // wall1 = get_wall_time();
//   // cpu1  = get_cpu_time();

//   // wall = wall1 - wall0;
//   // cpu = cpu1 - cpu0;

//   // // note: andoird log
//   // __android_log_print(ANDROID_LOG_INFO, "Convruntime", "Walltime readbuffer: %lf", wall);

//   // wall0 = get_wall_time();
//   // cpu0  = get_cpu_time();

//   // // clReleaseMemObject(d_input);
//   // // clReleaseMemObject(d_filter);
//   // // clReleaseMemObject(d_bias);
//   // // clReleaseMemObject(d_output);
//   // // clReleaseMemObject(d_dim_sizes);
//   // // clReleaseMemObject(d_dim_strides);
//   // // clReleaseProgram(program);
//   // clReleaseKernel(kernel);
//   // // clReleaseCommandQueue(queue);
//   // // clReleaseContext(context);

//   // // Stop timers
//   // wall1 = get_wall_time();
//   // cpu1  = get_cpu_time();

//   // wall = wall1 - wall0;
//   // cpu = cpu1 - cpu0;

//   // // note: andoird log
//   // __android_log_print(ANDROID_LOG_INFO, "Convruntime", "Walltime cleaning: %lf", wall);
// }

inline void OpenCLConv(const float* input_data, int input_size,
          const float* filter_data, int filter_size,
          const float* bias_data, const int bias_size,
          float* output_data, const int output_size,
          int stride_width, int stride_height, 
          int pad_width, int pad_height, 
          int* dim_sizes, int* dim_strides,
          float output_activation_min, float output_activation_max,
          cl_context context, cl_command_queue queue, cl_program program, cl_mem cl_mem_arr[6]) {
  cl_mem d_input = cl_mem_arr[0];
  cl_mem d_filter = cl_mem_arr[1];
  cl_mem d_bias = cl_mem_arr[2];
  cl_mem d_output = cl_mem_arr[3];
  cl_mem d_dim_sizes = cl_mem_arr[4];
  cl_mem d_dim_strides = cl_mem_arr[5];

  //add cl_event
  cl_event event_runkernel;
  cl_event event_mapinput;
  cl_event event_mapfilter;
  cl_event event_mapbias;
  cl_event event_mapoutput;
  cl_event event_unmapinput;
  cl_event event_unmapfilter;
  cl_event event_unmapbias;
  cl_event event_unmapoutput;
  cl_event event_writedimsizes;
  cl_event event_writedimstrides;

  cl_int err;
  // cl_kernel kernel;
  
  int batches = dim_sizes[3];
  int output_depth = dim_sizes[7];
  int output_height = dim_sizes[14];  
  int output_width = dim_sizes[13];

  const size_t local[2] = { 8, 32 };
  const size_t global[2] = { (size_t) (((output_depth*batches-1)/8+1)*8), (size_t) (((output_height*output_width-1)/32+1)*32) };

  int numchannel = dim_sizes[0];
  int addslot = (4-(numchannel%4))%4;
  numchannel = numchannel + addslot;
  input_size = (input_size/dim_sizes[0])*numchannel;
  filter_size = (filter_size/dim_sizes[0])*numchannel;

  float *inputfloat = (float*)clEnqueueMapBuffer(
              queue,
              d_input,
              CL_TRUE,
              CL_MAP_WRITE,
              0,
              input_size*sizeof(float),
              0, NULL, NULL, &err);
  float *filterfloat = (float*)clEnqueueMapBuffer(
              queue,
              d_filter,
              CL_TRUE,
              CL_MAP_WRITE,
              0,
              filter_size*sizeof(float),
              0, NULL, NULL, &err);
  // float *biasfloat = (float*)clEnqueueMapBuffer(
  //             queue,
  //             d_bias,
  //             CL_TRUE,
  //             CL_MAP_WRITE,
  //             0,
  //             bias_size*sizeof(float),
  //             0, NULL, NULL, &err);

  for(int i = 0,i2 = 0; i < input_size; i+=numchannel,i2+=dim_sizes[0]) {
    for(int j = 0; j < dim_sizes[0]; j++) {
      inputfloat[i+j] = input_data[i2+j];
    }
    for(int j = dim_sizes[0]; j < numchannel; j++) {
      inputfloat[i+j] = 0.0;
    }
  }
  for(int i = 0,i2 = 0; i < filter_size; i+=numchannel,i2+=dim_sizes[0]) {
    for(int j = 0; j < dim_sizes[0]; j++) {
      filterfloat[i+j] = filter_data[i2+j];
    }
    for(int j = dim_sizes[0]; j < numchannel; j++) {
      filterfloat[i+j] = 0.0;
    }
  }
  // for(int i = 0; i < bias_size; i++) {
  //   biasfloat[i] = bias_data[i];
  // }

  clEnqueueUnmapMemObject(queue,d_input,(void *) inputfloat,0, NULL, &event_unmapinput);
  clEnqueueUnmapMemObject(queue,d_filter,(void *) filterfloat,0, NULL, &event_unmapfilter);
  // clEnqueueUnmapMemObject(queue,d_bias,(void *) biasfloat,0, NULL, &event_unmapbias);

  err = clEnqueueWriteBuffer(queue, d_bias, CL_TRUE, 0,
                                 bias_size*sizeof(float), bias_data, 0, NULL, &event_unmapbias);

  dim_strides[1] = (dim_strides[1]/dim_sizes[0])*numchannel;
  dim_strides[2] = (dim_strides[2]/dim_sizes[0])*numchannel;
  dim_strides[3] = (dim_strides[3]/dim_sizes[0])*numchannel;
  dim_strides[5] = (dim_strides[5]/dim_sizes[0])*numchannel;
  dim_strides[6] = (dim_strides[6]/dim_sizes[0])*numchannel;
  dim_strides[7] = (dim_strides[7]/dim_sizes[0])*numchannel;
  dim_sizes[0] = numchannel;
  dim_sizes[4] = numchannel;

  err = clEnqueueWriteBuffer(queue, d_dim_sizes, CL_TRUE, 0,
                                 16*sizeof(int), dim_sizes, 0, NULL, &event_writedimsizes);
  err = clEnqueueWriteBuffer(queue, d_dim_strides, CL_TRUE, 0,
                                 16*sizeof(int), dim_strides, 0, NULL, &event_writedimstrides);
  // clFinish(queue);

  // // Stop timers
  // wall1 = get_wall_time();
  // cpu1  = get_cpu_time();

  // wall = wall1 - wall0;
  // cpu = cpu1 - cpu0;

  // // note: andoird log
  // __android_log_print(ANDROID_LOG_INFO, "Convruntime", "Walltime writebuffer: %lf", wall);

  // wall0 = get_wall_time();
  // cpu0  = get_cpu_time();

  err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_input);
  err  = clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_filter);
  err  = clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_bias);
  err  = clSetKernelArg(kernel, 3, sizeof(cl_mem), &d_output);
  err  = clSetKernelArg(kernel, 4, sizeof(int), &stride_width);
  err  = clSetKernelArg(kernel, 5, sizeof(int), &stride_height);
  err  = clSetKernelArg(kernel, 6, sizeof(int), &pad_width);
  err  = clSetKernelArg(kernel, 7, sizeof(int), &pad_height);
  err  = clSetKernelArg(kernel, 8, sizeof(cl_mem), &d_dim_sizes);
  err  = clSetKernelArg(kernel, 9, sizeof(cl_mem), &d_dim_strides);
  err  = clSetKernelArg(kernel, 10, sizeof(float), &output_activation_min);
  err  = clSetKernelArg(kernel, 11, sizeof(float), &output_activation_max);

  // // Stop timers
  // wall1 = get_wall_time();
  // cpu1  = get_cpu_time();

  // wall = wall1 - wall0;
  // cpu = cpu1 - cpu0;

  // // note: andoird log
  // __android_log_print(ANDROID_LOG_INFO, "Convruntime", "Walltime setkernelargs: %lf", wall);

  cl_event eventwaitlist[5] = {event_unmapinput,event_unmapfilter,event_unmapbias,event_writedimsizes,event_writedimstrides};

  clFinish(queue);

  double wall0 = get_wall_time();
  double cpu0  = get_cpu_time();

  err = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global, local, 0, NULL, NULL);

  clFinish(queue);

  // // Stop timers
  double wall1 = get_wall_time();
  double cpu1  = get_cpu_time();

  double wall = wall1 - wall0;
  double cpu = cpu1 - cpu0;

  // note: andoird log
  __android_log_print(ANDROID_LOG_INFO, "Convruntime", "runkernelOclConv: %lf", wall);

  // __android_log_print(ANDROID_LOG_INFO, "Convruntime", "runkernelConverror: %d", err);

  // wall0 = get_wall_time();
  // cpu0  = get_cpu_time();

  clEnqueueReadBuffer(queue, d_output, CL_TRUE, 0, output_size*sizeof(float), output_data, 0, NULL, NULL );
  
  // half *outputHalf = (half*)clEnqueueMapBuffer(
  //             queue,
  //             d_output,
  //             CL_TRUE,
  //             CL_MAP_READ,
  //             0,
  //             output_size*sizeof(half),
  //             1, &event_runkernel, NULL, NULL);

  // for(int i = 0; i < output_size; i++) {
  //   output_data[i] = (float) outputHalf[i];
  // }

  // clEnqueueUnmapMemObject(queue,d_output,(void *) outputHalf,0, NULL, NULL);

  clFinish(queue);
}

inline void Conv(const float* input_data, const Dims<4>& input_dims,
                 const float* filter_data, const Dims<4>& filter_dims,
                 const float* bias_data, const Dims<4>& bias_dims,
                 int stride_width, int stride_height, int pad_width,
                 int pad_height, TfLitePadding padding,
                 float output_activation_min, float output_activation_max,
                 float* output_data, const Dims<4>& output_dims,
                 float* im2col_data, const Dims<4>& im2col_dims) {
  // Start Timers
  double wall0 = get_wall_time();
  double cpu0  = get_cpu_time();

  const int batches = MatchingArraySize(input_dims, 3, output_dims, 3);
  const int input_depth = MatchingArraySize(input_dims, 0, filter_dims, 0);
  const int output_depth = MatchingArraySize(filter_dims, 3, output_dims, 0);
  const int input_height = ArraySize(input_dims, 2);
  const int input_width = ArraySize(input_dims, 1);
  const int filter_height = ArraySize(filter_dims, 2);
  const int filter_width = ArraySize(filter_dims, 1);
  const int output_height = ArraySize(output_dims, 2);
  const int output_width = ArraySize(output_dims, 1);

  EigenTensorConvFunctor<float> conv_functor;
  conv_functor(input_data, im2col_data, batches, input_height, input_width,
               input_depth, filter_data, filter_height, filter_width,
               output_depth, stride_height, stride_width, pad_height, pad_width,
               padding, output_data, output_height, output_width);

  optimized_ops::AddBiasAndEvalActivationFunction(
      bias_data, bias_dims, output_data, output_dims, output_activation_min,
      output_activation_max);

   // Stop timers
  double wall1 = get_wall_time();
  double cpu1  = get_cpu_time();

  double wall = wall1 - wall0;
  double cpu = cpu1 - cpu0;

  // note: andoird log
  __android_log_print(ANDROID_LOG_INFO, "Convruntime", "runkernelMultithread: %lf", wall);
}

inline void Conv2(const float* input_data, const Dims<4>& input_dims,
                 const float* filter_data, const Dims<4>& filter_dims,
                 const float* bias_data, const Dims<4>& bias_dims,
                 int stride_width, int stride_height, int pad_width,
                 int pad_height, float output_activation_min,
                 float output_activation_max, float* output_data,
                 const Dims<4>& output_dims, float* im2col_data,
                 const Dims<4>& im2col_dims) {
  (void)im2col_data;  // only used in optimized code.
  (void)im2col_dims;  // only used in optimized code.
  const int batches = MatchingArraySize(input_dims, 3, output_dims, 3);
  const int input_depth = MatchingArraySize(input_dims, 0, filter_dims, 0);
  const int output_depth = MatchingArraySize(filter_dims, 3, output_dims, 0);
  if (bias_data) {
    TFLITE_DCHECK_EQ(ArraySize(filter_dims, 3), ArraySize(bias_dims, 0));
  }
  const int input_height = ArraySize(input_dims, 2);
  const int input_width = ArraySize(input_dims, 1);
  const int filter_height = ArraySize(filter_dims, 2);
  const int filter_width = ArraySize(filter_dims, 1);
  const int output_height = ArraySize(output_dims, 2);
  const int output_width = ArraySize(output_dims, 1);
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
                  float input_value = input_data[Offset(input_dims, in_channel,
                                                        in_x, in_y, batch)];
                  float filter_value =
                      filter_data[Offset(filter_dims, in_channel, filter_x,
                                         filter_y, out_channel)];
                  total += (input_value * filter_value);
                }
              }
            }
          }
          float bias_value = 0.0f;
          if (bias_data) {
            bias_value = bias_data[Offset(bias_dims, out_channel, 0, 0, 0)];
          }
          output_data[Offset(output_dims, out_channel, out_x, out_y, batch)] =
              ActivationFunctionWithMinMax(total + bias_value,
                                           output_activation_min,
                                           output_activation_max);
        }
      }
    }
  }
}

void ConvApik(const float* input_data, 
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
        for (int out_channel = output_depth/2; out_channel < output_depth; ++out_channel) {
          const int in_x_origin = (out_x * stride_width) - pad_width;
          const int in_y_origin = (out_y * stride_height) - pad_height;
          float total = 0.0;
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
            = std::min(std::max(total + bias_value, output_activation_min), output_activation_max);
        }
      }
    }
  }
}

inline uint32_t findMemoryType2(VkPhysicalDevice physicalDevice, VkDeviceSize memorySize, VkMemoryPropertyFlags properties) {
        VkPhysicalDeviceMemoryProperties memoryProperties;

  vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memoryProperties);

  for (uint32_t i = 0; i < memoryProperties.memoryTypeCount; ++i) {
      if ((memorySize < memoryProperties.memoryHeaps[memoryProperties.memoryTypes[i].heapIndex].size) &&
          ((memoryProperties.memoryTypes[i].propertyFlags & properties) == properties))
          return i;
  }
  return -1;
}

inline void ConvOpenCL(const float* input_data, const Dims<4>& input_dims,
                 const float* filter_data, const Dims<4>& filter_dims,
                 const float* bias_data, const Dims<4>& bias_dims,
                 int stride_width, int stride_height, int pad_width,
                 int pad_height, TfLitePadding padding,
                 float output_activation_min, float output_activation_max,
                 float* output_data, const Dims<4>& output_dims,
                 float* im2col_data, const Dims<4>& im2col_dims,
                 cl_context context_cl, cl_command_queue queue, cl_program program, cl_mem cl_mem_arr[6], int buffsizes[4],
                 VkPhysicalDevice physicalDevice, VkDevice device, VkPipeline pipelineConv, VkPipeline pipelineMatmul, VkPipelineLayout pipelineLayoutConv, VkPipelineLayout pipelineLayoutMatmul, 
    VkDescriptorSetLayout descriptorSetLayoutConv, VkDescriptorSetLayout descriptorSetLayoutMatmul, VkQueue queueV, uint32_t queueFamilyIndex,
    VkCommandPool conv_commandPool, VkCommandBuffer conv_commandBuffer, VkBuffer conv_matrixA, VkBuffer conv_matrixSizes, VkDeviceMemory conv_bufferMemory) {
  
  // if(kernel == NULL) {
  //   // __android_log_print(ANDROID_LOG_INFO, "Convruntime", "runkernelmasuksekali");    
      
  //   kernel = clCreateKernel(program, "conv", NULL);

  //   // d_input = clCreateBuffer(context_cl, CL_MEM_READ_ONLY, buffsizes[0]*sizeof(half), NULL, NULL);
  //   // d_filter = clCreateBuffer(context_cl, CL_MEM_READ_ONLY, buffsizes[1]*sizeof(half), NULL, NULL);
  //   // d_bias = clCreateBuffer(context_cl, CL_MEM_READ_ONLY, buffsizes[2]*sizeof(half), NULL, NULL);
  //   // d_output = clCreateBuffer(context_cl, CL_MEM_WRITE_ONLY, buffsizes[3]*sizeof(half), NULL, NULL);
  //   // d_dim_sizes = clCreateBuffer(context_cl, CL_MEM_READ_ONLY, 16*sizeof(int), NULL, NULL);
  //   // d_dim_strides = clCreateBuffer(context_cl, CL_MEM_READ_ONLY, 16*sizeof(int), NULL, NULL);

  // }




  int inheightsize = input_dims.sizes[2];
  int inwidthsize = input_dims.sizes[1];
  int indepthsize = input_dims.sizes[0];
  int inbatchsize = input_dims.sizes[3];

  int strides0 = input_dims.strides[0];
  int strides1 = input_dims.strides[1];
  int strides2 = input_dims.strides[2];
  int strides3 = input_dims.strides[3];

  // __android_log_print(ANDROID_LOG_INFO, "Convruntime", "Input height: %d", inheightsize);
  // __android_log_print(ANDROID_LOG_INFO, "Convruntime", "Input width: %d", inwidthsize);
  // __android_log_print(ANDROID_LOG_INFO, "Convruntime", "Input depth: %d", indepthsize);
  // __android_log_print(ANDROID_LOG_INFO, "Convruntime", "Input batch: %d", inbatchsize);

  // __android_log_print(ANDROID_LOG_INFO, "Convruntime", "strides0: %d", strides0);
  // __android_log_print(ANDROID_LOG_INFO, "Convruntime", "strides1: %d", strides1);
  // __android_log_print(ANDROID_LOG_INFO, "Convruntime", "strides2: %d", strides2);
  // __android_log_print(ANDROID_LOG_INFO, "Convruntime", "strides3: %d", strides3);

  // // Start Timers
  // double wall0 = get_wall_time();
  // double cpu0  = get_cpu_time();

  const int batches = MatchingArraySize(input_dims, 3, output_dims, 3);
  const int input_depth = MatchingArraySize(input_dims, 0, filter_dims, 0);
  const int output_depth = MatchingArraySize(filter_dims, 3, output_dims, 0);
  const int input_height = ArraySize(input_dims, 2);
  const int input_width = ArraySize(input_dims, 1);
  const int filter_height = ArraySize(filter_dims, 2);
  const int filter_width = ArraySize(filter_dims, 1);
  const int output_height = ArraySize(output_dims, 2);
  const int output_width = ArraySize(output_dims, 1);

  int* sizes;
  int* strides;

  sizes = (int*)malloc(16*sizeof(int));
  strides = (int*)malloc(16*sizeof(int));

  //input
  sizes[0] = input_dims.sizes[0];
  sizes[1] = input_dims.sizes[1];
  sizes[2] = input_dims.sizes[2];
  sizes[3] = input_dims.sizes[3];
  strides[0] = input_dims.strides[0];
  strides[1] = input_dims.strides[1];
  strides[2] = input_dims.strides[2];
  strides[3] = input_dims.strides[3];

  //filter
  sizes[4] = filter_dims.sizes[0];
  sizes[5] = filter_dims.sizes[1];
  sizes[6] = filter_dims.sizes[2];
  sizes[7] = filter_dims.sizes[3];
  strides[4] = filter_dims.strides[0];
  strides[5] = filter_dims.strides[1];
  strides[6] = filter_dims.strides[2];
  strides[7] = filter_dims.strides[3];

  //bias
  sizes[8] = bias_dims.sizes[0];
  sizes[9] = bias_dims.sizes[1];
  sizes[10] = bias_dims.sizes[2];
  sizes[11] = bias_dims.sizes[3];
  strides[8] = bias_dims.strides[0];
  strides[9] = bias_dims.strides[1];
  strides[10] = bias_dims.strides[2];
  strides[11] = bias_dims.strides[3];

  //output
  sizes[12] = output_dims.sizes[0];
  sizes[13] = output_dims.sizes[1];
  sizes[14] = output_dims.sizes[2];
  sizes[15] = output_dims.sizes[3];
  strides[12] = output_dims.strides[0];
  strides[13] = output_dims.strides[1];
  strides[14] = output_dims.strides[2];
  strides[15] = output_dims.strides[3];

  int input_size = batches*input_width*input_height*input_depth;
  int filter_size = input_depth*output_depth*filter_width*filter_height;
  int bias_size = output_depth;
  int output_size = batches*output_width*output_height*output_depth;

  // __android_log_print(ANDROID_LOG_INFO, "VectorSize", "InputSizeconv: %d", input_size);
  // __android_log_print(ANDROID_LOG_INFO, "VectorSize", "OutputSizeconv: %d", output_size);
  // __android_log_print(ANDROID_LOG_INFO, "VectorSize", "FitlerSizeconv: %d", filter_size);
  // __android_log_print(ANDROID_LOG_INFO, "VectorSize", "BiasSizeconv: %d", bias_size);


  // if(commandBuffer == NULL) {
  //   uint32_t matrixASize = (uint32_t) (sizeof(float) *buffsizes[0]);
  //   uint32_t matrixBSize = (uint32_t) (sizeof(float) *(buffsizes[1] + buffsizes[2] + 2));
  //   uint32_t matrixCSize = (uint32_t) (sizeof(float) * buffsizes[3]);
  //   uint32_t matrixSizesSize = sizeof(int) * 40;

  //   VkBufferCreateInfo matrixACreateInfo = {};
  //   matrixACreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  //   matrixACreateInfo.size = matrixASize+matrixBSize+matrixCSize; // buffer size in bytes. 
  //   matrixACreateInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT; // buffer is used as a storage buffer.
  //   matrixACreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE; // buffer is exclusive to a single queue family at a time. 

  //   VK_CHECK_RESULT(vkCreateBuffer(device, &matrixACreateInfo, NULL, &matrixA)); // create buffer.

  //   VkBufferCreateInfo matrixSizesCreateInfo = {};
  //   matrixSizesCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  //   matrixSizesCreateInfo.size = matrixSizesSize; // buffer size in bytes. 
  //   matrixSizesCreateInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT; // buffer is used as a storage buffer.
  //   matrixSizesCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE; // buffer is exclusive to a single queue family at a time. 

  //   VK_CHECK_RESULT(vkCreateBuffer(device, &matrixSizesCreateInfo, NULL, &matrixSizes)); // create buffer.
    
  //   VkMemoryRequirements memoryRequirementsmatrixA, memoryRequirementsmatrixSizes;
  //   vkGetBufferMemoryRequirements(device, matrixA, &memoryRequirementsmatrixA);
  //   vkGetBufferMemoryRequirements(device, matrixSizes, &memoryRequirementsmatrixSizes);

  //   const VkDeviceSize memorySize = memoryRequirementsmatrixA.size+memoryRequirementsmatrixSizes.size;

  //   VkMemoryAllocateInfo allocateInfo = {};
  //   allocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  //   allocateInfo.allocationSize = memorySize; // specify required memory.

  //   allocateInfo.memoryTypeIndex = findMemoryType2(
  //       physicalDevice, memorySize, VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);

  //   VK_CHECK_RESULT(vkAllocateMemory(device, &allocateInfo, NULL, &bufferMemory));

  //   VK_CHECK_RESULT(vkBindBufferMemory(device, matrixA, bufferMemory, 0));
  //   VK_CHECK_RESULT(vkBindBufferMemory(device, matrixSizes, bufferMemory, matrixASize+matrixBSize+matrixCSize));

  //   VkCommandPoolCreateInfo commandPoolCreateInfo = {};
  //   commandPoolCreateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
  //   commandPoolCreateInfo.flags = 0;
  //   commandPoolCreateInfo.queueFamilyIndex = queueFamilyIndex;
  //   VK_CHECK_RESULT(vkCreateCommandPool(device, &commandPoolCreateInfo, NULL, &commandPool));

  //   VkCommandBufferAllocateInfo commandBufferAllocateInfo = {};
  //   commandBufferAllocateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  //   commandBufferAllocateInfo.commandPool = commandPool;

  //   commandBufferAllocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  //   commandBufferAllocateInfo.commandBufferCount = 1; // allocate a single command buffer. 
  //   VK_CHECK_RESULT(vkAllocateCommandBuffers(device, &commandBufferAllocateInfo, &commandBuffer)); // allocate command buffer.
  // }



  // // // Test half precision
  // half* inputHalf = (half*) malloc(input_size*sizeof(half));
  // half* filterHalf = (half*) malloc(filter_size*sizeof(half));
  // half* biasHalf = (half*) malloc(bias_size*sizeof(half));
  // for(int i = 0; i < std::max(std::max(input_size,filter_size),bias_size); i++) {
  //   // half halfTmp(matrix[i]);
  //   if(i < input_size)
  //     inputHalf[i] = half_cast<half>(input_data[i]);
  //   if(i < filter_size)
  //     filterHalf[i] = half_cast<half>(filter_data[i]);
  //   if(i < bias_size)
  //     biasHalf[i] = half_cast<half>(bias_data[i]);
  // }
  
  // // for(int i = 0; i < filter_size; i++) {
  // //   // half halfTmp(vector[i]);
  // //   filterHalf[i] = half_cast<half>(filter_data[i]);
  // // }
  
  // // for(int i = 0; i < bias_size; i++) {
  // //   // half halfTmp(vector[i]);
  // //   biasHalf[i] = half_cast<half>(bias_data[i]);
  // // }
  // half* outputHalf = (half*) malloc(output_size*sizeof(half)); 


  // OpenCLConv(inputHalf, input_size,
  //         filterHalf, filter_size,
  //         biasHalf, bias_size,
  //         outputHalf, output_size,
  //         stride_width, stride_height, 
  //         pad_width, pad_height, 
  //         sizes, strides,
  //         half_cast<half>(output_activation_min), half_cast<half>(output_activation_max),
  //         context_cl, queue, program);

    // OpenCLConv(input_data, input_size,
    //       filter_data, filter_size,
    //       bias_data, bias_size,
    //       output_data, output_size,
    //       stride_width, stride_height, 
    //       pad_width, pad_height, 
    //       sizes, strides,
    //       half_cast<half>(output_activation_min), half_cast<half>(output_activation_max),
    //       context_cl, queue, program, cl_mem_arr);

        // OpenCLConv(input_data, input_size,
        //   filter_data, filter_size,
        //   bias_data, bias_size,
        //   output_data, output_size,
        //   stride_width, stride_height, 
        //   pad_width, pad_height, 
        //   sizes, strides,
        //   output_activation_min, output_activation_max,
        //   context_cl, queue, program, cl_mem_arr);

  // for(int i = 0; i < output_size; i++) {
  //   // half halfTmp(vector[i]);
  //   output_data[i] = (float) outputHalf[i];
  // }

  // free(inputHalf);
  // free(filterHalf);
  // free(biasHalf);
  // free(outputHalf);

  // double wall0 = get_wall_time();
  // double cpu0  = get_cpu_time();

  vulkanTestConv(buffsizes, input_data, input_size,
          filter_data, filter_size,
          bias_data, bias_size,
          output_data, output_size,
          stride_width, stride_height, 
          pad_width, pad_height, 
          sizes, strides,
          output_activation_min, output_activation_max,
          physicalDevice, device, pipelineConv, pipelineLayoutConv, 
          descriptorSetLayoutConv, queueV, queueFamilyIndex,
          conv_commandPool, conv_commandBuffer, conv_matrixA, conv_matrixSizes, conv_bufferMemory);



  // // // Stop timers
  // // double wall1 = get_wall_time();
  // // double cpu1  = get_cpu_time();
  // // double wall = wall1 - wall0;
  // // double cpu = cpu1 - cpu0;
  // // __android_log_print(ANDROID_LOG_INFO, "VulkanConvDetail", "totalRuntime: %lf", wall);

  // Conv2(input_data, input_dims,
  //                filter_data, filter_dims,
  //                bias_data, bias_dims,
  //                stride_width, stride_height, pad_width,
  //                pad_height, output_activation_min,
  //                output_activation_max, output_data,
  //                output_dims, im2col_data,
  //                im2col_dims);

  // // Stop timers
  // double wall1 = get_wall_time();
  // double cpu1  = get_cpu_time();

  // double wall = wall1 - wall0;
  // double cpu = cpu1 - cpu0;

  // // note: andoird log
  // __android_log_print(ANDROID_LOG_INFO, "Convruntime", "runkernelnaive: %lf", wall);

  free(sizes);
  free(strides);

  // // note: andoird log
  // __android_log_print(ANDROID_LOG_INFO, "Convruntime", "Walltime: %lf", wall);
}

}  // namespace multithreaded_ops
}  // namespace tflite

#endif  // TENSORFLOW_CONTRIB_LITE_KERNELS_INTERNAL_MULTITHREAD_CONV
