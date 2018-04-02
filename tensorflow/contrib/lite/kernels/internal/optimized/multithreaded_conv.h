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
#include <shaderc/shaderc.hpp>

// note: timer
#include <time.h>
#include <sys/time.h>

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

const char *kernelSource =           "\n" \
"__kernel void conv(__global float* input_data,   \n" \
"          __global float* filter_data,   \n" \
"          __global float* bias_data,   \n" \
"          __global float* output_data,  \n" \
"          int stride_width, int stride_height,   \n" \
"          int pad_width, int pad_height,   \n" \
"          __global int* dim_sizes, __global int* dim_strides,  \n" \
"          float output_activation_min, float output_activation_max) {  \n" \
"  int gid = get_global_id(0);  \n" \
"  const int batches = dim_sizes[3];  \n" \
"  const int input_depth = dim_sizes[0];  \n" \
"  const int output_depth = dim_sizes[7];  \n" \
"  int batch = gid/output_depth;  \n" \
"  int out_channel = gid%output_depth;  \n" \
"  if(gid < batches*output_depth) {  \n" \
"    const int input_height = dim_sizes[2];  \n" \
"    const int input_width = dim_sizes[1];  \n" \
"    const int filter_height = dim_sizes[6];  \n" \
"    const int filter_width = dim_sizes[5];  \n" \
"    const int output_height = dim_sizes[14];  \n" \
"    const int output_width = dim_sizes[13];  \n" \
"    for (int out_y = 0; out_y < output_height; ++out_y) {  \n" \
"      for (int out_x = 0; out_x < output_width; ++out_x) {  \n" \
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
"      }  \n" \
"    }  \n" \
"  }  \n" \
"}  \n" \
"\n";


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
    int* dimSizes;
    int* dimStrides;
    float outputActivationMin = 0;
    float outputActivationMax = 0;
    VkDebugReportCallbackEXT debugReportCallback;
    std::vector<const char *> enabledLayers;
    bool enableValidationLayers = true;
    
public:
    void run(const float* input_data, const int input_size,
          const float* filter_data, const int filter_size,
          const float* bias_data, const int bias_size,
          float* output_data, const int output_size,
          int stride_width, int stride_height, 
          int pad_width, int pad_height, 
          const int* dim_sizes, const int* dim_strides,
          float output_activation_min, float output_activation_max) {
        
        matrixASize = (uint32_t) (sizeof(float) *dim_sizes[0]*dim_sizes[1]*dim_sizes[2]*dim_sizes[3]);
        matrixBSize = (uint32_t) (sizeof(float) *((dim_sizes[4]*dim_sizes[5]*dim_sizes[6]*dim_sizes[7]) +
            (dim_sizes[8]*dim_sizes[9]*dim_sizes[10]*dim_sizes[11]) + 2));
        matrixCSize = (uint32_t) (sizeof(float) * dim_sizes[12]*dim_sizes[13]*dim_sizes[14]*dim_sizes[15]);
        matrixSizesSize = sizeof(int) * 39;
        
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
        dimSizes = (int*) dim_sizes;
        dimStrides = (int*) dim_strides;
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

    static VKAPI_ATTR VkBool32 VKAPI_CALL debugReportCallbackFn(
        VkDebugReportFlagsEXT                       flags,
        VkDebugReportObjectTypeEXT                  objectType,
        uint64_t                                    object,
        size_t                                      location,
        int32_t                                     messageCode,
        const char*                                 pLayerPrefix,
        const char*                                 pMessage,
        void*                                       pUserData) {

        __android_log_print(ANDROID_LOG_INFO, "Vulkanerror", "Debug Report: %s: %s\n", pLayerPrefix, pMessage);

        return VK_FALSE;
     }


    void createInstance() {
        // std::vector<const char *> enabledExtensions;
        // if (enableValidationLayers) {
        //     uint32_t layerCount;
        //     vkEnumerateInstanceLayerProperties(&layerCount, NULL);

        //     std::vector<VkLayerProperties> layerProperties(layerCount);
        //     vkEnumerateInstanceLayerProperties(&layerCount, layerProperties.data());

        //     bool foundLayer = false;
        //     for (VkLayerProperties prop : layerProperties) {
                
        //         if (strcmp("VK_LAYER_LUNARG_standard_validation", prop.layerName) == 0) {
        //             foundLayer = true;
        //             break;
        //         }

        //     }
            
        //     if (!foundLayer) {
        //         throw std::runtime_error("Layer VK_LAYER_LUNARG_standard_validation not supported\n");
        //     }
        //     enabledLayers.push_back("VK_LAYER_LUNARG_standard_validation"); // Alright, we can use this layer.
            
        //     uint32_t extensionCount;
            
        //     vkEnumerateInstanceExtensionProperties(NULL, &extensionCount, NULL);
        //     std::vector<VkExtensionProperties> extensionProperties(extensionCount);
        //     vkEnumerateInstanceExtensionProperties(NULL, &extensionCount, extensionProperties.data());

        //     bool foundExtension = false;
        //     for (VkExtensionProperties prop : extensionProperties) {
        //         if (strcmp(VK_EXT_DEBUG_REPORT_EXTENSION_NAME, prop.extensionName) == 0) {
        //             foundExtension = true;
        //             break;
        //         }

        //     }

        //     if (!foundExtension) {
        //         throw std::runtime_error("Extension VK_EXT_DEBUG_REPORT_EXTENSION_NAME not supported\n");
        //     }
        //     enabledExtensions.push_back(VK_EXT_DEBUG_REPORT_EXTENSION_NAME);
        // }

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

        // // Give our desired layers and extensions to vulkan.
        // createInfo.enabledLayerCount = enabledLayers.size();
        // createInfo.ppEnabledLayerNames = enabledLayers.data();
        // createInfo.enabledExtensionCount = enabledExtensions.size();
        // createInfo.ppEnabledExtensionNames = enabledExtensions.data();

        VK_CHECK_RESULT(vkCreateInstance(
            &createInfo,
            NULL,
            &instance));

        // if (enableValidationLayers) {
        //     VkDebugReportCallbackCreateInfoEXT createInfo = {};
        //     createInfo.sType = VK_STRUCTURE_TYPE_DEBUG_REPORT_CALLBACK_CREATE_INFO_EXT;
        //     createInfo.flags = VK_DEBUG_REPORT_ERROR_BIT_EXT | VK_DEBUG_REPORT_WARNING_BIT_EXT | VK_DEBUG_REPORT_PERFORMANCE_WARNING_BIT_EXT;
        //     createInfo.pfnCallback = &debugReportCallbackFn;

        //     // We have to explicitly load this function.
        //     auto vkCreateDebugReportCallbackEXT = (PFN_vkCreateDebugReportCallbackEXT)vkGetInstanceProcAddr(instance, "vkCreateDebugReportCallbackEXT");
        //     if (vkCreateDebugReportCallbackEXT == nullptr) {
        //         throw std::runtime_error("Could not load vkCreateDebugReportCallbackEXT");
        //     }

        //     // Create and register callback.
        //     VK_CHECK_RESULT(vkCreateDebugReportCallbackEXT(instance, &createInfo, NULL, &debugReportCallback));
        // }
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
        // deviceCreateInfo.enabledLayerCount = enabledLayers.size();  // need to specify validation layers here as well.
        // deviceCreateInfo.ppEnabledLayerNames = enabledLayers.data();
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
        matrixACreateInfo.size = matrixASize+matrixBSize+matrixCSize; // buffer size in bytes. 
        matrixACreateInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT; // buffer is used as a storage buffer.
        matrixACreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE; // buffer is exclusive to a single queue family at a time. 

        VK_CHECK_RESULT(vkCreateBuffer(device, &matrixACreateInfo, NULL, &matrixA)); // create buffer.

        VkBufferCreateInfo matrixSizesCreateInfo = {};
        matrixSizesCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        matrixSizesCreateInfo.size = matrixSizesSize; // buffer size in bytes. 
        matrixSizesCreateInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT; // buffer is used as a storage buffer.
        matrixSizesCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE; // buffer is exclusive to a single queue family at a time. 

        VK_CHECK_RESULT(vkCreateBuffer(device, &matrixSizesCreateInfo, NULL, &matrixSizes)); // create buffer.

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
        
        VkMemoryRequirements memoryRequirementsmatrixA, memoryRequirementsmatrixSizes;
        vkGetBufferMemoryRequirements(device, matrixA, &memoryRequirementsmatrixA);
        vkGetBufferMemoryRequirements(device, matrixSizes, &memoryRequirementsmatrixSizes);

        // VkMemoryRequirements memoryRequirementsmatrixA, memoryRequirementsmatrixB, 
        //         memoryRequirementsmatrixC, memoryRequirementsmatrixSizes;
        // vkGetBufferMemoryRequirements(device, matrixA, &memoryRequirementsmatrixA);
        // vkGetBufferMemoryRequirements(device, matrixB, &memoryRequirementsmatrixB);
        // vkGetBufferMemoryRequirements(device, matrixSizes, &memoryRequirementsmatrixSizes);
        // vkGetBufferMemoryRequirements(device, matrixC, &memoryRequirementsmatrixC);
        
        
        // const VkDeviceSize memorySizematA = memoryRequirementsmatrixA.size;
        // const VkDeviceSize memorySizematB = memoryRequirementsmatrixB.size;
        // const VkDeviceSize memorySizematC = memoryRequirementsmatrixC.size; 
        // const VkDeviceSize memorySizeS = memoryRequirementsmatrixSizes.size;
        const VkDeviceSize memorySize = memoryRequirementsmatrixA.size+memoryRequirementsmatrixSizes.size;

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

        float* oActMinMaxtmp;
        VK_CHECK_RESULT(vkMapMemory(device, bufferMemory, 0, sizeof(float)*2, 0, (void **) &oActMinMaxtmp));
        
        oActMinMaxtmp[0] = outputActivationMin;
        oActMinMaxtmp[1] = outputActivationMax;

        vkUnmapMemory(device, bufferMemory);

        float* iDatatmp;
        VK_CHECK_RESULT(vkMapMemory(device, bufferMemory, sizeof(float)*2, inputSize, 0, (void **) &iDatatmp));
        
        std::memcpy(iDatatmp, inputData, inputSize);

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
        VK_CHECK_RESULT(vkMapMemory(device, bufferMemory, inputSize+(sizeof(float)*2)+filterSize+biasSize+outputSize, matrixSizesSize, 0, (void **) &stridepaddimstmp));
        
        stridepaddimstmp[0] = strideWidth;
        stridepaddimstmp[1] = strideHeight;
        stridepaddimstmp[2] = padWidth;
        stridepaddimstmp[3] = padHeight;
        stridepaddimstmp[36] = inputSize/sizeof(float);
        stridepaddimstmp[37] = filterSize/sizeof(float);
        stridepaddimstmp[38] = biasSize/sizeof(float);
        for(int i = 0; i < 16; i++) {
            stridepaddimstmp[i+4] = dimSizes[i];
            stridepaddimstmp[i+20] = dimStrides[i];
        }

        vkUnmapMemory(device, bufferMemory);

        VK_CHECK_RESULT(vkBindBufferMemory(device, matrixA, bufferMemory, 0));
        // VK_CHECK_RESULT(vkBindBufferMemory(device, matrixB, bufferMemory, matrixASize));
        VK_CHECK_RESULT(vkBindBufferMemory(device, matrixSizes, bufferMemory, matrixASize+matrixBSize+matrixCSize));
        // VK_CHECK_RESULT(vkBindBufferMemory(device, matrixC, bufferMemory, matrixASize+matrixBSize+matrixSizesSize));
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
        descriptorSetLayoutCreateInfo.bindingCount = 2; // only a single binding in this descriptor set layout. 
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

        // VkDescriptorBufferInfo descriptorBufferInfoMatB = {};
        // descriptorBufferInfoMatB.buffer = matrixB;
        // descriptorBufferInfoMatB.offset = 0;
        // descriptorBufferInfoMatB.range = VK_WHOLE_SIZE;

        VkDescriptorBufferInfo descriptorBufferInfoMatSizes = {};
        descriptorBufferInfoMatSizes.buffer = matrixSizes;
        descriptorBufferInfoMatSizes.offset = 0;
        descriptorBufferInfoMatSizes.range = VK_WHOLE_SIZE;

        // VkDescriptorBufferInfo descriptorBufferInfoMatC = {};
        // descriptorBufferInfoMatC.buffer = matrixC;
        // descriptorBufferInfoMatC.offset = 0;
        // descriptorBufferInfoMatC.range = VK_WHOLE_SIZE;

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

        vkUpdateDescriptorSets(device, 2, writeDescriptorSets, 0, NULL);
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

        // std::string source =
            // "#version 450 \n" \
            // "#extension GL_ARB_separate_shader_objects : enable \n" \
            // "layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in; \n" \
            // "layout(binding = 0) buffer floatBuffer { \n" \
            // "    float convFloatB[]; \n" \
            // "}; \n" \
            // "layout(binding = 1) readonly buffer intBuffer { \n" \
            // "    int convIntB[]; \n" \
            // "}; \n" \
            // "void main() { \n" \
            // "  int gid0 = int(gl_GlobalInvocationID.x); \n" \
            // "  int gid1 = int(gl_GlobalInvocationID.y); \n" \
            // "  int output_depth = convIntB[11]; \n" \
            // "  int output_width = convIntB[17]; \n" \
            // "  if((gid0*24+gid1) < 144) { \n" \
            // "        convFloatB[2 + convIntB[36] + convIntB[37] + convIntB[38] + (gid0*24+gid1)] = 20.0; \n" \
            // "  } \n" \
            // "}";
        std::string source =
        "#version 450 \n" \
        "#extension GL_ARB_separate_shader_objects : enable \n" \
        "layout(local_size_x = 8, local_size_y = 32, local_size_z = 1) in; \n" \
        "layout(binding = 0) buffer floatBuffer { \n" \
        "    float convFloatB[]; \n" \
        "}; \n" \
        "layout(binding = 1) readonly buffer intBuffer { \n" \
        "    int convIntB[]; \n" \
        "}; \n" \
        "void main() { \n" \
        "  int gid0 = int(gl_GlobalInvocationID.x); \n" \
        "  int gid1 = int(gl_GlobalInvocationID.y); \n" \
        "  int output_depth = convIntB[11]; \n" \
        "  int output_width = convIntB[17]; \n" \
        "  if((gid0 < convIntB[7]*output_depth) && (gid1 < convIntB[18]*output_width)) { \n" \
        "        int var1 = gid0/output_depth; \n" \
        "        int var2 = int(mod(gid1,output_width)); \n" \
        "        int var3 = int(mod(gid0,output_depth)); \n" \
        "        int var4 = gid1/output_width; \n" \
        "        float total = 0.0; \n" \
        "        for (int filter_y = 0; filter_y < convIntB[10]; ++filter_y) { \n" \
        "          for (int filter_x = 0; filter_x < convIntB[9]; ++filter_x) { \n" \
        "            for (int in_channel = 0; in_channel < convIntB[4]; ++in_channel) { \n" \
        "              int in_x = (var2 * convIntB[0] - convIntB[2]) + filter_x; \n" \
        "              int in_y = (var4 * convIntB[1] - convIntB[3]) + filter_y; \n" \
        "              if ((in_x >= 0) && (in_x < convIntB[5]) && (in_y >= 0) && \n" \
        "                  (in_y < convIntB[6])) { \n" \
        "                total += (convFloatB[2 + in_channel*convIntB[20] + in_x*convIntB[21] +in_y*convIntB[22] + var1*convIntB[23]] *  \n" \
        "                        convFloatB[convIntB[36] + 2 + in_channel*convIntB[24] + filter_x*convIntB[25] + filter_y*convIntB[26] + var3*convIntB[27]]); \n" \
        "              } \n" \
        "            } \n" \
        "          } \n" \
        "        } \n" \
        "        float bias_value = 0.0; \n" \
        "        if (convIntB[38] > 0) { \n" \
        "          bias_value = convFloatB[convIntB[36] + 2 + convIntB[37]+(var3*convIntB[28])]; \n" \
        "        } \n" \
        "        float max = total+bias_value; \n" \
        "        if(max < convFloatB[0]) max = convFloatB[0]; \n" \
        "        float min = max; \n" \
        "        if(min > convFloatB[1]) min = convFloatB[1]; \n" \
        "        convFloatB[2 + convIntB[36] + convIntB[37] + convIntB[38] + var3*convIntB[32] + var2*convIntB[33] + var4*convIntB[34] + var1*convIntB[35]] = min; \n" \
        "  } \n" \
        "}";
        // std::string source =
        //     "#version 450 \n" \
        //     "#extension GL_ARB_separate_shader_objects : enable \n" \
        //     "layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in; \n" \
        //     "layout(binding = 0) buffer floatBuffer { \n" \
        //     "    float convFloatB[]; \n" \
        //     "}; \n" \
        //     "layout(binding = 1) readonly buffer intBuffer { \n" \
        //     "    int convIntB[]; \n" \
        //     "}; \n" \
        //     "void main() { \n" \
        //     "  int gid0 = int(gl_GlobalInvocationID.x); \n" \
        //     "  int gid1 = int(gl_GlobalInvocationID.y); \n" \
        //     "  if((gid0 < convIntB[7]*"+to_string(dimSizes[7])+") && (gid1 < convIntB[18]*"+to_string(dimSizes[13])+")) { \n" \
        //     "        float total = 0.0; \n" \
        //     "        for (int filter_y = 0; filter_y < convIntB[10]; filter_y++) { \n" \
        //     "          for (int filter_x = 0; filter_x < convIntB[9]; filter_x++) { \n" \
        //     "            for (int in_channel = 0; in_channel < convIntB[4]; in_channel++) { \n" \
        //     "              int in_x = ((int(mod(gid1,"+to_string(dimSizes[13])+"))) * convIntB[0] - convIntB[2]) + filter_x; \n" \
        //     "              int in_y = ((gid1/"+to_string(dimSizes[13])+") * convIntB[1] - convIntB[3]) + filter_y; \n" \
        //     "              if ((in_x >= 0) && (in_x < convIntB[5]) && (in_y >= 0) && \n" \
        //     "                  (in_y < convIntB[6])) { \n" \
        //     "                total += (convFloatB[2 + in_channel*convIntB[20] + in_x*convIntB[21] +in_y*convIntB[22] + (gid0/"+to_string(dimSizes[7])+")*convIntB[23]] *  \n" \
        //     "                        convFloatB[convIntB[36] + 2 + in_channel*convIntB[24] + filter_x*convIntB[25] + filter_y*convIntB[26] + (int(mod(gid0,"+to_string(dimSizes[7])+")))*convIntB[27]]); \n" \
        //     "              } \n" \
        //     "            } \n" \
        //     "          } \n" \
        //     "        } \n" \
        //     "        float bias_value = 0.0; \n" \
        //     "        if (convIntB[38] > 0) { \n" \
        //     "          bias_value = convFloatB[convIntB[36] + 2 + convIntB[37])+((int(mod(gid0,"+to_string(dimSizes[7])+")))*convIntB[28])]; \n" \
        //     "        } \n" \
        //     "        float max = total+bias_value; \n" \
        //     "        if(max < convFloatB[0]) max = convFloatB[0]; \n" \
        //     "        float min = max; \n" \
        //     "        if(min > convFloatB[1]) min = convFloatB[1]; \n" \
        //     "        convFloatB[2 + convIntB[36] + convIntB[37] + convIntB[38] + (int(mod(gid0,"+to_string(dimSizes[7])+")))*convIntB[32] + (int(mod(gid1,"+to_string(dimSizes[13])+")))*convIntB[33] + (gid1/"+to_string(dimSizes[17])+")*convIntB[34] + (gid0/"+to_string(dimSizes[13])+")*convIntB[35]] = min; \n" \
        //     "  } \n" \
        //     "}";

            __android_log_print(ANDROID_LOG_INFO, "VulkanCode", "codeStr : %s", source.c_str());

            // "#version 450 \n" \
            // "#extension GL_ARB_separate_shader_objects : enable \n" \
            // "layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in; \n" \
            // "layout(binding = 0) readonly buffer inputData { \n" \
            // "    float iData[]; \n" \
            // "}; \n" \
            // "layout(binding = 1) readonly buffer filterBiasActivationMinMax { \n" \
            // "    float oActMin; \n" \
            // "    float oActMax; \n" \
            // "    float fData[]; \n" \
            // "}; \n" \
            // "layout(binding = 2) readonly buffer dimSizeStrideStridePad { \n" \
            // "    int strideWidth; \n" \
            // "    int strideHeight; \n" \
            // "    int padWidth; \n" \
            // "    int padHeight; \n" \
            // "    int dimSizes[16]; \n" \
            // "    int dimStrides[16]; \n" \
            // "}; \n" \
            // "layout(binding = 3) buffer outputData { \n" \
            // "    float oData[]; \n" \
            // "}; \n" \
            // "void main() { \n" \
            // "  int gid0 = int(gl_GlobalInvocationID.x); \n" \
            // "  int gid1 = int(gl_GlobalInvocationID.y); \n" \
            // "  int output_depth = dimSizes[7]; \n" \
            // "  int output_width = dimSizes[13]; \n" \
            // "  if((gid0 < dimSizes[3]*output_depth) && (gid1 < dimSizes[14]*output_width)) { \n" \
            // "        float total = 0.0; \n" \
            // "        for (int filter_y = 0; filter_y < dimSizes[6]; ++filter_y) { \n" \
            // "          for (int filter_x = 0; filter_x < dimSizes[5]; ++filter_x) { \n" \
            // "            for (int in_channel = 0; in_channel < dimSizes[0]; ++in_channel) { \n" \
            // "              int in_x = ((int(mod(gid1,output_width)) * strideWidth) - padWidth) + filter_x; \n" \
            // "              int in_y = (((gid1/output_width) * strideHeight) - padHeight) + filter_y; \n" \
            // "              if ((in_x >= 0) && (in_x < dimSizes[1]) && (in_y >= 0) && \n" \
            // "                  (in_y < dimSizes[2])) { \n" \
            // "                total += (iData[in_channel*dimStrides[0] + in_x*dimStrides[1] +in_y*dimStrides[2] + (gid0/output_depth)*dimStrides[3]] *  \n" \
            // "                        fData[in_channel*dimStrides[4] + filter_x*dimStrides[5] + filter_y*dimStrides[6] + int(mod(gid0,output_depth))*dimStrides[7]]); \n" \
            // "              } \n" \
            // "            } \n" \
            // "          } \n" \
            // "        } \n" \
            // "        float bias_value = 0.0; \n" \
            // "        if (dimSizes[8]*dimSizes[9]*dimSizes[10]*dimSizes[11] > 0) { \n" \
            // "          bias_value = fData[(dimSizes[4]*dimSizes[5]*dimSizes[6]*dimSizes[7])+(int(mod(gid0,output_depth))*dimStrides[8])]; \n" \
            // "        } \n" \
            // "        float max = total+bias_value; \n" \
            // "        if(max < oActMin) max = oActMin; \n" \
            // "        float min = max; \n" \
            // "        if(min > oActMax) min = oActMax; \n" \
            // "        oData[int(mod(gid0,output_depth))*dimStrides[12] + int(mod(gid1,output_width))*dimStrides[13] + \n" \
            // "                     (gid1/output_width)*dimStrides[14] + (gid0/output_depth)*dimStrides[15]] = min; \n" \
            // "  } \n" \
            // "}";


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
        
        __android_log_print(ANDROID_LOG_INFO, "VulkanConv", "codeSize : %d", createInfo.codeSize);

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
        vkCmdDispatch(commandBuffer, (batches*output_depth-1)/8+1, (output_height*output_width-1)/32+1, 1);

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

        __android_log_print(ANDROID_LOG_INFO, "VulkanConv", "runkernel: %lf", wall);
    }

    void getresult() {
        float *matCtmp;
        VK_CHECK_RESULT(vkMapMemory(device, bufferMemory, 
            inputSize+(sizeof(float)*2)+filterSize+biasSize, outputSize, 0, (void **)&matCtmp));
      
        // double sumC = 0.0;
        // for (int k = 0; k < outputSize/sizeof(float); k++) {
        //   sumC += matCtmp[k];
        //   if(k < 100) {
        //       __android_log_print(ANDROID_LOG_INFO, "VulkanConv", "Conv %d: %lf", k, matCtmp[k]);
        //   }
        // }

        std::memcpy(outputData, matCtmp, outputSize);

        // __android_log_print(ANDROID_LOG_INFO, "VulkanConv", "Conv sumC: %lf", sumC);

        vkUnmapMemory(device, bufferMemory);  
    }

    void cleanup() {
        // if (enableValidationLayers) {
        //     // destroy callback.
        //     auto func = (PFN_vkDestroyDebugReportCallbackEXT)vkGetInstanceProcAddr(instance, "vkDestroyDebugReportCallbackEXT");
        //     if (func == nullptr) {
        //         throw std::runtime_error("Could not load vkDestroyDebugReportCallbackEXT");
        //     }
        //     func(instance, debugReportCallback, NULL);
        // }

        // vkFreeMemory(device, bufferMemorymatA, NULL);
        // vkFreeMemory(device, bufferMemorymatB, NULL);
        // vkFreeMemory(device, bufferMemorymatC, NULL);
        // vkFreeMemory(device, bufferMemoryS, NULL);
        vkFreeMemory(device, bufferMemory, NULL);
        vkDestroyBuffer(device, matrixA, NULL);
        // vkDestroyBuffer(device, matrixB, NULL);
        // vkDestroyBuffer(device, matrixC, NULL);
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

inline void OpenCLConv(const float* input_data, const int input_size,
          const float* filter_data, const int filter_size,
          const float* bias_data, const int bias_size,
          float* output_data, const int output_size,
          int stride_width, int stride_height, 
          int pad_width, int pad_height, 
          const int* dim_sizes, const int* dim_strides,
          float output_activation_min, float output_activation_max,
          cl_context context, cl_command_queue queue, cl_program program) {
  cl_mem d_input;
  cl_mem d_filter;
  cl_mem d_bias;
  cl_mem d_output;
  cl_mem d_dim_sizes;
  cl_mem d_dim_strides;

  // cl_platform_id cpPlatform;
  // cl_device_id device_id;    
  // cl_context context;       
  // cl_command_queue queue;   
  // cl_program program;       
  cl_kernel kernel;

  size_t globalSize0, globalSize1, localSize0, localSize1;
  localSize0 = 32;
  localSize1 = 32;
  
  int batches = dim_sizes[3];
  int output_depth = dim_sizes[7];
  int output_height = dim_sizes[14];  
  int output_width = dim_sizes[13];

  __android_log_print(ANDROID_LOG_INFO, "Convdimension", "batches: %d", batches);
  __android_log_print(ANDROID_LOG_INFO, "Convdimension", "output_depth: %d", output_depth);
  __android_log_print(ANDROID_LOG_INFO, "Convdimension", "output_height: %d", output_height);
  __android_log_print(ANDROID_LOG_INFO, "Convdimension", "output_width: %d", output_width);

  globalSize0 = ceil(batches*output_depth/(localSize0*1.0))*localSize0;
  globalSize1 = ceil(output_height*output_width/(localSize1*1.0))*localSize1;

  const size_t local[2] = { localSize0, localSize1 };
  const size_t global[2] = { globalSize0, globalSize1 };

  cl_int err;

  // err = clGetPlatformIDs(1, &cpPlatform, NULL);
 
  // err = clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);

  // context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);

  // queue = clCreateCommandQueue(context, device_id, 0, &err);

  // program = clCreateProgramWithSource(context, 1,
  //                         (const char **) & kernelSource, NULL, &err);

  // clBuildProgram(program, 0, NULL, NULL, NULL, NULL);

  // Start Timers
  double wall0 = get_wall_time();
  double cpu0  = get_cpu_time();

  kernel = clCreateKernel(program, "conv", &err);

  // Stop timers
  double wall1 = get_wall_time();
  double cpu1  = get_cpu_time();

  double wall = wall1 - wall0;
  double cpu = cpu1 - cpu0;

  // note: andoird log
  __android_log_print(ANDROID_LOG_INFO, "Convruntime", "Walltime createkernel: %lf", wall);

  wall0 = get_wall_time();
  cpu0  = get_cpu_time();

  d_input = clCreateBuffer(context, CL_MEM_READ_ONLY, input_size*sizeof(float), NULL, NULL);
  d_filter = clCreateBuffer(context, CL_MEM_READ_ONLY, filter_size*sizeof(float), NULL, NULL);
  d_bias = clCreateBuffer(context, CL_MEM_READ_ONLY, bias_size*sizeof(float), NULL, NULL);
  d_output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, output_size*sizeof(float), NULL, NULL);
  d_dim_sizes = clCreateBuffer(context, CL_MEM_READ_ONLY, 16*sizeof(int), NULL, NULL);
  d_dim_strides = clCreateBuffer(context, CL_MEM_READ_ONLY, 16*sizeof(int), NULL, NULL);

  // Stop timers
  wall1 = get_wall_time();
  cpu1  = get_cpu_time();

  wall = wall1 - wall0;
  cpu = cpu1 - cpu0;

  //note: andoird log
  __android_log_print(ANDROID_LOG_INFO, "Convruntime", "Walltime createbuffer: %lf", wall);

  wall0 = get_wall_time();
  cpu0  = get_cpu_time();

  err = clEnqueueWriteBuffer(queue, d_input, CL_TRUE, 0,
                                 input_size*sizeof(float), input_data, 0, NULL, NULL);
  err = clEnqueueWriteBuffer(queue, d_filter, CL_TRUE, 0,
                                 filter_size*sizeof(float), filter_data, 0, NULL, NULL);
  err = clEnqueueWriteBuffer(queue, d_bias, CL_TRUE, 0,
                                 bias_size*sizeof(float), bias_data, 0, NULL, NULL);
  err = clEnqueueWriteBuffer(queue, d_dim_sizes, CL_TRUE, 0,
                                 16*sizeof(int), dim_sizes, 0, NULL, NULL);
  err = clEnqueueWriteBuffer(queue, d_dim_strides, CL_TRUE, 0,
                                 16*sizeof(int), dim_strides, 0, NULL, NULL);
  clFinish(queue);

  // Stop timers
  wall1 = get_wall_time();
  cpu1  = get_cpu_time();

  wall = wall1 - wall0;
  cpu = cpu1 - cpu0;

  // note: andoird log
  __android_log_print(ANDROID_LOG_INFO, "Convruntime", "Walltime writebuffer: %lf", wall);

  wall0 = get_wall_time();
  cpu0  = get_cpu_time();

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

  // Stop timers
  wall1 = get_wall_time();
  cpu1  = get_cpu_time();

  wall = wall1 - wall0;
  cpu = cpu1 - cpu0;

  // note: andoird log
  __android_log_print(ANDROID_LOG_INFO, "Convruntime", "Walltime setkernelargs: %lf", wall);

  wall0 = get_wall_time();
  cpu0  = get_cpu_time();

  err = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global, local, 0, NULL, NULL);

  clFinish(queue);

  // Stop timers
  wall1 = get_wall_time();
  cpu1  = get_cpu_time();

  wall = wall1 - wall0;
  cpu = cpu1 - cpu0;

  // note: andoird log
  __android_log_print(ANDROID_LOG_INFO, "Convruntime", "Walltime runkernel: %lf", wall);

  __android_log_print(ANDROID_LOG_INFO, "Convruntime", "Converror: %d", err);

  wall0 = get_wall_time();
  cpu0  = get_cpu_time();

  clEnqueueReadBuffer(queue, d_output, CL_TRUE, 0, output_size*sizeof(float), output_data, 0, NULL, NULL );
  clFinish(queue);

  // Stop timers
  wall1 = get_wall_time();
  cpu1  = get_cpu_time();

  wall = wall1 - wall0;
  cpu = cpu1 - cpu0;

  // note: andoird log
  __android_log_print(ANDROID_LOG_INFO, "Convruntime", "Walltime readbuffer: %lf", wall);

  wall0 = get_wall_time();
  cpu0  = get_cpu_time();

  clReleaseMemObject(d_input);
  clReleaseMemObject(d_filter);
  clReleaseMemObject(d_bias);
  clReleaseMemObject(d_output);
  clReleaseMemObject(d_dim_sizes);
  clReleaseMemObject(d_dim_strides);
  // clReleaseProgram(program);
  clReleaseKernel(kernel);
  // clReleaseCommandQueue(queue);
  // clReleaseContext(context);

  // Stop timers
  wall1 = get_wall_time();
  cpu1  = get_cpu_time();

  wall = wall1 - wall0;
  cpu = cpu1 - cpu0;

  // note: andoird log
  __android_log_print(ANDROID_LOG_INFO, "Convruntime", "Walltime cleaning: %lf", wall);
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
  __android_log_print(ANDROID_LOG_INFO, "Convruntime", "Walltime: %lf", wall);
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

inline void ConvOpenCL(const float* input_data, const Dims<4>& input_dims,
                 const float* filter_data, const Dims<4>& filter_dims,
                 const float* bias_data, const Dims<4>& bias_dims,
                 int stride_width, int stride_height, int pad_width,
                 int pad_height, TfLitePadding padding,
                 float output_activation_min, float output_activation_max,
                 float* output_data, const Dims<4>& output_dims,
                 float* im2col_data, const Dims<4>& im2col_dims,
                 cl_context context_cl, cl_command_queue queue, cl_program program) {
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

  // __android_log_print(ANDROID_LOG_INFO, "Convruntime", "Walltime: %lf", wall);

  // OpenCLConv(input_data, input_size,
  //         filter_data, filter_size,
  //         bias_data, bias_size,
  //         output_data, output_size,
  //         stride_width, stride_height, 
  //         pad_width, pad_height, 
  //         sizes, strides,
  //         output_activation_min, output_activation_max,
  //         context_cl, queue, program);

  vulkanTestConv(input_data, input_size,
          filter_data, filter_size,
          bias_data, bias_size,
          output_data, output_size,
          stride_width, stride_height, 
          pad_width, pad_height, 
          sizes, strides,
          output_activation_min, output_activation_max);

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
  // __android_log_print(ANDROID_LOG_INFO, "Convruntime", "Walltime: %lf", wall);
}

}  // namespace multithreaded_ops
}  // namespace tflite

#endif  // TENSORFLOW_CONTRIB_LITE_KERNELS_INTERNAL_MULTITHREAD_CONV
