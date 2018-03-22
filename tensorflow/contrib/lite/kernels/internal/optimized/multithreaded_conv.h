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
          "layout(local_size_x = 32, local_size_y = 8) in; \n" \
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
          "    uint row = gl_GlobalInvocationID.x; \n" \
          "    uint column = gl_GlobalInvocationID.y;  \n" \
          "    int M = S[0];  \n" \
          "    int K = S[1];  \n" \
          "    int N = S[2];  \n" \
          "    if ((row < M) && (column < N)) { \n" \
          "        float sum = 0.0; \n" \
          "        uint a_row_start_index = K * row; \n" \
          "        uint b_column_start_index = K * column; \n" \
          "        for(int i = 0; i < K; i++){ \n" \
          "            sum += A[a_row_start_index + i]*B[b_column_start_index + i]; \n" \
          "        } \n" \
          "        C[M * column + row] = sum; \n" \
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
        vkCmdDispatch(commandBuffer, (M-1)/32+1, (N-1)/8+1, 1);

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

        // float sum = 0.0;
        // for (int k = 0; k < matrixCSize / sizeof(float); k++) {
        //   sum += matCtmp[k];
          // __android_log_print(ANDROID_LOG_INFO, "VulkanTest", "Matmul %d: %lf", k, matCtmp[k]);
        // }

        // // __android_log_print(ANDROID_LOG_INFO, "VulkanTest", "Matmul sum: %lf", sum);

        // vkUnmapMemory(device, bufferMemory);


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

  __android_log_print(ANDROID_LOG_INFO, "Convruntime", "Input height: %d", inheightsize);
  __android_log_print(ANDROID_LOG_INFO, "Convruntime", "Input width: %d", inwidthsize);
  __android_log_print(ANDROID_LOG_INFO, "Convruntime", "Input depth: %d", indepthsize);
  __android_log_print(ANDROID_LOG_INFO, "Convruntime", "Input batch: %d", inbatchsize);

  __android_log_print(ANDROID_LOG_INFO, "Convruntime", "strides0: %d", strides0);
  __android_log_print(ANDROID_LOG_INFO, "Convruntime", "strides1: %d", strides1);
  __android_log_print(ANDROID_LOG_INFO, "Convruntime", "strides2: %d", strides2);
  __android_log_print(ANDROID_LOG_INFO, "Convruntime", "strides3: %d", strides3);

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

  OpenCLConv(input_data, input_size,
          filter_data, filter_size,
          bias_data, bias_size,
          output_data, output_size,
          stride_width, stride_height, 
          pad_width, pad_height, 
          sizes, strides,
          output_activation_min, output_activation_max,
          context_cl, queue, program);

  // Conv2(input_data, input_dims,
  //                filter_data, filter_dims,
  //                bias_data, bias_dims,
  //                stride_width, stride_height, pad_width,
  //                pad_height, output_activation_min,
  //                output_activation_max, output_data,
  //                output_dims, im2col_data,
  //                im2col_dims);

  // Stop timers
  double wall1 = get_wall_time();
  double cpu1  = get_cpu_time();

  double wall = wall1 - wall0;
  double cpu = cpu1 - cpu0;

  // note: andoird log
  __android_log_print(ANDROID_LOG_INFO, "Convruntime", "Walltime: %lf", wall);
}

}  // namespace multithreaded_ops
}  // namespace tflite

#endif  // TENSORFLOW_CONTRIB_LITE_KERNELS_INTERNAL_MULTITHREAD_CONV
