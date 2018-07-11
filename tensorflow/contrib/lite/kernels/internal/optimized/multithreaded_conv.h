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

#define kFloatWeightsPerNeonLane 4

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

      //  // Start Timers
      // double wall0 = get_wall_time();
      // double cpu0  = get_cpu_time();

      MatMulConvFunctor<Eigen::ThreadPoolDevice, T>()(device, output, input,
                                                      filter, dim_pair);
      //    // Stop timers
      // double wall1 = get_wall_time();
      // double cpu1  = get_cpu_time();

      // double wall = wall1 - wall0;
      // double cpu = cpu1 - cpu0;

      // // note: andoird log
      // __android_log_print(ANDROID_LOG_INFO, "Convruntime", "runkernelMultithreadConv1x1: %lf", wall);

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



// static cl_mem d_all = NULL;
static cl_kernel kernellocalall = NULL;
static cl_kernel kernellocalfilter = NULL;
static cl_kernel kernelmatmul = NULL;
static cl_kernel kernelconv = NULL;
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
    VkCommandPool commandPool0, VkCommandBuffer commandBuffer0, VkBuffer matrixA0, VkBuffer matrixB0, VkBuffer matrixC0, VkBuffer matrixSizes0, VkDeviceMemory bufferMemory0) {
        
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
        matrixB = matrixB0;
        matrixC = matrixC0;
        matrixSizes = matrixSizes0;
        bufferMemory = bufferMemory0;

        matrixASize = (uint32_t) (sizeof(float) *buffsizes[0]+buffsizes[1]);
        matrixBSize = (uint32_t) (sizeof(float) *buffsizes[2]);
        matrixCSize = (uint32_t) (sizeof(float) * buffsizes[3]);
        matrixSizesSize = (uint32_t) ((sizeof(int) * 40) + (sizeof(float) * 4));
        
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

        wall0 = get_wall_time();
        cpu0  = get_cpu_time();
        createBuffer();
        wall1 = get_wall_time();
        cpu1  = get_cpu_time();
        wall = wall1 - wall0;
        cpu = cpu1 - cpu0;
        __android_log_print(ANDROID_LOG_INFO, "VulkanConvDetail", "createBuffer: %lf", wall);

        wall0 = get_wall_time();
        cpu0  = get_cpu_time();
        createDescriptorSet();
        wall1 = get_wall_time();
        cpu1  = get_cpu_time();
        wall = wall1 - wall0;
        cpu = cpu1 - cpu0;
        __android_log_print(ANDROID_LOG_INFO, "VulkanConvDetail", "createDescriptorSet: %lf", wall);

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

    void createBuffer() {
        int numchannel = dimSizes[0];
        int addslot = (4-(numchannel%4))%4;
        numchannel = numchannel + addslot;
        inputSize = (uint32_t) (sizeof(float)*dimSizes[1]*dimSizes[2]*dimSizes[3]*numchannel);
        filterSize = (uint32_t) (sizeof(float)*dimSizes[5]*dimSizes[6]*dimSizes[7]*numchannel);

        float* iDatatmp;
        VK_CHECK_RESULT(vkMapMemory(device, bufferMemory, 0, inputSizeAll, 0, (void **) &iDatatmp));
        
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
        VK_CHECK_RESULT(vkMapMemory(device, bufferMemory, inputSizeAll, filterSizeAll, 0, (void **) &fDatatmp));
        
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
        VK_CHECK_RESULT(vkMapMemory(device, bufferMemory, inputSizeAll+filterSizeAll, biasSizeAll, 0, (void **) &bDatatmp));
        
        std::memcpy(bDatatmp, biasData, biasSize);

        vkUnmapMemory(device, bufferMemory);

        float* act;
        VK_CHECK_RESULT(vkMapMemory(device, bufferMemory, inputSizeAll+filterSizeAll+biasSizeAll+outputSizeAll, 4*sizeof(float), 0, (void **) &act));

        act[0] = outputActivationMin;
        act[1] = outputActivationMax;
        act[2] = 0;
        act[3] = 0;

        if((dimSizes[5] == 1) && (dimSizes[6] == 1) && (strideWidth == 1) && (strideHeight == 1) && (padWidth == 0) && (padHeight == 0)) {
          act[2] = 1;
        }
        
        vkUnmapMemory(device, bufferMemory);

        dimStrides[1] = (dimStrides[1]/dimSizes[0])*numchannel;
        dimStrides[2] = (dimStrides[2]/dimSizes[0])*numchannel;
        dimStrides[3] = (dimStrides[3]/dimSizes[0])*numchannel;
        dimStrides[5] = (dimStrides[5]/dimSizes[0])*numchannel;
        dimStrides[6] = (dimStrides[6]/dimSizes[0])*numchannel;
        dimStrides[7] = (dimStrides[7]/dimSizes[0])*numchannel;
        dimSizes[0] = numchannel;
        dimSizes[4] = numchannel;

        int d_output_depth = (((dimSizes[12]-1)/4+1)*4);

        //output
        dimStrides[13] = (dimStrides[13]/dimSizes[12])*d_output_depth/4;
        dimStrides[14] = (dimStrides[14]/dimSizes[12])*d_output_depth/4;
        dimStrides[15] = (dimStrides[15]/dimSizes[12])*d_output_depth/4;

        int* stridepaddimstmp;
        VK_CHECK_RESULT(vkMapMemory(device, bufferMemory, inputSizeAll+filterSizeAll+biasSizeAll+outputSizeAll+(4*sizeof(float)), 40*sizeof(int), 0, (void **) &stridepaddimstmp));
        
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
    }

    void createDescriptorSet() {
        VkDescriptorPoolSize descriptorPoolSize[2];

        descriptorPoolSize[0] = {};
        descriptorPoolSize[0].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        descriptorPoolSize[0].descriptorCount = 3;

        descriptorPoolSize[1] = {};
        descriptorPoolSize[1].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        descriptorPoolSize[1].descriptorCount = 1;

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
        writeDescriptorSets[3].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER; // storage buffer.
        writeDescriptorSets[3].pBufferInfo = &descriptorBufferInfoMatSizes;

        vkUpdateDescriptorSets(device, 4, writeDescriptorSets, 0, NULL);
    }

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

        int d_output_depth = (((output_depth-1)/4+1)*4);

        if((dimSizes[5] == 1) && (dimSizes[6] == 1) && (strideWidth == 1) && (strideHeight == 1) && (padWidth == 0) && (padHeight == 0)) {
          vkCmdDispatch(commandBuffer, (d_output_depth/4-1)/8+1, (output_height*output_width*batches-1)/32+1, 1);
        }
        else {
          vkCmdDispatch(commandBuffer, (d_output_depth/4-1)/8+1, (output_height-1)/8+1, (output_width-1)/8+1);
        }

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
        VK_CHECK_RESULT(vkMapMemory(device, bufferMemory,inputSizeAll+filterSizeAll+biasSizeAll, outputSizeAll, 0, (void **)&matCtmp));

        // std::memcpy(outputData, matCtmp, outputSize);

        int output_depth = dimSizes[7];
        int d_output_depth = (((output_depth-1)/4+1)*4);
        int output_size = outputSize/sizeof(float);

        for(int i = 0; i < output_size/output_depth; i++) {
          for(int j = 0; j < output_depth; j++) {
            outputData[i*output_depth + j] = matCtmp[i*d_output_depth + j];
          }
        }

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
    VkCommandPool conv_commandPool, VkCommandBuffer conv_commandBuffer, VkBuffer conv_matrixA, VkBuffer conv_matrixB, VkBuffer conv_matrixC, VkBuffer conv_matrixSizes, VkDeviceMemory conv_bufferMemory) {

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
          conv_commandPool, conv_commandBuffer, conv_matrixA, conv_matrixB, conv_matrixC, conv_matrixSizes, conv_bufferMemory);
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
  // cl_mem d_bias = cl_mem_arr[2];
  cl_mem d_output = cl_mem_arr[3];
  cl_mem d_dim_sizes = cl_mem_arr[4];
  cl_mem d_dim_strides = cl_mem_arr[5];

  //add cl_event
  cl_event event_runkernel;
  cl_event event_mapinput;
  cl_event event_mapfilter;
  // cl_event event_mapbias;
  cl_event event_mapoutput;
  cl_event event_unmapinput;
  cl_event event_unmapfilter;
  // cl_event event_unmapbias;
  cl_event event_unmapoutput;
  cl_event event_writedimsizes;
  cl_event event_writedimstrides;

  cl_int err;
  // cl_kernel kernel;
  
  int batches = dim_sizes[3];
  int output_depth = dim_sizes[7];
  int output_height = dim_sizes[14];  
  int output_width = dim_sizes[13];

  // const size_t local[2] = { 8, 16 };
  // const size_t global[2] = { (size_t) (((output_depth*batches-1)/8+1)*8), (size_t) (((output_height*output_width-1)/16+1)*16) };

  int numchannel = dim_sizes[0];
  int addslot = (4-(numchannel%4))%4;
  numchannel = numchannel + addslot;
  input_size = (input_size/dim_sizes[0])*numchannel;
  filter_size = (filter_size/dim_sizes[0])*numchannel;

  double wall01 = get_wall_time();
  double cpu01  = get_cpu_time();

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

  // // Stop timers
    // double wall11 = get_wall_time();
    // double cpu11  = get_cpu_time();

    // double wall1 = wall11 - wall01;
    // double cpu1 = cpu11 - cpu01;

    // __android_log_print(ANDROID_LOG_INFO, "Convruntime", "runkernelConverror: %d", err);

    // note: andoird log
    // __android_log_print(ANDROID_LOG_INFO, "Convruntime", "runkernelmemcpy: %lf", wall1);

  // clEnqueueUnmapMemObject(queue,d_bias,(void *) biasfloat,0, NULL, &event_unmapbias);

  // err = clEnqueueWriteBuffer(queue, d_bias, CL_TRUE, 0,
  //                                bias_size*sizeof(float), bias_data, 0, NULL, &event_unmapbias);



  dim_strides[1] = (dim_strides[1]/dim_sizes[0])*numchannel;
  dim_strides[2] = (dim_strides[2]/dim_sizes[0])*numchannel;
  dim_strides[3] = (dim_strides[3]/dim_sizes[0])*numchannel;
  dim_strides[5] = (dim_strides[5]/dim_sizes[0])*numchannel;
  dim_strides[6] = (dim_strides[6]/dim_sizes[0])*numchannel;
  dim_strides[7] = (dim_strides[7]/dim_sizes[0])*numchannel;
  dim_sizes[0] = numchannel;
  dim_sizes[4] = numchannel;

  int d_output_depth = (((output_depth-1)/4+1)*4);

  //output
  dim_strides[13] = (dim_strides[13]/dim_sizes[12])*d_output_depth/4;
  dim_strides[14] = (dim_strides[14]/dim_sizes[12])*d_output_depth/4;
  dim_strides[15] = (dim_strides[15]/dim_sizes[12])*d_output_depth/4;

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



  // // Stop timers
  // wall1 = get_wall_time();
  // cpu1  = get_cpu_time();

  // wall = wall1 - wall0;
  // cpu = cpu1 - cpu0;

  // // note: andoird log
  // __android_log_print(ANDROID_LOG_INFO, "Convruntime", "Walltime setkernelargs: %lf", wall);

  // cl_event eventwaitlist[4] = {event_unmapinput,event_unmapfilter,event_writedimsizes,event_writedimstrides};

  clFinish(queue);

  // double wall0 = get_wall_time();
  // double cpu0  = get_cpu_time();


  // cl_float4 omin = {output_activation_min, output_activation_min, output_activation_min, output_activation_min};
  // cl_float4 omax = {output_activation_max, output_activation_max, output_activation_max, output_activation_max};

  // double wall01 = get_wall_time();
  // double cpu01  = get_cpu_time();

  if((dim_sizes[6] == 1) && (dim_sizes[5] == 1) && (stride_width == 1) && (stride_height == 1) && (pad_width == 0) && (pad_height == 0)) {
    // cl_mem d_output2 = clCreateBuffer(context, CL_MEM_WRITE_ONLY, output_size/output_depth*d_output_depth*sizeof(float), NULL, NULL);

    int m_cols = dim_sizes[0];
    int m_rows = dim_sizes[1]*dim_sizes[2]*dim_sizes[3];
    int n_batch = dim_sizes[7];
    int bias_stride = dim_strides[8];

    err  = clSetKernelArg(kernelmatmul, 0, sizeof(cl_mem), &d_input);
    err  = clSetKernelArg(kernelmatmul, 1, sizeof(cl_mem), &d_filter);
    // err  = clSetKernelArg(kernelmatmul, 2, sizeof(cl_mem), &d_bias);
    err  = clSetKernelArg(kernelmatmul, 2, sizeof(cl_mem), &d_output);
    err  = clSetKernelArg(kernelmatmul, 3, sizeof(int), &m_rows);
    err  = clSetKernelArg(kernelmatmul, 4, sizeof(int), &m_cols);
    err  = clSetKernelArg(kernelmatmul, 5, sizeof(int), &n_batch);
    // err  = clSetKernelArg(kernelmatmul, 7, sizeof(int), &bias_stride);
    // err  = clSetKernelArg(kernelmatmul, 8, sizeof(cl_float4), &omin);
    // err  = clSetKernelArg(kernelmatmul, 9, sizeof(cl_float4), &omax);

    //conv matmul local
    // const size_t local[2] = { 8, 32 };
    // const size_t global[2] = { 8, (size_t) ((output_height*output_width*batches*output_depth-1)/32+1)*32 };  
    
    //conv matmul
    // const size_t local[2] = { 8, 32 };
    // const size_t global[2] = { (size_t) ((output_depth-1)/8+1)*8, (size_t) ((output_height*output_width*batches-1)/32+1)*32 };

    // conv matmul block
    const size_t local[2] = { 8, 32 };
    const size_t global[2] = { (size_t) ((d_output_depth/4-1)/8+1)*8, (size_t) ((output_height*output_width*batches-1)/32+1)*32 };

    double wall0 = get_wall_time();
    double cpu0  = get_cpu_time();

    err = clEnqueueNDRangeKernel(queue, kernelmatmul, 2, NULL, global, local, 0, NULL, NULL);

    clFinish(queue);

    // // Stop timers
    double wall1 = get_wall_time();
    double cpu1  = get_cpu_time();

    double wall = wall1 - wall0;
    double cpu = cpu1 - cpu0;

    // __android_log_print(ANDROID_LOG_INFO, "Convruntime", "runkernelConverror: %d", err);

    // note: andoird log
    __android_log_print(ANDROID_LOG_INFO, "Convruntime", "runkernelOclConvMatmul: %lf", wall);

    cl_float *host_result = (cl_float*)clEnqueueMapBuffer(
            queue,
            d_output,
            CL_TRUE,
            CL_MAP_READ,
            0,
            output_size/output_depth*d_output_depth*sizeof(float),
            0, NULL, NULL, NULL);

    for(int i = 0; i < output_size/output_depth; i++) {
      for(int j = 0; j < output_depth; j++) {
        output_data[i*output_depth + j] = host_result[i*d_output_depth + j];
      }
    }

    clEnqueueUnmapMemObject(queue,d_output,(void *) host_result,0, NULL, NULL);
    clFinish(queue);
    // clReleaseMemObject(d_output2);
  }
  else if((dim_sizes[6] < 8) && (dim_sizes[5] < 8) && (stride_width == 1) && (stride_height == 1) && (pad_width == 0) && (pad_height == 0)) {
    int xsize = ((output_width-1)/16+1)*16;
    int ysize = ((output_height-1)/8+1)*8;

    err  = clSetKernelArg(kernellocalall, 0, sizeof(cl_mem), &d_input);
    err  = clSetKernelArg(kernellocalall, 1, sizeof(cl_mem), &d_filter);
    err  = clSetKernelArg(kernellocalall, 2, sizeof(cl_mem), &d_output);
    err  = clSetKernelArg(kernellocalall, 3, sizeof(int), &stride_width);
    err  = clSetKernelArg(kernellocalall, 4, sizeof(int), &stride_height);
    err  = clSetKernelArg(kernellocalall, 5, sizeof(int), &pad_width);
    err  = clSetKernelArg(kernellocalall, 6, sizeof(int), &pad_height);
    err  = clSetKernelArg(kernellocalall, 7, sizeof(int), &xsize);
    err  = clSetKernelArg(kernellocalall, 8, sizeof(int), &ysize);
    err  = clSetKernelArg(kernellocalall, 9, sizeof(cl_mem), &d_dim_sizes);
    err  = clSetKernelArg(kernellocalall, 10, sizeof(cl_mem), &d_dim_strides);

    //conv baru with local
    const size_t local[2] = { 8, 16 };
    const size_t global[2] = { (size_t) ysize*batches, (size_t) xsize*d_output_depth/4 };
    
    double wall0 = get_wall_time();
    double cpu0  = get_cpu_time();
    
    err = clEnqueueNDRangeKernel(queue, kernellocalall, 2, NULL, global, local, 0, NULL, NULL);

    clFinish(queue);

    // // Stop timers
    double wall1 = get_wall_time();
    double cpu1  = get_cpu_time();

    double wall = wall1 - wall0;
    double cpu = cpu1 - cpu0;

    // __android_log_print(ANDROID_LOG_INFO, "Convruntime", "runkernelConverror: %d", err);

    // note: andoird log
    __android_log_print(ANDROID_LOG_INFO, "Convruntime", "runkernelOclConvlocalall: %lf", wall);

    cl_float *host_result = (cl_float*)clEnqueueMapBuffer(
            queue,
            d_output,
            CL_TRUE,
            CL_MAP_READ,
            0,
            output_size/output_depth*d_output_depth*sizeof(float),
            0, NULL, NULL, NULL);

    for(int i = 0; i < output_size/output_depth; i++) {
      for(int j = 0; j < output_depth; j++) {
        output_data[i*output_depth + j] = host_result[i*d_output_depth + j];
      }
    }

    clEnqueueUnmapMemObject(queue,d_output,(void *) host_result,0, NULL, NULL);
    clFinish(queue);

    // clEnqueueReadBuffer(queue, d_output, CL_TRUE, 0, output_size*sizeof(float), output_data, 0, NULL, NULL );
  }
  else {
    int xsize = ((output_width-1)/16+1)*16;
    int ysize = ((output_height-1)/8+1)*8;

    err  = clSetKernelArg(kernellocalfilter, 0, sizeof(cl_mem), &d_input);
    err  = clSetKernelArg(kernellocalfilter, 1, sizeof(cl_mem), &d_filter);
    err  = clSetKernelArg(kernellocalfilter, 2, sizeof(cl_mem), &d_output);
    err  = clSetKernelArg(kernellocalfilter, 3, sizeof(int), &stride_width);
    err  = clSetKernelArg(kernellocalfilter, 4, sizeof(int), &stride_height);
    err  = clSetKernelArg(kernellocalfilter, 5, sizeof(int), &pad_width);
    err  = clSetKernelArg(kernellocalfilter, 6, sizeof(int), &pad_height);
    err  = clSetKernelArg(kernellocalfilter, 7, sizeof(int), &xsize);
    err  = clSetKernelArg(kernellocalfilter, 8, sizeof(int), &ysize);
    err  = clSetKernelArg(kernellocalfilter, 9, sizeof(cl_mem), &d_dim_sizes);
    err  = clSetKernelArg(kernellocalfilter, 10, sizeof(cl_mem), &d_dim_strides);

    //conv baru with local
    const size_t local[2] = { 8, 16 };
    const size_t global[2] = { (size_t) ysize*batches, (size_t) xsize*d_output_depth/4 };
    
    double wall0 = get_wall_time();
    double cpu0  = get_cpu_time();
    
    err = clEnqueueNDRangeKernel(queue, kernellocalfilter, 2, NULL, global, local, 0, NULL, NULL);

    clFinish(queue);

    // // Stop timers
    double wall1 = get_wall_time();
    double cpu1  = get_cpu_time();

    double wall = wall1 - wall0;
    double cpu = cpu1 - cpu0;

    // __android_log_print(ANDROID_LOG_INFO, "Convruntime", "runkernelConverror: %d", err);

    // note: andoird log
    __android_log_print(ANDROID_LOG_INFO, "Convruntime", "runkernelOclConvlocalfilter: %lf", wall);

    cl_float *host_result = (cl_float*)clEnqueueMapBuffer(
            queue,
            d_output,
            CL_TRUE,
            CL_MAP_READ,
            0,
            output_size/output_depth*d_output_depth*sizeof(float),
            0, NULL, NULL, NULL);

    for(int i = 0; i < output_size/output_depth; i++) {
      for(int j = 0; j < output_depth; j++) {
        output_data[i*output_depth + j] = host_result[i*d_output_depth + j];
      }
    }

    clEnqueueUnmapMemObject(queue,d_output,(void *) host_result,0, NULL, NULL);
    clFinish(queue);

    // clEnqueueReadBuffer(queue, d_output, CL_TRUE, 0, output_size*sizeof(float), output_data, 0, NULL, NULL );
  }
  // else {
  //   err  = clSetKernelArg(kernelconv, 0, sizeof(cl_mem), &d_input);
  //   err  = clSetKernelArg(kernelconv, 1, sizeof(cl_mem), &d_filter);
  //   // err  = clSetKernelArg(kernelconv, 2, sizeof(cl_mem), &d_bias);
  //   err  = clSetKernelArg(kernelconv, 2, sizeof(cl_mem), &d_output);
  //   err  = clSetKernelArg(kernelconv, 3, sizeof(int), &stride_width);
  //   err  = clSetKernelArg(kernelconv, 4, sizeof(int), &stride_height);
  //   err  = clSetKernelArg(kernelconv, 5, sizeof(int), &pad_width);
  //   err  = clSetKernelArg(kernelconv, 6, sizeof(int), &pad_height);
  //   err  = clSetKernelArg(kernelconv, 7, sizeof(cl_mem), &d_dim_sizes);
  //   err  = clSetKernelArg(kernelconv, 8, sizeof(cl_mem), &d_dim_strides);
  //   // err  = clSetKernelArg(kernelconv, 10, sizeof(cl_float4), &omin);
  //   // err  = clSetKernelArg(kernelconv, 11, sizeof(cl_float4), &omax);

  //   //conv baru with local
  //   const size_t local[2] = { 8, 32 };
  //   const size_t global[2] = { (size_t) ((d_output_depth*batches/4-1)/8+1)*8, (size_t) ((output_width*output_height-1)/32+1)*32 };
    
  //   double wall0 = get_wall_time();
  //   double cpu0  = get_cpu_time();
    
  //   err = clEnqueueNDRangeKernel(queue, kernelconv, 2, NULL, global, local, 0, NULL, NULL);

  //   clFinish(queue);

  //   // // Stop timers
  //   double wall1 = get_wall_time();
  //   double cpu1  = get_cpu_time();

  //   double wall = wall1 - wall0;
  //   double cpu = cpu1 - cpu0;

  //   // __android_log_print(ANDROID_LOG_INFO, "Convruntime", "runkernelConverror: %d", err);

  //   // note: andoird log
  //   __android_log_print(ANDROID_LOG_INFO, "Convruntime", "runkernelOclConvNormal: %lf", wall);

  //   // clEnqueueReadBuffer(queue, d_output, CL_TRUE, 0, output_size*sizeof(float), output_data, 0, NULL, NULL );
  //   cl_float *host_result = (cl_float*)clEnqueueMapBuffer(
  //           queue,
  //           d_output,
  //           CL_TRUE,
  //           CL_MAP_READ,
  //           0,
  //           output_size/output_depth*d_output_depth*sizeof(float),
  //           0, NULL, NULL, NULL);

  //   for(int i = 0; i < output_size/output_depth; i++) {
  //     for(int j = 0; j < output_depth; j++) {
  //       output_data[i*output_depth + j] = host_result[i*d_output_depth + j];
  //     }
  //   }

  //   clEnqueueUnmapMemObject(queue,d_output,(void *) host_result,0, NULL, NULL);
  //   clFinish(queue);
  // }

      double wall11 = get_wall_time();
    double cpu11  = get_cpu_time();

    double wall1 = wall11 - wall01;
    double cpu1 = cpu11 - cpu01;

    // // __android_log_print(ANDROID_LOG_INFO, "Convruntime", "runkernelConverror: %d", err);

    // // // note: andoird log
    // __android_log_print(ANDROID_LOG_INFO, "Convruntime", "runkerneltotal: %lf", wall1);

  // else if((stride_width == 1) && (stride_height == 1) && (pad_width == 0) && (pad_height == 0)) {
  //   err  = clSetKernelArg(kernellocalall, 0, sizeof(cl_mem), &d_input);
  //   err  = clSetKernelArg(kernellocalall, 1, sizeof(cl_mem), &d_filter);
  //   err  = clSetKernelArg(kernellocalall, 2, sizeof(cl_mem), &d_bias);
  //   err  = clSetKernelArg(kernellocalall, 3, sizeof(cl_mem), &d_output);
  //   err  = clSetKernelArg(kernellocalall, 4, sizeof(int), &stride_width);
  //   err  = clSetKernelArg(kernellocalall, 5, sizeof(int), &stride_height);
  //   err  = clSetKernelArg(kernellocalall, 6, sizeof(int), &pad_width);
  //   err  = clSetKernelArg(kernellocalall, 7, sizeof(int), &pad_height);
  //   err  = clSetKernelArg(kernellocalall, 8, sizeof(cl_mem), &d_dim_sizes);
  //   err  = clSetKernelArg(kernellocalall, 9, sizeof(cl_mem), &d_dim_strides);
  //   err  = clSetKernelArg(kernellocalall, 10, sizeof(float), &output_activation_min);
  //   err  = clSetKernelArg(kernellocalall, 11, sizeof(float), &output_activation_max);

  //   //conv baru with local
  //   const size_t local[2] = { 8, 16 };
  //   const size_t global[2] = { (size_t) ((output_height-1)/8+1)*8*batches, (size_t) ((output_width-1)/16+1)*16*output_depth };
    
  //   double wall0 = get_wall_time();
  //   double cpu0  = get_cpu_time();

  //   err = clEnqueueNDRangeKernel(queue, kernellocalall, 2, NULL, global, local, 0, NULL, NULL);

  //   clFinish(queue);

  //   // // Stop timers
  //   double wall1 = get_wall_time();
  //   double cpu1  = get_cpu_time();

  //   double wall = wall1 - wall0;
  //   double cpu = cpu1 - cpu0;

  //   // note: andoird log
  //   __android_log_print(ANDROID_LOG_INFO, "Convruntime", "runkernelOclConv: %lf", wall);

  //   clEnqueueReadBuffer(queue, d_output, CL_TRUE, 0, output_size*sizeof(float), output_data, 0, NULL, NULL );
  // }
  

  // err = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global, local, 0, NULL, NULL);

  // clFinish(queue);

  // // // Stop timers
  // double wall1 = get_wall_time();
  // double cpu1  = get_cpu_time();

  // double wall = wall1 - wall0;
  // double cpu = cpu1 - cpu0;

  // // note: andoird log
  // __android_log_print(ANDROID_LOG_INFO, "Convruntime", "runkernelOclConv: %lf", wall);

  // __android_log_print(ANDROID_LOG_INFO, "Convruntime", "runkernelConverror: %d", err);

  // wall0 = get_wall_time();
  // cpu0  = get_cpu_time();

  // clEnqueueReadBuffer(queue, d_output, CL_TRUE, 0, output_size*sizeof(float), output_data, 0, NULL, NULL );
  
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

  // clFinish(queue);
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

     // Stop timers
  double wall1 = get_wall_time();
  double cpu1  = get_cpu_time();

  double wall = wall1 - wall0;
  double cpu = cpu1 - cpu0;

  // note: andoird log
  __android_log_print(ANDROID_LOG_INFO, "Convruntime", "runkernelMultithreadConv: %lf", wall);

  optimized_ops::AddBiasAndEvalActivationFunction(
      bias_data, bias_dims, output_data, output_dims, output_activation_min,
      output_activation_max);


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


      // Start Timers
  double wall0 = get_wall_time();
  double cpu0  = get_cpu_time();

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
          // float bias_value = 0.0f;
          // if (bias_data) {
          //   bias_value = bias_data[Offset(bias_dims, out_channel, 0, 0, 0)];
          // }
          output_data[Offset(output_dims, out_channel, out_x, out_y, batch)] = total;
              // ActivationFunctionWithMinMax(total + bias_value,
              //                              output_activation_min,
              //                              output_activation_max);
        }
      }
    }
  }
  // Stop timers
  double wall1 = get_wall_time();
  double cpu1  = get_cpu_time();

  double wall = wall1 - wall0;
  double cpu = cpu1 - cpu0;

  // note: andoird log
  __android_log_print(ANDROID_LOG_INFO, "Convruntime", "runkernelNaiveConv: %lf", wall);
}

// inline void ConvApik(const float* input_data, 
//           const float* filter_data, 
//           const float* bias_data, 
//           float* output_data,
//           int stride_width, int stride_height, 
//           int pad_width, int pad_height, 
//           const int* dim_sizes, const int* dim_strides,
//           float output_activation_min, float output_activation_max) {
//   const int batches = dim_sizes[3]; //MatchingArraySize(input_dims, 3, output_dims, 3);
//   const int input_depth = dim_sizes[0]; //MatchingArraySize(input_dims, 0, filter_dims, 0);
//   const int output_depth = dim_sizes[7]; //MatchingArraySize(filter_dims, 3, output_dims, 0);
//   // if (bias_data) {
//   //   TFLITE_DCHECK_EQ(ArraySize(filter_dims, 3), ArraySize(bias_dims, 0));
//   // }
//   const int input_height = dim_sizes[2]; //ArraySize(input_dims, 2);
//   const int input_width = dim_sizes[1]; //ArraySize(input_dims, 1);
//   const int filter_height = dim_sizes[6]; //ArraySize(filter_dims, 2);
//   const int filter_width = dim_sizes[5]; //ArraySize(filter_dims, 1);
//   const int output_height = dim_sizes[14]; //ArraySize(output_dims, 2);
//   const int output_width = dim_sizes[13]; //ArraySize(output_dims, 1);
//   for (int batch = 0; batch < batches; ++batch) {
//     for (int out_y = 0; out_y < output_height; ++out_y) {
//       for (int out_x = 0; out_x < output_width; ++out_x) {
//         for (int out_channel = output_depth/2; out_channel < output_depth; ++out_channel) {
//           const int in_x_origin = (out_x * stride_width) - pad_width;
//           const int in_y_origin = (out_y * stride_height) - pad_height;
//           float total = 0.0;
//           for (int filter_y = 0; filter_y < filter_height; ++filter_y) {
//             for (int filter_x = 0; filter_x < filter_width; ++filter_x) {
//               for (int in_channel = 0; in_channel < input_depth; ++in_channel) {
//                 const int in_x = in_x_origin + filter_x;
//                 const int in_y = in_y_origin + filter_y;
//                 // If the location is outside the bounds of the input image,
//                 // use zero as a default value.
//                 if ((in_x >= 0) && (in_x < input_width) && (in_y >= 0) &&
//                     (in_y < input_height)) {
//                   float input_value = input_data[in_channel*dim_strides[0] + in_x*dim_strides[1] + 
//                                                   in_y*dim_strides[2] + batch*dim_strides[3]];
//                   float filter_value =
//                       filter_data[in_channel*dim_strides[4] + filter_x*dim_strides[5] +
//                                          filter_y*dim_strides[6] + out_channel*dim_strides[7]];
//                   total += (input_value * filter_value);
//                 }
//               }
//             }
//           }
//           float bias_value = 0.0f;
//           if (bias_data) {
//             bias_value = bias_data[out_channel*dim_strides[8]];
//           }
//           output_data[out_channel*dim_strides[12] + out_x*dim_strides[13] + out_y*dim_strides[14] + batch*dim_strides[15]] 
//             = std::min(std::max(total + bias_value, output_activation_min), output_activation_max);
//         }
//       }
//     }
//   }
// }

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

inline void NeonMatrixBatchVectorMultiplyAccumulate(const float* matrix, int m_rows,
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
  __android_log_print(ANDROID_LOG_INFO, "Matmulruntime", "runkernelNEONConv1x1: %lf", wall);
}

inline void PortableMatrixBatchVectorMultiplyAccumulate(const float* matrix,
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
  __android_log_print(ANDROID_LOG_INFO, "Matmulruntime", "runkernelPortableConv1x1: %lf", wall);
}

inline void ConvOpenCL(const float* input_data, const Dims<4>& input_dims,
                 const float* filter_data, const float* filter_data2, const Dims<4>& filter_dims,
                 const float* bias_data, const Dims<4>& bias_dims,
                 int stride_width, int stride_height, int pad_width,
                 int pad_height, TfLitePadding padding,
                 float output_activation_min, float output_activation_max,
                 float* output_data, const Dims<4>& output_dims,
                 float* im2col_data, const Dims<4>& im2col_dims,
                 cl_context context_cl, cl_command_queue queue, cl_program program, cl_mem cl_mem_arr[6], int buffsizes[4],
                 VkPhysicalDevice physicalDevice, VkDevice device, VkPipeline pipelineConv, VkPipeline pipelineMatmul, VkPipelineLayout pipelineLayoutConv, VkPipelineLayout pipelineLayoutMatmul, VkPipeline pipelineConvMatmul, VkPipelineLayout pipelineLayoutConvMatmul,
    VkDescriptorSetLayout descriptorSetLayoutConv, VkDescriptorSetLayout descriptorSetLayoutMatmul, VkQueue queueV, uint32_t queueFamilyIndex,
    VkCommandPool conv_commandPool, VkCommandBuffer conv_commandBuffer, VkBuffer conv_matrixA, VkBuffer conv_matrixB, VkBuffer conv_matrixC, VkBuffer conv_matrixSizes, VkDeviceMemory conv_bufferMemory) {
  
  if((kernelconv == NULL) || (kernelmatmul == NULL) || (kernellocalfilter == NULL) || (kernellocalall == NULL)) {
    kernelconv = clCreateKernel(program, "convfloat", NULL);
    kernelmatmul = clCreateKernel(program, "convmatmulblock", NULL);
    kernellocalall = clCreateKernel(program, "convlocalall", NULL);
    kernellocalfilter = clCreateKernel(program, "convlocalfilter", NULL);
  }



  


  // untuk eksperimen 
  // Dims<4> input_dims,filter_dims,bias_dims,output_dims,im2col_dims;
  // float* im2col_data;
  // im2col_dims.sizes[0] = 0;
  // im2col_dims.sizes[1] = 0;
  // im2col_dims.sizes[2] = 0;
  // im2col_dims.sizes[3] = 0;

  // if(im2col_dims.sizes[0]*im2col_dims.sizes[1]*im2col_dims.sizes[2]*im2col_dims.sizes[3] > 10) {
  //   for(int i = 0; i < 10; i++) {
  //     __android_log_print(ANDROID_LOG_INFO, "Convruntime", "runkernelsim2coldata %d: %f", i, im2col_data[i]);
  //     im2col_data[i] = 0;
  //   }
  // }
  
  // __android_log_print(ANDROID_LOG_INFO, "Convruntime", "runkernelpadw: %d", pad_width);
  // __android_log_print(ANDROID_LOG_INFO, "Convruntime", "runkernelpadh: %d", pad_height);
  // __android_log_print(ANDROID_LOG_INFO, "Convruntime", "runkerneltfpad: %d", (int) padding);

  // int im2colsize = im2col_dims.sizes[0]*im2col_dims.sizes[1]*im2col_dims.sizes[2]*im2col_dims.sizes[3];
   // __android_log_print(ANDROID_LOG_INFO, "Convruntime", "runkernelsim2colsize0: %d", im2col_dims.sizes[0]);
   // __android_log_print(ANDROID_LOG_INFO, "Convruntime", "runkernelsim2colsize1: %d", im2col_dims.sizes[1]);
   // __android_log_print(ANDROID_LOG_INFO, "Convruntime", "runkernelsim2colsize2: %d", im2col_dims.sizes[2]);
   // __android_log_print(ANDROID_LOG_INFO, "Convruntime", "runkernelsim2colsize3: %d", im2col_dims.sizes[3]);

   // // __android_log_print(ANDROID_LOG_INFO, "Convruntime", "runkernelsim2colstride0: %d", im2col_dims.strides[0]);
   // // __android_log_print(ANDROID_LOG_INFO, "Convruntime", "runkernelsim2colstride1: %d", im2col_dims.strides[1]);
   // // __android_log_print(ANDROID_LOG_INFO, "Convruntime", "runkernelsim2colstride2: %d", im2col_dims.strides[2]);
   // // __android_log_print(ANDROID_LOG_INFO, "Convruntime", "runkernelsim2colstride3: %d", im2col_dims.strides[3]);

   //    __android_log_print(ANDROID_LOG_INFO, "Convruntime", "runkernelsinputsize0: %d", input_dims.sizes[0]);
   // __android_log_print(ANDROID_LOG_INFO, "Convruntime", "runkernelsinputsize1: %d", input_dims.sizes[1]);
   // __android_log_print(ANDROID_LOG_INFO, "Convruntime", "runkernelsinputsize2: %d", input_dims.sizes[2]);
   // __android_log_print(ANDROID_LOG_INFO, "Convruntime", "runkernelsinputsize3: %d", input_dims.sizes[3]);

   // // __android_log_print(ANDROID_LOG_INFO, "Convruntime", "runkernelsinputstride0: %d", input_dims.strides[0]);
   // // __android_log_print(ANDROID_LOG_INFO, "Convruntime", "runkernelsinputstride1: %d", input_dims.strides[1]);
   // // __android_log_print(ANDROID_LOG_INFO, "Convruntime", "runkernelsinputstride2: %d", input_dims.strides[2]);
   // // __android_log_print(ANDROID_LOG_INFO, "Convruntime", "runkernelsinputstride3: %d", input_dims.strides[3]);

   //    __android_log_print(ANDROID_LOG_INFO, "Convruntime", "runkernelsfiltersize0: %d", filter_dims.sizes[0]);
   // __android_log_print(ANDROID_LOG_INFO, "Convruntime", "runkernelsfiltersize1: %d", filter_dims.sizes[1]);
   // __android_log_print(ANDROID_LOG_INFO, "Convruntime", "runkernelsfiltersize2: %d", filter_dims.sizes[2]);
   // __android_log_print(ANDROID_LOG_INFO, "Convruntime", "runkernelsfiltersize3: %d", filter_dims.sizes[3]);

   // // __android_log_print(ANDROID_LOG_INFO, "Convruntime", "runkernelsfilterstride0: %d", filter_dims.strides[0]);
   // // __android_log_print(ANDROID_LOG_INFO, "Convruntime", "runkernelsfilterstride1: %d", filter_dims.strides[1]);
   // // __android_log_print(ANDROID_LOG_INFO, "Convruntime", "runkernelsfilterstride2: %d", filter_dims.strides[2]);
   // // __android_log_print(ANDROID_LOG_INFO, "Convruntime", "runkernelsfilterstride3: %d", filter_dims.strides[3]);

   // __android_log_print(ANDROID_LOG_INFO, "Convruntime", "runkernelsoutputsize0: %d", output_dims.sizes[0]);
   // __android_log_print(ANDROID_LOG_INFO, "Convruntime", "runkernelsoutputsize1: %d", output_dims.sizes[1]);
   // __android_log_print(ANDROID_LOG_INFO, "Convruntime", "runkernelsoutputsize2: %d", output_dims.sizes[2]);
   // __android_log_print(ANDROID_LOG_INFO, "Convruntime", "runkernelsoutputsize3: %d", output_dims.sizes[3]);

   //im2col0 = filter1xfilter2xfilter3, im2col1 = output1, im2col2 = output2, im2col3 = 1

  //   // input
  // input_dims.sizes[0] = 4;
  // input_dims.sizes[1] = 516;
  // input_dims.sizes[2] = 516;
  // input_dims.sizes[3] = 1;
  // input_dims.strides[0] = 1;
  // input_dims.strides[1] = input_dims.strides[0]*input_dims.sizes[0];
  // input_dims.strides[2] = input_dims.strides[1]*input_dims.sizes[1];
  // input_dims.strides[3] = input_dims.strides[2]*input_dims.sizes[2];

  // //filter
  // filter_dims.sizes[0] = input_dims.sizes[0];
  // filter_dims.sizes[1] = 5;
  // filter_dims.sizes[2] = 5;
  // filter_dims.sizes[3] = 4;
  // filter_dims.strides[0] = 1;
  // filter_dims.strides[1] = filter_dims.strides[0]*filter_dims.sizes[0];
  // filter_dims.strides[2] = filter_dims.strides[1]*filter_dims.sizes[1];
  // filter_dims.strides[3] = filter_dims.strides[2]*filter_dims.sizes[2];

  // //bias
  // bias_dims.sizes[0] = filter_dims.sizes[3];
  // bias_dims.sizes[1] = 1;
  // bias_dims.sizes[2] = 1;
  // bias_dims.sizes[3] = 1;
  // bias_dims.strides[0] = 1;
  // bias_dims.strides[1] = bias_dims.strides[0]*bias_dims.sizes[0];
  // bias_dims.strides[2] = bias_dims.strides[1]*bias_dims.sizes[1];
  // bias_dims.strides[3] = bias_dims.strides[2]*bias_dims.sizes[2];

  // //output
  // output_dims.sizes[0] = filter_dims.sizes[3];
  // output_dims.sizes[1] = input_dims.sizes[1] - filter_dims.sizes[1] + 1;
  // output_dims.sizes[2] = input_dims.sizes[2] - filter_dims.sizes[2] + 1;
  // output_dims.sizes[3] = input_dims.sizes[3];
  // output_dims.strides[0] = 1;
  // output_dims.strides[1] = output_dims.strides[0]*output_dims.sizes[0];
  // output_dims.strides[2] = output_dims.strides[1]*output_dims.sizes[1];
  // output_dims.strides[3] = output_dims.strides[2]*output_dims.sizes[2];

  // //im2col
  // im2col_dims.sizes[0] = filter_dims.sizes[0]*filter_dims.sizes[1]*filter_dims.sizes[2];
  // im2col_dims.sizes[1] = output_dims.sizes[1];
  // im2col_dims.sizes[2] = output_dims.sizes[2];
  // im2col_dims.sizes[3] = input_dims.sizes[3];
  // im2col_dims.strides[0] = 1;
  // im2col_dims.strides[1] = im2col_dims.strides[0]*im2col_dims.sizes[0];
  // im2col_dims.strides[2] = im2col_dims.strides[1]*im2col_dims.sizes[1];
  // im2col_dims.strides[3] = im2col_dims.strides[2]*im2col_dims.sizes[2];

  // TfLitePadding padding = kTfLitePaddingValid;
  // int stride_width = 1;
  // int stride_height = 1;
  // int pad_width = 0;
  // int pad_height = 0;

  // __android_log_print(ANDROID_LOG_INFO, "Convruntime", "runkernelstridew: %d", stride_width);
  // __android_log_print(ANDROID_LOG_INFO, "Convruntime", "runkernelstrideh: %d", stride_width);
  // __android_log_print(ANDROID_LOG_INFO, "Convruntime", "runkernelpadw: %d", pad_width);
  // __android_log_print(ANDROID_LOG_INFO, "Convruntime", "runkernelpadh: %d", pad_width);

  // int inheightsize = input_dims.sizes[2];
  // int inwidthsize = input_dims.sizes[1];
  // int indepthsize = input_dims.sizes[0];
  // int inbatchsize = input_dims.sizes[3];

  // int strides0 = input_dims.strides[0];
  // int strides1 = input_dims.strides[1];
  // int strides2 = input_dims.strides[2];
  // int strides3 = input_dims.strides[3];

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

/////////////////////////////////////////
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
  int im2col_size = im2col_dims.sizes[0]*im2col_dims.sizes[1]*im2col_dims.sizes[2]*im2col_dims.sizes[3];

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


  // untuk eksperimen
  // float* input = (float*)malloc(input_size*sizeof(float));
  // float* filter = (float*)malloc(filter_size*sizeof(float));
  // float* bias = (float*)malloc(bias_size*sizeof(float));
  // float* output = (float*)malloc(output_size*sizeof(float));
  // float* im2col = (float*)malloc(im2col_size*sizeof(float));

  // for(int i = 0; i < input_size; i++) {
  //   input[i] = 1;
  // }
  // for(int i = 0; i < filter_size; i++) {
  //   filter[i] = 1;
  // }
  // for(int i = 0; i < bias_size; i++) {
  //   bias[i] = 1;
  // }
  // for(int i = 0; i < im2col_size; i++) {
  //   im2col[i] = 1;
  // }

  // int m_cols = sizes[0];
  // int m_rows = sizes[1]*sizes[2]*sizes[3];
  // int n_batch = sizes[7];

  // OpenCLConv(input, input_size,
  //   filter, filter_size,
  //   bias, bias_size,
  //   output, output_size,
  //   stride_width, stride_height, 
  //   pad_width, pad_height, 
  //   sizes, strides,
  //   output_activation_min, output_activation_max,
  //   context_cl, queue, program, cl_mem_arr);

  // sleep(1);

  // double sum = 0.0;
  // for(int i = 0; i < output_size; i++) {
  //   sum += output[i];
  // }

  // __android_log_print(ANDROID_LOG_INFO, "Convruntime", "runkernelsumopencl: %lf", sum);

  // for(int i = 0; i < output_size; i++) {
  //   output[i] = 0;
  // }

  // NeonMatrixBatchVectorMultiplyAccumulate(input,m_rows,m_cols,filter,n_batch,output,1);
  // Conv(input, input_dims,
  //   filter, filter_dims,
  //   bias, bias_dims,
  //   stride_width, stride_height, 
  //   pad_width, pad_height, 
  //   padding,
  //   output_activation_min, output_activation_max,
  //   output, output_dims,
  //   im2col_data, im2col_dims);

  // sleep(1);

  // sum = 0.0;
  // for(int i = 0; i < output_size; i++) {
  //   sum += output[i];
  // }

  // __android_log_print(ANDROID_LOG_INFO, "Convruntime", "runkernelsummultithread: %lf", sum);

  // for(int i = 0; i < output_size; i++) {
  //   output[i] = 0;
  // }

  // PortableMatrixBatchVectorMultiplyAccumulate(input,m_rows,m_cols,filter,n_batch,output,1);
  // Conv2(input, input_dims,
  //   filter, filter_dims,
  //   bias, bias_dims,
  //   stride_width, stride_height, 
  //   pad_width, pad_height,
  //   output_activation_min, output_activation_max,
  //   output, output_dims,
  //   im2col_data, im2col_dims);

  // sleep(1);

  // sum = 0.0;
  // for(int i = 0; i < output_size; i++) {
  //   sum += output[i];
  // }

  // __android_log_print(ANDROID_LOG_INFO, "Convruntime", "runkernelsumnaive: %lf", sum);

  // for(int i = 0; i < output_size; i++) {
  //   output[i] = 0;
  // }



  // sum = 0.0;
  // for(int i = 0; i < output_size; i++) {
  //   sum += output[i];
  // }

  // __android_log_print(ANDROID_LOG_INFO, "Convruntime", "runkernelsumopencl: %lf", sum);

  // for(int i = 0; i < output_size; i++) {
  //   output[i] = 0;
  // }

  // free(input);
  // free(filter);
  // free(bias);
  // free(output);
  // free(im2col);

  if((sizes[6] == 1) && (sizes[5] == 1) && (stride_width == 1) && (stride_height == 1) && (pad_width == 0) && (pad_height == 0)) {
    OpenCLConv(input_data, input_size,
          filter_data, filter_size,
          bias_data, bias_size,
          output_data, output_size,
          stride_width, stride_height, 
          pad_width, pad_height, 
          sizes, strides,
          output_activation_min, output_activation_max,
          context_cl, queue, program, cl_mem_arr);

    optimized_ops::AddBiasAndEvalActivationFunction(
      bias_data, bias_dims, output_data, output_dims, output_activation_min,
      output_activation_max);
  }
  else if((sizes[6] < 8) && (sizes[5] < 8) && (stride_width == 1) && (stride_height == 1) && (pad_width == 0) && (pad_height == 0)) {
    OpenCLConv(input_data, input_size,
          filter_data, filter_size,
          bias_data, bias_size,
          output_data, output_size,
          stride_width, stride_height, 
          pad_width, pad_height, 
          sizes, strides,
          output_activation_min, output_activation_max,
          context_cl, queue, program, cl_mem_arr);

    optimized_ops::AddBiasAndEvalActivationFunction(
      bias_data, bias_dims, output_data, output_dims, output_activation_min,
      output_activation_max);
  }
  else {
      Conv(input_data, input_dims,
        filter_data2, filter_dims,
        bias_data, bias_dims,
        stride_width, stride_height, 
        pad_width, pad_height, 
        padding,
        output_activation_min, output_activation_max,
        output_data, output_dims,
        im2col_data, im2col_dims);
  }

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

  // if((filter_width == 1) && (filter_height == 1) && (stride_width == 1) && (stride_height == 1) && (pad_width == 0) && (pad_height == 0))
  // {__android_log_print(ANDROID_LOG_INFO, "VulkanConvDetail", "runkernelmasuksini1x1");
  // vulkanTestConv(buffsizes, input_data, input_size,
  //         filter_data, filter_size,
  //         bias_data, bias_size,
  //         output_data, output_size,
  //         stride_width, stride_height, 
  //         pad_width, pad_height, 
  //         sizes, strides,
  //         output_activation_min, output_activation_max,
  //         physicalDevice, device, pipelineConvMatmul, pipelineLayoutConvMatmul, 
  //         descriptorSetLayoutConv, queueV, queueFamilyIndex,
  //         conv_commandPool, conv_commandBuffer, conv_matrixA, conv_matrixB, conv_matrixC, conv_matrixSizes, conv_bufferMemory);
  // }
  // else
  // vulkanTestConv(buffsizes, input_data, input_size,
  //       filter_data, filter_size,
  //       bias_data, bias_size,
  //       output_data, output_size,
  //       stride_width, stride_height, 
  //       pad_width, pad_height, 
  //       sizes, strides,
  //       output_activation_min, output_activation_max,
  //       physicalDevice, device, pipelineConv, pipelineLayoutConv, 
  //       descriptorSetLayoutConv, queueV, queueFamilyIndex,
  //       conv_commandPool, conv_commandBuffer, conv_matrixA, conv_matrixB, conv_matrixC, conv_matrixSizes, conv_bufferMemory);

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
