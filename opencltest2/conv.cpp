#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <bits/stdc++.h>
#include <time.h>
#include <sys/time.h>
#include <bits/stdc++.h>

using namespace std;

template <int N>
struct Dims {
  int sizes[N];
  int strides[N];
};

template <int N>
int ArraySize(const Dims<N>& array, int index) {
  // TFLITE_DCHECK(index >= 0 && index < N);
  return array.sizes[index];
}

template <typename ArrayType1, typename ArrayType2>
int MatchingArraySize(const ArrayType1& array1, int index1,
                      const ArrayType2& array2, int index2) {
  // TFLITE_DCHECK_EQ(ArraySize(array1, index1), ArraySize(array2, index2));
  return ArraySize(array1, index1);
}

int Offset(const Dims<4>& dims, int i0, int i1, int i2, int i3) {
  // TFLITE_DCHECK(i0 >= 0 && i0 < dims.sizes[0]);
  // TFLITE_DCHECK(i1 >= 0 && i1 < dims.sizes[1]);
  // TFLITE_DCHECK(i2 >= 0 && i2 < dims.sizes[2]);
  // TFLITE_DCHECK(i3 >= 0 && i3 < dims.sizes[3]);
  return i0 * dims.strides[0] + i1 * dims.strides[1] + i2 * dims.strides[2] +
         i3 * dims.strides[3];
}

float ActivationFunctionWithMinMax(float x, float output_activation_min,
                                          float output_activation_max) {
  return std::min(std::max(x, output_activation_min), output_activation_max);
}

void ConvAsli(const float* input_data, const Dims<4>& input_dims,
                 const float* filter_data, const Dims<4>& filter_dims,
                 const float* bias_data, const Dims<4>& bias_dims,
                 int stride_width, int stride_height, int pad_width,
                 int pad_height, float output_activation_min,
                 float output_activation_max, float* output_data,
                 const Dims<4>& output_dims) {
                // , float* im2col_data,
                //  const Dims<4>& im2col_dims) {
  // (void)im2col_data;  // only used in optimized code.
  // (void)im2col_dims;  // only used in optimized code.
  const int batches = MatchingArraySize(input_dims, 3, output_dims, 3);
  const int input_depth = MatchingArraySize(input_dims, 0, filter_dims, 0);
  const int output_depth = MatchingArraySize(filter_dims, 3, output_dims, 0);
  // if (bias_data) {
  //   TFLITE_DCHECK_EQ(ArraySize(filter_dims, 3), ArraySize(bias_dims, 0));
  // }
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

const char *kernelSource =           "\n" \
"__kernel void convdef(__global float4* input_data,    \n" \
"          __constant float4* filter_data,    \n" \
"          __global float* bias_data,    \n" \
"          __global float* output_data,   \n" \
"          int stride_width, int stride_height,    \n" \
"          int pad_width, int pad_height,    \n" \
"          __global int16* dim_sizes0, __global int16* dim_strides0,   \n" \
"          float output_activation_min, float output_activation_max) {   \n" \
"     \n" \
"    int batchdepth = get_global_id(0);   \n" \
"    int widthheight = get_global_id(1);   \n" \
"    int16 dim_sizes = dim_sizes0[0];   \n" \
"    int output_depth = dim_sizes.s7;  \n" \
"    int output_height = dim_sizes.se;   \n" \
"    int batch = batchdepth/output_depth;   \n" \
"    int out_channel = batchdepth\%output_depth;   \n" \
"    int out_x = widthheight/output_height;   \n" \
"    int out_y = widthheight\%output_height;   \n" \
"    if((batch < dim_sizes.s3) && (out_x < dim_sizes.sd) && (out_y < output_height) && (out_channel < output_depth)) {   \n" \
"            int16 dim_strides = dim_strides0[0];   \n" \
"            float total = 0.0;   \n" \
"            for (int filter_y = 0; filter_y < dim_sizes.s6; ++filter_y) {   \n" \
"              for (int filter_x = 0; filter_x < dim_sizes.s5; ++filter_x) {   \n" \
"                for (int in_channel = 0; in_channel < dim_sizes.s0/4; ++in_channel) {   \n" \
"                  int in_x = (out_x * stride_width) - pad_width + filter_x;   \n" \
"                  int in_y = (out_y * stride_height) - pad_height + filter_y;   \n" \
"                  if ((in_x >= 0) && (in_x < dim_sizes.s1) && (in_y >= 0) &&   \n" \
"                      (in_y < dim_sizes.s2)) {   \n" \
"                    float4 input_value = input_data[in_channel*dim_strides.s0 + in_x*dim_strides.s1/4 + in_y*dim_strides.s2/4 + batch*dim_strides.s3/4];   \n" \
"                    float4 filter_value = filter_data[in_channel*dim_strides.s4 + filter_x*dim_strides.s5/4 + filter_y*dim_strides.s6/4 + out_channel*dim_strides.s7/4];  \n" \
"                    total += dot(input_value,filter_value);   \n" \
"                  }   \n" \
"                }   \n" \
"              }   \n" \
"            }   \n" \
"            float bias_value = 0.0;   \n" \
"            if (bias_data) {   \n" \
"              bias_value = bias_data[out_channel*dim_strides.s8];   \n" \
"            } \n" \
"            output_data[out_channel*dim_strides.sc + out_x*dim_strides.sd + out_y*dim_strides.se + batch*dim_strides.sf] = total + bias_value; \n" \
"    }  \n" \
"}   \n" \
"__kernel void convterbaik(__global float4* input_data,    \n" \
"          __constant float4* filter_data,    \n" \
"          __global float* bias_data,    \n" \
"          __global float* output_data,   \n" \
"          int stride_width, int stride_height,    \n" \
"          int pad_width, int pad_height,    \n" \
"          __global int16* dim_sizes0, __global int16* dim_strides0,   \n" \
"          float output_activation_min, float output_activation_max) {   \n" \
"     \n" \
"    int batchdepth = get_global_id(0);   \n" \
"    int widthheight = get_global_id(1);   \n" \
"    int local_y = get_local_id(0);   \n" \
"    int local_x = get_local_id(1);   \n" \
"    int16 dim_sizes = dim_sizes0[0];   \n" \
"    int filter_size_y = dim_sizes.s6;   \n" \
"    int filter_size_x = dim_sizes.s5;   \n" \
"    __local float4 localfilter[8][8]; \n" \
"    int output_depth = dim_sizes.s7;  \n" \
"    int output_height = dim_sizes.se;   \n" \
"    int batch = batchdepth/output_depth;   \n" \
"    int out_channel = batchdepth\%output_depth;   \n" \
"    int out_x = widthheight/output_height;   \n" \
"    int out_y = widthheight\%output_height;   \n" \
"    if((batch < dim_sizes.s3) && (out_x < dim_sizes.sd) && (out_y < output_height) && (out_channel < output_depth)) {   \n" \
"            int16 dim_strides = dim_strides0[0];   \n" \
"            float total = 0.0;   \n" \
"            for (int in_channel = 0; in_channel < dim_sizes.s0/4; ++in_channel) {   \n" \
"              if((local_y < filter_size_y) && (local_x < filter_size_x)) {   \n" \
"                localfilter[local_y][local_x] = filter_data[in_channel*dim_strides.s4 + local_x*dim_strides.s5/4 + local_y*dim_strides.s6/4 + out_channel*dim_strides.s7/4];   \n" \
"              }   \n" \
"              barrier(CLK_LOCAL_MEM_FENCE);   \n" \
"              for (int filter_y = 0; filter_y < dim_sizes.s6; ++filter_y) {   \n" \
"                for (int filter_x = 0; filter_x < dim_sizes.s5; ++filter_x) {   \n" \
"                  int in_x = (out_x * stride_width) - pad_width + filter_x;   \n" \
"                  int in_y = (out_y * stride_height) - pad_height + filter_y;   \n" \
"                  if ((in_x >= 0) && (in_x < dim_sizes.s1) && (in_y >= 0) &&   \n" \
"                      (in_y < dim_sizes.s2)) {   \n" \
"                    float4 input_value = input_data[in_channel*dim_strides.s0 + in_x*dim_strides.s1/4 + in_y*dim_strides.s2/4 + batch*dim_strides.s3/4];   \n" \
"                    float4 filter_value = localfilter[filter_y][filter_x];  \n" \
"                    total += dot(input_value,filter_value);   \n" \
"                  }   \n" \
"                }   \n" \
"              }   \n" \
"            }   \n" \
"            float bias_value = 0.0;   \n" \
"            if (bias_data) {   \n" \
"              bias_value = bias_data[out_channel*dim_strides.s8];   \n" \
"            } \n" \
"            output_data[out_channel*dim_strides.sc + out_x*dim_strides.sd + out_y*dim_strides.se + batch*dim_strides.sf] = total + bias_value; \n" \
"    }  \n" \
"}   \n" \
"__kernel void convlocal(__global float4* input_data,    \n" \
"          __constant float4* filter_data,    \n" \
"          __global float* bias_data,    \n" \
"          __global float* output_data,   \n" \
"          int stride_width, int stride_height,    \n" \
"          int pad_width, int pad_height,    \n" \
"          __global int16* dim_sizes0, __global int16* dim_strides0,   \n" \
"          float output_activation_min, float output_activation_max) {   \n" \
"     \n" \
"    int local_y = get_local_id(0);   \n" \
"    int local_x = get_local_id(1);   \n" \
"    int ychannel = get_global_id(0);   \n" \
"    int xchannel = get_global_id(1);   \n" \
"    int16 dim_sizes = dim_sizes0[0];   \n" \
"    int16 dim_strides = dim_strides0[0];   \n" \
"    int xsize = ((dim_sizes.sd-1)/16+1)*16;   \n" \
"    int ysize = ((dim_sizes.se-1)/8+1)*8;   \n" \
"    int out_x = xchannel\%xsize;   \n" \
"    int out_y = ychannel\%ysize;   \n" \
"    int out_channel = xchannel/xsize;   \n" \
"    int batch = ychannel/ysize;   \n" \
"    __local float4 localfilter[8][8]; \n" \
"    //__local float4 localinput[16][24]; \n" \
"      float total = 0.0;   \n" \
"      for (int in_channel = 0; in_channel < dim_sizes.s0/4; ++in_channel) {   \n" \
"        barrier(CLK_LOCAL_MEM_FENCE);   \n" \
"        //load input     \n" \
"        //if((out_x < dim_sizes.s1) && (out_y < dim_sizes.s2))\n" \
"         // localinput[local_y][local_x] = input_data[in_channel*dim_strides.s0 + out_x*dim_strides.s1/4 + out_y*dim_strides.s2/4 + batch*dim_strides.s3/4];\n" \
"        //if((out_x < dim_sizes.s1) && ((out_y+8) < dim_sizes.s2) && (local_y < dim_sizes.s6))\n" \
"        //  localinput[local_y+8][local_x] = input_data[in_channel*dim_strides.s0 + out_x*dim_strides.s1/4 + (out_y+8)*dim_strides.s2/4 + batch*dim_strides.s3/4];\n" \
"        //if(((out_x+16) < dim_sizes.s1) && (out_y < dim_sizes.s2) && (local_x < dim_sizes.s5))\n" \
"         // localinput[local_y][local_x+16] = input_data[in_channel*dim_strides.s0 + (out_x+16)*dim_strides.s1/4 + out_y*dim_strides.s2/4 + batch*dim_strides.s3/4];\n" \
"        //if(((out_x+16) < dim_sizes.s1) && ((out_y+8) < dim_sizes.s2) && (local_x < dim_sizes.s5) && (local_y < dim_sizes.s6))\n" \
"         // localinput[local_y+8][local_x+16] = input_data[in_channel*dim_strides.s0 + (out_x+16)*dim_strides.s1/4 + (out_y+8)*dim_strides.s2/4 + batch*dim_strides.s3/4];\n" \
"        //load filter  \n" \
"        if((local_y < dim_sizes.s6) && (local_x < dim_sizes.s5)) {   \n" \
"          localfilter[local_y][local_x] = filter_data[in_channel*dim_strides.s4 + local_x*dim_strides.s5/4 + local_y*dim_strides.s6/4 + out_channel*dim_strides.s7/4];   \n" \
"        }   \n" \
"        barrier(CLK_LOCAL_MEM_FENCE);   \n" \
"        if((out_x < dim_sizes.sd) && (out_y < dim_sizes.se)) {   \n" \
"          for (int filter_y = 0; filter_y < dim_sizes.s6; ++filter_y) {   \n" \
"            for (int filter_x = 0; filter_x < dim_sizes.s5; ++filter_x) {   \n" \
"              //int in_x = (local_x * stride_width) - pad_width + filter_x;   \n" \
"              //int in_y = (local_y * stride_height) - pad_height + filter_y;\n" \
"              int in_x = (out_x * stride_width) - pad_width + filter_x;   \n" \
"              int in_y = (out_y * stride_height) - pad_height + filter_y;   \n" \
"              if ((in_x >= 0) && (in_x < dim_sizes.s1) && (in_y >= 0) &&   \n" \
"                  (in_y < dim_sizes.s2)) {   \n" \
"                // float4 input_value = localinput[in_y][in_x];\n" \
"                float4 input_value = input_data[in_channel*dim_strides.s0 + in_x*dim_strides.s1/4 + in_y*dim_strides.s2/4 + batch*dim_strides.s3/4];   \n" \
"                float4 filter_value = localfilter[filter_y][filter_x];  \n" \
"                // float4 filter_value = filter_data[in_channel*dim_strides.s4 + filter_x*dim_strides.s5/4 + filter_y*dim_strides.s6/4 + out_channel*dim_strides.s7/4];  \n" \
"                total += dot(input_value,filter_value);   \n" \
"              }   \n" \
"            }   \n" \
"          }   \n" \
"        }   \n" \
"      }  \n" \
"      if((out_x < dim_sizes.sd) && (out_y < dim_sizes.se)) {   \n" \
"        float bias_value = 0.0;   \n" \
"        if (bias_data) {   \n" \
"          bias_value = bias_data[out_channel*dim_strides.s8];   \n" \
"        } \n" \
"        output_data[out_channel*dim_strides.sc + out_x*dim_strides.sd + out_y*dim_strides.se + batch*dim_strides.sf] = total + bias_value; \n" \
"      }   \n" \
"} \n" \
"__kernel void convmatmul4(__global float4* input_data,    \n" \
"          __constant float4* filter_data,    \n" \
"          __global float* bias_data,    \n" \
"          __global float4* output_data,   \n" \
"          int stride_width, int stride_height,    \n" \
"          int pad_width, int pad_height,    \n" \
"          __global int16* dim_sizes0, __global int16* dim_strides0,   \n" \
"          float output_activation_min, float output_activation_max) {   \n" \
"    int row = get_global_id(1); \n" \
"    int col = get_global_id(0)*4; \n" \
"    int16 dim_sizes = dim_sizes0[0];   \n" \
"    int16 dim_strides = dim_strides0[0];   \n" \
"    int m_cols = dim_sizes.s0; \n" \
"    int m_rows = dim_sizes.s1*dim_sizes.s2*dim_sizes.s3; \n" \
"    int n_batch = dim_sizes.s7; \n" \
"    if ((row < m_rows) && (col < n_batch)) { \n" \
"        float4 sum = {0.0, 0.0, 0.0, 0.0}; \n" \
"        for(int i = 0; i < m_cols/4; i++){ \n" \
"            sum.x += dot(input_data[row*m_cols/4 + i],filter_data[col*m_cols/4 + i]);\n" \
"            if ((col+1) < n_batch)\n" \
"            sum.y += dot(input_data[row*m_cols/4 + i],filter_data[(col+1)*m_cols/4 + i]);\n" \
"            if ((col+2) < n_batch)\n" \
"            sum.z += dot(input_data[row*m_cols/4 + i],filter_data[(col+2)*m_cols/4 + i]);\n" \
"            if ((col+3) < n_batch);\n" \
"            sum.w += dot(input_data[row*m_cols/4 + i],filter_data[(col+3)*m_cols/4 + i]);\n" \
"        } \n" \
"        float4 bias_value = {0.0, 0.0, 0.0, 0.0};\n" \
"        if (bias_data) {   \n" \
"          bias_value.x = bias_data[col*dim_strides.s8];   \n" \
"          if ((col+1) < n_batch)\n" \
"          bias_value.y = bias_data[(col+1)*dim_strides.s8];\n" \
"          if ((col+2) < n_batch)\n" \
"          bias_value.z = bias_data[(col+2)*dim_strides.s8];\n" \
"          if ((col+3) < n_batch);\n" \
"          bias_value.w = bias_data[(col+3)*dim_strides.s8];\n" \
"        } \n" \
"        output_data[row*((n_batch-1)/4+1) + (col/4)] = sum + bias_value;\n" \
"    } \n" \
"} \n" \
"__kernel void convmatmul(__global float4* input_data,    \n" \
"          __constant float4* filter_data,    \n" \
"          __global float* bias_data,    \n" \
"          __global float* output_data,   \n" \
"          int stride_width, int stride_height,    \n" \
"          int pad_width, int pad_height,    \n" \
"          __global int16* dim_sizes0, __global int16* dim_strides0,   \n" \
"          float output_activation_min, float output_activation_max) {   \n" \
"    int row = get_global_id(1); \n" \
"    int col = get_global_id(0); \n" \
"    int16 dim_sizes = dim_sizes0[0];   \n" \
"    int16 dim_strides = dim_strides0[0];   \n" \
"    int m_cols = dim_sizes.s0; \n" \
"    int m_rows = dim_sizes.s1*dim_sizes.s2*dim_sizes.s3; \n" \
"    int n_batch = dim_sizes.s5*dim_sizes.s6*dim_sizes.s7; \n" \
"    if ((row < m_rows) && (col < n_batch)) { \n" \
"        float sum = 0.0; \n" \
"        for(int i = 0; i < m_cols/4; i++){ \n" \
"            sum += dot(input_data[row*m_cols/4 + i],filter_data[col*m_cols/4 + i]);\n" \
"        } \n" \
"        float bias_value = 0.0;   \n" \
"        if (bias_data) {   \n" \
"          bias_value = bias_data[col*dim_strides.s8];   \n" \
"        } \n" \
"        output_data[row*n_batch + col] = sum + bias_value;\n" \
"    } \n" \
"} \n" \
"__kernel void convmatmullocal(__global float4* input_data,    \n" \
"          __constant float4* filter_data,    \n" \
"          __global float* bias_data,    \n" \
"          __global float* output_data,   \n" \
"          int stride_width, int stride_height,    \n" \
"          int pad_width, int pad_height,    \n" \
"          __global int16* dim_sizes0, __global int16* dim_strides0,   \n" \
"          float output_activation_min, float output_activation_max) {   \n" \
"    int rowcol = get_global_id(0); \n" \
"    int localidx0 = get_local_id(0); \n" \
"    int localidx = get_local_id(1); \n" \
"    int16 dim_sizes = dim_sizes0[0];   \n" \
"    int16 dim_strides = dim_strides0[0];   \n" \
"    int m_cols = dim_sizes.s0; \n" \
"    int m_rows = dim_sizes.s1*dim_sizes.s2*dim_sizes.s3; \n" \
"    int n_batch = dim_sizes.s5*dim_sizes.s6*dim_sizes.s7; \n" \
"    int row = rowcol/dim_sizes.s7; \n" \
"    int col = rowcol\%dim_sizes.s7; \n" \
"    __local float Aacc[8][32]; \n" \
"    if ((row < m_rows) && (col < n_batch)) { \n" \
"        float sum = 0.0; \n" \
"        int interval = (m_cols/4-1)/32+1; \n" \
"        int starti = localidx*interval; \n" \
"        for(int i = starti; i < min(m_cols/4,(starti+interval)); i++){ \n" \
"            sum += dot(input_data[row*m_cols/4 + i],filter_data[col*m_cols/4 + i]);\n" \
"        } \n" \
"        Aacc[localidx0][localidx] = sum; \n" \
"        barrier(CLK_LOCAL_MEM_FENCE);    \n" \
"        if(localidx == 0) { \n" \
"          float total = 0.0;    \n" \
"          for(int i = 0; i < 32; i++) {    \n" \
"            total += Aacc[localidx0][i]; \n" \
"          } \n" \
"          float bias_value = 0.0;   \n" \
"          if (bias_data) {   \n" \
"            bias_value = bias_data[col*dim_strides.s8];   \n" \
"          } \n" \
"          output_data[row*n_batch + col] = total + bias_value;\n" \
"        } \n" \
"    } \n" \
"} \n" \
"__kernel void convmatmulblock(__global float4* input_data,    \n" \
"          __constant float4* filter_data,    \n" \
"          __global float* bias_data,    \n" \
"          __global float4* output_data,   \n" \
"          int stride_width, int stride_height,    \n" \
"          int pad_width, int pad_height,    \n" \
"          __global int16* dim_sizes0, __global int16* dim_strides0,   \n" \
"          float output_activation_min, float output_activation_max) {   \n" \
"    const int row = get_local_id(1);  \n" \
"    const int col = get_local_id(0);  \n" \
"    const int globalRow = 32*get_group_id(1) + row;   \n" \
"    const int globalCol0 = 32*get_group_id(0) + row;  \n" \
"    const int globalCol = (8*get_group_id(0) + col)*4;    \n" \
"    int16 dim_sizes = dim_sizes0[0];   \n" \
"    int16 dim_strides = dim_strides0[0];   \n" \
"    int m_cols = dim_sizes.s0; \n" \
"    int m_rows = dim_sizes.s1*dim_sizes.s2*dim_sizes.s3; \n" \
"    int n_batch = dim_sizes.s7; \n" \
"    __local float4 Asub[32][8];      \n" \
"    __local float4 Bsub[8][32];      \n" \
"    float4 acc = { 0.0, 0.0, 0.0, 0.0 };      \n" \
"    const int numTiles = (m_cols/4-1)/8+1;      \n" \
"    for(int t = 0; t < numTiles; t++) {      \n" \
"        int tiledColRow = t*8 + col;       \n" \
"        if((globalRow < m_rows) && (tiledColRow < m_cols/4)) {     \n" \
"           Asub[row][col] = input_data[globalRow*(m_cols/4) + tiledColRow];     \n" \
"        }      \n" \
"        else {     \n" \
"           float4 tmp = { 0.0, 0.0, 0.0, 0.0 };  \n" \
"           Asub[row][col] = tmp;     \n" \
"        }     \n" \
"        if((globalCol0 < n_batch) && (tiledColRow < m_cols/4)) {     \n" \
"           Bsub[col][row] = filter_data[globalCol0*(m_cols/4) + tiledColRow];     \n" \
"        }      \n" \
"        else {     \n" \
"           float4 tmp = { 0.0, 0.0, 0.0, 0.0 };  \n" \
"           Bsub[col][row] = tmp;     \n" \
"        }     \n" \
"        barrier(CLK_LOCAL_MEM_FENCE);      \n" \
"       \n" \
"        for(int k = 0; k < 8; k++) { \n" \
"          float4 valA = Asub[row][k]; \n" \
"          acc.x += dot(valA,Bsub[k][col*4+0]); \n" \
"          acc.y += dot(valA,Bsub[k][col*4+1]); \n" \
"          acc.z += dot(valA,Bsub[k][col*4+2]); \n" \
"          acc.w += dot(valA,Bsub[k][col*4+3]); \n" \
"        } \n" \
"        barrier(CLK_LOCAL_MEM_FENCE);      \n" \
"    }   \n" \
"    \n" \
"    float4 bias_value = {0.0, 0.0, 0.0, 0.0};\n" \
"    if (bias_data) {   \n" \
"      if (globalCol < n_batch) \n" \
"        bias_value.x = bias_data[globalCol*dim_strides.s8];   \n" \
"      if ((globalCol+1) < n_batch) \n" \
"        bias_value.y = bias_data[(globalCol+1)*dim_strides.s8];\n" \
"      if ((globalCol+2) < n_batch) \n" \
"        bias_value.z = bias_data[(globalCol+2)*dim_strides.s8];\n" \
"      if ((globalCol+3) < n_batch) \n" \
"        bias_value.w = bias_data[(globalCol+3)*dim_strides.s8];\n" \
"    } \n" \
"    if((globalCol < n_batch) && (globalRow < m_rows)) {     \n" \
"      output_data[globalRow*((n_batch-1)/4+1) + globalCol/4] = acc + bias_value;     \n" \
"    }      \n" \
"}      \n" \
"\n";


// "__kernel void convlocal(__global float4* input_data,    \n" \
// "          __constant float4* filter_data,    \n" \
// "          __global float* bias_data,    \n" \
// "          __global float* output_data,   \n" \
// "          int stride_width, int stride_height,    \n" \
// "          int pad_width, int pad_height,    \n" \
// "          __global int16* dim_sizes0, __global int16* dim_strides0,   \n" \
// "          float output_activation_min, float output_activation_max) {   \n" \
// "     \n" \
// "    int local_y = get_local_id(0);   \n" \
// "    int local_x = get_local_id(1);   \n" \
// "    int out_y = get_global_id(0);   \n" \
// "    int out_x = get_global_id(1);   \n" \
// "    int filter_size_y = dim_sizes.s6;   \n" \
// "    int filter_size_x = dim_sizes.s5;   \n" \
// "    int16 dim_sizes = dim_sizes0[0];   \n" \
// "    __local float4 localfilter[filter_size_y][filter_size_x]; \n" \
// "    if((out_x < dim_sizes.sd) && (out_y < dim_sizes.s7)) {   \n" \
// "        int16 dim_strides = dim_strides0[0];   \n" \
// "        for (int batch = 0; batch < dim_sizes.s3; ++batch) { \n" \
// "          for (int out_channel = 0; out_channel < dim_sizes.s7; ++out_channel) { \n" \
// "            float total = 0.0;   \n" \
// "            for (int in_channel = 0; in_channel < dim_sizes.s0/4; ++in_channel) {   \n" \
// "              if((local_y < filter_size_y) && (local_x < filter_size_x)) {   \n" \
// "                localfilter[local_y][local_x] = filter_data[in_channel*dim_strides.s4 + local_x*dim_strides.s5/4 + local_y*dim_strides.s6/4 + out_channel*dim_strides.s7/4];   \n" \
// "              }   \n" \
// "              barrier(CLK_LOCAL_MEM_FENCE);   \n" \
// "              for (int filter_y = 0; filter_y < filter_size_y; ++filter_y) {   \n" \
// "                for (int filter_x = 0; filter_x < filter_size_x; ++filter_x) {   \n" \
// "                  int in_x = (out_x * stride_width) - pad_width + filter_x;   \n" \
// "                  int in_y = (out_y * stride_height) - pad_height + filter_y;   \n" \
// "                  if ((in_x >= 0) && (in_x < dim_sizes.s1) && (in_y >= 0) &&   \n" \
// "                      (in_y < dim_sizes.s2)) {   \n" \
// "                    float4 input_value = input_data[in_channel*dim_strides.s0 + in_x*dim_strides.s1/4 + in_y*dim_strides.s2/4 + batch*dim_strides.s3/4];   \n" \
// "                    float4 filter_value = localfilter[in_channel*dim_strides.s4 + filter_x*dim_strides.s5/4 + filter_y*dim_strides.s6/4 + out_channel*dim_strides.s7/4];  \n" \
// "                    total += dot(input_value,filter_value);   \n" \
// "                  }   \n" \
// "                }   \n" \
// "              }   \n" \
// "            }   \n" \
// "            float bias_value = 0.0;   \n" \
// "            if (bias_data) {   \n" \
// "              bias_value = bias_data[out_channel*dim_strides.s8];   \n" \
// "            } \n" \
// "            output_data[out_channel*dim_strides.sc + out_x*dim_strides.sd + out_y*dim_strides.se + batch*dim_strides.sf] = min(max(total + bias_value, output_activation_min), output_activation_max); \n" \
// "          }  \n" \
// "        }  \n" \
// "    }  \n" \
// "}   \n" \

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

void Conv(const float* input_data, 
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
            = total + bias_value;
        }
      }
    }
  }
}

// void Test(float* input_data,   
//           float* filter_data,   
//           float* bias_data,   
//           float* output_data,  
//           int stride_width, int stride_height,   
//           int pad_width, int pad_height,   
//           int* dim_sizes, int* dim_strides,  
//           float output_activation_min, float output_activation_max) {  
//   int out_channel = 1;
//   int out_y = 2;  
//   int out_x = 3;  
//   if((out_channel < dim_sizes[7]) && (out_y < dim_sizes[14]) && (out_x < dim_sizes[13])) {  
//       for (int batch = 0; batch < dim_sizes[3]; ++batch) { 
//         for (int filter_y = 0; filter_y < dim_sizes[6]; ++filter_y) {  
//           for (int filter_x = 0; filter_x < dim_sizes[5]; ++filter_x) {  
//             for (int in_channel = 0; in_channel < dim_sizes[0]; ++in_channel) {  
//               int in_x = (out_x * stride_width) - pad_width + filter_x;  
//               int in_y = (out_y * stride_height) - pad_height + filter_y;  
//               if ((in_x >= 0) && (in_x < dim_sizes[1]) && (in_y >= 0) &&  
//                   (in_y < dim_sizes[2])) {  
//                 float input_value = input_data[in_channel*dim_strides[0] + in_x*dim_strides[1] +   
//                                                 in_y*dim_strides[2] + batch*dim_strides[3]];  
//                 float filter_value =  
//                     filter_data[in_channel*dim_strides[4] + filter_x*dim_strides[5] +  
//                                        filter_y*dim_strides[6] + out_channel*dim_strides[7]];  
//                 total += (input_value * filter_value);  
//               }  
//             }  
//           }  
//         }  
//         float bias_value = 0.0f;  
//         if (bias_data) {  
//           bias_value = bias_data[out_channel*dim_strides[8]];  
//         }  
//         float max = total+bias_value; 
//         if(max < output_activation_min) max = output_activation_min; 
//         float min = max; 
//         if(min > output_activation_max) min = output_activation_max; 
//         output_data[out_channel*dim_strides[12] + out_x*dim_strides[13] +   
//                      out_y*dim_strides[14] + batch*dim_strides[15]] = min; 
//       }  
//   }  
// }

void OpenCLConv(const float* input_data, const int input_size,
          const float* filter_data, const int filter_size,
          const float* bias_data, const int bias_size,
          float* output_data, const int output_size,
          int stride_width, int stride_height, 
          int pad_width, int pad_height, 
          const int* dim_sizes, const int* dim_strides,
          float output_activation_min, float output_activation_max,
          cl_context context, cl_command_queue queue, cl_program program) {
  
  // float* input2 = (float*)malloc(input_size*sizeof(float));
  // float* filter2 = (float*)malloc(filter_size*sizeof(float));
  // float* bias2 = (float*)malloc(bias_size*sizeof(float));
  // int* dimsizes2 = (int*)malloc(16*sizeof(int)); 
  // int* dimstrides2 = (int*)malloc(16*sizeof(int));   

  cl_mem d_input;
  cl_mem d_filter;
  cl_mem d_bias;
  cl_mem d_output;
  cl_mem d_dim_sizes;
  cl_mem d_dim_strides;
  cl_kernel kernel;

  int batches = dim_sizes[3];
  int output_depth = dim_sizes[7];
  int output_height = dim_sizes[14];  
  int output_width = dim_sizes[13];

  float* output_data2 = (float*)malloc(output_size*sizeof(float));

  // cout << "Output h w: " << output_height << " " << output_width << endl;

  //conv peringkat 2
  // const size_t local[2] = { 8, 32 };
  // const size_t global[2] = { (size_t) ((output_depth*batches-1)/8+1)*8, (size_t) ((output_height*output_width-1)/32+1)*32 };

  //conv matmul local
  // const size_t local[2] = { 8, 32 };
  // const size_t global[2] = { (size_t) ((output_height*output_width*batches*output_depth-1)/8+1)*8, 32 };

  int d_output_depth = (((output_depth-1)/4+1)*4);

  //conv matmul
  // const size_t local[2] = { 8, 32 };
  // const size_t global[2] = { (size_t) ((output_depth-1)/8+1)*8, (size_t) ((output_height*output_width*batches-1)/32+1)*32 };

  //conv matmul4
  const size_t local[2] = { 8, 32 };
  const size_t global[2] = { (size_t) ((d_output_depth/4-1)/8+1)*8, (size_t) ((output_height*output_width*batches-1)/32+1)*32 };

  //conv baru
  // const size_t local[2] = { 32, 8 };
  // const size_t global[2] = { (size_t) ((output_depth*batches*output_height*output_width-1)/32+1)*32, 8 };

  //conv baru with local
  // const size_t local[2] = { 8, 16 };
  // const size_t global[2] = { (size_t) ((output_height-1)/8+1)*8*batches, (size_t) ((output_width-1)/16+1)*16*output_depth };

  cl_int err;

  kernel = clCreateKernel(program, "convmatmul4", &err);

  d_input = clCreateBuffer(context, CL_MEM_READ_ONLY, input_size*sizeof(float), NULL, NULL);
  d_filter = clCreateBuffer(context, CL_MEM_READ_ONLY, filter_size*sizeof(float), NULL, NULL);
  d_bias = clCreateBuffer(context, CL_MEM_READ_ONLY, bias_size*sizeof(float), NULL, NULL);
  d_output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, output_size/output_depth*d_output_depth*sizeof(float), NULL, NULL);
  // d_output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, output_size*sizeof(float), NULL, NULL);
  d_dim_sizes = clCreateBuffer(context, CL_MEM_READ_ONLY, 16*sizeof(int), NULL, NULL);
  d_dim_strides = clCreateBuffer(context, CL_MEM_READ_ONLY, 16*sizeof(int), NULL, NULL);

  err = clEnqueueWriteBuffer(queue, d_input, CL_TRUE, 0,
                                 input_size*sizeof(float), input_data, 0, NULL, NULL);
  clFinish(queue);
  err = clEnqueueWriteBuffer(queue, d_filter, CL_TRUE, 0,
                                 filter_size*sizeof(float), filter_data, 0, NULL, NULL);
  clFinish(queue);
  err = clEnqueueWriteBuffer(queue, d_bias, CL_TRUE, 0,
                                 bias_size*sizeof(float), bias_data, 0, NULL, NULL);
  clFinish(queue);
  err = clEnqueueWriteBuffer(queue, d_dim_sizes, CL_TRUE, 0,
                                 16*sizeof(int), dim_sizes, 0, NULL, NULL);
  clFinish(queue);
  err = clEnqueueWriteBuffer(queue, d_dim_strides, CL_TRUE, 0,
                                 16*sizeof(int), dim_strides, 0, NULL, NULL);

  clFinish(queue);

  // size_t argsize = sizeof(d_input)+sizeof(d_filter)+sizeof(d_bias)+sizeof(d_output)+sizeof(int)
  //                   +sizeof(int)+sizeof(int)+sizeof(int)+sizeof(d_dim_sizes)+sizeof(d_dim_strides)
  //                   +sizeof(float)+sizeof(float);
  // cout << "Arg Size: " << argsize << endl;

  err  = clSetKernelArg(kernel, 0, sizeof(d_input), &d_input);
  err  = clSetKernelArg(kernel, 1, sizeof(d_filter), &d_filter);
  err  = clSetKernelArg(kernel, 2, sizeof(d_bias), &d_bias);
  err  = clSetKernelArg(kernel, 3, sizeof(d_output), &d_output);
  err  = clSetKernelArg(kernel, 4, sizeof(int), &stride_width);
  err  = clSetKernelArg(kernel, 5, sizeof(int), &stride_height);
  err  = clSetKernelArg(kernel, 6, sizeof(int), &pad_width);
  err  = clSetKernelArg(kernel, 7, sizeof(int), &pad_height);
  err  = clSetKernelArg(kernel, 8, sizeof(d_dim_sizes), &d_dim_sizes);
  err  = clSetKernelArg(kernel, 9, sizeof(d_dim_strides), &d_dim_strides);
  err  = clSetKernelArg(kernel, 10, sizeof(float), &output_activation_min);
  err  = clSetKernelArg(kernel, 11, sizeof(float), &output_activation_max);

  //  Start Timers
  double wall0 = get_wall_time();
  double cpu0  = get_cpu_time();

  err = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global, local, 0, NULL, NULL);
  // Conv(input_data, filter_data, bias_data, output_data2, 1, 1, 0, 0, dim_sizes, dim_strides, 0.0, 1000.0);

  clFinish(queue);

     //  Stop timers
  double wall1 = get_wall_time();
  double cpu1  = get_cpu_time();  

  cout << "Wall Time runkernel = " << wall1 - wall0 << endl;
  cout << "CPU Time runkernel = " << cpu1  - cpu0  << endl;
  
  cout << "Kernel error: " << err << endl;

  // clEnqueueReadBuffer(queue, d_output, CL_TRUE, 0, output_size*sizeof(float), output_data, 0, NULL, NULL );
  
  cl_float *host_result = (cl_float*)clEnqueueMapBuffer(
              queue,
              d_output,
              CL_TRUE,
              CL_MAP_READ,
              0,
              output_size/output_depth*(((output_depth-1)/4+1)*4)*sizeof(float),
              0, NULL, NULL, NULL);

  for(int i = 0; i < output_size/output_depth; i++) {
    for(int j = 0; j < output_depth; j++) {
      output_data[i*output_depth + j] = host_result[i*d_output_depth + j];
    }
  }

  clEnqueueUnmapMemObject(queue,d_output,(void *) host_result,0, NULL, NULL);

  clFinish(queue);  

  // usleep(1000);

  // for(int i = 0; i < output_size; i++) {
  //   float tmp = output_data[i];
  //   output_data[i] = tmp + output_data2[i];
  // }
  // clEnqueueReadBuffer(queue, d_input, CL_TRUE, 0, input_size*sizeof(float), input2, 0, NULL, NULL );
  // clEnqueueReadBuffer(queue, d_filter, CL_TRUE, 0, filter_size*sizeof(float), filter2, 0, NULL, NULL );
  // clEnqueueReadBuffer(queue, d_bias, CL_TRUE, 0, bias_size*sizeof(float), bias2, 0, NULL, NULL );
  // clEnqueueReadBuffer(queue, d_dim_sizes, CL_TRUE, 0, 16*sizeof(int), dimsizes2, 0, NULL, NULL );
  // clEnqueueReadBuffer(queue, d_dim_strides, CL_TRUE, 0, 16*sizeof(int), dimstrides2, 0, NULL, NULL );

  // clFinish(queue);

  // cout << "Input from buffer: " << endl;
  // for(int i=0; i < input_size; i++) {
  //   cout << input2[i] << " ";
  // }
  // cout << endl;

  // cout << "Filter from buffer: " << endl;
  // for(int i=0; i < filter_size; i++) {
  //   cout << filter2[i] << " ";
  // }
  // cout << endl;

  // cout << "Bias from buffer: " << endl;
  // for(int i=0; i < bias_size; i++) {
  //   cout << bias2[i] << " ";
  // }
  // cout << endl;

  // cout << "Sizes from buffer: " << endl;
  // for(int i=0; i < 16; i++) {
  //   cout << dimsizes2[i] << " ";
  // }
  // cout << endl;

  // cout << "Strides from buffer: " << endl;
  // for(int i=0; i < 16; i++) {
  //   cout << dimstrides2[i] << " ";
  // }
  // cout << endl;
  free(output_data2);

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
}

int main() {
  int* sizes;
  int* strides;

  sizes = (int*)malloc(16*sizeof(int));
  strides = (int*)malloc(16*sizeof(int));

  Dims<4> input_dims,filter_dims,bias_dims,output_dims;
  
  //////////////////////////////////////////// channel, width, height, batch
  //input
  sizes[0] = 8;
  sizes[1] = 24;
  sizes[2] = 32;
  sizes[3] = 1;
  strides[0] = 1;
  strides[1] = strides[0]*sizes[0];
  strides[2] = strides[1]*sizes[1];
  strides[3] = strides[2]*sizes[2];

  //filter
  sizes[4] = 8;
  sizes[5] = 1;
  sizes[6] = 1;
  sizes[7] = 177;
  strides[4] = 1;
  strides[5] = strides[4]*sizes[4];
  strides[6] = strides[5]*sizes[5];
  strides[7] = strides[6]*sizes[6];

  //bias
  sizes[8] = 177;
  sizes[9] = 1;
  sizes[10] = 1;
  sizes[11] = 1;
  strides[8] = 1;
  strides[9] = strides[8]*sizes[8];
  strides[10] = strides[9]*sizes[9];
  strides[11] = strides[10]*sizes[10];

  //output
  sizes[12] = 177;
  sizes[13] = 24;
  sizes[14] = 32;
  sizes[15] = 1;
  strides[12] = 1;
  strides[13] = strides[12]*sizes[12];
  strides[14] = strides[13]*sizes[13];
  strides[15] = strides[14]*sizes[14];

  //////////////////////////////////////////////////////////

  input_dims.sizes[0] = sizes[0];
  input_dims.sizes[1] = sizes[1];
  input_dims.sizes[2] = sizes[2];
  input_dims.sizes[3] = sizes[3];
  input_dims.strides[0] = 1;
  input_dims.strides[1] = strides[1];
  input_dims.strides[2] = strides[2];
  input_dims.strides[3] = strides[3];

  //filter
  filter_dims.sizes[0] = sizes[4];
  filter_dims.sizes[1] = sizes[5];
  filter_dims.sizes[2] = sizes[6];
  filter_dims.sizes[3] = sizes[7];
  filter_dims.strides[0] = 1;
  filter_dims.strides[1] = strides[5];
  filter_dims.strides[2] = strides[6];
  filter_dims.strides[3] = strides[7];

  //bias
  bias_dims.sizes[0] = sizes[8];
  bias_dims.sizes[1] = sizes[9];
  bias_dims.sizes[2] = sizes[10];
  bias_dims.sizes[3] = sizes[11];
  bias_dims.strides[0] = 1;
  bias_dims.strides[1] = strides[9];
  bias_dims.strides[2] = strides[10];
  bias_dims.strides[3] = strides[11];

  //output
  output_dims.sizes[0] = sizes[12];
  output_dims.sizes[1] = sizes[13];
  output_dims.sizes[2] = sizes[14];
  output_dims.sizes[3] = sizes[15];
  output_dims.strides[0] = 1;
  output_dims.strides[1] = strides[13];
  output_dims.strides[2] = strides[14];
  output_dims.strides[3] = strides[15];

  int input_size = sizes[0]*sizes[1]*sizes[2]*sizes[3];
  int filter_size = sizes[4]*sizes[5]*sizes[6]*sizes[7];
  int bias_size = sizes[8]*sizes[9]*sizes[10]*sizes[11];
  int output_size = sizes[12]*sizes[13]*sizes[14]*sizes[15];

  float* input;
  float* output;
  float* filter;
  float* bias;

  input = (float*)malloc(input_size*sizeof(float));
  filter = (float*)malloc(filter_size*sizeof(float));
  bias = (float*)malloc(bias_size*sizeof(float));
  output = (float*)malloc(output_size*sizeof(float));

  for(int i = 0; i < input_size; i++) {
    input[i] = i*0.1;
  }
  for(int i = 0; i < filter_size; i++) {
    filter[i] = i*0.1;
  }
  for(int i = 0; i < bias_size; i++) {
    bias[i] = i;
  }

  cl_platform_id cpPlatform;
  cl_device_id device_id;    
  cl_context context;       
  cl_command_queue queue;   
  cl_program program;

  cl_int err;

  err = clGetPlatformIDs(1, &cpPlatform, NULL);
 
  err = clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);

  context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);

  queue = clCreateCommandQueue(context, device_id, 0, &err);

  program = clCreateProgramWithSource(context, 1,
                          (const char **) & kernelSource, NULL, &err);

  clBuildProgram(program, 0, NULL, NULL, NULL, NULL);

// printf("\t-------------------------\n");

//             cl_char string[10240] = {0};
//             // Get device name
//             err = clGetDeviceInfo(device_id, CL_DEVICE_NAME, sizeof(string), &string, NULL);
//             printf("\t\tName: %s\n", string);

//             // Get device OpenCL version
//             err = clGetDeviceInfo(device_id, CL_DEVICE_OPENCL_C_VERSION, sizeof(string), &string, NULL);
//             printf("\t\tVersion: %s\n", string);

//             // Get Max. Compute units
//             cl_uint num;
//             err = clGetDeviceInfo(device_id, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint), &num, NULL);
//             printf("\t\tMax. Compute Units: %d\n", num);

//             // Get local memory size
//             cl_ulong mem_size;
//             err = clGetDeviceInfo(device_id, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(cl_ulong), &mem_size, NULL);
//             printf("\t\tLocal Memory Size: %lu KB\n", mem_size/1024);

//             // Get global memory size
//             err = clGetDeviceInfo(device_id, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(cl_ulong), &mem_size, NULL);
//             printf("\t\tGlobal Memory Size: %lu MB\n", mem_size/(1024*1024));

//             // Get maximum buffer alloc. size
//             err = clGetDeviceInfo(device_id, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(cl_ulong), &mem_size, NULL);
//             printf("\t\tMax Alloc Size: %lu MB\n", mem_size/(1024*1024));

//             // Get work-group size information
//             size_t size;
//             err = clGetDeviceInfo(device_id, CL_DEVICE_MAX_PARAMETER_SIZE, sizeof(size_t), &size, NULL);
//             printf("\t\tMax Param Size: %ld\n", size);

//             err = clGetDeviceInfo(device_id, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &size, NULL);
//             printf("\t\tMax Work-group Total Size: %ld\n", size);

//             // Find the maximum dimensions of the work-groups
//             err = clGetDeviceInfo(device_id, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(cl_uint), &num, NULL);
//             // Get the max. dimensions of the work-groups
//             size_t dims[num];
//             err = clGetDeviceInfo(device_id, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(dims), &dims, NULL);
//             printf("\t\tMax Work-group Dims: ( ");
//             for (size_t k = 0; k < num; k++)
//             {
//                 printf("%ld ", dims[k]);
//             }
//             printf(")\n");

//             printf("\t-------------------------\n");

  for(int i = 0; i < 1; i++) {

    //  Start Timers
    double wall0 = get_wall_time();
    double cpu0  = get_cpu_time();

    // ConvAsli(input, input_dims,
    //              filter, filter_dims,
    //              bias, bias_dims,
    //              1, 1, 0,
    //              0, 0.0,
    //              1000.0, output,
    //              output_dims);
    // Conv(input, filter, bias, output, 1, 1, 0, 0, sizes, strides, 0.0, 1000.0);
    OpenCLConv(input, input_size, filter, filter_size, bias, bias_size, output, output_size, 
      1, 1, 0, 0, sizes, strides, 0.0, 1000.0,
      context, queue, program);

    //  Stop timers
    double wall1 = get_wall_time();
    double cpu1  = get_cpu_time();  

    double sum2 = 0.0;
    for(int i = 0; i < output_size; i++) {
      sum2 += output[i]; 
      if(i < 100)
      cout << output[i] << " ";
    }
    cout << endl;
    cout << "Wall Time = " << wall1 - wall0 << endl;
    cout << "CPU Time  = " << cpu1  - cpu0  << endl;
    cout << "Sum = " << sum2  << endl;

  }

  // int tmp = 10;

  // cout << "Coba convert size_t dari int 10:  " << (size_t) tmp << endl; 

  clReleaseProgram(program);
  clReleaseCommandQueue(queue);
  clReleaseContext(context);

  return 0;
}