#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <bits/stdc++.h>
#include <time.h>
#include <sys/time.h>

using namespace std;

const char *kernelSource =           "\n" \                               
"#pragma OPENCL EXTENSION cl_khr_fp64 : enable                        \n" \
"__kernel void transpose(__global float* input, __global float* output, \n" \
"    int rows, int cols) {          \n" \
"   int gid = get_global_id(0);                                       \n" \
"  \n" \
"   if(gid < rows*cols) {                                                    \n" \
"      int i = gid/cols; \n" \
"      int j = gid%cols;   \n" \
"      const float in_value = input[gid]; \n" \
"      output[j*rows + i] = in_value; \n" \
"   }                                                                 \n" \
"} \n" \                                                                    
"\n";

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

void TransposeFloatTensor(float* input, int rows, int cols, float* output) {
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      const float in_value = input[i * cols + j];
      output[j * rows + i] = in_value;
    }
  }
}                                                            
                                                                    

void OpenCLTransposeFloatTensor(const float* input, int rows, int cols, float* output) {
  cl_mem d_a;
  cl_mem d_b;

  cl_platform_id cpPlatform;
  cl_device_id device_id;    
  cl_context context;       
  cl_command_queue queue;   
  cl_program program;       
  cl_kernel kernel;

  size_t globalSize, localSize;
  cl_int err;

  err = clGetPlatformIDs(1, &cpPlatform, NULL);
 
  err = clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);

  clGetDeviceInfo(device_id, CL_DEVICE_MAX_WORK_GROUP_SIZE,  
         sizeof(localSize), &localSize, NULL);
  globalSize = ceil(rows*cols/(localSize*1.0))*localSize;

  context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);

  queue = clCreateCommandQueue(context, device_id, 0, &err);

  program = clCreateProgramWithSource(context, 1,
                          (const char **) & kernelSource, NULL, &err);

  clBuildProgram(program, 0, NULL, NULL, NULL, NULL);

  kernel = clCreateKernel(program, "transpose", &err);

  d_a = clCreateBuffer(context, CL_MEM_READ_ONLY, rows*cols*sizeof(float), NULL, NULL);
  d_b = clCreateBuffer(context, CL_MEM_WRITE_ONLY, rows*cols*sizeof(float), NULL, NULL);

  err = clEnqueueWriteBuffer(queue, d_a, CL_TRUE, 0,
                                 rows*cols*sizeof(float), input, 0, NULL, NULL);

  err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_a);
  err  = clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_b);
  err  = clSetKernelArg(kernel, 2, sizeof(int), &rows);
  err  = clSetKernelArg(kernel, 3, sizeof(int), &cols);

  err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &globalSize, &localSize, 0, NULL, NULL);

  clFinish(queue);

  clEnqueueReadBuffer(queue, d_b, CL_TRUE, 0, rows*cols*sizeof(float), output, 0, NULL, NULL );

  clReleaseMemObject(d_a);
  clReleaseMemObject(d_b);
  clReleaseProgram(program);
  clReleaseKernel(kernel);
  clReleaseCommandQueue(queue);
  clReleaseContext(context);
}

int main() {
  float* h_a;
  float* h_b;

  long mat_size = 100000000;

  h_a = (float*)malloc(mat_size*sizeof(float));
  h_b = (float*)malloc(mat_size*sizeof(float));

  for(int i = 0; i < mat_size; i++ )
    {
        h_a[i] = i/100.0;
    }

  //  Start Timers
  double wall0 = get_wall_time();
  double cpu0  = get_cpu_time();

  // TransposeFloatTensor(h_a,1000,100000,h_b);
  OpenCLTransposeFloatTensor(h_a,1000,100000,h_b);

  //  Stop timers
  double wall1 = get_wall_time();
  double cpu1  = get_cpu_time();

  // for(int i = 0; i < mat_size; i++) {
  //   cout << h_a[i] << " ";
  // }
  // cout << endl << endl;

  for(int i = 0; i < 10; i++) {
    cout << h_b[i] << " ";
  }
  cout << endl;

  cout << "Wall Time = " << wall1 - wall0 << endl;
  cout << "CPU Time  = " << cpu1  - cpu0  << endl;

  return 0;
}