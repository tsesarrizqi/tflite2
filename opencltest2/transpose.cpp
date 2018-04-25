#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <bits/stdc++.h>
#include <time.h>
#include <sys/time.h>

using namespace std;

// const char *kernelSource =           "\n" \ 
// "#define BLOCK_DIM 16  \n" \
// "  \n" \
// "__kernel void transpose(__global float *odata, __global float *idata, int width, int height, __local float* block)  \n" \
// "{  \n" \
// "  unsigned int xIndex = get_global_id(0);  \n" \
// "  unsigned int yIndex = get_global_id(1);  \n" \
// "    \n" \
// "  if((xIndex < width) && (yIndex < height))  \n" \
// "  {  \n" \
// "    unsigned int index_in = yIndex * width + xIndex;  \n" \
// "    block[get_local_id(1)*(BLOCK_DIM+1)+get_local_id(0)] = idata[index_in];  \n" \
// "  }  \n" \
// "  \n" \
// "  barrier(CLK_LOCAL_MEM_FENCE);  \n" \
// "  \n" \
// "  // write the transposed matrix tile to global memory  \n" \
// "  xIndex = get_group_id(1) * BLOCK_DIM + get_local_id(0);  \n" \
// "  yIndex = get_group_id(0) * BLOCK_DIM + get_local_id(1);  \n" \
// "  if((xIndex < height) && (yIndex < width))  \n" \
// "    {  \n" \
// "    unsigned int index_out = yIndex * height + xIndex;  \n" \
// "    odata[index_out] = block[get_local_id(0)*(BLOCK_DIM+1)+get_local_id(1)];  \n" \
// "  }  \n" \
// "}  \n" \
// "\n";

const char *kernelSource =           "\n" \                               
"__kernel void transpose(__global float4* input, __global float4* output,\n" \
"    int rows, int cols) {         \n" \
"   int row4 = get_global_id(0);                                      \n" \
"   int col4 = get_global_id(1);                                      \n" \
"   const float4 in_value1 = input[(row4*4+0)*(cols/4)+col4];\n" \
"   const float4 in_value2 = input[(row4*4+1)*(cols/4)+col4];\n" \
"   const float4 in_value3 = input[(row4*4+2)*(cols/4)+col4];\n" \
"   const float4 in_value4 = input[(row4*4+3)*(cols/4)+col4];\n" \
"   float4 acc = {0.0f, 0.0f, 0.0f, 0.0f};\n" \
"   acc.x = in_value1.x; \n" \
"   acc.y = in_value2.x; \n" \
"   acc.z = in_value3.x;\n" \
"   acc.w = in_value4.x;\n" \
"   output[(col4*4+0)*(rows/4) + row4] = acc;\n" \
"   acc.x = in_value1.y; \n" \
"   acc.y = in_value2.y; \n" \
"   acc.z = in_value3.y;\n" \
"   acc.w = in_value4.y;\n" \
"   output[(col4*4+1)*(rows/4) + row4] = acc;\n" \
"   acc.x = in_value1.z; \n" \
"   acc.y = in_value2.z; \n" \
"   acc.z = in_value3.z;\n" \
"   acc.w = in_value4.z;\n" \
"   output[(col4*4+2)*(rows/4) + row4] = acc;\n" \
"   acc.x = in_value1.w; \n" \
"   acc.y = in_value2.w; \n" \
"   acc.z = in_value3.w;\n" \
"   acc.w = in_value4.w;\n" \
"   output[(col4*4+3)*(rows/4) + row4] = acc;\n" \
"}      \n" \                                                              
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
                
void transpose_scalar_block(float *A, float *B, const int n, const int m, const int block_size_row, const int block_size_col) {
    #pragma omp parallel for
    for(int i=0; i<block_size_row; i++) {
        for(int j=0; j<block_size_col; j++) {
            B[j*n + i] = A[i*m +j];
        }
    }
}

void transpose_block(float *A, float *B, const int n, const int m, const int block_size) {
    #pragma omp parallel for
    for(int i=0; i<n; i+=block_size) {
        for(int j=0; j<m; j+=block_size) {
            transpose_scalar_block(&A[i*m +j], &B[j*n + i], n, m, fmin(block_size,n-i), fmin(block_size,m-j));
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

  // err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *) &d_b);
  // err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *) &d_a);
  // err |= clSetKernelArg(kernel, 2, sizeof(int), &cols);
  // err |= clSetKernelArg(kernel, 3, sizeof(int), &rows);
  // err |= clSetKernelArg(kernel, 4, 272*sizeof(float), 0);

  const size_t local[2] = { 4, 16 };
  const size_t global[2] = { rows/4, cols/4 };

  clFinish(queue);

  //  Start Timers
  double wall0 = get_wall_time();
  double cpu0  = get_cpu_time();

  err = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global, local, 0, NULL, NULL);
  cout << "Error: " << err << endl;

  clFinish(queue);

  //  Stop timers
  double wall1 = get_wall_time();
  double cpu1  = get_cpu_time();

  cout << "Wall Time Kernel = " << wall1 - wall0 << endl;
  cout << "CPU Time Kernel = " << cpu1  - cpu0  << endl;

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

  long mat_size = 2064384;

  h_a = (float*)malloc(mat_size*sizeof(float));
  h_b = (float*)malloc(mat_size*sizeof(float));

  for(int i = 0; i < mat_size; i++ )
    {
        h_a[i] = i/100.0;
    }

  //  Start Timers
  double wall0 = get_wall_time();
  double cpu0  = get_cpu_time();

  // TransposeFloatTensor(h_a,1008,2048,h_b);
  // transpose_block(h_a,h_b,1008,2048,16);
  OpenCLTransposeFloatTensor(h_a,1008,2048,h_b);

  //  Stop timers
  double wall1 = get_wall_time();
  double cpu1  = get_cpu_time();

  for(int i = 0; i < 10; i++) {
    cout << h_b[i] << " ";
  }
  cout << endl;

  double sum = 0;
  for(int i = 0; i < mat_size; i++) {
    sum += h_b[i];
  }
  cout << "Sum: " << sum;
  cout << endl;

  cout << "Wall Time = " << wall1 - wall0 << endl;
  cout << "CPU Time  = " << cpu1  - cpu0  << endl;

  return 0;
}