#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <bits/stdc++.h>

// #define MEM_SIZE (128)
// #define MAX_SOURCE_SIZE (0x100000)

using namespace std;

// const char *kernelSource =                                       "\n" \
// "#pragma OPENCL EXTENSION cl_khr_fp64 : enable                    \n" \
// "__kernel void vecAdd(  __global double *a,                       \n" \
// "                       __global double *b,                       \n" \
// "                       __global double *c,                       \n" \
// "                       const unsigned int n)                    \n" \
// "{                                                               \n" \
// "    //Get our global thread ID                                  \n" \
// "    int id = get_global_id(0);                                  \n" \
// "                                                                \n" \
// "    //Make sure we do not go out of bounds                      \n" \
// "    if (id < n)                                                 \n" \
// "        c[0] = c[0] + (a[id] * b[id]);                            \n" \
// "}                                                               \n" \
//                                                                 "\n" ;

const char *kernelSource =                                                 "\n" \
"#pragma OPENCL EXTENSION cl_khr_fp64 : enable                              \n" \
"__kernel void vecAdd(__global double* a_vec, __global double* b_vec,       \n" \
"    __global double* output, __local double* partial_dot) {                \n" \
"   int gid = get_global_id(0);                                             \n" \
"   int lid = get_local_id(0);                                              \n" \
"   int group_size = get_local_size(0);                                     \n" \
"                                                                           \n" \
"   /* Place product of global values into local memory */                  \n" \
"   partial_dot[lid] = a_vec[gid] * b_vec[gid];                             \n" \
"   barrier(CLK_LOCAL_MEM_FENCE);                                           \n" \
"                                                                           \n" \
"   /* Repeatedly add values in local memory */                             \n" \
"   for(int i = group_size/2; i>0; i >>= 1) {                               \n" \
"      if(lid < i) {                                                        \n" \
"         partial_dot[lid] += partial_dot[lid + i];                         \n" \
"      }                                                                    \n" \
"      barrier(CLK_LOCAL_MEM_FENCE);                                        \n" \
"   }                                                                       \n" \
"                                                                           \n" \
"   /* Transfer final result to global memory */                            \n" \
"   if(lid == 0) {                                                          \n" \
"      output[get_group_id(0)] = dot(partial_dot[0], (double)(1.0f));       \n" \
"   }                                                                       \n" \
"}                                                                          \n" \
                                                                           "\n" ;

int main( int argc, char* argv[] )
{
    unsigned int n = 1280000;
 
    double *h_a;
    double *h_b;
    double *h_c;
 
    cl_mem d_a;
    cl_mem d_b;
    cl_mem d_c;
 
    cl_platform_id cpPlatform;
    cl_device_id device_id;    
    cl_context context;       
    cl_command_queue queue;   
    cl_program program;       
    cl_kernel kernel;    

    size_t bytes = n*sizeof(double);
 
    h_a = (double*)malloc(bytes);
    h_b = (double*)malloc(bytes);
    h_c = (double*)malloc(20000*sizeof(double));
 
    int i;
    for( i = 0; i < n; i++ )
    {
        h_a[i] = 1.0;
        h_b[i] = 1.0;
    }
 
    size_t globalSize, localSize;
    cl_int err;
 
    localSize = 64;
 
    globalSize = 1280000;

    int num_group = globalSize/localSize;
    // std::ceil(n/(float)localSize)*localSize;
 
    err = clGetPlatformIDs(1, &cpPlatform, NULL);
 
    err = clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);
  
    context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
  
    queue = clCreateCommandQueue(context, device_id, 0, &err);
 
    program = clCreateProgramWithSource(context, 1,
                            (const char **) & kernelSource, NULL, &err);
 
    clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
 
    kernel = clCreateKernel(program, "vecAdd", &err);
 
    d_a = clCreateBuffer(context, CL_MEM_READ_ONLY, bytes, NULL, NULL);
    d_b = clCreateBuffer(context, CL_MEM_READ_ONLY, bytes, NULL, NULL);
    d_c = clCreateBuffer(context, CL_MEM_READ_WRITE, num_group*sizeof(double), NULL, NULL);
 
    err = clEnqueueWriteBuffer(queue, d_a, CL_TRUE, 0,
                                   bytes, h_a, 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(queue, d_b, CL_TRUE, 0,
                                   bytes, h_b, 0, NULL, NULL);
 
    err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_a);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_b);
    err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_c);
    err |= clSetKernelArg(kernel, 3, localSize*sizeof(double), NULL);
    // err |= clSetKernelArg(kernel, 3, sizeof(unsigned int), &n);
  
    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &globalSize, &localSize, 0, NULL, NULL);
 
    clFinish(queue);
 
    clEnqueueReadBuffer(queue, d_c, CL_TRUE, 0, num_group*sizeof(double), h_c, 0, NULL, NULL );
 
    double sum = 0;
    for(i=0; i<num_group; i++) {
        // std::cout << h_c[i] << " ";
        sum += h_c[i];
    }
    // std::cout << std::endl;
    printf("Dot product result: %f\n", sum);
    // std::cout << std::endl;

    clReleaseMemObject(d_a);
    clReleaseMemObject(d_b);
    clReleaseMemObject(d_c);
    clReleaseProgram(program);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
 
    free(h_a);
    free(h_b);
    free(h_c);
 
    return 0;
}

// int main( int argc, char* argv[] )
// {
//     unsigned int n = 1280000;
 
//     double *h_a;
//     double *h_b;

//     size_t bytes = n*sizeof(double);
 
//     h_a = (double*)malloc(bytes);
//     h_b = (double*)malloc(bytes);
     
//     int i;
//     for( i = 0; i < n; i++ )
//     {
//         h_a[i] = 1.0;
//         h_b[i] = 1.0;
//     }

//     double sum = 0;
//     for(int i=0; i < n; i++) {
//         sum += h_a[i] * h_b[i];    
//     }

//     printf("Dot product result: %f\n", sum);
// }
    

// int main() {

// 	cl_device_id device_id = NULL;
// 	cl_context context = NULL;
// 	cl_command_queue command_queue = NULL;
// 	cl_mem memobj = NULL;
// 	cl_program program = NULL;
// 	cl_kernel kernel = NULL;
// 	cl_platform_id platform_id = NULL;
// 	cl_uint ret_num_devices;
// 	cl_uint ret_num_platforms;
// 	cl_int ret;
	 
// 	char string[MEM_SIZE];
	 
// 	FILE *fp;
// 	char fileName[] = "./hello.cl";
// 	char *source_str;
// 	size_t source_size;
	 
// 	/* Load the source code containing the kernel*/
// 	fp = fopen(fileName, "r");
// 	if (!fp) {
// 	fprintf(stderr, "Failed to load kernel.\n");
// 	exit(1);
// 	}
// 	source_str = (char*)malloc(MAX_SOURCE_SIZE);
// 	source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
// 	fclose(fp);
	
// 	cout << source_str << endl << source_size << endl;

// 	// PLATFORM LEVEL 
// 	/* Get Platform and Device Info */
// 	ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
// 	ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, &ret_num_devices);
	 
// 	/* Create OpenCL context */
// 	context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);
	 

// 	// RUNTIME LEVEL 
// 	/* Create Command Queue Object */
// 	command_queue = clCreateCommandQueue(context, device_id, 0, &ret);
	 
// 	/* Create Memory Buffer Object */
// 	memobj = clCreateBuffer(context, CL_MEM_READ_WRITE,MEM_SIZE * sizeof(char), NULL, &ret);
	 
// 	//Create Kernel Program Object from the source (a program consist of kernels) 
// 	program = clCreateProgramWithSource(context, 1, (const char **)&source_str,
// 	(const size_t *)&source_size, &ret);
	 
// 	/* Build Kernel Program */
// 	ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
	 
// 	/* Create OpenCL Kernel Object */
// 	/* A kernel object encapsulates the specific __kernel
// 	   function declared in a program and the argument values to be used when executing this
// 	   __kernel function. */
// 	kernel = clCreateKernel(program, "hello", &ret);
	 
// 	/* Set OpenCL Kernel Parameters */
// 	/* Pass the Buffer object as kernel argument */
// 	ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&memobj);
	 
// 	/* Execute OpenCL Kernel using single work item */
// 	ret = clEnqueueTask(command_queue, kernel, 0, NULL,NULL);
	 
// 	/* Copy results from the memory buffer */
// 	ret = clEnqueueReadBuffer(command_queue, memobj, CL_TRUE, 0,
// 	MEM_SIZE * sizeof(char),string, 0, NULL, NULL);
	 
// 	/* Display Result */
// 	puts(string);
	 
// 	/* Finalization */
//  /* issues all previously queued OpenCL commands in command_queue to the device associated
//     with command_queue. clFlush only guarantees that all queued commands to command_queue
//     will eventually be submitted to the appropriate device. */
// 	ret = clFlush(command_queue);
//  /* wait until all enqueued command executed/finished */
// 	ret = clFinish(command_queue);
// 	ret = clReleaseKernel(kernel);
// 	ret = clReleaseProgram(program);
// 	ret = clReleaseMemObject(memobj);
// 	ret = clReleaseCommandQueue(command_queue);
// 	ret = clReleaseContext(context);
	 
// 	free(source_str);

// 	return 0;
// }

// int main()
// {
// cl_device_id device_id = NULL;
// cl_context context = NULL;
// cl_command_queue command_queue = NULL;
// cl_mem memobj = NULL;
// cl_program program = NULL;
// cl_kernel kernel = NULL;
// cl_platform_id platform_id = NULL;
// cl_uint ret_num_devices;
// cl_uint ret_num_platforms;
// cl_int ret;
 
// char string[MEM_SIZE];
 
// FILE *fp;
// char fileName[] = "./hello.cl";
// char *source_str;
// size_t source_size;
 
// /* Load the source code containing the kernel*/
// fp = fopen(fileName, "r");
// if (!fp) {
// fprintf(stderr, "Failed to load kernel.\n");
// exit(1);
// }
// source_str = (char*)malloc(MAX_SOURCE_SIZE);
// source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
// fclose(fp);
 
// /* Get Platform and Device Info */
// ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
// ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, &ret_num_devices);
 
// /* Create OpenCL context */
// context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);
 
// /* Create Command Queue */
// command_queue = clCreateCommandQueue(context, device_id, 0, &ret);
 
// /* Create Memory Buffer */
// memobj = clCreateBuffer(context, CL_MEM_READ_WRITE,MEM_SIZE * sizeof(char), NULL, &ret);
 
// /* Create Kernel Program from the source */
// program = clCreateProgramWithSource(context, 1, (const char **)&source_str,
// (const size_t *)&source_size, &ret);
 
// /* Build Kernel Program */
// ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
 
// /* Create OpenCL Kernel */
// kernel = clCreateKernel(program, "hello", &ret);
 
// /* Set OpenCL Kernel Parameters */
// ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&memobj);
 
// /* Execute OpenCL Kernel */
// ret = clEnqueueTask(command_queue, kernel, 0, NULL,NULL);
 
// /* Copy results from the memory buffer */
// ret = clEnqueueReadBuffer(command_queue, memobj, CL_TRUE, 0,
// MEM_SIZE * sizeof(char),string, 0, NULL, NULL);
 
// /* Display Result */
// puts(string);
 
// /* Finalization */
// ret = clFlush(command_queue);
// ret = clFinish(command_queue);
// ret = clReleaseKernel(kernel);
// ret = clReleaseProgram(program);
// ret = clReleaseMemObject(memobj);
// ret = clReleaseCommandQueue(command_queue);
// ret = clReleaseContext(context);
 
// free(source_str);
 
// return 0;
// }