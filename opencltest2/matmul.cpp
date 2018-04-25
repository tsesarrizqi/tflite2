#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <bits/stdc++.h>
#include <time.h>
#include <sys/time.h>
#include <clblast.h>
#include <half.hpp>

#define TSM 128                // The tile-size in dimension M
#define TSN 128                // The tile-size in dimension N
#define TSK 16                 // The tile-size in dimension K
#define WPTM 8                 // The work-per-thread in dimension M
#define WPTN 8                 // The work-per-thread in dimension N
#define RTSM (TSM/WPTM)        // The reduced tile-size in dimension M
#define RTSN (TSN/WPTN)        // The reduced tile-size in dimension N
#define LPTA ((TSK*TSM)/(RTSM*RTSN)) // Loads-per-thread for A
#define LPTB ((TSK*TSN)/(RTSM*RTSN)) // Loads-per-thread for B

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
// #define WIDTH 4
// #if WIDTH == 1
//     typedef float floatX;
// #elif WIDTH == 2
//     typedef float2 floatX;
// #elif WIDTH == 4
//     typedef float4 floatX;
// #endif

using namespace std;
using namespace clblast;
using namespace half_float;

// const char *kernelSource3 =           "\n" \ 
// "__kernel void matrixVectorMul(__global float4* result,    \n" \
// "    __global float4* matrix,    \n" \
// "    __global float4* vector,     \n" \
// "    int m_cols,    \n" \
// "    int m_rows,    \n" \
// "    int n_batch)    \n" \
// "{  \n" \
// "    int row = get_local_id(0)*4; \n" \
// "    int col = get_local_id(1); \n" \
// "    if ((row < m_rows) && (col < n_batch)) { \n" \
// "        float4 sum = {0.0, 0.0, 0.0, 0.0}; \n" \
// "        for(int i = 0; i < m_cols/4; i++){ \n" \
// "            float4 currb = matrix[i];\n" \
// "            sum.x += dot(matrix[(row*m_cols/4) + i],currb);\n" \
// "            sum.y += dot(matrix[((row+1)*m_cols/4) + i],currb); \n" \
// "            sum.z += dot(matrix[((row+2)*m_cols/4) + i],currb);\n" \
// "            sum.w += dot(matrix[((row+3)*m_cols/4) + i],currb);\n" \
// "        } \n" \
// "        result[row/4] = sum; \n" \
// "    } \n" \
// "}";

// const char *kernelSource =           "\n" \                               
// "#pragma OPENCL EXTENSION cl_khr_fp64 : enable      \n" \
// "__kernel void matrixVectorMul(__global float* result,    \n" \
// "    __global float* matrix,    \n" \
// "    __global float* vector,     \n" \
// "    int m_cols,    \n" \
// "    int m_rows,    \n" \
// "    int n_batch)    \n" \
// "{    \n" \
// "  float acc;   \n" \
// "  for (int b = 0; b < n_batch; b++) {\n" \
// "    for (int r = 0; r < m_rows; r++) {\n" \
// "      acc = 0.0; \n" \
// "      for (int c = 0; c < m_cols; c++) {\n" \
// "        acc += matrix[r*m_cols + c] * vector[b*m_cols + c];\n" \
// "      }\n" \
// "      result[b*m_cols + r] = acc;\n" \
// "    }\n" \
// "  }\n" \
// "} \n" \   
// "\n";

// const char *kernelSource2 =           "\n" \                               
// "__kernel void matrixVectorMul(__global float* resultVector,     \n" \
// "    __global float* matrixA,     \n" \
// "    __global float* vectorB,      \n" \
// "    int width_A,     \n" \
// "    int height_A,     \n" \
// "    int width_B)     \n" \
// "{     \n" \
// "    int idx = get_global_id(0);\n" \
// "    int idx2 = get_global_id(1);      \n" \
// "      \n" \
// "    if((idx < height_A) && (idx2 < width_B)) {  \n" \
// "        float value = 0.0f;     \n" \
// "        for (int k = 0; k < width_A; ++k) {     \n" \
// "            value += matrixA[idx * width_A + k] * vectorB[idx2*width_A+k];     \n" \
// "        }     \n" \
// "        resultVector[idx2*height_A+idx] = value;     \n" \
// "   }     \n" \   
// "}  \n" \   
// "\n";

// const char *kernelSource2 =           "\n" \
// "#pragma OPENCL EXTENSION cl_khr_fp64 : enable      \n" \
// "__kernel void matrixVectorMul(__global float* C, \n" \
// "                      const __global float* A, \n" \
// "                      const __global float* B, \n" \
// "                      int K, int M, int N) { \n" \
// "     \n" \
// "    const int row = get_local_id(0); // Local row ID (max: 8) \n" \
// "    const int col = get_local_id(1); // Local col ID (max: 32) \n" \
// "    const int globalRow = 8*get_group_id(0) + row; // Row ID of C (0..M) \n" \
// "    const int globalCol = 32*get_group_id(1) + col; // Col ID of C (0..N) \n" \
// "  \n" \
// "      __local float Asub[32][8]; \n" \
// "      __local float Bsub[32][8]; \n" \
// "    \n" \
// "      float4 acc = { 0.0f, 0.0f, 0.0f, 0.0f }; \n" \
// "       \n" \
// "      const int numTiles = ((K-1)/32)+1; \n" \
// "      for (int t=0; t<numTiles; t++) { \n" \
// "    \n" \
// "          const int tiledRow = 8*t + row; \n" \
// "          const int tiledCol = 32*t + col; \n" \
// "          if((tiledCol < K) && (globalRow < M)) {\n" \
// "            Asub[col][row] = A[globalRow*K + tiledCol]; \n" \
// "          }  \n" \
// "          else {   \n" \
// "            Asub[col][row] = 0.0; \n" \
// "          }  \n" \
// "          if((tiledRow < K) && (globalCol < N)) {\n" \
// "            Bsub[col][row] = B[globalCol*K + tiledRow]; \n" \
// "          }  \n" \
// "          else {   \n" \
// "            Bsub[col][row] = 0.0; \n" \
// "          }  \n" \
// "    \n" \
// "          barrier(CLK_LOCAL_MEM_FENCE); \n" \
// "    \n" \
// "          for (int k=0; k<32; k++) { \n" \
// "              acc += Asub[k][row] * Bsub[col][k]; \n" \
// "          } \n" \
// "    \n" \
// "          barrier(CLK_LOCAL_MEM_FENCE); \n" \
// "      } \n" \
// "    \n" \
// "      if((globalRow < M) && (globalCol < N)) { \n" \
// "          C[globalCol*M + globalRow] = acc; \n" \
// "      }\n" \
// "} \n" \
// "\n";

// const char *kernelSource2 =           "\n" \
// "__kernel void matrixVectorMul(__global float* C,  \n" \
// "                      const __global float* A,  \n" \
// "                      const __global float* B,  \n" \
// "                      int K, int M, int N) {  \n" \
// "      \n" \
// "    const int row = get_local_id(0); // Local row ID (max: 32)  \n" \
// "    const int col = get_local_id(1); // Local col ID (max: 32)  \n" \
// "    const int globalRow = 32*get_group_id(0) + row; // Row ID of C (0..M)  \n" \
// "    const int globalCol = 32*get_group_id(1) + col; // Col ID of C (0..N)  \n" \
// "   \n" \
// "      __local float Asub[32][32];  \n" \
// "      __local float Bsub[32][32];  \n" \
// "     \n" \
// "      float acc[8];\n" \
// "      for (int w=0; w<8; w++) {\n" \
// "          acc[w] = 0.0f;\n" \
// "      }  \n" \
// "        \n" \
// "      const int numTiles = ((K-1)/32)+1;  \n" \
// "      for (int t=0; t<numTiles; t++) {  \n" \
// "        for (int w=0; w<8; w++) {\n" \
// "          const int tiledRow = 32*t + row;  \n" \
// "          const int tiledCol = 32*t + col;  \n" \
// "          if(((tiledCol+w*4) < K) && (globalRow < M)) { \n" \
// "            Asub[col + w*4][row] = A[globalRow*K + tiledCol + w*4];  \n" \
// "          }   \n" \
// "          else {    \n" \
// "            Asub[col + w*4][row] = 0.0;  \n" \
// "          }   \n" \
// "          if((tiledRow < K) && ((globalCol + w*4) < N)) { \n" \
// "            Bsub[col + w*4][row] = B[(globalCol + w*4)*K + tiledRow];  \n" \
// "          }   \n" \
// "          else {    \n" \
// "            Bsub[col + w*4][row] = 0.0;  \n" \
// "          }   \n" \
// "        }\n" \
// "          barrier(CLK_LOCAL_MEM_FENCE);  \n" \
// "     \n" \
// "          for (int k=0; k<32; k++) {  \n" \
// "            for (int w=0; w<8; w++) {\n" \
// "              acc[w] += Asub[k][row] * Bsub[col + w*4][k];\n" \
// "            }  \n" \
// "          }  \n" \
// "     \n" \
// "          barrier(CLK_LOCAL_MEM_FENCE);  \n" \
// "      }  \n" \
// "      for (int w=0; w<8; w++) {\n" \
// "        if((globalRow < M) && ((globalCol + w*4) < N)) {  \n" \
// "            C[(globalCol + w*4)*M + globalRow] = acc[w];  \n" \
// "        }\n" \
// "      } \n" \
// "}\n" \  
// "\n";


// const char *kernelSource2 =           "\n" \
// "#pragma OPENCL EXTENSION cl_khr_fp64 : enable      \n" \
// "__kernel void matrixVectorMul(__global float* C, \n" \
// "                      const __global float* A, \n" \
// "                      const __global float* B, \n" \
// "                      int K, int M, int N) { \n" \
// "     \n" \
// "    const int row = get_local_id(0); // Local row ID (max: 8) \n" \
// "    const int col = get_local_id(1); // Local col ID (max: 32) \n" \
// "    const int globalRow = 32*get_group_id(0) + row; // Row ID of C (0..M) \n" \
// "    const int globalCol = 32*get_group_id(1) + col; // Col ID of C (0..N) \n" \
// "  \n" \
// "      __local float Asub[32][32]; \n" \
// "      __local float Bsub[32][32]; \n" \
// "    \n" \
// "      float acc = 0.0; \n" \
// "       \n" \
// "      const int numTiles = ((K-1)/32)+1; \n" \
// "      for (int t=0; t<numTiles; t++) { \n" \
// "    \n" \
// "          const int tiledRow = 32*t + row; \n" \
// "          const int tiledCol = 32*t + col; \n" \
// "          if((tiledCol < K) && (globalRow < M)) {\n" \
// "            Asub[col][row] = A[globalRow*K + tiledCol]; \n" \
// "          }  \n" \
// "          else {   \n" \
// "            Asub[col][row] = 0.0; \n" \
// "          }  \n" \
// "          if((tiledRow < K) && (globalCol < N)) {\n" \
// "            Bsub[col][row] = B[globalCol*K + tiledRow]; \n" \
// "          }  \n" \
// "          else {   \n" \
// "            Bsub[col][row] = 0.0; \n" \
// "          }  \n" \
// "    \n" \
// "          barrier(CLK_LOCAL_MEM_FENCE); \n" \
// "    \n" \
// "          for (int k=0; k<32; k++) { \n" \
// "              acc += Asub[k][row] * Bsub[col][k]; \n" \
// "          } \n" \
// "    \n" \
// "          barrier(CLK_LOCAL_MEM_FENCE); \n" \
// "      } \n" \
// "    \n" \
// "      if((globalRow < M) && (globalCol < N)) { \n" \
// "          C[globalCol*M + globalRow] = acc; \n" \
// "      }\n" \
// "} \n" \
// "\n";

// const char *kernelSource =           "\n" \
// "#define TSM 128                // The tile-size in dimension M \n" \
// "#define TSN 128                // The tile-size in dimension N \n" \
// "#define TSK 16                 // The tile-size in dimension K \n" \
// "#define WPTM 8                 // The work-per-thread in dimension M \n" \
// "#define WPTN 8                 // The work-per-thread in dimension N \n" \
// "#define RTSM (TSM/WPTM)        // The reduced tile-size in dimension M \n" \
// "#define RTSN (TSN/WPTN)        // The reduced tile-size in dimension N \n" \
// "#define LPTA ((TSK*TSM)/(RTSM*RTSN)) // Loads-per-thread for A \n" \
// "#define LPTB ((TSK*TSN)/(RTSM*RTSN)) // Loads-per-thread for B \n" \
// "  \n" \
// "// Use 2D register blocking (further increase in work per thread) \n" \
// "__kernel void matrixVectorMul(__global float* C,   \n" \
// "                      const __global float* A,   \n" \
// "                      const __global float* B,   \n" \
// "                      int K, int M, int N) {   \n" \
// "     \n" \
// "    // Thread identifiers \n" \
// "    const int tidm = get_local_id(0); // Local row ID (max: TSM/WPTM) \n" \
// "    const int tidn = get_local_id(1); // Local col ID (max: TSN/WPTN) \n" \
// "    const int offsetM = TSM*get_group_id(0); // Work-group offset \n" \
// "    const int offsetN = TSN*get_group_id(1); // Work-group offset \n" \
// "  \n" \
// "    // Local memory to fit a tile of A and B \n" \
// "    __local float Asub[TSK][TSM]; \n" \
// "    __local float Bsub[TSN][TSK+2]; \n" \
// "  \n" \
// "    // Allocate register space \n" \
// "    float Areg; \n" \
// "    float Breg[WPTN]; \n" \
// "    float acc[WPTM][WPTN]; \n" \
// "  \n" \
// "    // Initialise the accumulation registers \n" \
// "    for (int wm=0; wm<WPTM; wm++) { \n" \
// "        for (int wn=0; wn<WPTN; wn++) { \n" \
// "            acc[wm][wn] = 0.0f; \n" \
// "        } \n" \
// "    } \n" \
// "     \n" \
// "    // Loop over all tiles \n" \
// "    int numTiles = K/TSK; \n" \
// "    for (int t=0; t<numTiles; t++) { \n" \
// "  \n" \
// "        // Load one tile of A and B into local memory \n" \
// "        for (int la=0; la<LPTA; la++) { \n" \
// "            int tid = tidn*RTSM + tidm; \n" \
// "            int id = la*RTSN*RTSM + tid; \n" \
// "            int row = id % TSM; \n" \
// "            int col = id / TSM; \n" \
// "            int tiledIndex = TSK*t + col; \n" \
// "            Asub[col][row] = A[tiledIndex*M + offsetM + row]; \n" \
// "            Bsub[row][col] = B[tiledIndex*N + offsetN + row]; \n" \
// "        } \n" \
// "         \n" \
// "        // Synchronise to make sure the tile is loaded \n" \
// "        barrier(CLK_LOCAL_MEM_FENCE); \n" \
// "  \n" \
// "        // Loop over the values of a single tile \n" \
// "        for (int k=0; k<TSK; k++) { \n" \
// "  \n" \
// "            // Cache the values of Bsub in registers \n" \
// "            for (int wn=0; wn<WPTN; wn++) { \n" \
// "                int col = tidn + wn*RTSN; \n" \
// "                Breg[wn] = Bsub[col][k]; \n" \
// "            } \n" \
// "  \n" \
// "            // Perform the computation \n" \
// "            for (int wm=0; wm<WPTM; wm++) { \n" \
// "                int row = tidm + wm*RTSM; \n" \
// "                Areg = Asub[k][row]; \n" \
// "                for (int wn=0; wn<WPTN; wn++) { \n" \
// "                    acc[wm][wn] += Areg * Breg[wn]; \n" \
// "                } \n" \
// "            } \n" \
// "        } \n" \
// "  \n" \
// "        // Synchronise before loading the next tile \n" \
// "        barrier(CLK_LOCAL_MEM_FENCE); \n" \
// "    } \n" \
// "  \n" \
// "    // Store the final results in C \n" \
// "    for (int wm=0; wm<WPTM; wm++) { \n" \
// "        int globalRow = offsetM + tidm + wm*RTSM; \n" \
// "        for (int wn=0; wn<WPTN; wn++) { \n" \
// "            int globalCol = offsetN + tidn + wn*RTSN; \n" \
// "            C[globalCol*M + globalRow] = acc[wm][wn]; \n" \
// "        } \n" \
// "    } \n" \
// "} \n" \
// "\n";

//5.3248e+06
// kecepatan current 0.005 kernel
const char *kernelSource2 =           "\n" \
"#define TS 32      \n" \
"#define WIDTH 4      \n" \
"__kernel void matrixVectorMulF4(__global float4* result,    \n" \
"    const __global float4* matrix,    \n" \
"    const __global float4* vector,     \n" \
"    int m_cols,    \n" \
"    int m_rows,    \n" \
"    int n_batch)    \n" \
"{  \n" \
"    int row = get_global_id(0)*4; \n" \
"    int localidx0 = get_local_id(0); \n" \
"    int localidx = get_local_id(1); \n" \
"    __local float4 Aacc[8][32]; \n" \
"    if (row < m_rows) { \n" \
"        float4 sum = {0.0, 0.0, 0.0, 0.0}; \n" \
"        int starti = localidx*(m_cols/128); \n" \
"        for(int i = starti; i < (starti+(m_cols/128)); i++){ \n" \
"            float4 currb = vector[i];\n" \
"            sum.x += dot(matrix[(row*m_cols/4) + i],currb);\n" \
"            sum.y += dot(matrix[((row+1)*m_cols/4) + i],currb); \n" \
"            sum.z += dot(matrix[((row+2)*m_cols/4) + i],currb);\n" \
"            sum.w += dot(matrix[((row+3)*m_cols/4) + i],currb);\n" \
"        } \n" \
"        Aacc[localidx0][localidx] = sum; \n" \
"        barrier(CLK_LOCAL_MEM_FENCE);    \n" \
"        if(localidx == 0) { \n" \
"          float4 total = {0.0, 0.0, 0.0, 0.0};    \n" \
"          for(int i = 0; i < 32; i++) {    \n" \
"            total += Aacc[localidx0][i]; \n" \
"          } \n" \
"          result[row/4] = total;\n" \
"        } \n" \
"    } \n" \
"} \n" \
"__kernel void matrixVectorMulF16(__global float4* result,    \n" \
"    const __global float16* matrix,    \n" \
"    const __global float16* vector,     \n" \
"    int m_cols,    \n" \
"    int m_rows,    \n" \
"    int n_batch)    \n" \
"{  \n" \
"    int row = get_global_id(0)*4; \n" \
"    int localidx0 = get_local_id(0); \n" \
"    int localidx = get_local_id(1); \n" \
"    __local float4 Aacc[8][32]; \n" \
"    if (row < m_rows) { \n" \
"        float4 sum = {0.0, 0.0, 0.0, 0.0}; \n" \
"        int starti = localidx*(m_cols/512); \n" \
"        for(int i = starti; i < (starti+(m_cols/512)); i++){ \n" \
"            float16 currb = vector[i];\n" \
"            float16 curra1 = matrix[(row*m_cols/16) + i];\n" \
"            float16 curra2 = matrix[((row+1)*m_cols/16) + i];\n" \
"            float16 curra3 = matrix[((row+2)*m_cols/16) + i];\n" \
"            float16 curra4 = matrix[((row+3)*m_cols/16) + i];\n" \
"            sum.x += dot(curra1.s0123,currb.s0123)+dot(curra1.s4567,currb.s4567)+dot(curra1.s89ab,currb.s89ab)+dot(curra1.scdef,currb.scdef);\n" \
"            sum.y += dot(curra2.s0123,currb.s0123)+dot(curra2.s4567,currb.s4567)+dot(curra2.s89ab,currb.s89ab)+dot(curra2.scdef,currb.scdef); \n" \
"            sum.z += dot(curra3.s0123,currb.s0123)+dot(curra3.s4567,currb.s4567)+dot(curra3.s89ab,currb.s89ab)+dot(curra3.scdef,currb.scdef);\n" \
"            sum.w += dot(curra4.s0123,currb.s0123)+dot(curra4.s4567,currb.s4567)+dot(curra4.s89ab,currb.s89ab)+dot(curra4.scdef,currb.scdef);\n" \
"        } \n" \
"        Aacc[localidx0][localidx] = sum; \n" \
"        barrier(CLK_LOCAL_MEM_FENCE);    \n" \
"        if(localidx == 0) { \n" \
"          float4 total = {0.0, 0.0, 0.0, 0.0};    \n" \
"          for(int i = 0; i < 32; i++) {    \n" \
"            total += Aacc[localidx0][i]; \n" \
"          } \n" \
"          result[row/4] = total;\n" \
"        } \n" \
"    } \n" \
"} \n" \
"__kernel void matrixVectorMul(__global float4* C,         \n" \
"                      const __global float4* A,         \n" \
"                      const __global float4* B,         \n" \
"                      int K, int M, int N) {         \n" \
"    const int row = get_local_id(0); // Local row ID (max: TS/WIDTH)      \n" \
"    const int col = get_local_id(1); // Local col ID (max: TS)      \n" \
"    const int globalRow = (TS/WIDTH)*get_group_id(0) + row; // 0..M/WIDTH      \n" \
"    const int globalCol = TS*get_group_id(1) + col; // 0..N      \n" \
"    __local float4 Asub[TS][TS/WIDTH];      \n" \
"    __local float4 Bsub[TS][TS/WIDTH];      \n" \
"    float4 acc = { 0.0, 0.0, 0.0, 0.0 };      \n" \
"    const int numTiles = K/TS;      \n" \
"    for (int t=0; t<numTiles; t++) {      \n" \
"        const int tiledRow = (TS/WIDTH)*t + row;      \n" \
"        const int tiledCol = TS*t + col;      \n" \
"        if(globalRow < (M/WIDTH)) {     \n" \
"          Asub[col][row] = A[tiledCol*(M/WIDTH) + globalRow];     \n" \
"        }      \n" \
"        else {     \n" \
"           float4 tmp = { 0.0, 0.0, 0.0, 0.0 };  \n" \
"           Asub[col][row] = tmp;     \n" \
"        }     \n" \
"        if(globalCol < N) {     \n" \
"          Bsub[col][row] = B[globalCol*(K/WIDTH) + tiledRow];     \n" \
"        }      \n" \
"        else {     \n" \
"           float4 tmp = { 0.0, 0.0, 0.0, 0.0 };  \n" \
"           Bsub[col][row] = tmp;     \n" \
"        }     \n" \
"        barrier(CLK_LOCAL_MEM_FENCE);      \n" \
"       \n" \
"        if(globalCol < N) {     \n" \
"          float4 vecA, vecB;      \n" \
"          float valB;      \n" \
"          for (int k=0; k<TS/WIDTH; k++) {      \n" \
"              vecB = Bsub[col][k];      \n" \
"              for (int w=0; w<WIDTH; w++) {      \n" \
"                  vecA = Asub[WIDTH*k + w][row];      \n" \
"                  switch (w) {      \n" \
"                      case 0: valB = vecB.x; break;      \n" \
"                      case 1: valB = vecB.y; break;      \n" \
"                      case 2: valB = vecB.z; break;      \n" \
"                      case 3: valB = vecB.w; break;      \n" \
"                  }      \n" \
"                  acc.x += vecA.x * valB;      \n" \
"                  acc.y += vecA.y * valB;      \n" \
"                  acc.z += vecA.z * valB;      \n" \
"                  acc.w += vecA.w * valB;      \n" \
"              }     \n" \
"          }      \n" \
"        }     \n" \
"        barrier(CLK_LOCAL_MEM_FENCE);      \n" \
"    }      \n" \
"        \n" \
"    if((globalCol < N) && (globalRow < (M/WIDTH))) {     \n" \
"      C[globalCol*(M/WIDTH) + globalRow] = acc;     \n" \
"    }      \n" \
"         \n" \
"}      \n" \
"__kernel void matmul2(__global float* result,    \n" \
"    const __global float4* matrix,    \n" \
"    const __global float4* vector,     \n" \
"    int m_cols,    \n" \
"    int m_rows,    \n" \
"    int n_batch)    \n" \
"{  \n" \
"    int row = get_global_id(0); \n" \
"    int col = get_global_id(1); \n" \
"    if ((row < m_rows) && (col < n_batch)) { \n" \
"        float sum = 0.0; \n" \
"        for(int i = 0; i < m_cols/4; i++){ \n" \
"            sum += dot(matrix[row*m_cols/4 + i],vector[i*n_batch + col]);\n" \
"        } \n" \
"        result[row*n_batch + col] = sum;\n" \
"    } \n" \
"} \n" \
"\n";

// const char *kernelSource2 =           "\n" \
// "#define TS 32      \n" \
// "#define WIDTH 8      \n" \
// "       \n" \
// "__kernel void matrixVectorMul(__global float8* C,         \n" \
// "                      const __global float8* A,         \n" \
// "                      const __global float8* B,         \n" \
// "                      int K, int M, int N) {         \n" \
// "          \n" \
// "    // Thread identifiers      \n" \
// "    const int row = get_local_id(0); // Local row ID (max: TS/WIDTH)      \n" \
// "    const int col = get_local_id(1); // Local col ID (max: TS)      \n" \
// "    const int globalRow = (TS/WIDTH)*get_group_id(0) + row; // 0..M/WIDTH      \n" \
// "    const int globalCol = TS*get_group_id(1) + col; // 0..N      \n" \
// "       \n" \
// "    // Local memory to fit a tile of TS*TS elements of A and B      \n" \
// "    __local float8 Asub[TS][TS/WIDTH];      \n" \
// "    __local float8 Bsub[TS][TS/WIDTH];      \n" \
// "       \n" \
// "    // Initialise the accumulation registers      \n" \
// "    float8 acc = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };      \n" \
// "          \n" \
// "    // Loop over all tiles      \n" \
// "    const int numTiles = K/TS;      \n" \
// "    for (int t=0; t<numTiles; t++) {      \n" \
// "       \n" \
// "        // Load one tile of A and B into local memory      \n" \
// "        const int tiledRow = (TS/WIDTH)*t + row;      \n" \
// "        const int tiledCol = TS*t + col;      \n" \
// "        if(globalRow < (M/WIDTH)) {     \n" \
// "          Asub[col][row] = A[tiledCol*(M/WIDTH) + globalRow];     \n" \
// "        }      \n" \
// "        else {     \n" \
// "           float8 tmp = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };  \n" \
// "           Asub[col][row] = tmp;     \n" \
// "        }     \n" \
// "        if(globalCol < N) {     \n" \
// "          Bsub[col][row] = B[globalCol*(K/WIDTH) + tiledRow];     \n" \
// "        }      \n" \
// "        else {     \n" \
// "           float8 tmp = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };  \n" \
// "           Bsub[col][row] = tmp;     \n" \
// "        }     \n" \
// "             \n" \
// "        // Synchronise to make sure the tile is loaded      \n" \
// "        barrier(CLK_LOCAL_MEM_FENCE);      \n" \
// "       \n" \
// "        if(globalCol < N) {     \n" \
// "          // Perform the computation for a single tile      \n" \
// "          float8 vecA, vecB;      \n" \
// "          float valB;      \n" \
// "          for (int k=0; k<TS/WIDTH; k++) {      \n" \
// "              vecB = Bsub[col][k];      \n" \
// "              for (int w=0; w<WIDTH; w++) {      \n" \
// "                  vecA = Asub[WIDTH*k + w][row];      \n" \
// "                  switch (w) {      \n" \
// "                      case 0: valB = vecB.s0; break;      \n" \
// "                      case 1: valB = vecB.s1; break;      \n" \
// "                      case 2: valB = vecB.s2; break;      \n" \
// "                      case 3: valB = vecB.s3; break;\n" \
// "                      case 4: valB = vecB.s4; break;      \n" \
// "                      case 5: valB = vecB.s5; break;      \n" \
// "                      case 6: valB = vecB.s6; break;      \n" \
// "                      case 7: valB = vecB.s7; break;      \n" \
// "                  }      \n" \
// "                  acc.s0 += vecA.s0 * valB;      \n" \
// "                  acc.s1 += vecA.s1 * valB;      \n" \
// "                  acc.s2 += vecA.s2 * valB;      \n" \
// "                  acc.s3 += vecA.s3 * valB;      \n" \
// "                  acc.s4 += vecA.s4 * valB;      \n" \
// "                  acc.s5 += vecA.s5 * valB;      \n" \
// "                  acc.s6 += vecA.s6 * valB;      \n" \
// "                  acc.s7 += vecA.s7 * valB;      \n" \
// "              }     \n" \
// "          }      \n" \
// "        }     \n" \
// "      \n" \
// "        // Synchronise before loading the next tile      \n" \
// "        barrier(CLK_LOCAL_MEM_FENCE);      \n" \
// "    }      \n" \
// "        \n" \
// "    if((globalCol < N) && (globalRow < (M/WIDTH))) {     \n" \
// "      // Store the final results in C      \n" \
// "      C[globalCol*(M/WIDTH) + globalRow] = acc;     \n" \
// "    }      \n" \
// "         \n" \
// "}      \n" \
// "\n";

// #define DOT(a,b) \
//     (a.S0 * b.S0 + a.S1 * b.S1 + a.S2 * b.S2 + a.S3 * b.S3 \
//     +a.S4 * b.S4 + a.S5 * b.S5 + a.S6 * b.S6 + a.S7 * b.S7) 

// #define SUM(a) \
//     (a.S0 + a.S1 + a.S2 + a.S3 + a.S4 + a.S5 + a.S6 + a.S7)

// __kernel void gemm_tn (
//     __global const T * restrict A,
//     int lda,    // column stride in elements for matrix A
//     __global const T * restrict B,
//     int ldb,    // row stride in elements for matrix B
//     __global T * restrict C,
//     int ldc,    // column stride in elements for matrix C
//     int k        // number of columns/rows in a matrix
// )
// {
//     const int i = get_global_id(0) * 2;
//     const int j = get_global_id(1) * 2;
    
//     float4 ab = (float4)0.0f;

//     for (int l = 0; l < k; l += 4)
//     {
//         float4 a0 = vload4(0, &A[i * k]);
//         float4 a1 = vload4(0, &A[(i+1) * k]);
//         float4 b0 = vload4(0, &B[j * k]);
//         float4 b1 = vload4(0, &B[(j+1) * k]);

//         ab += ( float4 ) ( dot (a0 , b0 ), dot (a0 , b1 ), dot (a1 , b0 ), dot (a1 , b1 ));
        
//         A += 4; 
//         B += 4;
//     }

//     /*for(int ib = 0; ib < 2; ib++) {
//         for(int jb = 0; jb < 2; jb++) {
//             C[(i+ib) * k + (j+jb)] = SUM(sum[ib][jb]);
//         }
//     }*/
//     vstore2(ab.s01, 0, &C[i * k + j]);
//     vstore2(ab.s23, 0, &C[(i+1) * k + j]);
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

void PortableMatrixBatchVectorMultiplyAccumulate(float* matrix,
                                                 int m_rows, int m_cols,
                                                 float* vector,
                                                 int n_batch, float* result,
                                                 int result_stride) {
  // vector per kolom
  // matrix per baris
  // result per kolom
  float* result_in_batch = result;
  for (int b = 0; b < n_batch; b++) {
    float* matrix_ptr = matrix;
    for (int r = 0; r < m_rows; r++) {
      float* vector_in_batch = vector + b * m_cols;
      for (int c = 0; c < m_cols; c++) {
        *result_in_batch += *matrix_ptr++ * *vector_in_batch++;
      }
      result_in_batch += result_stride;
    }
  }
}                                                            
                                                                    
void TransposeFloatTensor(float* input, int rows, int cols, float* output) {
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      const float in_value = input[i * cols + j];
      output[j * rows + i] = in_value;
    }
  }
}

// void OpenCLPortableMatrixBatchVectorMultiplyAccumulate(const float* matrix,
//                                                  int m_rows, int m_cols,
//                                                  const float* vector,
//                                                  int n_batch, float* result,
//                                                  int result_stride,
//                                                  cl_context context,
//                                                  cl_command_queue queue,
//                                                  cl_program program) {
//   cl_mem d_a;
//   cl_mem d_b;
//   cl_mem d_c;
   
//   // cl_program program;       
//   cl_kernel kernel;

//   // size_t globalSize, localSize;
//   cl_int err;

//   // program = clCreateProgramWithSource(context, 1,
//   //                         (const char **) & kernelSource, NULL, &err);

//   // clBuildProgram(program, 0, NULL, NULL, NULL, NULL);

//   kernel = clCreateKernel(program, "matrixVectorMul", &err);

//   d_a = clCreateBuffer(context, CL_MEM_READ_ONLY, m_rows*m_cols*sizeof(float), NULL, NULL);
//   d_b = clCreateBuffer(context, CL_MEM_READ_ONLY, m_cols*n_batch*sizeof(float), NULL, NULL);
//   d_c = clCreateBuffer(context, CL_MEM_READ_WRITE, m_rows*n_batch*sizeof(float), NULL, NULL);

//   err = clEnqueueWriteBuffer(queue, d_a, CL_TRUE, 0,
//                                  m_rows*m_cols*sizeof(float), matrix, 0, NULL, NULL);
//   err = clEnqueueWriteBuffer(queue, d_b, CL_TRUE, 0,
//                                  m_cols*n_batch*sizeof(float), vector, 0, NULL, NULL);

//   err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_c);
//   err  = clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_a);
//   err  = clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_b);
//   err  = clSetKernelArg(kernel, 3, sizeof(int), &m_cols);
//   err  = clSetKernelArg(kernel, 4, sizeof(int), &m_rows);
//   err  = clSetKernelArg(kernel, 5, sizeof(int), &n_batch);

//   // const int TS = 32;
//   // const size_t local[2] = { 32, 32 };
//   // const size_t global[2] = { ((m_rows-1)/32+1)*32, ((n_batch-1)/32+1)*32 };

//   const size_t local[2] = { (size_t) (TSM/WPTM), (size_t) (TSN/WPTN) }; // Or { RTSM, RTSN };
//   const size_t global[2] = { (size_t) (m_rows/WPTM), (size_t) (n_batch/WPTN) };

//   cout << local[0] << " " << local[1] << " " << global[0] << " " << global[1] << " " << endl;

//   // const size_t local[2] = { TS, TS/8 };
//   // const size_t global[2] = { ((m_rows-1)/32+1)*32, ((n_batch-1)/32+1)*4 };

//   // // err = clEnqueueTask(queue, kernel, 0, NULL,NULL);
//   err = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global, local, 0, NULL, NULL);
//   // // cout << err << endl;

//   clFinish(queue);

//   cout << err << endl;

//   clEnqueueReadBuffer(queue, d_c, CL_TRUE, 0, m_rows*n_batch*sizeof(float), result, 0, NULL, NULL );

//   // 0.219554
//   clReleaseMemObject(d_a);
//   clReleaseMemObject(d_b);
//   clReleaseMemObject(d_c);
//   // clReleaseProgram(program);
//   clReleaseKernel(kernel);
// }

void OpenCLPortableMatrixBatchVectorMultiplyAccumulate3(const float* matrix,
                                                 int m_rows, int m_cols,
                                                 const float* vector,
                                                 int n_batch, float* result,
                                                 int result_stride,
                                                 cl_context context,
                                                 cl_command_queue queue,
                                                 cl_program program) {
  cl_mem d_a;
  cl_mem d_b;
  cl_mem d_c;    
  cl_kernel kernel;
  cl_int err;

  //convert float to half
  int matrixsize = m_rows*m_cols*sizeof(float);
  int vectorsize = m_cols*n_batch*sizeof(float);
  int resultsize = m_rows*n_batch*sizeof(float);

  //  Start Timers
  double wall0 = get_wall_time();
  double cpu0  = get_cpu_time();

  kernel = clCreateKernel(program, "matmul2", &err);

  //  Stop timers
  double wall1 = get_wall_time();
  double cpu1  = get_cpu_time();

  cout << "Wall Time Createkernel = " << wall1 - wall0 << endl; 

  wall0 = get_wall_time();
  cpu0  = get_cpu_time();

  d_a = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, matrixsize, (float*) matrix, NULL);
  d_b = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, vectorsize, (float*) vector, NULL);
  d_c = clCreateBuffer(context, CL_MEM_WRITE_ONLY, resultsize, NULL, NULL);

  // //  Stop timers
  wall1 = get_wall_time();
  cpu1  = get_cpu_time();

  cout << "Wall Time Createbuffer = " << wall1 - wall0 << endl; 

  //  Start Timers
  wall0 = get_wall_time();
  cpu0  = get_cpu_time();

  // err = clEnqueueWriteBuffer(queue, d_a, CL_TRUE, 0,
  //                                matrixsize, (float*) matrix, 0, NULL, NULL);
  // err = clEnqueueWriteBuffer(queue, d_b, CL_TRUE, 0,
  //                                vectorsize, (float*) vector, 0, NULL, NULL);
  // clFinish(queue);
  //  Stop timers
  wall1 = get_wall_time();
  cpu1  = get_cpu_time();

  cout << "Wall Time Writebuffer = " << wall1 - wall0 << endl; 

  err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_c);
  err  = clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_a);
  err  = clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_b);
  err  = clSetKernelArg(kernel, 3, sizeof(int), &m_cols);
  err  = clSetKernelArg(kernel, 4, sizeof(int), &m_rows);
  err  = clSetKernelArg(kernel, 5, sizeof(int), &n_batch);

  const size_t local[2] = { 8, 32 };
  const size_t global[2] = { (size_t) (((m_rows-1)/8+1)*8), (size_t) (((n_batch-1)/32+1)*32) };

  //  Start Timers
  wall0 = get_wall_time();
  cpu0  = get_cpu_time();

  err = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global, local, 0, NULL, NULL);

  clFinish(queue);
  cout << "Runkernel error: " << err << endl;

  //  Stop timers
  wall1 = get_wall_time();
  cpu1  = get_cpu_time();

  cout << "Wall Time Runkernel = " << wall1 - wall0 << endl;  

  //  Start Timers
  wall0 = get_wall_time();
  cpu0  = get_cpu_time();

  clEnqueueReadBuffer(queue, d_c, CL_TRUE, 0, resultsize, result, 0, NULL, NULL );

  clFinish(queue);

  //  Stop timers
  wall1 = get_wall_time();
  cpu1  = get_cpu_time();

  cout << "Wall Time Readbuffer = " << wall1 - wall0 << endl; 

  wall0 = get_wall_time();
  cpu0  = get_cpu_time();

  clReleaseMemObject(d_a);
  clReleaseMemObject(d_b);
  clReleaseMemObject(d_c);
  clReleaseKernel(kernel);

  //  Stop timers
  wall1 = get_wall_time();
  cpu1  = get_cpu_time();

  cout << "Wall Time Cleaning = " << wall1 - wall0 << endl; 
}

// void OpenCLPortableMatrixBatchVectorMultiplyAccumulate2(const float* matrix,
//                                                  int m_rows, int m_cols,
//                                                  const float* vector,
//                                                  int n_batch, float* result,
//                                                  int result_stride,
//                                                  cl_context context,
//                                                  cl_command_queue queue,
//                                                  cl_program program) {
//   cl_mem d_a;
//   cl_mem d_b;
//   cl_mem d_c;    
//   cl_kernel kernel;
//   cl_int err;

//   // //convert float to half
//   int matrixsize = m_rows*m_cols*sizeof(float);
//   int vectorsize = m_cols*n_batch*sizeof(float);
//   int resultsize = m_rows*n_batch*sizeof(float);

//   // // Test half precision
//   // half* matrixHalf = (half*) malloc(m_rows*m_cols*sizeof(half));
//   // for(int i = 0; i < m_rows*m_cols; i++) {
//   //   // half halfTmp(matrix[i]);
//   //   matrixHalf[i] = half_cast<half>(matrix[i]);
//   // }
//   // half* vectorHalf = (half*) malloc(m_cols*n_batch*sizeof(half));
//   // for(int i = 0; i < m_cols*n_batch; i++) {
//   //   // half halfTmp(vector[i]);
//   //   vectorHalf[i] = half_cast<half>(vector[i]);
//   // }
//   // half* resultHalf = (half*) malloc(m_rows*n_batch*sizeof(half));
//   // ///////////////////////////////////////


//   //  Start Timers
//   double wall0 = get_wall_time();
//   double cpu0  = get_cpu_time();

//   kernel = clCreateKernel(program, "matrixVectorMul", &err);

//   //  Stop timers
//   double wall1 = get_wall_time();
//   double cpu1  = get_cpu_time();

//   cout << "Wall Time Createkernel = " << wall1 - wall0 << endl; 

//   wall0 = get_wall_time();
//   cpu0  = get_cpu_time();

//   d_a = clCreateBuffer(context, CL_MEM_READ_ONLY, matrixsize, NULL, NULL);
//   d_b = clCreateBuffer(context, CL_MEM_READ_ONLY, vectorsize, NULL, NULL);
//   d_c = clCreateBuffer(context, CL_MEM_WRITE_ONLY, resultsize, NULL, NULL);

//   // //  Stop timers
//   wall1 = get_wall_time();
//   cpu1  = get_cpu_time();

//   cout << "Wall Time Createbuffer = " << wall1 - wall0 << endl; 

//   //  Start Timers
//   wall0 = get_wall_time();
//   cpu0  = get_cpu_time();

//   // cl_float *host_a = (cl_float*)clEnqueueMapBuffer(
//   //             queue,
//   //             d_a,
//   //             CL_TRUE,
//   //             CL_MAP_WRITE,
//   //             0,
//   //             sizeof(float)*m_rows*m_cols,
//   //             0, NULL, NULL, NULL);
//   // cl_float *host_b = (cl_float*)clEnqueueMapBuffer(
//   //             queue,
//   //             d_b,
//   //             CL_TRUE,
//   //             CL_MAP_WRITE,
//   //             0,
//   //             sizeof(float)*m_cols*n_batch,
//   //             0, NULL, NULL, NULL);

//   // clFinish(queue);

//   // std::memcpy(host_a, matrix, m_rows*m_cols*sizeof(float));
//   // std::memcpy(host_b, vector, m_cols*n_batch*sizeof(float));

//   // clEnqueueUnmapMemObject(queue,d_a,(void *) host_a,0, NULL, NULL);
//   // clEnqueueUnmapMemObject(queue,d_b,(void *) host_b,0, NULL, NULL);

//   err = clEnqueueWriteBuffer(queue, d_a, CL_TRUE, 0,
//                                  matrixsize, (float*) matrix, 0, NULL, NULL);
//   err = clEnqueueWriteBuffer(queue, d_b, CL_TRUE, 0,
//                                  vectorsize, (float*) vector, 0, NULL, NULL);
//   clFinish(queue);
//   //  Stop timers
//   wall1 = get_wall_time();
//   cpu1  = get_cpu_time();

//   cout << "Wall Time Readbuffer = " << wall1 - wall0 << endl; 

//   err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_c);
//   err  = clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_a);
//   err  = clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_b);
//   err  = clSetKernelArg(kernel, 3, sizeof(int), &m_cols);
//   err  = clSetKernelArg(kernel, 4, sizeof(int), &m_rows);
//   err  = clSetKernelArg(kernel, 5, sizeof(int), &n_batch);

//   // const int TS = 32;
//   // const size_t local[2] = { (size_t) (TS/4), TS };
//   // const size_t global[2] = { (size_t) (((m_rows-1)/8+1)*8), TS };

//   const size_t local[2] = { 16, 32 };
//   const size_t global[2] = { (size_t) (((m_rows-1)/16+1)*16), 32 };

//   //  Start Timers
//   wall0 = get_wall_time();
//   cpu0  = get_cpu_time();

//   err = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global, local, 0, NULL, NULL);


//   clFinish(queue);
//   cout << "Runkernel error: " << err << endl;

//   //  Stop timers
//   wall1 = get_wall_time();
//   cpu1  = get_cpu_time();

//   cout << "Wall Time Runkernel = " << wall1 - wall0 << endl;  

//   //  Start Timers
//   wall0 = get_wall_time();
//   cpu0  = get_cpu_time();

//   clEnqueueReadBuffer(queue, d_c, CL_TRUE, 0, resultsize, result, 0, NULL, NULL );

//   clFinish(queue);

//   //  Stop timers
//   wall1 = get_wall_time();
//   cpu1  = get_cpu_time();

//   cout << "Wall Time Readbuffer = " << wall1 - wall0 << endl; 

//   // // // half halfTmp2 = half_cast<half>(resultHalf);
//   // for(int i = 0; i < m_rows*n_batch; i++) {
//   //   result[i] = resultHalf[i];
//   // }

//   wall0 = get_wall_time();
//   cpu0  = get_cpu_time();

//   // free(matrixHalf);
//   // free(vectorHalf);
//   // free(resultHalf);
//   clReleaseMemObject(d_a);
//   clReleaseMemObject(d_b);
//   clReleaseMemObject(d_c);
//   clReleaseKernel(kernel);

//   //  Stop timers
//   wall1 = get_wall_time();
//   cpu1  = get_cpu_time();

//   cout << "Wall Time Cleaning = " << wall1 - wall0 << endl; 
// }

int main() {
  cl_platform_id cpPlatform;
  cl_device_id device_id;    
  cl_context context;       
  cl_command_queue queue;
  cl_program program;

  // size_t globalSize, localSize;
  cl_int err;
  cl_mem d_a;
  cl_mem d_b;
  cl_mem d_c;
   
  // cl_program program;       
  cl_kernel kernel;

  // 0.0409939 detik
  err = clGetPlatformIDs(1, &cpPlatform, NULL); //ngising
 
  err = clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_GPU, 1, &device_id, NULL); //ngising

  // 0.151142
  context = clCreateContext(0, 1, &device_id, NULL, NULL, &err); //ngising

  queue = clCreateCommandQueue(context, device_id, 0, &err);

  program = clCreateProgramWithSource(context, 1,
                          (const char **) & kernelSource2, NULL, &err);

  clBuildProgram(program, 0, NULL, NULL, NULL, NULL);


  // cl_device_fp_config config;
  // clGetDeviceInfo(device_id, CL_DEVICE_HALF_FP_CONFIG, sizeof(cl_device_fp_config), &config, NULL);
  // cout << "FP16 support: " << config << endl;



  float* h_a;
  float* h_b;
  // float* h_b2;
  float* h_at;
  // float* h_bt;
  // float* h_c;

  int m_rows = 201;
  int m_cols = 104;
  int n_batch = 10;

  h_a = (float*)malloc(m_rows*m_cols*sizeof(float));
  h_b = (float*)malloc(m_cols*n_batch*sizeof(float));
  // h_b2 = (float*)malloc(2048*sizeof(float));
  h_at = (float*)malloc(m_rows*m_cols*sizeof(float));
  // h_bt = (float*)malloc(10000*sizeof(float));
  // h_c = (float*)malloc(10000*sizeof(float));

  for(int i = 0; i < m_rows*m_cols; i++ )
    {
        h_a[i] = 1;
    }
  for(int i = 0; i < m_cols*n_batch; i++ )
  {
        h_b[i] = 1;
  }

  //  Start Timers
  double wall0 = get_wall_time();
  double cpu0  = get_cpu_time();
  
  // std::memcpy(h_b2, h_b, 2048*sizeof(float));

  //  Stop timers
  double wall1 = get_wall_time();
  double cpu1  = get_cpu_time();

  cout << "Wall Time Memcpy = " << wall1 - wall0 << endl;
  cout << "CPU Time Memcpy = " << cpu1  - cpu0  << endl;

  // TransposeFloatTensor(h_a, m_rows, m_cols, h_at);

  // d_a = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, m_rows*m_cols*sizeof(float), NULL, NULL);
  // d_b = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, m_cols*n_batch*sizeof(float), NULL, NULL);
  // d_c = clCreateBuffer(context, CL_MEM_READ_WRITE, m_rows*n_batch*sizeof(float), NULL, NULL);

  // cl_float *host_a = (cl_float*)clEnqueueMapBuffer(
  //             queue,
  //             d_a,
  //             CL_TRUE,
  //             CL_MAP_WRITE,
  //             0,
  //             sizeof(float)*m_rows*m_cols,
  //             0, NULL, NULL, NULL);
  // cl_float *host_b = (cl_float*)clEnqueueMapBuffer(
  //             queue,
  //             d_b,
  //             CL_TRUE,
  //             CL_MAP_WRITE,
  //             0,
  //             sizeof(float)*m_cols*n_batch,
  //             0, NULL, NULL, NULL);

  // clFinish(queue);

  // std::memcpy(host_a, h_a, m_rows*m_cols*sizeof(float));
  // std::memcpy(host_b, h_b, m_cols*n_batch*sizeof(float));

  // clEnqueueUnmapMemObject(queue,d_a,(void *) host_a,0, NULL, NULL);
  // clEnqueueUnmapMemObject(queue,d_b,(void *) host_b,0, NULL, NULL);

  // clFinish(queue);

    // const size_t m = 1008;
    // const size_t n = 2048;
    // const float alpha = 1.0f;
    // const float beta = 0.0f;
    // const auto a_ld = 2048;
    // const size_t x_st = 1;
    // const size_t y_st = 1;

  cl_ulong size;
  clGetDeviceInfo(device_id, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(cl_ulong), &size, 0);
  cout << "Local size: " << size << endl;

  for(int i = 0; i < 1; i++) {
    float* h_c = (float*)malloc(m_rows*n_batch*sizeof(float));
      
    
    // TransposeFloatTensor(h_b, 128, 128, h_bt);



    //  Start Timers
    wall0 = get_wall_time();
    cpu0  = get_cpu_time();

    // Gemv(Layout::kRowMajor, Transpose::kNo,
    //             m, n,
    //             alpha,
    //             d_a, 0, a_ld,
    //             d_b, 0, x_st,
    //             beta,
    //             d_c, 0, y_st,
    //             &queue, NULL);
    // clFinish(queue);

    // PortableMatrixBatchVectorMultiplyAccumulate(h_a,m_rows,m_cols,h_b,n_batch,h_c,1);
    // OpenCLPortableMatrixBatchVectorMultiplyAccumulate(h_at,128,128,h_bt,128,h_c,1,context,queue,program);
    // OpenCLPortableMatrixBatchVectorMultiplyAccumulate2(h_at,m_rows,m_cols,h_b,n_batch,h_c,1,context,queue,program);
    OpenCLPortableMatrixBatchVectorMultiplyAccumulate3(h_a,m_rows,m_cols,h_b,n_batch,h_c,1,context,queue,program);

    //  Stop timers
    wall1 = get_wall_time();
    cpu1  = get_cpu_time();


    // clEnqueueReadBuffer(queue, d_c, CL_TRUE, 0, m_rows*n_batch*sizeof(float), h_c, 0, NULL, NULL );

    // clFinish(queue);

    // clReleaseMemObject(d_a);
    // clReleaseMemObject(d_b);
    // clReleaseMemObject(d_c);

    double sum = 0;
    for(int i = 0; i < m_rows*n_batch; i++) {
      sum += h_c[i];
      if(i > 908)
      cout << h_c[i] << " ";
    }
    cout << endl;
    cout << "Sum:" << sum << endl;

    cout << "Wall Time = " << wall1 - wall0 << endl;
    cout << "CPU Time  = " << cpu1  - cpu0  << endl;

    // sum = 0;
    // for(int i = 0; i < 2048; i++) {
    //   sum += h_b[i];
    //   // cout << h_c[i] << " ";
    // }
    // // cout << endl;
    // cout << "Sum1:" << sum << endl;
    
    // sum = 0;
    // for(int i = 0; i < 2048; i++) {
    //   sum += h_b2[i];
    //   // cout << h_c[i] << " ";
    // }
    // // cout << endl;
    // cout << "Sum2:" << sum << endl;

    free(h_c);
  }
  free(h_a);
  free(h_at);
  free(h_b);
    //   clReleaseMemObject(d_a);
    // clReleaseMemObject(d_b);
    // clReleaseMemObject(d_c);
  clReleaseProgram(program);
  clReleaseCommandQueue(queue);
  clReleaseContext(context); //ngising

  return 0;
}