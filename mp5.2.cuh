// MP Scan
// Given a list (lst) of length n
// Output its prefix sum = {lst[0], lst[0] + lst[1], lst[0] + lst[1] + ...
// +
// lst[n-1]}

#include <wb.h>

#define BLOCK_SIZE 1024 //@@ You can change this

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

__global__ void addToAll(float *input, float *S, int len){
  if(blockIdx.x >=1){
    int idx = 2*blockDim.x*blockIdx.x + threadIdx.x;
    if(idx < len)
      input[idx] += S[blockIdx.x-1];
    if(idx+blockDim.x < len)
      input[idx+blockDim.x] += S[blockIdx.x-1];
  }
}
__global__ void scan2(float *input, float *output, int len) {
  __shared__ float XY[2*BLOCK_SIZE];
  int idx = 2*blockDim.x*blockIdx.x + threadIdx.x;
  if(idx < len) XY[threadIdx.x] = input[idx];
  else XY[threadIdx.x] = 0;
  if(idx + blockDim.x < len) XY[threadIdx.x+blockDim.x] = input[idx+blockDim.x];
  else XY[threadIdx.x+blockDim.x] = 0;
  __syncthreads();
  
  for(unsigned int stride = 1; stride <= blockDim.x; stride *= 2){
    __syncthreads();
    int index  = (threadIdx.x+1)*2*stride-1;
    if(index < 2*BLOCK_SIZE)
      XY[index] += XY[index-stride];
  }  
  
  for(unsigned int stride = blockDim.x/2; stride >= 1; stride /= 2){
    __syncthreads();
    int index  = (threadIdx.x+1)*2*stride-1;
    if(index +stride < 2*BLOCK_SIZE)
      XY[index+stride] += XY[index];
  }
  __syncthreads();
  if(idx < len)
    output[idx] = XY[threadIdx.x];    
  if(idx+blockDim.x < len)
    output[idx+blockDim.x] = XY[threadIdx.x+blockDim.x]; 
}
__global__ void scan(float *input, float *output, float * S, int len) {
  //@@ Modify the body of this function to complete the functionality of
  //@@ the scan on the device
  //@@ You may need multiple kernel calls; write your kernels before this
  //@@ function and call them from the host
  __shared__ float XY[2*BLOCK_SIZE];
  int idx = 2*blockDim.x*blockIdx.x + threadIdx.x;
  if(idx < len)
    XY[threadIdx.x] = input[idx];
  else
    XY[threadIdx.x] = 0;
  if(idx + blockDim.x < len)
    XY[threadIdx.x+blockDim.x] = input[idx+blockDim.x];
  else
    XY[threadIdx.x+blockDim.x] = 0;
  __syncthreads();
  
  
  for(unsigned int stride = 1; stride <= blockDim.x; stride *= 2){
    __syncthreads();
    int index  = (threadIdx.x+1)*2*stride-1;
    if(index < 2*BLOCK_SIZE)
      XY[index] += XY[index-stride];
  }  
  
  
  for(unsigned int stride = blockDim.x/2; stride >= 1; stride /= 2){
    __syncthreads();
    int index  = (threadIdx.x+1)*2*stride-1;
    if(index +stride < 2*BLOCK_SIZE)
      XY[index+stride] += XY[index];
  }
  __syncthreads();
  
  
  if(idx < len)
    output[idx] = XY[threadIdx.x];
   
  if(idx+blockDim.x < len)
    output[idx+blockDim.x] = XY[threadIdx.x+blockDim.x];
  
  
  __syncthreads();
  if(threadIdx.x == blockDim.x-1)
    S[blockIdx.x] = XY[2*BLOCK_SIZE-1]; 
  
}

int main(int argc, char **argv) {
  wbArg_t args;
  float *hostInput;  // The input 1D list
  float *hostOutput; // The output list
  float *deviceInput;
  float *deviceOutput;
  float *tempSum; //temporary Sum array for the second phase
  float *tempSumHost;
  int numElements; // number of elements in the list
  int numBlocks; //number of blocks
  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &numElements);
  hostOutput = (float *)malloc(numElements * sizeof(float));
  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The number of input elements in the input is ",
        numElements);

  wbTime_start(GPU, "Allocating GPU memory.");
  wbCheck(cudaMalloc((void **)&deviceInput, numElements * sizeof(float)));
  wbCheck(cudaMalloc((void **)&deviceOutput, numElements * sizeof(float)));
  wbCheck(cudaMalloc((void **)&tempSum, ceil(numElements/(2.0*BLOCK_SIZE)) * sizeof(float)));
  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Clearing output memory.");
  wbCheck(cudaMemset(deviceOutput, 0, numElements * sizeof(float)));
  wbTime_stop(GPU, "Clearing output memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
  wbCheck(cudaMemcpy(deviceInput, hostInput, numElements * sizeof(float),
                     cudaMemcpyHostToDevice));
  wbTime_stop(GPU, "Copying input memory to the GPU.");

  //@@ Initialize the grid and block dimensions here
  dim3 dimBlock(BLOCK_SIZE,1,1);
  dim3 dimGrid(ceil(numElements/(2.0*BLOCK_SIZE)),1,1);
  wbTime_start(Compute, "Performing CUDA computation");
  //@@ Modify this to complete the functionality of the scan
  //@@ on the deivce
  scan<<<dimGrid, dimBlock>>>(deviceInput,deviceOutput,tempSum,numElements);
  numBlocks = ceil(numElements/(2.0*BLOCK_SIZE));
  scan2<<<ceil(numBlocks/(2.0*BLOCK_SIZE)), BLOCK_SIZE>>>(tempSum,tempSum,ceil(numElements/(2.0*BLOCK_SIZE)));
  numBlocks = ceil(numElements/(2.0*BLOCK_SIZE));
  if(numBlocks > 1)
    addToAll<<<numBlocks, BLOCK_SIZE>>>(deviceOutput,tempSum, numElements);
  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  wbCheck(cudaMemcpy(hostOutput, deviceOutput, numElements * sizeof(float),
                     cudaMemcpyDeviceToHost));
  tempSumHost = (float *)malloc(ceil(numElements/(2.0*BLOCK_SIZE)) * sizeof(float));
  wbCheck(cudaMemcpy(tempSumHost, tempSum, ceil(numElements/(2.0*BLOCK_SIZE)) * sizeof(float),
                     cudaMemcpyDeviceToHost));

  wbTime_stop(Copy, "Copying output memory to the CPU");

  wbTime_start(GPU, "Freeing GPU Memory");
  cudaFree(deviceInput);
  cudaFree(deviceOutput);
  cudaFree(tempSum);
  wbTime_stop(GPU, "Freeing GPU Memory");

  wbSolution(args, hostOutput, numElements);

  free(hostInput);
  free(hostOutput);
  free(tempSumHost);
  return 0;
}
