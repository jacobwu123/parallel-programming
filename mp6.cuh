// Histogram Equalization

#include <wb.h>
#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)
#define HISTOGRAM_LENGTH 256
#define BLOCK_SIZE 1024


//@@ insert code here
__global__ void castImage(float* imageOld, unsigned char* imageNew, int size){
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if(i < size){
    imageNew[i] = (unsigned char)(255*imageOld[i]);
  }
}
__global__ void toGrayScale(unsigned char*image, unsigned char*imageGray,int sizeImage, int sizeGray){
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if(3*i+3 < sizeImage){
    unsigned char r = image[3*i];
    unsigned char g = image[3*i+1];
    unsigned char b = image[3*i+2];
    if(i < sizeGray)
      imageGray[i] = 0.21*r + 0.71*g + 0.07*b;
  }
}
__global__ void histo(unsigned char*image, unsigned int* histogram, int sizeImage){
  int i = blockDim.x*blockIdx.x+threadIdx.x;
  __shared__ unsigned int histo_s[HISTOGRAM_LENGTH];
  if(threadIdx.x < HISTOGRAM_LENGTH)
    histo_s[threadIdx.x] = 0;
  __syncthreads();

  for(unsigned int j = i;j < sizeImage;j+=blockDim.x*gridDim.x){
    atomicAdd(&(histo_s[image[j]]),1);
  }
  __syncthreads();
  
  if(threadIdx.x<HISTOGRAM_LENGTH)
    atomicAdd(&(histogram[threadIdx.x]),histo_s[threadIdx.x]);
  
  if (i<HISTOGRAM_LENGTH) {
		printf("hist (bx,tx)=(%d,%d): %u\n", blockIdx.x, threadIdx.x, histogram[threadIdx.x]);
	}
}

__global__ void computeCdf(unsigned int* histogram, float* cdf, int sizeImage){
  int i = blockDim.x*blockIdx.x+threadIdx.x;
  __shared__ float p[HISTOGRAM_LENGTH];
  
  if(i < HISTOGRAM_LENGTH)
    p[i] = (float)histogram[i]*1.0/sizeImage; 
  __syncthreads();

  for(unsigned int stride =1;stride <= HISTOGRAM_LENGTH/2;stride*= 2 ){
    __syncthreads();
    unsigned int idx = (threadIdx.x+1)*2*stride-1;
    if(idx < HISTOGRAM_LENGTH)
      p[idx] += p[idx-stride];
  }
  
  for(unsigned int stride = HISTOGRAM_LENGTH/4;stride >= 1;stride /= 2){
    __syncthreads();
    unsigned int idx = (threadIdx.x+1)*2*stride-1;
    if(idx+stride < HISTOGRAM_LENGTH)
      p[idx+stride] += p[idx];
  }
  __syncthreads();
  
  if(i < HISTOGRAM_LENGTH)
    cdf[i] = p[i];
  
  
	if (i < HISTOGRAM_LENGTH)
		printf("cdf[%d]: %f\n", i, cdf[i]);
}

__global__ void equalization(unsigned char * image, float* cdf, int sizeImage){
  int i = blockDim.x*blockIdx.x+threadIdx.x;
  float cdfmin = cdf[0];
  for(int idx = i; idx < sizeImage;idx+= blockDim.x*gridDim.x){
    float temp=255*(cdf[image[idx]] - cdfmin)/(1.0 - cdfmin);
    if (temp < 0)
      temp = 0;
    if(temp > 255.0)
      temp = 255.0;
    image[i] = (unsigned char)temp;
  }

}

__global__ void castBack(unsigned char*image, float* Output,int sizeImage){
  int i = blockDim.x*blockIdx.x+threadIdx.x;
  if(i < sizeImage)
    Output[i] = (float)(image[i]/255.0);
  if (i == 0) { printf("unsignedCharToFloat i: %d\t", i); 
               printf("input[i]: %u\t", image[i]); 
               printf("output[i]: %f\n", Output[i]); }
}

int main(int argc, char **argv) {
  wbArg_t args;
  int imageWidth;
  int imageHeight;
  int imageChannels;
  wbImage_t inputImage;
  wbImage_t outputImage;
  float *hostInputImageData;
  float *hostOutputImageData;
  const char *inputImageFile;
  

  //@@ Insert more code here
  float *deviceInputImageData;
  unsigned char * castedImage;
  unsigned char * grayImage;
  unsigned int* histogram;
  float* cdf;
  float *deviceOutputImageData;
  
  unsigned char * hostCastedImage;
  unsigned char* hostGrayImage;
  unsigned int* hostHistogram;
  float* hostCdf;
  
  args = wbArg_read(argc, argv); /* parse the input arguments */
  
  inputImageFile = wbArg_getInputFile(args, 0);

    
  wbTime_start(Generic, "Importing data and creating memory on host");
  inputImage = wbImport(inputImageFile);
  imageWidth = wbImage_getWidth(inputImage);
  imageHeight = wbImage_getHeight(inputImage);
  imageChannels = wbImage_getChannels(inputImage);
  outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);
  hostInputImageData = wbImage_getData(inputImage);
  hostOutputImageData = wbImage_getData(outputImage);
  wbTime_stop(Generic, "Importing data and creating memory on host");
  int imageSize = imageWidth*imageHeight*imageChannels;
  int graySize = imageWidth*imageHeight;
  hostCastedImage = (unsigned char *)malloc( imageSize* sizeof(unsigned char));
  hostGrayImage = (unsigned char *)malloc(graySize* sizeof(unsigned char));
  hostHistogram = (unsigned int *)malloc(HISTOGRAM_LENGTH * sizeof(unsigned int));
  hostCdf = (float *)malloc(HISTOGRAM_LENGTH * sizeof(float));
  //@@ insert code here
  
  wbTime_start(GPU, "Allocating GPU memory.");
  cudaMalloc((void**)&deviceInputImageData, imageSize*sizeof(float));
  cudaMalloc((void**)&deviceOutputImageData, imageSize*sizeof(float));
  cudaMalloc((void**)&castedImage, imageSize*sizeof(unsigned char));
  cudaMalloc((void**)&grayImage, graySize*sizeof(unsigned char));
  cudaMalloc((void**)&histogram,HISTOGRAM_LENGTH*sizeof(unsigned int));
  cudaMalloc((void**)&cdf,HISTOGRAM_LENGTH*sizeof(float));
  wbTime_stop(GPU, "Allocating GPU memory.");  
  
  wbTime_start(GPU, "Copying input memory to the GPU.");
  wbCheck(cudaMemcpy(deviceInputImageData, hostInputImageData, imageSize* sizeof(float),
                     cudaMemcpyHostToDevice));
  wbTime_stop(GPU, "Copying input memory to the GPU.");
  
  //@@ Initialize the grid and block dimensions here
  dim3 dimBlock(BLOCK_SIZE,1,1);
  dim3 dimGrid(ceil(imageSize/(1.0*BLOCK_SIZE)),1,1);
  wbTime_start(Compute, "Performing CUDA computation");
  castImage<<<dimGrid,dimBlock>>>(deviceInputImageData, castedImage,imageSize);
  toGrayScale<<<dimGrid,dimBlock>>>(castedImage, grayImage,imageSize, graySize);
  histo<<<dimGrid,dimBlock>>>(grayImage, histogram, graySize);
  computeCdf<<<dimGrid,dimBlock>>>(histogram, cdf, graySize);
  equalization<<<dimGrid,dimBlock>>>(castedImage, cdf, imageSize);
  castBack<<<dimGrid, dimBlock>>>(castedImage,deviceOutputImageData,imageSize);
  wbTime_start(Copy, "Copying output memory to the CPU");
  wbCheck(cudaMemcpy(hostOutputImageData, deviceOutputImageData, imageSize * sizeof(float),
                     cudaMemcpyDeviceToHost));
//  wbCheck(cudaMemcpy(hostCastedImage, castedImage, imageSize * sizeof(unsigned char),
//                     cudaMemcpyDeviceToHost));
//  wbCheck(cudaMemcpy(hostGrayImage, grayImage, graySize * sizeof(unsigned char),
//                     cudaMemcpyDeviceToHost));
//  wbCheck(cudaMemcpy(hostHistogram, histogram, HISTOGRAM_LENGTH * sizeof(unsigned int),
//                     cudaMemcpyDeviceToHost));
  wbCheck(cudaMemcpy(hostHistogram, histogram, HISTOGRAM_LENGTH * sizeof(unsigned int),
                     cudaMemcpyDeviceToHost));
  wbTime_stop(Copy, "Copying output memory to the CPU");
  wbSolution(args, outputImage);

  printf("hostHist,%d\n",hostHistogram[0]);
  
  //@@ insert code here
  wbTime_start(GPU, "Freeing GPU Memory");
  cudaFree(deviceInputImageData);
  cudaFree(deviceOutputImageData);
  cudaFree(castedImage);
  cudaFree(grayImage);
  cudaFree(histogram);
  cudaFree(cdf);
  wbTime_stop(GPU, "Freeing GPU Memory");
  
  free(hostInputImageData);
  free(hostOutputImageData);
  free(hostCastedImage);
  free(hostGrayImage);
  free(hostHistogram);
  free(hostCdf);
  
  return 0;
}
