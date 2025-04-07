#include <unistd.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

// STB image library
#define STB_IMAGE_IMPLEMENTATION
#include "lib/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "lib/stb_image_write.h"

// CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#include "lib/helper_cuda.h"

// Constants
#define HISTOGRAM_LEVELS 256
#define COLOR_CHANNELS 3
#define NUM_BANKS 16
#define LOG_NUM_BANKS 4

// Settings
#define SAVE_TIMING_STATS
#define WRITE_OUTPUT_IMAGE

// Macros
#define CLAMP(a, min, max) ((a) < (min) ? (min) : ((a) > (max) ? (max) : (a)))
#define CLAMP255(a) CLAMP(a, 0, 255)
#define CONFLICT_FREE_OFFSET(n) ((n) >> NUM_BANKS + (n) >> (2 * LOG_NUM_BANKS))

void calculateHistogram(unsigned char *deviceImage, int imageWidthPixel, int imageHeightPixel, unsigned int *histogram);
__global__ void calculateHistogram_kernel(unsigned char *imageData, const int imageWidth, const int imageHeight, unsigned int *sharedHistogram);

void calculateCumulativeDistribution(unsigned int *histogram, unsigned int *cumulativeDistributionHistogram);
__global__ void calculateCumulativeDistribution_kernel(unsigned int *deviceInHistogram, unsigned int *deviceOutHistogram);

void calculateNewLuminances(unsigned char *newLuminances,  int imageWidthPixel, int imageHeightPixel, unsigned int *cumulativeDistributionHistogram);
__global__ void findMin_kernel(unsigned int *deviceCumulativeDistributionHistogram, unsigned int *minimum);
__global__ void calculateNewLuminances_kernel(unsigned char *deviceNewLuminances, unsigned int imageSize, unsigned int *cdf, unsigned int *cdfmin);

void equalize(unsigned char *deviceImage, int imageWidthPixel, int imageHeightPixel, unsigned char *newLuminances);
__global__ void equalize_kernel(unsigned char *deviceImage, int imageWidthPixel, int imageHeightPixel, int threadIdOffset, unsigned char *deviceNewLuminances);

void printHistogram(unsigned int *histogram);
void printKernelRuntime(float elapsedTimeMS);

struct execution_result
{
    int width;
    int height;
    float hist;
    float cdf;
    float equalize;
    float sum;
    float total;
};

int main(int argc, char *args[])
{
    if (argc != 3)
    {
        printf("Error: Invalid amount of arguments. [%d]\n", argc);
        return EXIT_FAILURE;
    }

    char *imageInPath = args[1];
    char *imageOutPath = args[2];

    // Load image
    int imageWidthPixel, imageHeightPixel, cpp, imageSizeBytes;
    unsigned char *image = stbi_load(imageInPath, &imageWidthPixel, &imageHeightPixel, &cpp, COLOR_CHANNELS);
    if (image == NULL)
    {
        printf("Error: Couldn't load image\n");
        return EXIT_FAILURE;
    }
    if (cpp != COLOR_CHANNELS)
    {
        printf("Error: Image is not RGB\n");
        return EXIT_FAILURE;
    }

    imageSizeBytes = imageWidthPixel * imageHeightPixel * COLOR_CHANNELS * sizeof(unsigned char);

    int device;
    struct cudaDeviceProp props;
    cudaGetDeviceProperties(&props, cudaGetDevice(&device));

    // Create time events
    cudaEvent_t startMain, stopMain,
                startTimeHistogramMS, stopTimeHistogramMS, 
                startTimeCumulativeMS, stopTimeCumulativeMS, 
                startTimeLuminancesMS, stopTimeLuminancesMS,
                startTimeEqualizeMS, stopTimeEqualizeMS;

    cudaEventCreate(&startMain);
    cudaEventCreate(&stopMain);
    cudaEventCreate(&startTimeHistogramMS);
    cudaEventCreate(&stopTimeHistogramMS);
    cudaEventCreate(&startTimeCumulativeMS);
    cudaEventCreate(&stopTimeCumulativeMS);
    cudaEventCreate(&startTimeLuminancesMS);
    cudaEventCreate(&stopTimeLuminancesMS);
    cudaEventCreate(&startTimeEqualizeMS);
    cudaEventCreate(&stopTimeEqualizeMS);

    float elapsedTimeMain = 0, 
          elapsedTimeHistogramMS = 0, 
          elapsedTimeLuminancesMS = 0, 
          elapsedTimeCumulativeMS = 0, 
          elapsedTimeEqualizeMS = 0;

    cudaEventRecord(startMain);

    // copy the initial image to the GPU
    unsigned char *deviceImage;
    cudaMalloc((void **)&deviceImage, imageSizeBytes);
    cudaMemcpy(deviceImage, image, imageSizeBytes, cudaMemcpyHostToDevice);

    // STEP 1: Image to YUV and compute the histogram
    cudaEventRecord(startTimeHistogramMS);
    unsigned int *histogram = (unsigned int *)malloc(HISTOGRAM_LEVELS * sizeof(unsigned int));
    calculateHistogram(deviceImage, imageWidthPixel, imageHeightPixel, histogram);
    cudaEventRecord(stopTimeHistogramMS);

    // STEP 2: Compute the cumulative distribution of the histogram
    cudaEventRecord(startTimeCumulativeMS);
    unsigned int *cumulativeDistributionHistogram = (unsigned int *)malloc(HISTOGRAM_LEVELS * sizeof(unsigned int));
    calculateCumulativeDistribution(histogram, cumulativeDistributionHistogram);
    cudaEventRecord(stopTimeCumulativeMS);

    // STEP 3: Computation Of New Pixel Intensities 
    cudaEventRecord(startTimeLuminancesMS);
    unsigned char *newLuminances = (unsigned char *)malloc(HISTOGRAM_LEVELS * sizeof(unsigned char));
    calculateNewLuminances(newLuminances, imageWidthPixel, imageHeightPixel, cumulativeDistributionHistogram);
    cudaEventRecord(stopTimeLuminancesMS);

    // STEP 4: Transform the original image using the scaled cumulative distribution as the transformation function
    cudaEventRecord(stopTimeCumulativeMS);
    equalize(deviceImage, imageWidthPixel, imageHeightPixel, newLuminances);
    cudaEventRecord(stopTimeEqualizeMS);

    // recover data from the GPU to the CPU allocated memory
    cudaMemcpy(image, deviceImage, imageSizeBytes, cudaMemcpyDeviceToHost);
    getLastCudaError("retrieving data from GPU failed in: main()");

    // End the time recording and calculate elapsed times
    cudaEventRecord(stopMain);
    cudaEventSynchronize(stopMain);

    cudaEventElapsedTime(&elapsedTimeMain, startMain, stopMain);
    cudaEventElapsedTime(&elapsedTimeHistogramMS, startTimeHistogramMS, stopTimeHistogramMS);
    cudaEventElapsedTime(&elapsedTimeLuminancesMS, startTimeLuminancesMS, stopTimeLuminancesMS);
    cudaEventElapsedTime(&elapsedTimeCumulativeMS, startTimeCumulativeMS, stopTimeCumulativeMS);
    cudaEventElapsedTime(&elapsedTimeEqualizeMS, startTimeEqualizeMS, stopTimeEqualizeMS);

    elapsedTimeCumulativeMS += elapsedTimeLuminancesMS;

// Output timing stats to file //////////////////////////////////////////////////////////////////////////
#ifdef SAVE_TIMING_STATS
    // execution stats
    struct execution_result result;
    result.width = imageWidthPixel;
    result.height = imageHeightPixel;
    result.hist = elapsedTimeHistogramMS;
    result.cdf = elapsedTimeCumulativeMS;
    result.equalize = elapsedTimeEqualizeMS;
    result.sum = elapsedTimeHistogramMS + elapsedTimeCumulativeMS + elapsedTimeEqualizeMS;
    result.total = elapsedTimeMain;

    FILE *timingFile = fopen("./timing_stats/timing_stats_parallel.txt", "a");
    fprintf(timingFile, "--------------- HISTOGRAM EQUALIZATION - Parallel ---------------\n");
    fprintf(timingFile, "--------------- %s ---------------\n", imageInPath);
    fprintf(timingFile, "Image width: %d\n", imageWidthPixel);
    fprintf(timingFile, "Image height: %d\n", imageHeightPixel);
    fprintf(timingFile, "Histogram: %f ms\n", result.hist);
    fprintf(timingFile, "CDF: %f ms\n", result.cdf);
    fprintf(timingFile, "Equalize: %f ms\n", result.equalize);
    fprintf(timingFile, "Total time: %f ms\n", result.total);
    fprintf(timingFile, "Sum of all times: %f ms\n", result.sum);
    fprintf(timingFile, "-----------------------------------------------------\n");
    fprintf(timingFile, "\n");
    fclose(timingFile);
#endif

#ifdef WRITE_OUTPUT_IMAGE
    // write output image:
    stbi_write_png(imageOutPath, imageWidthPixel, imageHeightPixel, COLOR_CHANNELS, image, imageWidthPixel * COLOR_CHANNELS);
#endif

    // Clean-up events
    cudaEventDestroy(startMain);
    cudaEventDestroy(stopMain);
    cudaEventDestroy(startTimeHistogramMS);
    cudaEventDestroy(stopTimeHistogramMS);
    cudaEventDestroy(startTimeCumulativeMS);
    cudaEventDestroy(stopTimeCumulativeMS);
    cudaEventDestroy(startTimeLuminancesMS);
    cudaEventDestroy(stopTimeLuminancesMS);
    cudaEventDestroy(startTimeEqualizeMS);
    cudaEventDestroy(stopTimeEqualizeMS);

    cudaFree(deviceImage);

    stbi_image_free(image);
    free(histogram);
    free(cumulativeDistributionHistogram);

    return EXIT_SUCCESS;
}

void calculateHistogram(unsigned char *deviceImage, int imageWidthPixel, int imageHeightPixel, unsigned int *histogram)
{
    // pointer to the histogram on the GPU
    unsigned int *deviceHistogram;
    cudaMalloc((void **)&deviceHistogram, HISTOGRAM_LEVELS * sizeof(unsigned int));
    cudaMemset(deviceHistogram, 0, HISTOGRAM_LEVELS * sizeof(unsigned int));
    getLastCudaError("setting up GPU data faled in: calculateHistogram()");

    // set up the grid and block size
    dim3 gridSize(ceil(imageWidthPixel * imageHeightPixel / (float)HISTOGRAM_LEVELS));
    dim3 blockSize(HISTOGRAM_LEVELS);

    // runs KERNEL
    calculateHistogram_kernel<<<gridSize, blockSize>>>(deviceImage, imageWidthPixel, imageHeightPixel, deviceHistogram);
    getLastCudaError("calculateHistogram_kernel() execution failed");

    // recover data from the GPU to the CPU allocated memory
    cudaMemcpy(histogram, deviceHistogram, HISTOGRAM_LEVELS * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    getLastCudaError("retrieving data from GPU failed in: calculateHistogram()");

    cudaFree(deviceHistogram);
    getLastCudaError("freeing memory in calculateHistogram() failed");
}

__global__ void calculateHistogram_kernel(unsigned char *imageData, const int imageWidth, const int imageHeight, unsigned int *sharedHistogram)
{
    // find index of the pixel of the thread
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int indexOffset = blockDim.x * gridDim.x;

    // check current y levels and increment corresponding values
    int imagePixelSize = imageWidth * imageHeight;
    while (index < imagePixelSize)
    {
        unsigned int pixelIdx = index * COLOR_CHANNELS;

        // RBG to YUV conversion
        float r = (float)imageData[pixelIdx + 0];
        float g = (float)imageData[pixelIdx + 1];
        float b = (float)imageData[pixelIdx + 2];

        imageData[pixelIdx + 0] = (unsigned char) CLAMP255((    0.299f * r +    0.587f * g +    0.114f * b) +   0.0f);
        imageData[pixelIdx + 1] = (unsigned char) CLAMP255((-0.168736f * r - 0.331264f * g +      0.5f * b) + 128.0f);
        imageData[pixelIdx + 2] = (unsigned char) CLAMP255((      0.5f * r - 0.418688f * g - 0.081312f * b) + 128.0f);

        atomicAdd(&sharedHistogram[imageData[pixelIdx]], 1);
        index += indexOffset;
    }
}

void calculateCumulativeDistribution(unsigned int *histogram, unsigned int *cumulativeDistributionHistogram)
{
    // pointer to the input histogram on the GPU
    unsigned int *deviceInHistogram;
    cudaMalloc((void **)&deviceInHistogram, HISTOGRAM_LEVELS * sizeof(unsigned int));
    cudaMemcpy(deviceInHistogram, histogram, HISTOGRAM_LEVELS * sizeof(unsigned int), cudaMemcpyHostToDevice);
    // pointer to the output histogram on the GPU
    unsigned int *deviceOutHistogram;
    cudaMalloc((void **)&deviceOutHistogram, HISTOGRAM_LEVELS * sizeof(unsigned int));
    getLastCudaError("setting up GPU data faled in: calculateCumulativeDistribution()");

    // set up the grid and block size
    dim3 gridSize(1);
    dim3 blockSize(32);

    // runs KERNEL
    calculateCumulativeDistribution_kernel<<<gridSize, blockSize>>>(deviceInHistogram, deviceOutHistogram);
    getLastCudaError("calculateCumulativeDistribution_kernel() execution failed");

    // recover data from the GPU to the CPU allocated memory
    cudaMemcpy(cumulativeDistributionHistogram, deviceOutHistogram, HISTOGRAM_LEVELS * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    getLastCudaError("retrieving data from GPU failed in: calculateCumulativeDistribution()");

    cudaFree(deviceInHistogram);
    cudaFree(deviceOutHistogram);
    getLastCudaError("freeing memory in calculateCumulativeDistribution() failed");
}

// algorithm explained: [https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda]
__global__ void calculateCumulativeDistribution_kernel(unsigned int *deviceInHistogram, unsigned int *deviceOutHistogram)
{
    if (threadIdx.x == 0) // Only one thread does the work
    {
        deviceOutHistogram[0] = deviceInHistogram[0];
        for (int i = 1; i < HISTOGRAM_LEVELS; i++)
        {
            deviceOutHistogram[i] = deviceOutHistogram[i - 1] + deviceInHistogram[i];
        }
    }
}

void calculateNewLuminances(unsigned char *newLuminances,  int imageWidthPixel, int imageHeightPixel, unsigned int *cumulativeDistributionHistogram)
{
    // pointer to the cumulative distribution histogram on the GPU
    unsigned int *deviceCumulativeDistributionHistogram;
    cudaMalloc((void **)&deviceCumulativeDistributionHistogram, HISTOGRAM_LEVELS * sizeof(unsigned int));
    cudaMemcpy(deviceCumulativeDistributionHistogram, cumulativeDistributionHistogram, HISTOGRAM_LEVELS * sizeof(unsigned int), cudaMemcpyHostToDevice);
    
    // pointer to the new luminances on the GPU
    unsigned char *deviceNewLuminances;
    cudaMalloc((void **)&deviceNewLuminances, HISTOGRAM_LEVELS * sizeof(unsigned char));

    // pointer to the non zero minimum in the cumulative distribution on the GPU
    unsigned int *cdfmin;
    cudaMalloc((void **)&cdfmin, sizeof(unsigned int));
    getLastCudaError("setting up GPU data faled in: equalize()");

    dim3 gridSize(1);
    dim3 blockSize(HISTOGRAM_LEVELS); 
    dim3 blockSizeMin(32);

    findMin_kernel<<<gridSize, blockSizeMin>>>(deviceCumulativeDistributionHistogram, cdfmin);
    getLastCudaError("findMin_kernel() execution failed");

    calculateNewLuminances_kernel<<<gridSize, blockSize>>>(deviceNewLuminances, imageWidthPixel * imageHeightPixel, deviceCumulativeDistributionHistogram, cdfmin);
    getLastCudaError("calculateNewLuminances_kernel() execution failed");

    // recover data from the GPU to the CPU allocated memory
    cudaMemcpy(newLuminances, deviceNewLuminances, HISTOGRAM_LEVELS * sizeof(unsigned char), cudaMemcpyDeviceToHost);
    getLastCudaError("retrieving data from GPU failed in: calculateCumulativeDistribution()");

    // clean up
    cudaFree(deviceCumulativeDistributionHistogram);
    cudaFree(cdfmin);
    cudaFree(deviceNewLuminances);
    getLastCudaError("freeing memory in luminances() failed");
}

__global__ void findMin_kernel(unsigned int *deviceCumulativeDistributionHistogram, unsigned int *minimum)
{
    if (threadIdx.x == 0)
    {
        *minimum = 0;
        for (int i = 0; *minimum == 0 && i < HISTOGRAM_LEVELS; i++)
        {
            *minimum = deviceCumulativeDistributionHistogram[i];
        }
    }
}

__global__ void calculateNewLuminances_kernel(unsigned char *deviceNewLuminances, unsigned int imageSize, unsigned int *cdf, unsigned int *cdfmin)
{
    deviceNewLuminances[threadIdx.x] =  (unsigned char) CLAMP255(floor(((float)(cdf[threadIdx.x] - *cdfmin) / (float)(imageSize - *cdfmin)) * (HISTOGRAM_LEVELS - 1.0)));
}

void equalize(unsigned char *deviceImage, int imageWidthPixel, int imageHeightPixel, unsigned char *newLuminances)
{
    // pointer to the new luminances on the GPU
    unsigned char *deviceNewLuminances;
    cudaMalloc((void **)&deviceNewLuminances, HISTOGRAM_LEVELS * sizeof(unsigned char));
    cudaMemcpy(deviceNewLuminances, newLuminances, HISTOGRAM_LEVELS * sizeof(unsigned char), cudaMemcpyHostToDevice);

    dim3 gridSizeEqualize(ceil(imageWidthPixel * imageHeightPixel) / 256.0);
    dim3 blockSizeEqualize(256);

    // pointer to the thread id offset on new iteration
    int threadIdOffset = blockSizeEqualize.x * gridSizeEqualize.x;

    equalize_kernel<<<gridSizeEqualize, blockSizeEqualize>>>(deviceImage, imageWidthPixel, imageHeightPixel, threadIdOffset, deviceNewLuminances);
    getLastCudaError("equalize_kernel() execution failed");

    cudaFree(deviceNewLuminances);
    getLastCudaError("freeing memory in luminances() failed");
}

__global__ void equalize_kernel(unsigned char *deviceImage, int imageWidthPixel, int imageHeightPixel, int threadIdOffset, unsigned char *deviceNewLuminances)
{
    int threadId = threadIdx.x + blockIdx.x * blockDim.x;

    while (threadId < imageWidthPixel * imageHeightPixel)
    {
        unsigned int pixelIdx = threadId * COLOR_CHANNELS;

        // YUV to RGB conversion
        float y = (float)deviceNewLuminances[deviceImage[pixelIdx]];
        float u = (float)deviceImage[pixelIdx + 1] - 128.0f;
        float v = (float)deviceImage[pixelIdx + 2] - 128.0f;

        deviceImage[pixelIdx + 0] = (unsigned char)(CLAMP255((float)(y + 1.402f * v)));
        deviceImage[pixelIdx + 1] = (unsigned char)(CLAMP255((float)(y - 0.344136f * u - 0.714136f * v)));
        deviceImage[pixelIdx + 2] = (unsigned char)(CLAMP255((float)(y + 1.772f * u)));

        threadId += threadIdOffset;
    }
}

void printHistogram(unsigned int *histogram)
{
    for (int i = 0; i < HISTOGRAM_LEVELS; i++)
    {
        printf("%i = %llu\n", i, histogram[i]);
    }
}

void printKernelRuntime(float elapsedTimeMS)
{
    printf("Kerner run time: %3.3f ms\n", elapsedTimeMS);
}
