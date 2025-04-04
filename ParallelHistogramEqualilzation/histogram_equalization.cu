#include <unistd.h>
#include <stdlib.h>
#include <math.h>

#include <cuda_runtime.h>
#include <cuda.h>
#include "helper_cuda.h"

// STB image library
#define STB_IMAGE_IMPLEMENTATION
#include "lib/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "lib/stb_image_write.h"

// Constants
#define HISTOGRAM_LEVELS 256
#define COLOR_CHANNELS 3

// Settings
#define SAVE_TIMING_STATS
#define WRITE_OUTPUT_IMAGE

// Macros
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define CLAMP(a, min, max) ((a) < (min) ? (min) : ((a) > (max) ? (max) : (a)))
#define CLAMP255(a) CLAMP(a, 0, 255)

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

unsigned int findMin(unsigned int *cdf)
{
    unsigned int min = 0;
    for (int i = 0; min == 0 && i < HISTOGRAM_LEVELS; i++)
    {
        min = cdf[i];
    }
    return min;
}

unsigned char scale(unsigned int cdf, unsigned int cdfmin, unsigned int imageSize)
{
    int scale = CLAMP255(floor(((float)(cdf - cdfmin) / (float)(imageSize - cdfmin)) * (HISTOGRAM_LEVELS - 1.0)));
    return (unsigned char) scale;
}

void RGBtoYUV(unsigned char *image, int width, int height)
{
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            unsigned int pixelIdx = (y * width + x) * COLOR_CHANNELS;

            float r = (float)image[pixelIdx + 0];
            float g = (float)image[pixelIdx + 1];
            float b = (float)image[pixelIdx + 2];

            // YUV conversion formula
            unsigned char y = (unsigned char) CLAMP255((    0.299f * r +    0.587f * g +    0.114f * b) +   0.0f);
            unsigned char u = (unsigned char) CLAMP255((-0.168736f * r - 0.331264f * g +      0.5f * b) + 128.0f);
            unsigned char v = (unsigned char) CLAMP255((      0.5f * r - 0.418688f * g - 0.081312f * b) + 128.0f);

            // assign YUV values back to the image
            image[pixelIdx + 0] = y;
            image[pixelIdx + 1] = u;
            image[pixelIdx + 2] = v;
        }
    }
}

void CalculateHistogram(unsigned char *image, int width, int height, unsigned int *histogram)
{
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            unsigned int pixelIdx = (y * width + x) * COLOR_CHANNELS;
            histogram[image[pixelIdx]]++;
        }
    }
}

void CalculateCDF(unsigned int *histogram, unsigned int *cdf)
{
    cdf[0] = histogram[0];
    for (int i = 1; i < HISTOGRAM_LEVELS; i++)
    {
        cdf[i] = cdf[i - 1] + histogram[i];
    }
}

void Equalize(unsigned char *image, int width, int height, unsigned int *cdf)
{
    unsigned int imageSize = width * height;
    unsigned int cdfmin = findMin(cdf);

    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            unsigned int pixelIdx = (y * width + x) * COLOR_CHANNELS;
            image[pixelIdx] = scale(cdf[image[pixelIdx]], cdfmin, imageSize);
        }
    }
}

void YUVtoRGB(unsigned char *image, int width, int height)
{
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            unsigned int pixelIdx = (y * width + x) * COLOR_CHANNELS;

            float y = (float)image[pixelIdx + 0];
            float u = (float)image[pixelIdx + 1];
            float v = (float)image[pixelIdx + 2];

            // RBG conversion formula
            u -= 128.0f;
            v -= 128.0f;

            unsigned char r = (unsigned char) CLAMP255((1.0f * y +      0.0f * u +    1.402f * v));
            unsigned char g = (unsigned char) CLAMP255((1.0f * y - 0.344136f * u - 0.714136f * v));
            unsigned char b = (unsigned char) CLAMP255((1.0f * y +    1.772f * u +      0.0f * v));

            // assign YUV values back to the image
            image[pixelIdx + 0] = r;
            image[pixelIdx + 1] = g;
            image[pixelIdx + 2] = b;
        }
    }
}

int main(int argc, char *args[])
{
    if (argc != 3)
    {
        printf("Error: Invalid amount of arguments. [%d]\n", argc);
        exit(1);
    }

    char *imageInPath = args[1];
    char *imageOutPath = args[2];

    // Read image from file
    int imageWidthPixel, imageHeightPixel, cpp;
    unsigned char *image = stbi_load(imageInPath, &imageWidthPixel, &imageHeightPixel, &cpp, COLOR_CHANNELS);
    if (image == NULL)
    {
        printf("Error in loading the image\n");
        return EXIT_FAILURE;
    }
    if (cpp != COLOR_CHANNELS)
    {
        printf("Error: Image is not RGB\n");
        return EXIT_FAILURE;
    }

    // Allocate memory for raw output image data, histogram, and CDF
    unsigned int *histogram = (unsigned int *) calloc(HISTOGRAM_LEVELS, sizeof(unsigned int));
    unsigned int *CDF = (unsigned int *) calloc(HISTOGRAM_LEVELS, sizeof(unsigned int));

    // Create time events
    cudaEvent_t startMain, stopMain,
                startTimeRGBtoYUV, stopTimeRGBtoYUV, 
                startTimeHistogramMS, stopTimeHistogramMS, 
                startTimeCumulativeMS, stopTimeCumulativeMS, 
                startTimeEqualizeMS, stopTimeEqualizeMS, 
                startTimeYUVtoRGB, stopTimeYUVtoRGB;

    cudaEventCreate(&startMain);
    cudaEventCreate(&stopMain);
    cudaEventCreate(&startTimeRGBtoYUV);
    cudaEventCreate(&stopTimeRGBtoYUV);
    cudaEventCreate(&startTimeHistogramMS);
    cudaEventCreate(&stopTimeHistogramMS);
    cudaEventCreate(&startTimeCumulativeMS);
    cudaEventCreate(&stopTimeCumulativeMS);
    cudaEventCreate(&startTimeEqualizeMS);
    cudaEventCreate(&stopTimeEqualizeMS);
    cudaEventCreate(&startTimeYUVtoRGB);
    cudaEventCreate(&stopTimeYUVtoRGB);

    float elapsedTimeRGBtoYUV = 0,
          elapsedTimeHistogramMS= 0,
          elapsedTimeCumulativeMS= 0, 
          elapsedTimeEqualizeMS= 0,
          elapsedTimeYUVtoRGB= 0,
          elapsedMain= 0;

    cudaEventRecord(startMain);

    // 1. Transform the image from RGB to YUV
    cudaEventRecord(startTimeRGBtoYUV);
    RGBtoYUV(image, imageWidthPixel, imageHeightPixel);
    cudaEventRecord(stopTimeRGBtoYUV);

    // 2. Compute the luminance histogram
    cudaEventRecord(startTimeHistogramMS);
    CalculateHistogram(image, imageWidthPixel, imageHeightPixel, histogram);
    cudaEventRecord(stopTimeHistogramMS);

    // 3. Calculate the cumulative histogram
    cudaEventRecord(startTimeCumulativeMS);
    CalculateCDF(histogram, CDF);
    cudaEventRecord(stopTimeCumulativeMS);

    // 4. Calculate new pixel luminances from original luminance based on the histogram equalization formula
    // 5. Assign new luminance to each pixel
    cudaEventRecord(startTimeEqualizeMS);
    Equalize(image, imageWidthPixel, imageHeightPixel, CDF);
    cudaEventRecord(stopTimeEqualizeMS);

    // 6. Convert the image back to RGB colour space
    cudaEventRecord(startTimeYUVtoRGB);
    YUVtoRGB(image, imageWidthPixel, imageHeightPixel);
    cudaEventRecord(stopTimeYUVtoRGB);

    // End the time recording and calculate elapsed times
    cudaEventRecord(stopMain);
    cudaEventSynchronize(stopMain);

    cudaEventElapsedTime(&elapsedMain, startMain, stopMain);
    cudaEventElapsedTime(&elapsedTimeRGBtoYUV, startTimeRGBtoYUV, stopTimeRGBtoYUV);
    cudaEventElapsedTime(&elapsedTimeHistogramMS, startTimeHistogramMS, stopTimeHistogramMS);
    cudaEventElapsedTime(&elapsedTimeCumulativeMS, startTimeCumulativeMS, stopTimeCumulativeMS);
    cudaEventElapsedTime(&elapsedTimeEqualizeMS, startTimeEqualizeMS, stopTimeEqualizeMS);
    cudaEventElapsedTime(&elapsedTimeYUVtoRGB, startTimeYUVtoRGB, stopTimeYUVtoRGB);
    
    elapsedTimeHistogramMS += elapsedTimeRGBtoYUV; // add RGB to YUV time to compare with CUDA implementation
    elapsedTimeEqualizeMS += elapsedTimeYUVtoRGB; // add YUV to RGB time to compare with CUDA implementation


// Output timing stats to file //////////////////////////////////////////////////////////////////////////
#ifdef SAVE_TIMING_STATS
    struct execution_result result;
    result.width = imageWidthPixel;
    result.height = imageHeightPixel;
    result.hist = elapsedTimeHistogramMS;
    result.cdf = elapsedTimeCumulativeMS;
    result.equalize = elapsedTimeEqualizeMS;
    result.sum = elapsedTimeHistogramMS + elapsedTimeCumulativeMS + elapsedTimeEqualizeMS;
    result.total = elapsedMain;

    FILE *timingFile = fopen("./timing_stats/timing_stats_serial.txt", "a");
    fprintf(timingFile, "--------------- HISTOGRAM EQUALIZATION - Serial ---------------\n");
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
    // Write output image:
    stbi_write_png(imageOutPath, imageWidthPixel, imageHeightPixel, COLOR_CHANNELS, image, imageWidthPixel * COLOR_CHANNELS);
#endif

    // Clean-up events
    cudaEventDestroy(startMain);
    cudaEventDestroy(stopMain);
    cudaEventDestroy(startTimeRGBtoYUV);
    cudaEventDestroy(stopTimeRGBtoYUV);
    cudaEventDestroy(startTimeHistogramMS);
    cudaEventDestroy(stopTimeHistogramMS);
    cudaEventDestroy(startTimeCumulativeMS);
    cudaEventDestroy(stopTimeCumulativeMS);
    cudaEventDestroy(startTimeEqualizeMS);
    cudaEventDestroy(stopTimeEqualizeMS);
    cudaEventDestroy(startTimeYUVtoRGB);
    cudaEventDestroy(stopTimeYUVtoRGB);

    // Free memory
    stbi_image_free(image);
    free(histogram);
    free(CDF);

    return EXIT_SUCCESS;
}