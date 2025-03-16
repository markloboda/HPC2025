// LOCAL RUNNING
// gcc -lm --openmp -g3 -O0 seam_carving.c -o seam_carving.out; ./seam_carving.out ./test_images/720x480.png ./output_images/720x480.png 720

// SYSTEM LIBS //////////////////////////////////////////////////////////////////////////
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <limits.h>
#include <math.h>
#include <string.h>
#include <stdbool.h>


// IMPORTED LIBS //////////////////////////////////////////////////////////////////////////
#define STB_IMAGE_IMPLEMENTATION
#include "lib/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "lib/stb_image_write.h"


// MACROS ////////////////////////////////////////////////////////////////////////////////
#define max(a,b) \
   ({ __typeof__ (a) _a = (a); \
       __typeof__ (b) _b = (b); \
     _a > _b ? _a : _b; })

#define min(a,b) \
   ({ __typeof__ (a) _a = (a); \
       __typeof__ (b) _b = (b); \
     _a < _b ? _a : _b; })


// CONSTANTS //////////////////////////////////////////////////////////////////////////////
#define STB_COLOR_CHANNELS 0 // 0 for dynamic color channels
#define SEAM -1

int outputDebugCount = 0;

// Main image object
typedef struct _Image_
{
    unsigned char* data;
    char* fPath;
    int width;
    int height;
    int channelCount;
} Image;

Image imageIn, imageOut, debugImage;

typedef struct __ImageProcessData__
{
    unsigned int* imgEnergy;
    unsigned int* imgSeam;
    int width;
    int height;
} ImageProcessData;

unsigned char *getPixel(Image* image, int x, int y)
{
    if (x >= image->width || y >= image->height || x < 0 || y < 0) {
        return NULL;
    }

    const int pixelCount = image->width * image->height * image->channelCount;
    const int pixelPos = (y * image->width + x) * image->channelCount;

    if (pixelPos >= pixelCount || pixelPos < 0) {
        return NULL;
    }

    return &image->data[pixelPos];
}

unsigned char* calcPixelE(Image* image, int x, int y)  // Only used for energy calculation
{
    // if x and y outside bounds, use the closest pixel
    if (x < 0) x++;
    if (y < 0) y++;
    if (x == image->width) x--;
    if (y == image->height) y--;

    return getPixel(image, x, y);
}

unsigned int calcEnergyPixel(Image* image, int x, int y)
{
    int energy = 0;
    for (int rgbChannel = 0; rgbChannel < image->channelCount; rgbChannel++)
    {
        int Gx = - calcPixelE(image, x - 1, y - 1)[rgbChannel] -
                 2 * calcPixelE(image, x - 1, y)[rgbChannel] -
                 calcPixelE(image, x - 1, y + 1)[rgbChannel] +
                 calcPixelE(image, x + 1, y - 1)[rgbChannel] +
                 2 * calcPixelE(image, x + 1, y)[rgbChannel] +
                 calcPixelE(image, x + 1, y + 1)[rgbChannel];

        int Gy = calcPixelE(image, x - 1, y - 1)[rgbChannel] +
                 2 * calcPixelE(image, x, y - 1)[rgbChannel] +
                 calcPixelE(image, x + 1, y - 1)[rgbChannel] -
                 calcPixelE(image, x - 1, y + 1)[rgbChannel] -
                 2 * calcPixelE(image, x, y + 1)[rgbChannel] -
                 calcPixelE(image, x + 1, y + 1)[rgbChannel];

        energy += sqrt(pow(Gx, 2) + pow(Gy, 2));
    }

    return energy / image->channelCount;
}

void calcEnergyFull(Image* image, unsigned int* energyImage)
{
    for (int y = 0; y < image->height; y++)
    {
        for (int x = 0; x < image->width; x++)
        {
            unsigned int pixelPos = y * image->width + x;
            energyImage[pixelPos] = calcEnergyPixel(image, x, y);
        }
    }
}

unsigned int getPixelIdx(ImageProcessData* data, int x, int y)
{
    return y * data->width + x;
}

void getPixelPos(unsigned int idx, ImageProcessData* data, int* x, int* y)
{
    *x = idx % data->width;
    *y = idx / data->width;
}

unsigned int getPixelEnergy(ImageProcessData* data, bool fromEnergy, int* x, int* y)
{
    unsigned int *arr;
    if (fromEnergy)
    {
        arr = data->imgEnergy;
    }
    else
    {
        arr = data->imgSeam;
    }

    if (x < 0 || y < 0 || *x >= data->width || *y >= data->height)
    {
        return UINT_MAX;
    }

    int energy = arr[getPixelIdx(data, *x, *y)];

    int startX = *x;
    while (energy == SEAM)
    {
        *x++;
        if (*x >= data->width)
        {
            *x = startX - 1;
            break;
        }
        energy = arr[getPixelIdx(data, *x, *y)];
    }

    energy = arr[getPixelIdx(data, *x, *y)];
    while (energy == SEAM)
    {
        *x--;
        if (*x < 0)
        {
            printf("Error: No valid pixels found in row [%d]\n", *y);
            exit(EXIT_FAILURE);
        }
        energy = arr[getPixelIdx(data, *x, *y)];
    }

    return energy;
}

void seamIdentification(ImageProcessData* data)
{
    // Starts at the bottom (the bottom row is skipped)
    for (int y = data->height - 2; y >= 0; y--)
    {
        for (int x = 0; x < data->width; x++)
        {
            if (data->imgEnergy[getPixelIdx(data, x, y)] == SEAM) continue;

            int xPtr = x;
            int yPtr = y + 1;
            int rightEnergy = getPixelEnergy(data, true, &xPtr, &yPtr);

            xPtr = x - 1;
            int centerEnergy = getPixelEnergy(data, true, &xPtr, &yPtr);

            xPtr = x - 1;
            int leftEnergy = getPixelEnergy(data, true, &xPtr, &yPtr);

            data->imgSeam[getPixelIdx(data, x, y)] = data->imgEnergy[getPixelIdx(data, x, y)] +
                min(min(leftEnergy, centerEnergy), rightEnergy);
        }
    }
}

void seamRemoval(ImageProcessData* data)
{
    // Find the minimum energy in the top row
    int curX = 0;
    for (int x = 1; x < data->width; x++)
    {
        if (data->imgSeam[getPixelIdx(data, x, 0)] < data->imgSeam[getPixelIdx(data, curX, 0)])
        {
            curX = x;
        }
    }

    int idx = getPixelIdx(data, curX, 0);
    data->imgEnergy[idx] = SEAM;
    data->imgSeam[idx] = SEAM;

    for (int y = 1; y < data->height; y++)
    {
        int yPtr = y + 1;
        int xPtr = curX;
        int rightEnergy = getPixelEnergy(data, false, &xPtr, &yPtr);

        xPtr = xPtr - 1;
        int centerEnergy = getPixelEnergy(data, false, &xPtr, &yPtr);

        xPtr = xPtr - 1;
        int leftEnergy = getPixelEnergy(data, false, &xPtr, &yPtr);

        if (leftEnergy < centerEnergy && leftEnergy < rightEnergy)
        {
            curX = xPtr - 1;
        }
        else if (rightEnergy < centerEnergy && rightEnergy < leftEnergy)
        {
            curX = xPtr + 1;
        }

        idx = getPixelIdx(data, curX, y);
        data->imgEnergy[idx] = SEAM;
        data->imgSeam[idx] = SEAM;
    }
}

void outputDebugImage(ImageProcessData* processData)
{
    debugImage.width = processData->width;
    debugImage.height = processData->height;
    debugImage.channelCount = 3;
    debugImage.data = (unsigned char *) malloc(
        sizeof(unsigned char *) * processData->width * processData->height * debugImage.channelCount);

    for (int y = 0; y < processData->height; y++)
    {
        for (int x = 0; x < processData->width; x++)
        {
            unsigned int pixelPos = y * processData->width + x;
            unsigned char* pixel = &debugImage.data[pixelPos * debugImage.channelCount];

            if (processData->imgSeam[pixelPos] == SEAM)
            {
                pixel[0] = 180;
                pixel[1] = 0;
                pixel[2] = 0;
            }
            else
            {
                pixel[0] = processData->imgEnergy[pixelPos];
                pixel[1] = processData->imgEnergy[pixelPos];
                pixel[2] = processData->imgEnergy[pixelPos];
            }
        }
    }

    // Use path from imageOut.fPath and append debug_$count
    char *debugPath = (char *) malloc(sizeof(char) * (strlen(imageOut.fPath) + 10));
    sprintf(debugPath, "%s_debug_%d.png", imageOut.fPath, outputDebugCount++);
    stbi_write_png(debugPath,
        debugImage.width,
        debugImage.height,
        debugImage.channelCount,
        debugImage.data,
        debugImage.width * debugImage.channelCount);

    printf("Debug image written to %s\n", debugPath);

    free(debugImage.data);
}

void copyToOutput(Image* imageIn, Image* imageOut, ImageProcessData* processData)
{
    unsigned char* outputPos = imageOut->data;
    for (int pixelPos = 0; pixelPos < imageIn->width * imageIn->height; pixelPos++)
    {
        if (processData->imgSeam[pixelPos] != SEAM)
        {
            for (int channel = 0; channel < imageIn->channelCount; channel++)
            {
                int channelInPos = pixelPos * imageOut->channelCount + channel;
                *outputPos = imageIn->data[channelInPos];
                outputPos++;
            }
        }
    }
}

int main(int argc, char *args[])
{
    // Read arguments
    if (argc != 4)
    {
        printf("Error: Invalid amount of arguments. [%d]\n", argc);
        exit(EXIT_FAILURE);
    }

    imageIn.fPath = args[1];
    imageOut.fPath = args[2];
    imageOut.width = atoi(args[3]);
    //imageOut.height = atoi(args[4]);  // Height stays the same

    // Load image
    imageIn.data = stbi_load(imageIn.fPath,
                             &imageIn.width,
                             &imageIn.height,
                             &imageIn.channelCount,
                             STB_COLOR_CHANNELS);

    if (imageIn.data == NULL)
    {
        printf("Error: Couldn't load image\n");
        exit(EXIT_FAILURE);
    }
    printf("Loaded image %s of size %dx%d.\n", imageIn.fPath, imageIn.width, imageIn.height);

    imageOut.height = imageIn.height; // Height stays the same

    if (imageOut.width > imageIn.width) return EXIT_FAILURE;  // Output width should be smaller than input width

    //printMatrix(&imageIn.image, imageIn.width * image.channelCount, image.height);

    // Process image //////////////////////////////////////////////////////////////////////////
    ImageProcessData processData;
    processData.width = imageIn.width;
    processData.height = imageIn.height;
    processData.imgEnergy = (unsigned int *) malloc(sizeof(unsigned int) * processData.width * processData.height);
    processData.imgSeam = (unsigned int *) malloc(sizeof(unsigned int) * processData.width * processData.height);

    /// Energy Calculation Full - Assign energy value for every pixel in the image
    calcEnergyFull(&imageIn, processData.imgEnergy);

    int seamCount = imageIn.width - imageOut.width;
    for (int i = 0; i < seamCount; i++) {
        seamIdentification(&processData);
        seamRemoval(&processData);
    }

    outputDebugImage(&processData);

    // Output image //////////////////////////////////////////////////////////////////////////
    double startOutput = omp_get_wtime();

    const int numPixelsOutput = imageOut.width * imageOut.height;
    imageOut.channelCount = imageIn.channelCount;
    imageOut.data = (unsigned char *) malloc(
        sizeof(unsigned char *) * numPixelsOutput * imageOut.channelCount);

    copyToOutput(&imageIn, &imageOut, &processData);

    double stopOutput = omp_get_wtime();

    // Write the output image to file
    stbi_write_png(imageOut.fPath, imageOut.width, imageOut.height,
        imageOut.channelCount, imageOut.data,
        imageOut.width * imageOut.channelCount);

    // printf(" -> total time: %f s\n", stopEnergy - startEnergy +
    //      stopSeamId - startSeamId +
    //      stopSeamRemove - startSeamRemove +
    //      stopOutput - startOutput);


    // FREE ///////////////////////////////////////////////////////////////////////////////
    free(processData.imgEnergy);
    free(processData.imgSeam);
    free(imageOut.data);
    stbi_image_free(imageIn.data);

    return EXIT_SUCCESS;
}