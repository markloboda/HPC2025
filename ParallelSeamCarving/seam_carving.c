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
#define SEAM INT_MAX

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
    unsigned char* image;
    unsigned int* imgEnergy;
    unsigned int* imgSeam;
    int width;
    int height;
    int channelCount;
} ImageProcessData;

unsigned int getPixelIdx(int x, int y, int width, int channel)
{
    return (y * width + x) * channel;
}

unsigned int getPixelIdx(int x, int y, int width)
{
    return getPixelIdx(x, y, width, 1);
}

unsigned char *getPixel(unsigned char *data, int x, int y, int width, int height, int channel)
{
    if (x >= width || y >= height || x < 0 || y < 0) {
        return NULL;
    }

    const int pixelIdx = getPixelIdx(x, y, width, height, channel);
    return &data[pixelIdx];
}

unsigned int getPixelValue(unsigned char *data, int x, int y, int width, int height, int channel)  // Only used for energy calculation
{
    // if x and y outside bounds, use the closest pixel
    if (x < 0) x++;
    if (y < 0) y++;
    if (x == width) x--;
    if (y == height) y--;

    return getPixel(data, x, y, width, height, channel)[channel];
}

unsigned int getPixelEnergy(unsigned int* data, int x, int y, int width, int height)
{
    if (x < 0 || y < 0 || x >= width || y >= height)
    {
        return UINT_MAX;
    }

    return data[getPixelIdx(x, y, width, height)];
}

unsigned int calcEnergyPixel(unsigned char *data, int x, int y, int width, int height, int channelCount)
{
    int energy = 0;
    for (int rgbChannel = 0; rgbChannel < channelCount; rgbChannel++)
    {
        int Gx = - getPixelValue(data, x - 1, y - 1, width, height, rgbChannel) -
                   2 * getPixelValue(data, x - 1, y, width, height, rgbChannel) -
                   getPixelValue(data, x - 1, y + 1, width, height, rgbChannel) +
                   getPixelValue(data, x + 1, y - 1, width, height, rgbChannel) +
                   2 * getPixelValue(data, x + 1, y, width, height, rgbChannel) +
                   getPixelValue(data, x + 1, y + 1, width, height, rgbChannel);

        int Gy = getPixelValue(data, x - 1, y - 1, width, height, rgbChannel) +
                 2 * getPixelValue(data, x, y - 1, width, height, rgbChannel) +
                 getPixelValue(data, x + 1, y - 1, width, height, rgbChannel) -
                 getPixelValue(data, x - 1, y + 1, width, height, rgbChannel) -
                 2 * getPixelValue(data, x, y + 1, width, height, rgbChannel) -
                 getPixelValue(data, x + 1, y + 1, width, height, rgbChannel);

        energy += sqrt(pow(Gx, 2) + pow(Gy, 2));
    }

    return energy / channelCount;
}

void calcEnergyFull(ImageProcessData* data)
{
    for (int y = 0; y < data->height; y++)
    {
        for (int x = 0; x < data->width; x++)
        {
            unsigned int pixelPos = y * data->width + x;
            data->imgEnergy[pixelPos] = calcEnergyPixel(data->image, x, y, data->width, data->height, data->channelCount);
        }
    }
}

void seamIdentification(ImageProcessData* data)
{
    // Starts at the bottom (the bottom row is skipped)
    for (int y = data->height - 2; y >= 0; y--)
    {
        for (int x = 0; x < data->width; x++)
        {
            if (data->imgEnergy[getPixelIdx(x, y, data->width, data->channelCount)] == SEAM) continue;

            int rightEnergy =  getPixelEnergy(data, x + 1, y + 1, data->width, data->height);
            int centerEnergy = getPixelEnergy(data, x    , y + 1, data->width, data->height);
            int leftEnergy =   getPixelEnergy(data, x - 1, y + 1, data->width, data->height);

            data->imgSeam[getPixelIdx(x, y, data->width, data->channelCount)] = 
                data->imgEnergy[getPixelIdx(data, x, y, data->width, data->channelCount)] +
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
        if (data->imgSeam[getPixelIdx(x, 0, data->width, data->channelCount)] 
                < data->imgSeam[getPixelIdx(curX, 0, data->width, data->channelCount)])
        {
            curX = x;
        }
    }

    int idx = getPixelIdx(curX, 0, data->width, data->channelCount);
    data->imgEnergy[idx] = SEAM;
    data->imgSeam[idx] = SEAM;

    for (int y = 1; y < data->height; y++)
    {
        int rightEnergy = getPixelEnergy(data->imgSeam, curX + 1, y + 1, data->width, data->height);
        int centerEnergy = getPixelEnergy(data->imgSeam, curX, y + 1, data->width, data->height);
        int leftEnergy = getPixelEnergy(data->imgSeam, curX - 1, y + 1, data->width, data->height);

        if (leftEnergy < centerEnergy && leftEnergy < rightEnergy)
        {
            curX -= 1;
        }
        else if (rightEnergy < centerEnergy && rightEnergy < leftEnergy)
        {
            curX += 1;
        }

        idx = getPixelIdx(curX, y, data->width, data->channelCount);
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