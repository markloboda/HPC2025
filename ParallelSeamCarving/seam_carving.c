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
#define ENERGY_CHANNEL_COUNT 1

int outputDebugCount = 0;

typedef struct __ImageProcessData__
{
    unsigned char* img;
    unsigned int* imgEnergy;
    unsigned int* imgSeam;
    int width;
    int height;
    int channelCount;
} ImageProcessData;

unsigned int getPixelIdxC(int x, int y, int width, int channelCount)
{
    return (y * width + x) * channelCount;
}

unsigned int getPixelIdx(int x, int y, int width)
{
    return getPixelIdxC(x, y, width, ENERGY_CHANNEL_COUNT);
}

unsigned char *getPixel(unsigned char *data, int x, int y, int width, int height, int channelCount)
{
    if (x >= width || y >= height || x < 0 || y < 0) {
        return NULL;
    }

    const int pixelIdx = getPixelIdxC(x, y, width, channelCount);
    return &data[pixelIdx];
}

unsigned char *getPixelE(unsigned char *data, int x, int y, int width, int height, int channelCount)  // Only used for energy calculation
{
    // if x and y outside bounds, use the closest pixel
    if (x < 0) x = 0;
    if (y < 0) y = 0;
    if (x >= width) x = width - 1;
    if (y >= height) y = height - 1;

    return getPixel(data, x, y, width, height, channelCount);
}

unsigned int getPixelEnergy(unsigned int* data, int x, int y, int width, int height)
{
    if (x < 0 || y < 0 || x >= width || y >= height)
    {
        return UINT_MAX;
    }

    return data[getPixelIdx(x, y, width)];
}

unsigned int calculatePixelEnergy(unsigned char *data, int x, int y, int width, int height, int channelCount)
{
    int energy = 0;
    for (int rgbChannel = 0; rgbChannel < channelCount; rgbChannel++)
    {
        int Gx = -     getPixelE(data, x - 1, y - 1, width, height, channelCount)[rgbChannel]
                 - 2 * getPixelE(data, x - 1,     y, width, height, channelCount)[rgbChannel]
                 -     getPixelE(data, x - 1, y + 1, width, height, channelCount)[rgbChannel]
                 +     getPixelE(data, x + 1, y - 1, width, height, channelCount)[rgbChannel]
                 + 2 * getPixelE(data, x + 1,     y, width, height, channelCount)[rgbChannel]
                 +     getPixelE(data, x + 1, y + 1, width, height, channelCount)[rgbChannel];

        int Gy =       getPixelE(data, x - 1, y - 1, width, height, channelCount)[rgbChannel]
                 + 2 * getPixelE(data,     x, y - 1, width, height, channelCount)[rgbChannel]
                 +     getPixelE(data, x + 1, y - 1, width, height, channelCount)[rgbChannel]
                 -     getPixelE(data, x - 1, y + 1, width, height, channelCount)[rgbChannel]
                 - 2 * getPixelE(data,     x, y + 1, width, height, channelCount)[rgbChannel]
                 -     getPixelE(data, x + 1, y + 1, width, height, channelCount)[rgbChannel];

        energy += sqrt(pow(Gx, 2) + pow(Gy, 2));
    }

    return energy / channelCount;
}

void calcEnergyFull(ImageProcessData* data)
{
    // Free if already allocated
    if (data->imgEnergy != NULL)
    {
        free(data->imgEnergy);
    }

    data->imgEnergy = (unsigned int *) malloc(sizeof(unsigned int) * data->width * data->height);

    for (int y = 0; y < data->height; y++)
    {
        for (int x = 0; x < data->width; x++)
        {
            unsigned int pixelIdx = getPixelIdx(x, y, data->width);
            unsigned int energy = calculatePixelEnergy(data->img, x, y, data->width, data->height, data->channelCount);
            data->imgEnergy[pixelIdx] = energy;
        }
    }
}

void updateEnergy(ImageProcessData* data)
{
    calcEnergyFull(data);
}

/// @brief Update data.imgSeam with the cumulative energy of the cheapest path from the bottom to top
/// @param data
void seamIdentification(ImageProcessData* data)
{
    // Free if already allocated
    if (data->imgSeam != NULL)
    {
        free(data->imgSeam);
    }

    data->imgSeam = (unsigned int *) malloc(sizeof(unsigned int) * data->width * data->height);

    // Starts at the bottom (the bottom row is skipped)
    for (int y = data->height - 2; y >= 0; y--)
    {
        if (y == 0)
        {
            int a = 0;
        }
        for (int x = 0; x < data->width; x++)
        {
            unsigned int leftEnergy =   getPixelEnergy(data->imgSeam, x - 1, y + 1, data->width, data->height);
            unsigned int centerEnergy = getPixelEnergy(data->imgSeam, x    , y + 1, data->width, data->height);
            unsigned int rightEnergy =  getPixelEnergy(data->imgSeam, x + 1, y + 1, data->width, data->height);

            unsigned int curEnergy = getPixelEnergy(data->imgEnergy, x, y, data->width, data->height);
            unsigned int minEnergy = min(leftEnergy, min(centerEnergy, rightEnergy));

            data->imgSeam[getPixelIdx(x, y, data->width)] = curEnergy + minEnergy;
        }
    }
}

void seamAnnotate(ImageProcessData* data)
{
    // Find the minimum energy in the top row
    int curX = 0;
    for (int x = 1; x < data->width; x++)
    {
        if (data->imgSeam[getPixelIdx(x, 0, data->width)] < data->imgSeam[getPixelIdx(curX, 0, data->width)])
        {
            curX = x;
        }
    }

    int idx = getPixelIdxC(curX, 0, data->width, data->channelCount);
    data->imgEnergy[idx] = SEAM;
    data->imgSeam[idx] = SEAM;

    for (int y = 1; y < data->height; y++)
    {
        // Set SEAM
        unsigned int idx = getPixelIdx(curX, y, data->width);
        data->imgSeam[idx] = SEAM;
        data->imgEnergy[idx] = SEAM;

        // Find the minimum energy in the next row
        unsigned int leftEnergy =   getPixelEnergy(data->imgSeam, curX - 1, y + 1, data->width, data->height);
        unsigned int centerEnergy = getPixelEnergy(data->imgSeam, curX    , y + 1, data->width, data->height);
        unsigned int rightEnergy =  getPixelEnergy(data->imgSeam, curX + 1, y + 1, data->width, data->height);

        // Select next X
        if (leftEnergy < centerEnergy && leftEnergy < rightEnergy)
        {
            curX = curX - 1;
        }
        else if (rightEnergy < centerEnergy && rightEnergy < leftEnergy)
        {
            curX = curX + 1;
        }
    }
}

void seamRemove(ImageProcessData* processData)
{
    processData->width = processData->width - 1;
    processData->height = processData->height;
    processData->channelCount = processData->channelCount;

    unsigned int pixelCount = processData->width * processData->height;

    unsigned char* image = (unsigned char *) malloc(sizeof(unsigned char) * pixelCount * processData->channelCount);

    // Copy image data without seam
    unsigned int pixelPos = 0;
    for (int y = 0; y < processData->height; y++)
    {
        for (int x = 0; x < processData->width; x++)
        {
            unsigned int pixelIdx = getPixelIdx(x, y, processData->width);
            if (processData->imgEnergy[pixelIdx] != SEAM)
            {
                unsigned int pixelIdxC = getPixelIdxC(x, y, processData->width, processData->channelCount);
                for (int channel = 0; channel < processData->channelCount; channel++)
                {
                    image[pixelPos] = processData->img[pixelIdxC + channel];
                    pixelPos++;
                }
            }
        }
    }

    free(processData->img);

    processData->img = image;
}

void outputDebugImage(ImageProcessData* processData, char* imageOutPath)
{
    int debugWidth = processData->width + 1; // + 1 because it was reduced by 1 in refreshProcessData
    int debugHeight = processData->height;
    int debugChannelCount = 3;
    unsigned char *debugImgData = (unsigned char *) malloc(sizeof(unsigned char *) * debugWidth * debugHeight * debugChannelCount);

    for (int y = 0; y < processData->height; y++)
    {
        for (int x = 0; x < processData->width; x++)
        {
            unsigned int pixelPos = getPixelIdx(x, y, debugWidth);
            unsigned char *pixel = &debugImgData[getPixelIdxC(x, y, debugWidth, debugChannelCount)];

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
    stbi_write_png(imageOutPath,
        debugWidth,
        debugHeight,
        debugChannelCount,
        debugImgData,
        debugWidth * debugChannelCount);
}

int main(int argc, char *args[])
{
    // Read arguments
    if (argc != 4)
    {
        printf("Error: Invalid amount of arguments. [%d]\n", argc);
        exit(EXIT_FAILURE);
    }

    char *imageInPath = args[1];
    char *imageOutPath = args[2];
    int outputWidth = atoi(args[3]);
    int outputHeight; // = atoi(args[4]); // Height stays the same

    ImageProcessData processData;
    processData.img = NULL;
    processData.imgEnergy = NULL;
    processData.imgSeam = NULL;

    // Load image //////////////////////////////////////////////////////////////////////////
    processData.img = stbi_load(imageInPath,
                             &processData.width,
                             &processData.height,
                             &processData.channelCount,
                             STB_COLOR_CHANNELS);

    if (processData.img == NULL)
    {
        printf("Error: Couldn't load image\n");
        exit(EXIT_FAILURE);
    }
    printf("Loaded image %s of size %dx%d.\n", imageInPath, processData.width, processData.height);

    if (outputWidth > processData.width)
    {
        printf("Error: Output width should be smaller than input width\n");
        return EXIT_FAILURE;
    }
    outputHeight = processData.height;

    // Process image //////////////////////////////////////////////////////////////////////////
    int seamCount = processData.width - outputWidth;

    calcEnergyFull(&processData);
    for (int i = 0; i < seamCount; i++)
    {
        if (i > 0) {
            updateEnergy(&processData);
        }
        seamIdentification(&processData);
        seamAnnotate(&processData);
        seamRemove(&processData);
    }

    char *debugPath = (char *) malloc(sizeof(char) * (strlen(imageOutPath) + 10));
    sprintf(debugPath, "%s_debug_%d.png", imageOutPath, outputDebugCount++);

    outputDebugImage(&processData, debugPath);

    free(debugPath);
    free(processData.imgEnergy);
    free(processData.imgSeam);

    // Output image //////////////////////////////////////////////////////////////////////////
    double startOutput = omp_get_wtime();

    stbi_write_png(imageOutPath,
                   processData.width,
                   processData.height,
                   processData.channelCount,
                   processData.img,
                   processData.width * processData.channelCount);

    double stopOutput = omp_get_wtime();

    // FREE ///////////////////////////////////////////////////////////////////////////////

    stbi_image_free(processData.img);

    return EXIT_SUCCESS;
}