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
#define SEAM UINT_MAX - 1
#define ENERGY_CHANNEL_COUNT 1
#define UNDEFINED_UINT UINT_MAX

// USER DEFINES ////////////////////////////////////////////////////////////////////////////
// #define SAVE_TIMING_STATS
// #define SAVE_DEBUG_IMAGE
#define BAR_WIDTH 50


int outputDebugCount = 0;

typedef struct __ImageProcessData__
{
    unsigned char* img;
    unsigned int* imgEnergy;
    unsigned int* imgSeam;
    int* seamPath;
    int width;
    int height;
    int channelCount;
} ImageProcessData;

typedef struct __TimingStats__
{
    double totalProcessingTime;
    double energyCalculations;
    double seamIdentifications;
    double seamAnnotates;
    double seamRemoves;
} TimingStats;

// FUNCTIONS //////////////////////////////////////////////////////////////////////////////
/// @brief Get the index of a pixel given the dimensions and channel count
static inline unsigned int getPixelIdxC(int x, int y, int width, int channelCount)
{
    return (y * width + x) * channelCount;
}

/// @brief Get the index of a pixel given the dimensions
static inline unsigned int getPixelIdx(int x, int y, int width)
{
    return getPixelIdxC(x, y, width, ENERGY_CHANNEL_COUNT);
}

static inline void getPixelPos(unsigned int idx, int width, int* x, int* y)
{
    *x = idx % width;
    *y = idx / width;
}

/// @brief Get the pixel data at the given position
static inline unsigned char *getPixel(unsigned char *data, int x, int y, int width, int height, int channelCount)
{
    if (x >= width || y >= height || x < 0 || y < 0)
    {
        return NULL;
    }

    const int pixelIdx = getPixelIdxC(x, y, width, channelCount);
    return &data[pixelIdx];
}

/// @brief Get the pixel data at the given position (with bounds check)
static inline unsigned char *getPixelE(unsigned char *data, int x, int y, int width, int height, int channelCount)  // Only used for energy calculation
{
    // if x and y outside bounds, use the closest pixel
    if (x < 0)       x = 0;
    if (y < 0)       y = 0;
    if (x >= width)  x = width - 1;
    if (y >= height) y = height - 1;

    return getPixel(data, x, y, width, height, channelCount);
}

/// @brief Get the energy pixel data at the given position
static inline unsigned int getEnergyPixel(unsigned int* data, int x, int y, int width, int height)
{
    // if x and y outside bounds, return undefined
    if (x < 0 || y < 0 || x >= width || y >= height)
    {
        return UNDEFINED_UINT;
    }

    return data[getPixelIdx(x, y, width)];
}

/// @brief Get the energy pixel data at the given position (with bounds check)
static inline unsigned int getEnergyPixelE(unsigned int* data, int x, int y, int width, int height)
{
    unsigned int energy = getEnergyPixel(data, x, y, width, height);
    if (energy == UNDEFINED_UINT)
    {
        return INT_MAX;
    }

    return energy;
}

static inline bool isSeam(ImageProcessData* data, int x, int y)
{
    if (data->seamPath == NULL || y >= data->height)
    {
        return false;
    }

    return data->seamPath[y] == x;
}

/// @brief Calculate the energy of a pixel using the sobel operator
static inline unsigned int calculatePixelEnergy(unsigned char *data, int x, int y, int width, int height, int channelCount)
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

        int Gy = +     getPixelE(data, x - 1, y - 1, width, height, channelCount)[rgbChannel]
                 + 2 * getPixelE(data,     x, y - 1, width, height, channelCount)[rgbChannel]
                 +     getPixelE(data, x + 1, y - 1, width, height, channelCount)[rgbChannel]
                 -     getPixelE(data, x - 1, y + 1, width, height, channelCount)[rgbChannel]
                 - 2 * getPixelE(data,     x, y + 1, width, height, channelCount)[rgbChannel]
                 -     getPixelE(data, x + 1, y + 1, width, height, channelCount)[rgbChannel];

        energy += sqrt(pow(Gx, 2) + pow(Gy, 2));
    }

    return energy / channelCount;
}

/// @brief Update the energy of all pixels in the image
static inline void updateEnergyFull(ImageProcessData* data)
{
    // Free if already allocated
    if (data->imgEnergy != NULL)
    {
        free(data->imgEnergy);
    }

    // Allocate space for energy and calculate energy for each pixel
    data->imgEnergy = (unsigned int *) malloc(sizeof(unsigned int) * data->width * data->height);

    /// Testing:
    // - Tested looping with one for loop through all data but is consistently slower in parallel and in sequential.
    // - Tested collapse(2) but also seems to be slower.
    #pragma omp parallel for
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

/// @brief Update the energy of the pixels on the seam
void updateEnergyOnSeam(ImageProcessData* data)
{
    unsigned int *imgEnergyNew = (unsigned int *) malloc(sizeof(unsigned int) * data->width * data->height);

    int oldWidth = data->width + 1;

    #pragma omp parallel for
    for (int y = 0; y < data->height; y++)
    {
        for (int x = 0; x < oldWidth; x++)
        {
            // Get data.
            int seamX0 = y > 0 ? data->seamPath[y - 1] : INT_MAX;
            int seamX1 = data->seamPath[y];
            int seamX2 = y < data->height - 1 ? data->seamPath[y + 1] : INT_MAX;

            int insertOffsetX = -(x > seamX1);
            int idx = getPixelIdx(x + insertOffsetX, y, data->width);

            // Should recalculate.
            int difX0 = abs(x - seamX0);
            int difX1 = abs(x - seamX1);
            int difX2 = abs(x - seamX2);
            bool shouldRecalculate = difX0 <= 1 || difX1 <= 1 || difX2 <= 1;

            // Recalculate and/or insert.
            if (shouldRecalculate)
            {
                int newX, newY;
                getPixelPos(idx, data->width, &newX, &newY);
                imgEnergyNew[idx] = calculatePixelEnergy(data->img, newX, newY, data->width, data->height, data->channelCount);
            }
            else
            {
                imgEnergyNew[idx] = data->imgEnergy[getPixelIdx(x, y, oldWidth)];
            }
        }
    }

    free(data->imgEnergy);

    data->imgEnergy = imgEnergyNew;
}

/// @brief Calculate the cumulative energy of the image from the bottom to the top
void seamIdentification(ImageProcessData* data)
{
    // Free if already allocated
    if (data->imgSeam != NULL)
    {
        free(data->imgSeam);
    }

    // Allocate space for seam and calculate cumulative energy for each pixel
    data->imgSeam = (unsigned int *) malloc(sizeof(unsigned int) * data->width * data->height);

    #pragma omp parallel for
    for (int y = data->height - 2; y >= 0; y--)
    {
        for (int x = 0; x < data->width; x++)
        {
            unsigned int leftEnergy =   getEnergyPixelE(data->imgSeam, x - 1, y + 1, data->width, data->height);
            unsigned int centerEnergy = getEnergyPixelE(data->imgSeam, x    , y + 1, data->width, data->height);
            unsigned int rightEnergy =  getEnergyPixelE(data->imgSeam, x + 1, y + 1, data->width, data->height);

            unsigned int curEnergy = getEnergyPixelE(data->imgEnergy, x, y, data->width, data->height);
            unsigned int minEnergy = min(leftEnergy, min(centerEnergy, rightEnergy));

            data->imgSeam[getPixelIdx(x, y, data->width)] = curEnergy + minEnergy;
        }
    }
}

/// @brief Annotate the seam in the image (with SEAM value)
void seamAnnotate(ImageProcessData* data)
{
    // Allocate memory for seam path
    if (data->seamPath != NULL)
    {
        free(data->seamPath);
    }
    data->seamPath = (int *) malloc(sizeof(int) * data->height);

    // Find the minimum energy in the top row
    int curX = 0;
    for (int x = 1; x < data->width; x++)
    {
        if (data->imgSeam[getPixelIdx(x, 0, data->width)] < data->imgSeam[getPixelIdx(curX, 0, data->width)])
        {
            curX = x;
        }
    }

    // Set SEAM
    data->seamPath[0] = curX;

    for (int y = 0; y < data->height - 1; y++)
    {
        // Find the minimum energy in the next row
        unsigned int leftEnergy =   getEnergyPixelE(data->imgSeam, curX - 1, y + 1, data->width, data->height);
        unsigned int centerEnergy = getEnergyPixelE(data->imgSeam, curX    , y + 1, data->width, data->height);
        unsigned int rightEnergy =  getEnergyPixelE(data->imgSeam, curX + 1, y + 1, data->width, data->height);

        // Select next X
        if (leftEnergy < centerEnergy && leftEnergy < rightEnergy)
        {
            curX = curX - 1;
        }
        else if (rightEnergy < centerEnergy && rightEnergy < leftEnergy)
        {
            curX = curX + 1;
        }

        // Set SEAM
        data->seamPath[y + 1] = curX;
    }
}

/// @brief Remove the seam from the image
void seamRemove(ImageProcessData* processData)
{
    // Allocate space for new image
    unsigned int newWidth = processData->width - 1;
    unsigned int pixelCount = newWidth * processData->height;
    unsigned char* image = (unsigned char *) malloc(sizeof(unsigned char) * pixelCount * processData->channelCount);

    // Copy image data without seam
    unsigned int pixelPos = 0;
    for (int y = 0; y < processData->height; y++)
    {
        for (int x = 0; x < processData->width; x++)
        {
            if (!isSeam(processData, x, y))
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

    // Free old image and set new image
    free(processData->img);

    // Update process data
    processData->img = image;
    processData->width = processData->width - 1;
    processData->height = processData->height;
    processData->channelCount = processData->channelCount;
}

#ifdef SAVE_DEBUG_IMAGE
/// @brief Output the debug image with the seam annotated and energy values
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
            unsigned char *pixel = &debugImgData[getPixelIdxC(x, y, debugWidth, debugChannelCount)];

            if (isSeam(processData, x, y))
            {
                pixel[0] = 180;
                pixel[1] = 0;
                pixel[2] = 0;
            }
            else
            {
                unsigned int pixelPos = getPixelIdx(x, y, debugWidth);
                pixel[0] = processData->imgEnergy[pixelPos];
                pixel[1] = processData->imgEnergy[pixelPos];
                pixel[2] = processData->imgEnergy[pixelPos];
            }
        }
    }

    // Use path from imageOut.fPath and append debug_$count
    stbi_write_png(imageOutPath, debugWidth, debugHeight, debugChannelCount, debugImgData, debugWidth * debugChannelCount);
    free(debugImgData);

    outputDebugCount++;
}
#endif

#ifdef BAR_WIDTH
/// @brief Update the loading bar
void updatePrintLoadingBar(int progress, int total)
{
    static int lastProgress = -1;
    if (lastProgress != -1) {
        printf("\033[F"); // Move cursor up one line
    }

    int percent = (progress * 100) / total;
    int filled = (progress * BAR_WIDTH) / total;

    printf("[");  // Carriage return to overwrite line
    for (int i = 0; i < filled; i++) printf("=");
    for (int i = filled; i < BAR_WIDTH; i++) printf(" ");
    printf("] %d%%\n", percent);

    fflush(stdout);
    lastProgress = progress;
}
#endif

int main(int argc, char *args[])
{
    // Read arguments
    if (argc != 4)
    {
        printf("Error: Invalid amount of arguments. [%d]\n", argc);
        exit(EXIT_FAILURE);
    }
    printf("Arguments: imageInPath=%s, imageOutPath=%s, outputWidth=%s\n", args[1], args[2], args[3]);

    // Parse arguments /////////////////////////////////////////////////////////////////////
    char *imageInPath = args[1];
    char *imageOutPath = args[2];
    int outputWidth = atoi(args[3]);
    int outputHeight; // = atoi(args[4]); // Height stays the same

    // Setup processing data struct //////////////////////////////////////////////////////
    ImageProcessData processData;
    processData.img = NULL;
    processData.imgEnergy = NULL;
    processData.imgSeam = NULL;
    processData.seamPath = NULL;

    // Load image //////////////////////////////////////////////////////////////////////////
    processData.img = stbi_load(imageInPath, &processData.width, &processData.height, &processData.channelCount, STB_COLOR_CHANNELS);
    if (processData.img == NULL)
    {
        printf("Error: Couldn't load image\n");
        exit(EXIT_FAILURE);
    }
    printf("Loaded image %s of size %dx%d.\n", imageInPath, processData.width, processData.height);

    if (outputWidth >= processData.width)
    {
        printf("Error: Output width should be smaller than input width\n");
        return EXIT_FAILURE;
    }
    outputHeight = processData.height;

    // Process image //////////////////////////////////////////////////////////////////////////
    TimingStats timingStats;
    double startTotalProcessingTime = omp_get_wtime();
    int seamCount = processData.width - outputWidth;
    // printf("Seam count: %d\n", seamCount);

    double startEnergyTime = omp_get_wtime();
    updateEnergyFull(&processData);
    double stopEnergyTime = omp_get_wtime();
    timingStats.energyCalculations += stopEnergyTime - startEnergyTime;
    for (int i = 0; i < seamCount; i++)
    {
        // printf("Processing seam %d/%d\n", i + 1, seamCount);

        // Energy step
        startEnergyTime = omp_get_wtime();
        if (i > 0) {
            updateEnergyOnSeam(&processData);
        }
        stopEnergyTime = omp_get_wtime();
        timingStats.energyCalculations += stopEnergyTime - startEnergyTime;

        // Seam identification step
        double startSeamTime = omp_get_wtime();
        seamIdentification(&processData);
        double stopSeamTime = omp_get_wtime();
        timingStats.seamIdentifications += stopSeamTime - startSeamTime;

        // Seam annotate step
        double startAnnotateTime = omp_get_wtime();
        seamAnnotate(&processData);
        double stopAnnotateTime = omp_get_wtime();
        timingStats.seamAnnotates += stopAnnotateTime - startAnnotateTime;

        // Seam remove step
        double startSeamRemoveTime = omp_get_wtime();
        seamRemove(&processData);
        double stopSeamRemoveTime = omp_get_wtime();
        timingStats.seamRemoves += stopSeamRemoveTime - startSeamRemoveTime;

#ifdef BAR_WIDTH
        updatePrintLoadingBar(i + 1, seamCount);
#endif
    }
    double stopTotalProcessingTime = omp_get_wtime();
    timingStats.totalProcessingTime = stopTotalProcessingTime - startTotalProcessingTime;

    // Output debug image //////////////////////////////////////////////////////////////////////////
#ifdef SAVE_DEBUG_IMAGE
    char debugImageOutPath[100];
    sprintf(debugImageOutPath, "%s/debug_%d.png", "debug_images", outputDebugCount);
    outputDebugImage(&processData, debugImageOutPath);
#endif

    // Free process data //////////////////////////////////////////////////////////////////////////
    free(processData.imgEnergy);
    free(processData.imgSeam);

    // Output image //////////////////////////////////////////////////////////////////////////
    stbi_write_png(imageOutPath,
                   processData.width,
                   processData.height,
                   processData.channelCount,
                   processData.img,
                   processData.width * processData.channelCount);

    printf("Output image %s of size %dx%d.\n", imageOutPath, processData.width, processData.height);

    stbi_image_free(processData.img);

    // Output timing stats //////////////////////////////////////////////////////////////////////////
    printf("--------------- Timing Stats ---------------\n");
    printf("Total Processing Time: %f s\n", timingStats.totalProcessingTime);
    printf("Energy Calculations: %f s [%f \%]\n", timingStats.energyCalculations, timingStats.energyCalculations / timingStats.totalProcessingTime * 100);
    printf("Seam Identifications: %f s [%f \%]\n", timingStats.seamIdentifications, timingStats.seamIdentifications / timingStats.totalProcessingTime * 100);
    printf("Seam Annotates: %f s [%f \%]\n", timingStats.seamAnnotates, timingStats.seamAnnotates / timingStats.totalProcessingTime * 100);
    printf("Seam Removes: %f s [%f \%]\n", timingStats.seamRemoves, timingStats.seamRemoves / timingStats.totalProcessingTime * 100);

    // Output timing stats to file //////////////////////////////////////////////////////////////////////////
#ifdef SAVE_TIMING_STATS
    FILE *timingFile = fopen("timing_stats/timing_stats_parallel.txt", "a");
    fprintf(timingFile, "--------------- %s ---------------\n", imageInPath);
    fprintf(timingFile, "Arguments: imageInPath=%s, imageOutPath=%s, outputWidth=%s\n", args[1], args[2], args[3]);
    fprintf(timingFile, "--------------- Timing Stats ---------------\n");
    fprintf(timingFile, "Total Processing Time: %f s\n", timingStats.totalProcessingTime);
    fprintf(timingFile, "Energy Calculations: %f s [%f \%]\n", timingStats.energyCalculations, timingStats.energyCalculations / timingStats.totalProcessingTime * 100);
    fprintf(timingFile, "Seam Identifications: %f s [%f \%]\n", timingStats.seamIdentifications, timingStats.seamIdentifications / timingStats.totalProcessingTime * 100);
    fprintf(timingFile, "Seam Annotates: %f s [%f \%]\n", timingStats.seamAnnotates, timingStats.seamAnnotates / timingStats.totalProcessingTime * 100);
    fprintf(timingFile, "Seam Removes: %f s [%f \%]\n", timingStats.seamRemoves, timingStats.seamRemoves / timingStats.totalProcessingTime * 100);
    fprintf(timingFile, "\n");
    fclose(timingFile);
#endif

    return EXIT_SUCCESS;
}