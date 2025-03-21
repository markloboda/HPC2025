
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
#define SIM_NUM_SEAM_REMOVAL 8
#define STRIP_HEIGHT 15 // Keep the STRIP_HEIGHT odd

// USER DEFINES ////////////////////////////////////////////////////////////////////////////
// #define SAVE_TIMING_STATS
// #define SAVE_DEBUG_IMAGE
#define RENDER_LOADING_BAR_WIDTH 50


int outputDebugCount = 0;

typedef struct __ImageProcessData__
{
    unsigned char* img;
    unsigned int* imgEnergy;
    unsigned int* imgSeam;
    int** seamPath;
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
static inline unsigned char *getPixelE(unsigned char *data, int x, int y, int width, int height, int channelCount, int limitLowX, int limitHighX)  // Only used for energy calculation
{
    // if x and y outside bounds, use the closest pixel
    if (x < limitLowX)   x = limitLowX;
    if (y < 0)           y = 0;
    if (x >= limitHighX) x = limitHighX - 1;
    if (y >= height)     y = height - 1;

    return getPixel(data, x, y, width, height, channelCount);
}

/// @brief Get the energy pixel data at the given position
static inline unsigned int getEnergyPixel(unsigned int* data, int x, int y, int width, int height, int limitLowX, int limitHighX)
{
    // if x and y outside bounds, return undefined
    if (x < limitLowX|| y < 0 || x >= limitHighX || y >= height)
    {
        return UNDEFINED_UINT;
    }

    return data[getPixelIdx(x, y, width)];
}

/// @brief Get the energy pixel data at the given position (with bounds check for horizontal stripes)
static inline unsigned int getEnergyPixelEStripe(unsigned int* data, int x, int y, int width, int height, int limitLowX, int limitHighX)
{
    unsigned int energy = getEnergyPixel(data, x, y, width, height, limitLowX, limitHighX);
    if (energy == UNDEFINED_UINT)
    {
        return INT_MAX;
    }

    return energy;
}

/// @brief Wrapper function for function for getting energy pixel data without horizontal stripes limits
static inline unsigned int getEnergyPixelE(unsigned int* data, int x, int y, int width, int height)
{
    return getEnergyPixelEStripe(data, x, y, width, height, 0, width);
}

/// @brief Returns true if the observed position with coordinates x and y is considered as part of seam
static inline bool isSeam(ImageProcessData* data, int x, int y, int seanIdx)
{
    if (data->seamPath == NULL || y >= data->height || seanIdx >= SIM_NUM_SEAM_REMOVAL)
    {
        return false;
    }

    return data->seamPath[seanIdx][y] == x;
}

/// @brief Calculate the energy of a pixel using the sobel operator
static inline unsigned int calculatePixelEnergyStripe(unsigned char *data, int x, int y, int width, int height, int channelCount, int limitLowX, int limitHighX)
{
    int energyTotal = 0;
    for (int rgbChannel = 0; rgbChannel < channelCount; rgbChannel++)
    {
        int Gx = -     getPixelE(data, x - 1, y - 1, width, height, channelCount, limitLowX, limitHighX)[rgbChannel]
                 - 2 * getPixelE(data, x - 1,     y, width, height, channelCount, limitLowX, limitHighX)[rgbChannel]
                 -     getPixelE(data, x - 1, y + 1, width, height, channelCount, limitLowX, limitHighX)[rgbChannel]
                 +     getPixelE(data, x + 1, y - 1, width, height, channelCount, limitLowX, limitHighX)[rgbChannel]
                 + 2 * getPixelE(data, x + 1,     y, width, height, channelCount, limitLowX, limitHighX)[rgbChannel]
                 +     getPixelE(data, x + 1, y + 1, width, height, channelCount, limitLowX, limitHighX)[rgbChannel];

        int Gy = +     getPixelE(data, x - 1, y - 1, width, height, channelCount, limitLowX, limitHighX)[rgbChannel]
                 + 2 * getPixelE(data,     x, y - 1, width, height, channelCount, limitLowX, limitHighX)[rgbChannel]
                 +     getPixelE(data, x + 1, y - 1, width, height, channelCount, limitLowX, limitHighX)[rgbChannel]
                 -     getPixelE(data, x - 1, y + 1, width, height, channelCount, limitLowX, limitHighX)[rgbChannel]
                 - 2 * getPixelE(data,     x, y + 1, width, height, channelCount, limitLowX, limitHighX)[rgbChannel]
                 -     getPixelE(data, x + 1, y + 1, width, height, channelCount, limitLowX, limitHighX)[rgbChannel];

        energyTotal += sqrt(pow(Gx, 2) + pow(Gy, 2));
    }
    int energy = energy / channelCount;
    return energy;
}

/// @brief Wrapper function without horizontal stripe limits
static inline unsigned int calculatePixelEnergy(unsigned char *data, int x, int y, int width, int height, int channelCount)
{
    return calculatePixelEnergyStripe(data, x, y, width, height, channelCount, 0, width);
}

#ifdef SAVE_DEBUG_IMAGE
/// @brief Output the debug image with the seam annotated and energy values
void outputDebugImage(ImageProcessData* processData, char* imageOutPath)
{
    int debugWidth = processData->width + SIM_NUM_SEAM_REMOVAL;
    int debugHeight = processData->height;
    int debugChannelCount = processData->channelCount;
    unsigned char *debugImgData = (unsigned char *) malloc(sizeof(unsigned char *) * debugWidth * debugHeight * debugChannelCount);

    for (int y = 0; y < processData->height; y++)
    {
        int seamPassedCount = 0;
        for (int x = 0; x < processData->width; x++)
        {
            unsigned char *pixel = &debugImgData[getPixelIdxC(x, y, debugWidth, debugChannelCount)];

            if (isSeam(processData, x, y, seamPassedCount))
            {
                pixel[0] = 180;
                pixel[1] = 0;
                pixel[2] = 0;
                seamPassedCount++;
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

/// @brief Calculate the energy of all pixels in the image
static inline void calculateEnergyFull(ImageProcessData* data)
{
    // Allocate space for energy and calculate energy for each pixel
    data->imgEnergy = (unsigned int *) malloc(sizeof(unsigned int) * data->width * data->height);

    /// Parallel:
    // - Tested looping with one for loop through all data but is consistently slower in parallel and in sequential.
    // - Tested collapse(2) but also seems to be slower.
    // - Standard approach is probably the best, as each thread gets a couple of rows (as cache lines) and every pixel calculation is independent
    // #pragma omp parallel for
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

/// @brief Update the energy of the pixels on the seam instead of updating the whole energy image
void updateEnergyOnSeam(ImageProcessData* data)
{
    unsigned int *imgEnergyNew = (unsigned int *) malloc(sizeof(unsigned int) * data->width * data->height);

    int oldWidth = data->width + SIM_NUM_SEAM_REMOVAL;
    const int stripWidth = oldWidth / SIM_NUM_SEAM_REMOVAL;

    /// Parallel:
    // #pragma omp parallel for
    for (int y = 0; y < data->height; y++)
    {
        for (int x = 0; x < oldWidth; x++)
        {
            int stripIdx = x / stripWidth;
            int lowX = stripIdx * stripWidth;
            int highX = lowX + stripWidth;
            int seamPassedCount = 0;

            // Get data.
            int seamX0 = y > 0 ? data->seamPath[stripIdx][y - 1] : INT_MAX;
            int seamX1 = data->seamPath[stripIdx][y];
            int seamX2 = y < data->height - 1 ? data->seamPath[stripIdx][y + 1] : INT_MAX;

            int insertOffsetX = x > seamX1 ? stripIdx + 1 : stripIdx;  // Each time we pass seam, the offset ticks up

            int idx = getPixelIdx(x - insertOffsetX, y, data->width);

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
                imgEnergyNew[idx] = calculatePixelEnergyStripe(data->img, newX, newY, data->width, data->height, data->channelCount, lowX, highX);
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

void updateEnergyOnSeamUpgrade(ImageProcessData* data)
{
    int oldWidth = data->width + SIM_NUM_SEAM_REMOVAL;
    const int stripWidth = oldWidth / SIM_NUM_SEAM_REMOVAL;

    /// Parallel:
    // TODO: The elements write into the same array. This causes problems as the energy on the right can get updated before the energy on the left.
    // Have to probably loop through x first and synchronize on each step of x.
    // #pragma omp parallel for
    for (int y = 0; y < data->height; y++)
    {
        for (int stripIdx = 0; stripIdx < SIM_NUM_SEAM_REMOVAL; stripIdx++)
        {
            for (int x = data->seamPath[stripIdx][y] - 2; x < oldWidth; x++)
            {
                // Get data.
                int seamX0 = y > 0 ? data->seamPath[stripIdx][y - 1] : INT_MAX;
                int seamX1 = data->seamPath[stripIdx][y];
                int seamX2 = y < data->height - 1 ? data->seamPath[stripIdx][y + 1] : INT_MAX;

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
                    data->imgEnergy[idx] = calculatePixelEnergy(data->img, newX, newY, data->width, data->height, data->channelCount);
                }
                else
                {
                    data->imgEnergy[idx] = data->imgEnergy[getPixelIdx(x, y, oldWidth)];
                }
            }
        }
    }
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

    // Fill bottom row with energy values
    // #pragma omp parallel
    for (int x = 0; x < data->width; x++)
    {
        data->imgSeam[getPixelIdx(x, data->height - 1, data->width)] = getEnergyPixelE(data->imgEnergy, x, data->height - 1, data->width, data->height);
    }

    /// Parallel:
    // - each row has to be calculated before starting the next row, we can only parallelize calc of a row
    for (int y = data->height - 2; y >= 0; y--)
    {
        // #pragma omp parallel for
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

/// @brief Calculate the cumulative energy of the image from the bottom to the top using the triangle approach to parallelization
void triangleSeamIdentification(ImageProcessData* data)
{
    if (data->imgSeam != NULL) {
        free(data->imgSeam);
    }

    data->imgSeam = (unsigned int *) malloc(sizeof(unsigned int) * data->width * data->height);

    // Fill bottom row with energy values
    for (int x = 0; x < data->width; x++)
    {
        data->imgSeam[getPixelIdx(x, data->height - 1, data->width)] = getEnergyPixelE(data->imgEnergy, x, data->height - 1, data->width, data->height);
    }

    // Separate steps by horizontal STRIPS of height STRIP_HEIGHT
    // (skip the bottom row as it is already correct)
    for (int stripBottom = data->height - 2; stripBottom > 0; stripBottom -= STRIP_HEIGHT)
    {
        int triangleWidth = STRIP_HEIGHT * 2;
        int triangleCount = (data->width + triangleWidth - 1) / triangleWidth;

        // Calculate each up pointing triangle in the strip
        // #pragma omp parallel for
        for (int triangleIdx = 0; triangleIdx < triangleCount; triangleIdx++)
        {
            for (int yLocal = 0; yLocal < STRIP_HEIGHT; yLocal++)
            {
                int y = stripBottom - yLocal;
                if (y < 0) break;

                int xStart = triangleIdx * triangleWidth + yLocal;
                int xEnd = min(xStart + triangleWidth - 2 * yLocal, data->width);
                for (int x = xStart; x < xEnd; x++)
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


        // Calculate each down pointing triangle in the strip
        // (bottom triangles start off the image to the left)
        int bottomPointTriangleLeftStartX = -STRIP_HEIGHT;
        triangleCount = (-bottomPointTriangleLeftStartX + data->width + triangleWidth - 1) / triangleWidth;
        // #pragma omp parallel for
        for (int triangleIdx = 0; triangleIdx < triangleCount; triangleIdx++)
        {
            for (int yLocal = 0; yLocal < STRIP_HEIGHT; yLocal++)
            {
                int y = stripBottom - yLocal;
                if (y < 0) break;

                int invYLocal = STRIP_HEIGHT - yLocal - 1;  // Inverted yLocal as the triangle is pointing down
                int xStart = bottomPointTriangleLeftStartX + triangleIdx * triangleWidth + invYLocal;
                int xEnd = min(xStart + triangleWidth - 2 * invYLocal, data->width);
                int clampedXStart = max(0, xStart);
                for (int x = clampedXStart; x < xEnd; x++)
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
    }
}

/// @brief Annotate the seam in the image
void seamAnnotate(ImageProcessData* data)
{
    const int stripWidth = data->width / SIM_NUM_SEAM_REMOVAL;

    // #pragma omp parallel for
    for (int seamIdx = 0; seamIdx < SIM_NUM_SEAM_REMOVAL; seamIdx++)
    {
        int lowX = seamIdx * stripWidth;
        int highX = lowX + stripWidth;

        // Allocate memory for seam path
        if (data->seamPath[seamIdx] != NULL)
        {
                free(data->seamPath[seamIdx]);
        }
        data->seamPath[seamIdx] = (int *) malloc(sizeof(int) * data->height);

        // Find the minimum energy in the top row on each image strip
        int curX = lowX;
        const int top_row = 0;
        for (int x = lowX + 1; x < highX; x++)
        {
            if (data->imgSeam[getPixelIdx(x, top_row, data->width)] < data->imgSeam[getPixelIdx(curX, top_row, data->width)])
            {
                curX = x;
            }
        }

        // Set SEAM
        data->seamPath[seamIdx][top_row] = curX;

        for (int y = 0; y < data->height - 1; y++)
        {
            // Find the minimum energy in the next row
            unsigned int leftEnergy =   getEnergyPixelEStripe(data->imgSeam, curX - 1, y + 1, data->width, data->height, lowX, highX);
            unsigned int centerEnergy = getEnergyPixelEStripe(data->imgSeam, curX    , y + 1, data->width, data->height, lowX, highX);
            unsigned int rightEnergy =  getEnergyPixelEStripe(data->imgSeam, curX + 1, y + 1, data->width, data->height, lowX, highX);

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
            data->seamPath[seamIdx][y + 1] = curX;
        }
    }
}

/// @brief Remove the seam from the image
void seamRemove(ImageProcessData* processData)
{
    // TODO: optimization: move pixels to the left (those after removed seam), then track the number of removed seams for index calc

    // Allocate space for new image
    unsigned int newWidth = processData->width - SIM_NUM_SEAM_REMOVAL;
    unsigned int pixelCount = newWidth * processData->height;
    unsigned char* image = (unsigned char *) malloc(sizeof(unsigned char) * pixelCount * processData->channelCount);
    
    // Copy image data without seam
    /// Parallel:
    // - standard for parallel, as the copying of whole lines is nicely divided between threads
    // #pragma omp parallel for
    for (int y = 0; y < processData->height; y++)
    {
        int seanPassedCount = 0;
        for (int x = 0; x < processData->width; x++)
        {
            if (!isSeam(processData, x, y, seanPassedCount))
            {
                unsigned int pixelIdxC = getPixelIdxC(x, y, processData->width, processData->channelCount);
                for (int channel = 0; channel < processData->channelCount; channel++)
                {
                    unsigned int pixelPos = getPixelIdxC(x - seanPassedCount, y, newWidth, processData->channelCount);
                    image[pixelPos + channel] = processData->img[pixelIdxC + channel];
                }
            }
            else
                seanPassedCount++;
        }
    }

    // Free old image and set new image
    free(processData->img);

    // Update process data
    processData->img = image;
    processData->width = newWidth;
    processData->height = processData->height;
    processData->channelCount = processData->channelCount;
}

#ifdef RENDER_LOADING_BAR_WIDTH
/// @brief Update the loading bar
void updatePrintLoadingBar(int progress, int total)
{
    static int lastProgress = -1;
    if (lastProgress != -1) {
        printf("\033[F"); // Move cursor up one line
    }

    int percent = (progress * 100) / total;
    int filled = (progress * RENDER_LOADING_BAR_WIDTH) / total;

    printf("[");  // Carriage return to overwrite line
    for (int i = 0; i < filled; i++) printf("=");
    for (int i = filled; i < RENDER_LOADING_BAR_WIDTH; i++) printf(" ");
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
    printf("Arguments: imageInPath=%s, imageOutPath=%s, seamCount=%s\n", args[1], args[2], args[3]);

    // Parse arguments /////////////////////////////////////////////////////////////////////
    char *imageInPath = args[1];
    char *imageOutPath = args[2];
    int seamCount = atoi(args[3]);
    int outputHeight; // = atoi(args[4]); // Height stays the same

    // Setup processing data struct //////////////////////////////////////////////////////
    ImageProcessData processData;
    processData.img = NULL;
    processData.imgEnergy = NULL;
    processData.imgSeam = NULL;
    processData.seamPath = (int**) malloc(sizeof(int*) * SIM_NUM_SEAM_REMOVAL);

    // Load image //////////////////////////////////////////////////////////////////////////
    processData.img = stbi_load(imageInPath, &processData.width, &processData.height, &processData.channelCount, STB_COLOR_CHANNELS);
    if (processData.img == NULL)
    {
        printf("Error: Couldn't load image\n");
        exit(EXIT_FAILURE);
    }
    printf("Loaded image %s of size %dx%d.\n", imageInPath, processData.width, processData.height);

    if (seamCount >= processData.width || seamCount < 0)
    {
        printf("Error: Incorrect value for number of seams.\n");
        return EXIT_FAILURE;
    }
    if (seamCount % SIM_NUM_SEAM_REMOVAL || processData.width % SIM_NUM_SEAM_REMOVAL)
    {   // TODO fix: the num of max num of simultaneously removed seams has to divide num of total removed seams 
        printf("Error: seamCount and image width should be divisible by the SIM_NUM_SEAM_REMOVAL!\n");
        return EXIT_FAILURE;
    }

    outputHeight = processData.height;

    // Process image //////////////////////////////////////////////////////////////////////////
    TimingStats timingStats;
    double startTotalProcessingTime = omp_get_wtime();
    // printf("Seam count: %d\n", seamCount);

    double startEnergyTime = omp_get_wtime();
    calculateEnergyFull(&processData);
    double stopEnergyTime = omp_get_wtime();
    timingStats.energyCalculations += stopEnergyTime - startEnergyTime;
    for (int passIdx = 0; passIdx < seamCount / SIM_NUM_SEAM_REMOVAL; passIdx++)
    {
        // printf("Processing seam %d/%d\n", i + 1, seamCount);

        // Energy step
        startEnergyTime = omp_get_wtime();
        if (passIdx != 0) {
            updateEnergyOnSeam(&processData);
            // updateEnergyOnSeamUpgrade(&processData);
        }
        stopEnergyTime = omp_get_wtime();
        timingStats.energyCalculations += stopEnergyTime - startEnergyTime;

        // Seam identification step
        double startSeamTime = omp_get_wtime();
        seamIdentification(&processData);
        // triangleSeamIdentification(&processData);
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

#ifdef RENDER_LOADING_BAR_WIDTH
        updatePrintLoadingBar(passIdx + 1, seamCount / SIM_NUM_SEAM_REMOVAL);
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
    free(processData.seamPath);

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
    FILE *timingFile = fopen("../timing_stats/timing_stats_parallel.txt", "a");
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
