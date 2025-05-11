
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

// CUDA
#include <cuda_runtime.h>
#include <cuda.h>
#include "lib/helper_cuda.h"

// GIF
#include "lib/gif.h"

// STB image library
#define STB_IMAGE_IMPLEMENTATION
#include "lib/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "lib/stb_image_write.h"

// Constants
#define MAX_SIM_STEPS 5000
#define NUM_FRAMES_CAPTURED 50  // Total frames caputed
#define DELTA_t 1
#define Du 0.16f
#define Dv 0.08f
#define F 0.060f
#define k 0.062f

#define U_INSIDE 0.75f
#define V_INSIDE 0.25f
#define U_OUTSIDE 1.0f
#define V_OUTSIDE 0.0f

#define COLOR_CHANNELS 1

// Settings
#define SAVE_TIMING_STATS
// #define WRITE_OUTPUT_IMAGE
// #define WRITE_OUTPUT_GIF

// CUDA settings
#define BLOCK_SIZE 16

typedef struct _Cell_ {
    float U;  // Concentration of species U
    float V;  // Concentration of species V
} Cell;

struct execution_result
{
    int size;
    float total;
};

void swapDeviceGridPtr(Cell** firstGridPtr, Cell** secondGridPtr)
{
    Cell* tmp = *firstGridPtr;
    *firstGridPtr = *secondGridPtr;
    *secondGridPtr = tmp;
}

void allocateDeviceGrid(Cell** deviceGridDataPtr, int gridSize)
{
    checkCudaErrors(cudaMalloc((void **)deviceGridDataPtr, gridSize * gridSize * sizeof(Cell)));
    getLastCudaError("Failed to allocate grid memory.");
}

__device__ void grayScottSimStep(Cell* grid, int gridSize, int x, int y, float& newU, float& newV)
{
    int idx;
    idx = y * gridSize + x;
    Cell origin = grid[idx];
    idx = y * gridSize + ((x - 1 + gridSize) % gridSize);
    Cell left = grid[idx];
    idx = y * gridSize + ((x + 1) % gridSize);
    Cell right = grid[idx];
    idx = ((y - 1 + gridSize) % gridSize) * gridSize + x;
    Cell up = grid[idx];
    idx = ((y + 1) % gridSize) * gridSize + x;
    Cell down = grid[idx];

    float deltaSqrU = right.U + left.U + down.U + up.U - 4 * origin.U;
    float deltaSqrV = right.V + left.V + down.V + up.V - 4 * origin.V;

    float uVSqr = origin.U * origin.V * origin.V;

    newU = origin.U + DELTA_t * (-uVSqr + F * (1 - origin.U) + Du * deltaSqrU);
    newV = origin.V + DELTA_t * ( uVSqr - (F + k) * origin.V + Dv * deltaSqrV);
}

#ifdef WRITE_OUTPUT_IMAGE
void write_output_image_frame(int step, int gridSize, Cell* gridData, Cell* deviceGridData)
{
    // recover data from the GPU to the CPU allocated memory
    int gridDataSizeBytes = gridSize * gridSize * sizeof(Cell);
    checkCudaErrors(cudaMemcpy(gridData, deviceGridData, gridDataSizeBytes, cudaMemcpyDeviceToHost));
    getLastCudaError("Retrieving data from GPU failed");

    char outputImageFpath[100];
    snprintf(outputImageFpath, sizeof(outputImageFpath), "%s%d%s%d%s%d%s", "./output_images/", gridSize, "x", gridSize, "/", step, ".png");

    unsigned char gridVImage[gridSize * gridSize];
    for (int y = 0; y < gridSize; y++)
    {
        for (int x = 0; x < gridSize; x++)
        {
            gridVImage[y * gridSize + x] = (unsigned char) (255 * gridData[y * gridSize + x].V);
        }
    }

    stbi_write_png(outputImageFpath, gridSize, gridSize, COLOR_CHANNELS, gridVImage, gridSize * COLOR_CHANNELS);
}
#endif

#ifdef WRITE_OUTPUT_GIF
void write_output_gif_frame(int step, int gridSize, Cell* gridData, Cell* deviceGridData, GifWriter* gifWriter)
{
    // recover data from the GPU to the CPU allocated memory
    int gridDataSizeBytes = gridSize * gridSize * sizeof(Cell);
    checkCudaErrors(cudaMemcpy(gridData, deviceGridData, gridDataSizeBytes, cudaMemcpyDeviceToHost));
    getLastCudaError("Retrieving data from GPU failed");

    int outColorChannels = 4;
    unsigned char* frame = new unsigned char[gridSize * gridSize * outColorChannels];
    for (int y = 0; y < gridSize; y++)
    {
        for (int x = 0; x < gridSize; x++)
        {
            unsigned char val = (unsigned char) (255 * gridData[y * gridSize + x].V);
            int idx = (y * gridSize + x) * outColorChannels;

            frame[idx] = val; // R
            frame[idx + 1] = val; // G
            frame[idx + 2] = val; // B
            frame[idx + 3] = 255; // A
        }
    }

    GifWriteFrame(gifWriter, frame, gridSize, gridSize, 4);
    delete[] frame;
}
#endif

__global__ void grayScottSimStep_kernel(Cell* deviceGrid, Cell* deviceGridTmp, int gridSize)
{
    // find index of the pixel of the thread
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    // int pixelPos = threadIdx.x + blockIdx.x * blockDim.x;
    // int x = pixelPos % gridSize;
    // int y = pixelPos / gridSize;

    if (x < gridSize && y < gridSize)
    {
        float newU, newV;
        grayScottSimStep(deviceGrid, gridSize, x, y, newU, newV);

        int idx = y * gridSize + x;
        deviceGridTmp[idx].U = newU;
        deviceGridTmp[idx].V = newV;
    }
}

void grayScottSolver(Cell* gridData, int gridSize)
{
    // Copy/allocate the initial grids to the GPU
    int gridDataSizeBytes = gridSize * gridSize * sizeof(Cell);

    Cell* deviceGridData;
    allocateDeviceGrid(&deviceGridData, gridSize);
    checkCudaErrors(cudaMemcpy(deviceGridData, gridData, gridDataSizeBytes, cudaMemcpyHostToDevice));
    getLastCudaError("Failed to copy initial grid to device.");

    Cell* deviceGridDataTmp;
    allocateDeviceGrid(&deviceGridDataTmp, gridSize);

#ifdef WRITE_OUTPUT_GIF
    GifWriter gifWriter;
    char outputGifFpath[100];
    snprintf(outputGifFpath, sizeof(outputGifFpath), "%s%d%s%d%s%d%s", "./output_gifs/", gridSize, "x", gridSize, "/", MAX_SIM_STEPS, ".gif");
    GifBegin(&gifWriter, outputGifFpath, gridSize, gridSize, 0);
#endif

    // set up the grid and block size
    dim3 cudaBlockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 cudaGridSize((gridSize + BLOCK_SIZE - 1) / BLOCK_SIZE,
                      (gridSize + BLOCK_SIZE - 1) / BLOCK_SIZE);
    // dim3 cudaBlockSize(BLOCK_SIZE * BLOCK_SIZE);
    // dim3 cudaGridSize(gridSize * gridSize / (BLOCK_SIZE * BLOCK_SIZE));

    // runs KERNEL
    for (int step = 0; step < MAX_SIM_STEPS; step++)
    {
        // 1. Calculate new grid values
        grayScottSimStep_kernel<<<cudaGridSize, cudaBlockSize>>>(deviceGridData, deviceGridDataTmp, gridSize);
        getLastCudaError("grayScottSimStep_kernel() execution failed");

        // 2. Switch current grid to new grid
        swapDeviceGridPtr(&deviceGridData, &deviceGridDataTmp);

#ifdef WRITE_OUTPUT_IMAGE
        if ((step + 1) % (MAX_SIM_STEPS / NUM_FRAMES_CAPTURED) == 0)
        {
            write_output_image_frame((step + 1), gridSize, gridData, deviceGridData);
        }
#endif
#ifdef WRITE_OUTPUT_GIF
        if ((step + 1) % (MAX_SIM_STEPS / NUM_FRAMES_CAPTURED)  == 0)
        {
            write_output_gif_frame((step + 1), gridSize, gridData, deviceGridData, &gifWriter);
        }
#endif
    }

#ifdef WRITE_OUTPUT_GIF
    GifEnd(&gifWriter);
#endif

    // recover data from the GPU to the CPU allocated memory
    checkCudaErrors(cudaMemcpy(gridData, deviceGridData, gridDataSizeBytes, cudaMemcpyDeviceToHost));
    getLastCudaError("Retrieving data from GPU failed");

    cudaFree(deviceGridData);
    cudaFree(deviceGridDataTmp);
    getLastCudaError("Freeing memory failed");
}

void initGrid(Cell* gridData, int gridSize)
{
    for (int y = 0; y < gridSize; y++)
    {
        for (int x = 0; x < gridSize; x++)
        {
            if ((x > (int)(gridSize * 3.0/8) && x < (int)(gridSize * 5.0/8)) &&
                (y > (int)(gridSize * 3.0/8) && y < (int)(gridSize * 5.0/8)))
            {
                gridData[y * gridSize + x].U = U_INSIDE;
                gridData[y * gridSize + x].V = V_INSIDE;
            }
            else
            {
                gridData[y * gridSize + x].U = U_OUTSIDE;
                gridData[y * gridSize + x].V = V_OUTSIDE;
            }
        }
    }
}

int main(int argc, char *args[])
{
    if (argc != 2)
    {
        printf("Error: Invalid amount of arguments. [%d]\n", argc);
        exit(EXIT_FAILURE);
    }

    int gridSize = atoi(args[1]);

    int gridDataSizeBytes = gridSize * gridSize * sizeof(Cell);

    // Reserve space for grids and initialize them
    Cell* gridData = (Cell*) malloc(gridDataSizeBytes);
    initGrid(gridData, gridSize);


#ifdef WRITE_OUTPUT_IMAGE
    // Create dirs if they do not exist
    {
        struct stat output_images_st = {0};
        if (stat("./output_images", &output_images_st) == -1) {
            mkdir("./output_images", 0700);
        }
        struct stat st = {0};
        char outDirFpath[50];
        snprintf(outDirFpath, sizeof(outDirFpath), "%s%d%s%d", "./output_images/", gridSize, "x", gridSize);
        if (stat(outDirFpath, &st) == -1) {
            mkdir(outDirFpath, 0700);
        }
    }
#endif
#ifdef WRITE_OUTPUT_GIF
    {
        // Create dirs if they do not exist
        struct stat output_gifs_st = {0};
        if (stat("./output_gifs", &output_gifs_st) == -1) {
            mkdir("./output_gifs", 0700);
        }
        struct stat st = {0};
        char outGifDirFpath[50];
        snprintf(outGifDirFpath, sizeof(outGifDirFpath), "%s%d%s%d", "./output_gifs/", gridSize, "x", gridSize);
        if (stat(outGifDirFpath, &st) == -1) {
            mkdir(outGifDirFpath, 0700);
        }
    }
#endif

    // Create time events
    cudaEvent_t startMain, stopMain;

    cudaEventCreate(&startMain);
    cudaEventCreate(&stopMain);

    float elapsedMain = 0;

    cudaEventRecord(startMain);

    // Main algorithm ///////////////////////////////////////////////////////////////////////////////////
    grayScottSolver(gridData, gridSize);
    /////////////////////////////////////////////////////////////////////////////////////////////////////

    // End the time recording and calculate elapsed times
    cudaEventRecord(stopMain);
    cudaEventSynchronize(stopMain);

    cudaEventElapsedTime(&elapsedMain, startMain, stopMain);


// Output timing stats to file //////////////////////////////////////////////////////////////////////////
#ifdef SAVE_TIMING_STATS
    struct execution_result result;
    result.size = gridSize;
    result.total = elapsedMain;

    // Create dir timing_stats if it does not exist
    struct stat timing_stats_st = {0};
    if (stat("./timing_stats", &timing_stats_st) == -1) {
        mkdir("./timing_stats", 0700);
    }

    FILE *timingFile = fopen("./timing_stats/timing_stats_parallel.txt", "a");
    fprintf(timingFile, "-------------- HISTOGRAM EQUALIZATION - Parallel -------------\n");
    fprintf(timingFile, "------------------------- %d%s%d ----------------------\n", gridSize, "x", gridSize);
    fprintf(timingFile, "Grid size: %d\n", result.size);
    fprintf(timingFile, "Total time: %f ms\n", result.total);
    fprintf(timingFile, "-----------------------------------------------------\n");
    fprintf(timingFile, "\n");
    fclose(timingFile);
#endif

    // Clean-up events
    cudaEventDestroy(startMain);
    cudaEventDestroy(stopMain);

    // Free memory
    free(gridData);

    return EXIT_SUCCESS;
}

