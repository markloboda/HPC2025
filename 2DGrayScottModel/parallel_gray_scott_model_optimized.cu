
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <thread>

// CUDA
#include <cuda_runtime.h>
#include <cuda.h>
#include "lib/helper_cuda.h"

// STB image library
#define STB_IMAGE_IMPLEMENTATION
#include "lib/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "lib/stb_image_write.h"

// GIF
#include "lib/gif.h"

// Constants
#define MAX_SIM_STEPS 5000
#define NUM_FRAMES_CAPTURED 50  // Total frames caputed
#define NUM_GPUS 2  // Only two GPUs are needed for HW
#define DELTA_t 1
#define Du 0.16f
#define Dv 0.08f
#define F 0.060f
#define k 0.062f
#define SHARED_GRID_SIZE (BLOCK_SIZE + 2) // Shared memory size (including halo cells)

#define U_INSIDE 0.75f
#define V_INSIDE 0.25f
#define U_OUTSIDE 1.0f
#define V_OUTSIDE 0.0f

#define COLOR_CHANNELS 1

// Settings
#define SAVE_TIMING_STATS
// #define WRITE_OUTPUT_IMAGE
#define WRITE_OUTPUT_GIF

// CUDA settings
#define BLOCK_SIZE 16

typedef struct _Cell_ {
    float2 UV;
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

void setDeviceCuda(int deviceIdx)
{
    checkCudaErrors(cudaSetDevice(deviceIdx));
    getLastCudaError("cudaSetDevice() failed");
}

__device__ void grayScottSimStep(Cell sharedGrid[SHARED_GRID_SIZE * SHARED_GRID_SIZE], int x, int y, float& newU, float& newV)
{
    Cell origin = sharedGrid[y * SHARED_GRID_SIZE + x];
    Cell left = sharedGrid[y * SHARED_GRID_SIZE + (x - 1)];
    Cell right = sharedGrid[y * SHARED_GRID_SIZE + (x + 1)];
    Cell up = sharedGrid[(y - 1) * SHARED_GRID_SIZE + x];
    Cell down = sharedGrid[(y + 1) * SHARED_GRID_SIZE + x];

    float2 originUV = origin.UV;
    float2 leftUV = left.UV;
    float2 rightUV = right.UV;
    float2 upUV = up.UV;
    float2 downUV = down.UV;

    float deltaSqrU = leftUV.x + rightUV.x + upUV.x + downUV.x - 4 * originUV.x;
    float deltaSqrV = leftUV.y + rightUV.y + upUV.y + downUV.y - 4 * originUV.y;

    float uVSqr = origin.UV.x * origin.UV.y * origin.UV.y;

    newU = originUV.x + DELTA_t * (-uVSqr + F * (1 - originUV.x) + Du * deltaSqrU);
    newV = originUV.y + DELTA_t * ( uVSqr - (F + k) * originUV.y + Dv * deltaSqrV);
}

void recoverGridDataDevice(int deviceIdx, int gridSize, Cell* gridData, Cell* deviceGridData)
{
    setDeviceCuda(deviceIdx);

    int gridDataDeviceSizeBytes = (gridSize * gridSize / NUM_GPUS) * sizeof(Cell);
    checkCudaErrors(cudaMemcpy(gridData, deviceGridData, gridDataDeviceSizeBytes, cudaMemcpyDeviceToHost));
    getLastCudaError("Retrieving data from GPU failed");
}

void recoverGridData(int gridSize, Cell* gridData, Cell* deviceGridData[])
{
    // Combine the data from both GPUs
    std::thread threads[NUM_GPUS];
    for (int deviceIdx = 0; deviceIdx < NUM_GPUS; deviceIdx++)
    {
        // recover data from the GPU to the CPU allocated memory (only the important part)
        int gridOffset = deviceIdx * gridSize * gridSize / NUM_GPUS;
        threads[deviceIdx] = std::thread(recoverGridDataDevice, deviceIdx, gridSize, &gridData[gridOffset], &deviceGridData[deviceIdx][gridOffset]);
    }

    for (int deviceIdx = 0; deviceIdx < NUM_GPUS; deviceIdx++)
    {
        threads[deviceIdx].join();
    }
}

#ifdef WRITE_OUTPUT_IMAGE
void write_output_image_frame(int step, int gridSize, Cell* gridData, Cell* deviceGridData[])
{
    recoverGridData(gridSize, gridData, deviceGridData);

    char outputImageFpath[100];
    snprintf(outputImageFpath, sizeof(outputImageFpath), "%s%d%s%d%s%d%s", "./output_images/", gridSize, "x", gridSize, "/", step, ".png");

    unsigned char gridVImage[gridSize * gridSize];
    for (int y = 0; y < gridSize; y++)
    {
        for (int x = 0; x < gridSize; x++)
        {
            gridVImage[y * gridSize + x] = (unsigned char) (255 * gridData[y * gridSize + x].UV.y);
        }
    }

    stbi_write_png(outputImageFpath, gridSize, gridSize, COLOR_CHANNELS, gridVImage, gridSize * COLOR_CHANNELS);
}
#endif

#ifdef WRITE_OUTPUT_GIF
void write_output_gif_frame(int step, int gridSize, Cell* gridData, Cell* deviceGridData[], GifWriter* gifWriter)
{
    recoverGridData(gridSize, gridData, deviceGridData);

    int outColorChannels = 4;
    unsigned char* frame = new unsigned char[gridSize * gridSize * outColorChannels];
    for (int y = 0; y < gridSize; y++)
    {
        for (int x = 0; x < gridSize; x++)
        {
            unsigned char val = (unsigned char) (255 * gridData[y * gridSize + x].UV.y);
            int idx = (y * gridSize + x) * outColorChannels;

            frame[idx] = val; // R
            frame[idx + 1] = val; // G
            frame[idx + 2] = val; // B
            frame[idx + 3] = 255; // A
        }
    }

    GifWriteFrame(gifWriter, frame, gridSize, gridSize, 10);
    delete[] frame;
}
#endif

__global__ void grayScottSimStep_kernel(int deviceIdx, Cell* deviceGrid, Cell* deviceGridTmp, int gridOffsetHeight, int gridSize)
{
    // find index of the pixel of the thread
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int gx = blockIdx.x * blockDim.x + threadIdx.x;
    int gy = blockIdx.y * blockDim.y + threadIdx.y + gridOffsetHeight;

    // find the index of the neighboring pixels (faster than using modulo on every access)
    int left  = (gx == 0          ? gridSize-1 : gx-1);
    int right = (gx == gridSize-1 ?          0 : gx+1);
    int up    = (gy == 0          ? gridSize-1 : gy-1);
    int down  = (gy == gridSize-1 ?          0 : gy+1);

    __shared__ Cell sharedGrid[SHARED_GRID_SIZE * SHARED_GRID_SIZE];

    int deviceGridWidth = gridSize;
    int deviceGridHeight = gridSize / NUM_GPUS;
    if (gx < deviceGridWidth && gy < deviceGridHeight + gridOffsetHeight)
    {
        int x = tx + 1;
        int y = ty + 1;

        // load data into shared memory (core and halo cells)
        sharedGrid[y * SHARED_GRID_SIZE + x] = deviceGrid[gy * gridSize + gx];
        int idx;
        if (tx == 0) // left edge
        {
            idx = gy * gridSize + left;
            sharedGrid[y * SHARED_GRID_SIZE + x - 1] = deviceGrid[idx];
        }
        if (tx == BLOCK_SIZE - 1) // right edge
        {
            idx = gy * gridSize + right;
            sharedGrid[y * SHARED_GRID_SIZE + x + 1] = deviceGrid[idx];
        }
        if (ty == 0) // top edge
        {
            idx = up * gridSize + gx;
            sharedGrid[(y - 1) * SHARED_GRID_SIZE + x] = deviceGrid[idx];
        }
        if (ty == BLOCK_SIZE - 1) // bottom edge
        {
            idx = down * gridSize + gx;
            sharedGrid[(y + 1) * SHARED_GRID_SIZE + x] = deviceGrid[idx];
        }

        __syncthreads();

        // calculate new values
        float newU, newV;
        grayScottSimStep(sharedGrid, x, y, newU, newV);

        idx = gy * gridSize + gx;
        deviceGridTmp[idx].UV.x = newU;
        deviceGridTmp[idx].UV.y = newV;
    }
}

void setupDevice(int deviceIdx, Cell* gridData, int gridSize, Cell** deviceGridDataOut, Cell** deviceGridDataTmpOut)
{
    int gridDataSizeBytes = gridSize * gridSize * sizeof(Cell);

    setDeviceCuda(deviceIdx);

    // Setup peer access
    if (NUM_GPUS > 1)
    {
        int canAccessPeer = 0;
        int peerDeviceIdx = deviceIdx ^ 1;
        cudaDeviceCanAccessPeer(&canAccessPeer, deviceIdx, peerDeviceIdx);
        if (canAccessPeer)
        {
            cudaDeviceEnablePeerAccess(peerDeviceIdx, 0);
        }
        else
        {
            fprintf(stderr, "No p2p device access.\n");
            exit(EXIT_FAILURE);
        }
    }

    // Copy/allocate the initial grids to the GPU
    Cell* deviceGridData;
    allocateDeviceGrid(&deviceGridData, gridSize);
    checkCudaErrors(cudaMemcpy(deviceGridData, gridData, gridDataSizeBytes, cudaMemcpyHostToDevice));
    getLastCudaError("Failed to copy initial grid to device.");

    Cell* deviceGridDataTmp;
    allocateDeviceGrid(&deviceGridDataTmp, gridSize);

    *deviceGridDataOut = deviceGridData;
    *deviceGridDataTmpOut = deviceGridDataTmp;
}

void runKernel(int deviceIdx, Cell* deviceGridData, Cell* deviceGridDataTmp, int gridOffsetHeight, int gridSize)
{
    setDeviceCuda(deviceIdx);

    // set up the grid and block size
    dim3 cudaBlockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 cudaGridSize(((gridSize + BLOCK_SIZE - 1) / BLOCK_SIZE),
                      ((gridSize + BLOCK_SIZE - 1) / BLOCK_SIZE) / NUM_GPUS);

    // 1. Calculate new grid values
    grayScottSimStep_kernel<<<cudaGridSize, cudaBlockSize>>>(deviceIdx, deviceGridData, deviceGridDataTmp, gridOffsetHeight, gridSize);
    getLastCudaError("grayScottSimStep_kernel() execution failed");
}

void grayScottSolver(Cell* gridData, int gridSize)
{
    int deviceCount;
    checkCudaErrors(cudaGetDeviceCount(&deviceCount));
    getLastCudaError("cudaGetDeviceCount() failed");
    if (deviceCount < NUM_GPUS)
    {
        fprintf(stderr, "Not enough devices found: %d\n", deviceCount);
        exit(EXIT_FAILURE);
    }

    std::thread threads[NUM_GPUS];
    Cell* deviceGridData[NUM_GPUS];
    Cell* deviceGridDataTmp[NUM_GPUS];

    // Setup each device
    for (int deviceIdx = 0; deviceIdx < NUM_GPUS; deviceIdx++)
    {
        threads[deviceIdx] = std::thread(setupDevice, deviceIdx, gridData, gridSize, &deviceGridData[deviceIdx], &deviceGridDataTmp[deviceIdx]);
    }

    for (int deviceIdx = 0; deviceIdx < NUM_GPUS; deviceIdx++)
    {
        threads[deviceIdx].join();
    }

#ifdef WRITE_OUTPUT_GIF
    GifWriter gifWriter;
    char outputGifFpath[100];
    snprintf(outputGifFpath, sizeof(outputGifFpath), "%s%d%s%d%s%d%s", "./output_gifs/", gridSize, "x", gridSize, "/", MAX_SIM_STEPS, ".gif");
    GifBegin(&gifWriter, outputGifFpath, gridSize, gridSize, 0);

    // write the first frame
    write_output_gif_frame(0, gridSize, gridData, deviceGridData, &gifWriter);
#endif

    // runs KERNELs
    for (int step = 0; step < MAX_SIM_STEPS; step++)
    {
        // Run kernel on each device
        for (int deviceIdx = 0; deviceIdx < NUM_GPUS; deviceIdx++)
        {
            int gridOffsetHeight = deviceIdx * gridSize / NUM_GPUS;
            threads[deviceIdx] = std::thread(runKernel, deviceIdx, deviceGridData[deviceIdx], deviceGridDataTmp[deviceIdx], gridOffsetHeight, gridSize);
        }

        for (int deviceIdx = 0; deviceIdx < NUM_GPUS; deviceIdx++)
        {
            threads[deviceIdx].join();
        }

        // Synchronize devices
        for (int deviceIdx = 0; deviceIdx < NUM_GPUS; deviceIdx++)
        {
            swapDeviceGridPtr(&deviceGridData[deviceIdx], &deviceGridDataTmp[deviceIdx]);
        }

        if (NUM_GPUS > 1)
        {
            int gridDataDeviceSizeBytes = (gridSize * gridSize / NUM_GPUS) * sizeof(Cell);
            recoverGridData(gridSize, gridData, deviceGridData);

            for (int deviceIdx = 0; deviceIdx < NUM_GPUS; deviceIdx++)
            {
                // int peerDeviceIdx = deviceIdx ^ 1;
                // int gridOffset = deviceIdx * gridSize * gridSize / NUM_GPUS;

                // checkCudaErrors(cudaMemcpyPeer(&deviceGridData[peerDeviceIdx][gridOffset], peerDeviceIdx, &deviceGridData[deviceIdx][gridOffset], deviceIdx, gridDataDeviceSizeBytes));
                // checkCudaErrors(cudaMemcpy(&(deviceGridData[peerDeviceIdx][gridOffset]), &(deviceGridData[deviceIdx][gridOffset]), gridDataDeviceSizeBytes, cudaMemcpyDefault));

                setDeviceCuda(deviceIdx);
                int peerDeviceIdx = deviceIdx ^ 1;
                int gridOffset = peerDeviceIdx * gridSize * gridSize / NUM_GPUS;
                checkCudaErrors(cudaMemcpy(&deviceGridData[deviceIdx][gridOffset], &gridData[gridOffset], gridDataDeviceSizeBytes, cudaMemcpyHostToDevice));
                getLastCudaError("Retrieving data from GPU failed");

                getLastCudaError("Transfer of data to peer failed.");
            }
        }

        // Write outputs
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

    recoverGridData(gridSize, gridData, deviceGridData);

    for (int deviceIdx = 0; deviceIdx < NUM_GPUS; deviceIdx++)
    {
        setDeviceCuda(deviceIdx);
        cudaFree(deviceGridData[deviceIdx]);
        cudaFree(deviceGridDataTmp[deviceIdx]);
    }
    getLastCudaError("Freeing memory failed");

    if (NUM_GPUS > 1)
    {
        checkCudaErrors(cudaSetDevice(0));
        checkCudaErrors(cudaDeviceDisablePeerAccess(1));
        checkCudaErrors(cudaSetDevice(1));
        checkCudaErrors(cudaDeviceDisablePeerAccess(0));
    }
}

void initGrid(Cell* gridData, int gridSize)
{
    for (int y = 0; y < gridSize; y++)
    {
        for (int x = 0; x < gridSize; x++)
        {
            if ((x > (int)(gridSize * 3.0f/8) && x < (int)(gridSize * 5.0f/8)) &&
                (y > (int)(gridSize * 3.0f/8) && y < (int)(gridSize * 5.0f/8)))
            {
                gridData[y * gridSize + x].UV.x = U_INSIDE;
                gridData[y * gridSize + x].UV.y = V_INSIDE;
            }
            else
            {
                gridData[y * gridSize + x].UV.x = U_OUTSIDE;
                gridData[y * gridSize + x].UV.y = V_OUTSIDE;
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
    {
        // Create dirs if they do not exist
        struct stat output_images_st = {0};
        if (stat("./output_images", &output_images_st) == -1) {
            mkdir("./output_images", 0700);
        }
        struct stat st = {0};
        char outImageDirFpath[50];
        snprintf(outImageDirFpath, sizeof(outImageDirFpath), "%s%d%s%d", "./output_images/", gridSize, "x", gridSize);
        if (stat(outImageDirFpath, &st) == -1) {
            mkdir(outImageDirFpath, 0700);
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

    checkCudaErrors(cudaSetDevice(0));
    cudaEventCreate(&startMain);
    cudaEventCreate(&stopMain);

    float elapsedMain = 0;

    cudaEventRecord(startMain);

    // Main algorithm ///////////////////////////////////////////////////////////////////////////////////
    grayScottSolver(gridData, gridSize);
    /////////////////////////////////////////////////////////////////////////////////////////////////////

    // End the time recording and calculate elapsed times
    checkCudaErrors(cudaSetDevice(0));
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

    FILE *timingFile = fopen("./timing_stats/timing_stats_optimized.txt", "a");
    fprintf(timingFile, "-------------- HISTOGRAM EQUALIZATION - Optimized -------------\n");
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

