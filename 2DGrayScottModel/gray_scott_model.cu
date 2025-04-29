
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cuda_runtime.h>
#include <cuda.h>
#include "lib/helper_cuda.h"

// STB image library
#define STB_IMAGE_IMPLEMENTATION
#include "lib/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "lib/stb_image_write.h"

// Constants
#define MAX_SIM_STEPS 5000
#define DELTA_t 1
#define Du 0.16
#define Dv 0.08
#define F 0.060
#define k 0.062

#define U_INSIDE 0.75
#define V_INSIDE 0.25
#define U_OUTSIDE 1.0
#define V_OUTSIDE 0.0

#define COLOR_CHANNELS 1

// Settings
#define SAVE_TIMING_STATS
// #define WRITE_OUTPUT_IMAGE

typedef struct _Cell_ {
    float U;  // Concentration of species U
    float V;  // Concentration of species V
} Cell;

struct execution_result
{
    int size;
    float total;
};

void grayScottSimStep(Cell** grid, Cell** gridOut, int gridSize)
{
    for (int y = 0; y < gridSize; y++)
    {
        for (int x = 0; x < gridSize; x++)
        {
            int left = (x + gridSize) % gridSize;
            int right = x % gridSize;
            int up = (y + gridSize) % gridSize;
            int down = y % gridSize;

            float deltaSqrU = grid[y][right].U + 
                              grid[y][left].U + 
                              grid[down][x].U + 
                              grid[up][x].U - 
                              4 * grid[y][x].U;

            float deltaSqrV = grid[y][right].V + 
                              grid[y][left].V + 
                              grid[down][x].V + 
                              grid[up][x].V - 
                              4 * grid[y][x].V;
                              
            float uVSqr = grid[y][x].U * grid[y][x].V * grid[y][x].V;

            float newU = grid[y][x].U + DELTA_t * (-uVSqr + F * (1 - grid[y][x].U) + Du + deltaSqrU);
            float newV = grid[y][x].V + DELTA_t * ( uVSqr - (F + k) * grid[y][x].V + Dv + deltaSqrV);

            gridOut[y][x].U = newU;
            gridOut[y][x].V = newV;
        }
    }
}

void swapGridPtr(Cell*** firstGridPtr, Cell*** secondGridPtr)
{
    Cell** tmp = *firstGridPtr;
    *firstGridPtr = *secondGridPtr;
    *secondGridPtr = tmp;
}

void initGrid(Cell** grid, int gridSize)
{ 
    for (int y = 0; y < gridSize; y++)
    {
        for (int x = 0; x < gridSize; x++)
        {
            if (((int)(gridSize * (3/8)) < x < (int)(gridSize * (5/8))) &&
                (gridSize * (3/8) < y < gridSize * (5/8)))
            {
                grid[y][x].U = U_INSIDE;
                grid[y][x].V = V_INSIDE;
            }
            else
            {
                grid[y][x].U = U_OUTSIDE;
                grid[y][x].V = V_OUTSIDE;
            }
        }
    }
}

void allocateGrid(Cell** gridDataPtr, Cell*** gridPtr, int gridSize)
{
    *gridDataPtr = (Cell*) malloc(gridSize * gridSize * sizeof(Cell));
    *gridPtr = (Cell**) malloc(gridSize * sizeof(Cell*));
    // Now we do not have to calc pixel position every time
    for (int i = 0; i < gridSize; i++)
        *gridPtr[i] = &(*gridDataPtr)[i * gridSize];
}

int main(int argc, char *args[])
{
    if (argc != 2)
    {
        printf("Error: Invalid amount of arguments. [%d]\n", argc);
        exit(EXIT_FAILURE);
    }

    int gridSize = atoi(args[1]);

    // Reserve space for grids and initialize them
    Cell* gridData;
    Cell** grid;
    allocateGrid(&gridData, &grid, gridSize);
    initGrid(grid, gridSize);

    Cell* gridDataTmp;
    Cell** gridTmp;
    allocateGrid(&gridDataTmp, &gridTmp, gridSize);


#ifdef WRITE_OUTPUT_IMAGE
    // Create dirs if they do not exist
    struct stat st = {0};
    if (stat("./output_images", &st) == -1) {
        mkdir("./output_images", 0700);
    }
    struct stat st = {0};
    char outDirFpath[50];
    snprintf(outDirFpath, sizeof(outDirFpath), "%s%s%s%s", "./output_images/", gridSize, "x", gridSize);
    if (stat(outDirFpath, &st) == -1) {
        mkdir(outDirFpath, 0700);
    }
#endif

    // Create time events
    cudaEvent_t startMain, stopMain;

    cudaEventCreate(&startMain);
    cudaEventCreate(&stopMain);

    float elapsedMain = 0;

    cudaEventRecord(startMain);

    for (int step = 0; step < MAX_SIM_STEPS; step++)
    {
        // 1. Calculate new grid values
        grayScottSimStep(grid, gridTmp, gridSize);
        // 2. Switch current grid to new grid
        swapGridPtr(&grid, &gridTmp);


#ifdef WRITE_OUTPUT_IMAGE
        char outputImageFpath[100];
        snprintf(outputImageFpath, sizeof(outputImageFpath), "%s%s%s%s", outDirFpath, "/", step, ".png");

        char gridVImage[gridSize * gridSize];
        for (int y = 0; y < gridSize; y++)
        {
            for (int x = 0; x < gridSize; x++)
            {
                // gridVImage[x + y * gridSize] = 255 - 255 * (grid[y][x].U - grid[y][x].V);
                gridVImage[x + y * gridSize] = (char) (255 * grid[y][x].V);
            }
        }

        stbi_write_png("./output_images/", gridSize, gridSize, COLOR_CHANNELS, gridVData, gridSize * COLOR_CHANNELS);
#endif
    }

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
    struct stat st = {0};
    if (stat("./timing_stats", &st) == -1) {
        mkdir("./timing_stats", 0700);
    }

    FILE *timingFile = fopen("./timing_stats/timing_stats_serial.txt", "a");
    fprintf(timingFile, "--------------- HISTOGRAM EQUALIZATION - Serial ---------------\n");
    fprintf(timingFile, "--------------- %s ---------------\n", gridSize);
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
    free(grid);
    free(gridData);

    return EXIT_SUCCESS;
}

