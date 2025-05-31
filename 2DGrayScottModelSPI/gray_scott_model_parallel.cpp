
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

// MPI
#include "mpi.h"

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

typedef struct _Cell_ {
    float U;  // Concentration of species U
    float V;  // Concentration of species V
} Cell;

struct execution_result
{
    int size;
    int numCores;
    float total;
};

void swapGridPtr(Cell** firstGridPtr, Cell** secondGridPtr)
{
    Cell* tmp = *firstGridPtr;
    *firstGridPtr = *secondGridPtr;
    *secondGridPtr = tmp;
}

void grayScottSimStep(Cell* grid, Cell* gridOut, int gridSize, int localHeight, const Cell* upRow, const Cell* downRow)
{
    for (int y = 0; y < localHeight; ++y)
    {
        for (int x = 0; x < gridSize; ++x)
        {
            int left = (x - 1 + gridSize) % gridSize;
            int right = (x + 1) % gridSize;

            // Row above
            const Cell& up = (y == 0) ? upRow[x] : grid[(y - 1) * gridSize + x];
            // Row below
            const Cell& down = (y == localHeight - 1) ? downRow[x] : grid[(y + 1) * gridSize + x];
            // Current row
            const Cell& center = grid[y * gridSize + x];
            const Cell& leftC = grid[y * gridSize + left];
            const Cell& rightC = grid[y * gridSize + right];

            float deltaSqrU = leftC.U + rightC.U + up.U + down.U - 4 * center.U;
            float deltaSqrV = leftC.V + rightC.V + up.V + down.V - 4 * center.V;

            float uVSqr = center.U * center.V * center.V;

            float newU = center.U + DELTA_t * (-uVSqr + F * (1.0f - center.U) + Du * deltaSqrU);
            float newV = center.V + DELTA_t * ( uVSqr - (F + k) * center.V + Dv * deltaSqrV);

            gridOut[y * gridSize + x].U = newU;
            gridOut[y * gridSize + x].V = newV;
        }
    }
}

#ifdef WRITE_OUTPUT_IMAGE
void write_output_image_frame(int step, int gridSize, Cell* gridData)
{
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
void write_output_gif_frame(int step, int gridSize, Cell* gridData, GifWriter* gifWriter)
{
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

    GifWriteFrame(gifWriter, frame, gridSize, gridSize, 10);
    delete[] frame;
}
#endif

void grayScottSolver(Cell* grid, Cell* gridTmp, int mpiRank, int mpiSize, int localHeight, int gridSize, char* outDirFpath)
{
#ifdef WRITE_OUTPUT_GIF
    GifWriter gifWriter;
    char outputGifFpath[100];
    snprintf(outputGifFpath, sizeof(outputGifFpath), "%s%d%s%d%s%d%s", "./output_gifs/", gridSize, "x", gridSize, "/", MAX_SIM_STEPS, ".gif");
    GifBegin(&gifWriter, outputGifFpath, gridSize, gridSize, 0);
#endif
    int localSize = localHeight * gridSize;
    Cell* localGrid = grid + mpiRank * localSize;
    Cell* localTmp = gridTmp + mpiRank * localSize;

    Cell* recvUp = new Cell[sizeof(Cell) * gridSize];
    Cell* recvDown = new Cell[sizeof(Cell) * gridSize];

    for (int step = 0; step < MAX_SIM_STEPS; step++)
    {
        // Padding rows; wrap-around
        int upRank = (mpiRank == 0) ? mpiSize - 1 : mpiRank - 1;
        int downRank = (mpiRank == mpiSize - 1) ? 0 : mpiRank + 1;

        MPI_Sendrecv(localGrid, gridSize * sizeof(Cell), MPI_BYTE, upRank, 0,
                     recvDown, gridSize * sizeof(Cell), MPI_BYTE, downRank, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        MPI_Sendrecv(localGrid + (localHeight - 1) * gridSize, gridSize * sizeof(Cell), MPI_BYTE, downRank, 1,
                     recvUp, gridSize * sizeof(Cell), MPI_BYTE, upRank, 1,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // Gray-Scott simulation step
        grayScottSimStep(localGrid, gridTmp, gridSize, localHeight, recvUp, recvDown);

        // Swap the grid pointers
        swapGridPtr(&localGrid, &gridTmp);

#if defined(WRITE_OUTPUT_IMAGE) || defined(WRITE_OUTPUT_GIF)
        // Gather full grid
        if ((step + 1) % (MAX_SIM_STEPS / NUM_FRAMES_CAPTURED) == 0)
        {
            if (mpiRank == 0)
            {
                memcpy(grid, localGrid, localSize * sizeof(Cell));
                for (int r = 1; r < mpiSize; ++r)
                {
                    MPI_Recv(grid + r * localSize, localSize * sizeof(Cell), MPI_BYTE, r, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                }

    #ifdef WRITE_OUTPUT_IMAGE
                write_output_image_frame((step + 1), gridSize, grid);
    #endif
    #ifdef WRITE_OUTPUT_GIF
                write_output_gif_frame((step + 1), gridSize, grid, &gifWriter);
    #endif
            }
            else
            {
                MPI_Send(localGrid, localSize * sizeof(Cell), MPI_BYTE, 0, 2, MPI_COMM_WORLD);
            }
        }
#endif
    }

    free(recvUp);
    free(recvDown);

#ifdef WRITE_OUTPUT_GIF
if (mpiRank == 0)
{
    GifEnd(&gifWriter);
}
#endif

    // If rank 0, write the final grid to output
    if (mpiRank == 0)
    {
        memcpy(grid, localGrid, localSize * sizeof(Cell));
        for (int r = 1; r < mpiSize; ++r)
        {
            MPI_Recv(grid + r * localSize, localSize * sizeof(Cell), MPI_BYTE, r, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }
    else
    {
        // Send the final grid to rank 0
        MPI_Send(localGrid, localSize * sizeof(Cell), MPI_BYTE, 0, 2, MPI_COMM_WORLD);
    }
}

void initGrid(Cell* grid, int gridSize)
{
    for (int y = 0; y < gridSize; y++)
    {
        for (int x = 0; x < gridSize; x++)
        {
            int idx = y * gridSize + x;
            if ((x > (int)(gridSize * 3.0/8) && x < (int)(gridSize * 5.0/8)) &&
                (y > (int)(gridSize * 3.0/8) && y < (int)(gridSize * 5.0/8)))
            {
                grid[idx].U = U_INSIDE;
                grid[idx].V = V_INSIDE;
            }
            else
            {
                grid[idx].U = U_OUTSIDE;
                grid[idx].V = V_OUTSIDE;
            }
        }
    }
}

void allocateGrid(Cell** gridPtr, int gridSize)
{
    *gridPtr = (Cell*) malloc(gridSize * gridSize * sizeof(Cell));
}

int main(int argc, char *args[])
{
    if (argc != 3)
    {
        printf("Error: Invalid amount of arguments. [%d]\n", argc);
        exit(EXIT_FAILURE);
    }

    // Initialize MPI
    MPI_Init(&argc, &args);
    int mpiRank, mpiSize;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpiSize);

    int gridSize = atoi(args[1]);
    if (gridSize % mpiSize != 0)
    {
        if (mpiRank == 0) fprintf(stderr, "Grid size must be divisible by number of processes.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    int localHeight = gridSize / mpiSize;

    // Reserve space for grids and initialize them
    Cell* grid;
    allocateGrid(&grid, gridSize);
    initGrid(grid, gridSize);

    Cell* gridTmp;
    allocateGrid(&gridTmp, gridSize);

    char outDirFpath[50];
#ifdef WRITE_OUTPUT_IMAGE
    {
        // Create dirs if they do not exist
        struct stat output_images_st = {0};
        if (stat("./output_images", &output_images_st) == -1) {
            mkdir("./output_images", 0700);
        }
        struct stat st = {0};
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

    double startTime = MPI_Wtime();

    // Main algorithm ///////////////////////////////////////////////////////////////////////////////////
    grayScottSolver(grid, gridTmp, mpiRank, mpiSize, localHeight, gridSize, outDirFpath);
    /////////////////////////////////////////////////////////////////////////////////////////////////////

    // End the time recording and calculate elapsed times
    double elapsedMain = MPI_Wtime() - startTime;

// Output timing stats to file //////////////////////////////////////////////////////////////////////////
#ifdef SAVE_TIMING_STATS
    struct execution_result result;
    result.size = gridSize;
    result.numCores = mpiSize;
    result.total = elapsedMain;

    // Create dir timing_stats if it does not exist
    struct stat timing_stats_st = {0};
    if (stat("./timing_stats", &timing_stats_st) == -1) {
        mkdir("./timing_stats", 0700);
    }

    FILE *timingFile = fopen("./timing_stats/timing_stats_parallel.txt", "a");
    fprintf(timingFile, "----------------- GRAY SCOTT - Parallel -----------------\n");
    fprintf(timingFile, "------------------------- %d%s%d ----------------------\n", gridSize, "x", gridSize);
    fprintf(timingFile, "Grid size: %d\n", result.size);
    fprintf(timingFile, "Number of cores: %d\n", result.numCores);
    fprintf(timingFile, "Total time: %f ms\n", result.total);
    fprintf(timingFile, "-----------------------------------------------------\n");
    fprintf(timingFile, "\n");
    fclose(timingFile);
#endif

    // Free memory
    free(grid);
    free(gridTmp);

    // Finalize MPI
    MPI_Finalize();

    return EXIT_SUCCESS;
}

