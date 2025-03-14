// LOCAL RUNNING
// gcc -lm --openmp -g3 -O0 seam_carving.c -o seam_carving.out; ./seam_carving.out ./test_images/720x480.png ./output_images/720x480.png 720

// SYSTEM LIBS //////////////////////////////////////////////////////////////////////////
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <limits.h>
#include <math.h>
#include <string.h>


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
#define REMOVED_SEAMS 128
#define SEAM -1


// Main image object
typedef struct _Image_
{
    union
    {
        unsigned char* imgC;
        unsigned int* imgI;  // Energy values have larger range than char data type
    };
    char* fPath;
    int width;
    int height;
    int channelCount;

} Image;

Image imageIn, imageOut, energyImage, seamIdImage, seamMaskImage;


/*
*  Returns the address of minimum element between three neighbouring elements
*  Parameter: middle element
*/
unsigned int* getMinAddr(unsigned int* midElement, int xMid, int width)
{
    unsigned int* min_addr = midElement;

    if (xMid - 1 >= 0 && *(midElement - 1) < *midElement)
    {
        min_addr = midElement - 1;
    }

    if (xMid + 1 < width && *(midElement + 1) < *midElement)
    {
        min_addr = midElement + 1;
    }
    
    return min_addr;
}

unsigned int* minElementRowAddr(unsigned int* row, int length)
{
    unsigned int* min_addr = row;

    for (int pos = 1; pos < length; pos++)
    {
        if (row[pos] < *min_addr)
            min_addr = &row[pos];
    }

    return min_addr;
}

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

    return &image->imgC[pixelPos];
}

unsigned int getPixelEnergy(Image* image, int x, int y)
{
    // Pixels outside the bounds and should be ignored
    if (x >= image->width || x < 0) {
        return INT_MAX;
    }

    const int pixelCount = image->width * image->height;
    const int pixelPos = y * image->width + x;

    if (pixelPos >= pixelCount || pixelPos < 0) {
        return -1; // Err
    }

    return image->imgI[pixelPos];
}

unsigned char* getPixelE(Image* image, int x, int y)  // Only used for energy calculation
{
    // if x and y outside bounds, use the closest pixel
    if (x < 0) x++;
    if (y < 0) y++;
    if (x == image->width) x--;
    if (y == image->height) y--;

    return getPixel(image, x, y);
}

void printMatrix(unsigned char* matrix, int width, int height)
{
    int row, column;
    for (row = 0; row < height; row++)
    {
        for(column = 0; column < width; column++)
        {
            int elementPos = row * width + column;
            printf("%d   ", matrix[elementPos]);
        }
        printf("\n");
    }
}

unsigned int calcEnergyPixel(Image* image, int x, int y)
{
    int energy = 0;
    for (int rgbChannel = 0; rgbChannel < image->channelCount; rgbChannel++)
    {
        int Gx = - getPixelE(image, x - 1, y - 1)[rgbChannel] -
                 2 * getPixelE(image, x - 1, y)[rgbChannel] -
                 getPixelE(image, x - 1, y + 1)[rgbChannel] +
                 getPixelE(image, x + 1, y - 1)[rgbChannel] +
                 2 * getPixelE(image, x + 1, y)[rgbChannel] + 
                 getPixelE(image, x + 1, y + 1)[rgbChannel];

        int Gy = getPixelE(image, x - 1, y - 1)[rgbChannel] +
                 2 * getPixelE(image, x, y - 1)[rgbChannel] +
                 getPixelE(image, x + 1, y - 1)[rgbChannel] -
                 getPixelE(image, x - 1, y + 1)[rgbChannel] -
                 2 * getPixelE(image, x, y + 1)[rgbChannel] -
                 getPixelE(image, x + 1, y + 1)[rgbChannel];

        energy += sqrt(pow(Gx, 2) + pow(Gy, 2));
    }

    return energy / image->channelCount;
}

void calcEnergy(Image* image, Image* energyImage)
{
    for (int y = 0; y < image->height; y++)
    {
        for (int x = 0; x < image->width; x++)
        {
            int pixelPos = y * (image->width) + x;
            energyImage->imgI[pixelPos] = calcEnergyPixel(image, x, y);
        }
    }
}

unsigned int seamIdentificationPixel(Image* image, int x, int y)
{   
    unsigned int currEnergy = getPixelEnergy(image, x, y);
    // Minimum between three neighbouring values
    unsigned int minEnergy = min(getPixelEnergy(image, x - 1, y + 1), 
                                min(getPixelEnergy(image, x, y + 1), 
                                    getPixelEnergy(image, x + 1, y + 1)));
    return currEnergy + minEnergy;
}

void seamIdentification(Image* image)
{
    // Starts at the bottom, the firts bottom row is skipped
    for (int y = image->height - 2; y >= 0; y--)
    {
        for (int x = 0; x < image->width; x++)
        {
            int pixelPos = y * (image->width) + x;
            image->imgI[pixelPos] = seamIdentificationPixel(image, x, y);
        }
    }
}

void seamRemoval(Image* image)
{
    unsigned int* minElementAddr = minElementRowAddr(image->imgI, image->width);
    *minElementAddr = SEAM;

    for (int y = 0; y < image->height - 1; y++)
    {
        unsigned int* midElementAddr = minElementAddr + image->width;  // Descend one row
        unsigned int midElementX = (midElementAddr - image->imgI) % image->width;
        minElementAddr = getMinAddr(midElementAddr, midElementX, image->width);
    }
}

void seamRemovalAll(Image* image, int widthOut)
{
    int numSeams = image->width - widthOut;

    for (int seam = 0; seam < numSeams; seam++)
    {
        seamRemoval(image);
    }
}

void copyToOutput(Image* imageIn, Image* imageOut, Image* seamMaskImage)
{
    unsigned char* outputPos = imageOut->imgC;
    for (int pixelPos = 0; pixelPos < imageIn->width * imageIn->height; pixelPos++)
    {
        if (seamMaskImage->imgI[pixelPos] > 0)
        {
            for (int channel = 0; channel < imageIn->channelCount; channel++)
            {
                int channelInPos = pixelPos * imageOut->channelCount + channel;
                *outputPos = imageIn->imgC[channelInPos];
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
    imageIn.imgC = stbi_load(imageIn.fPath,
                             &imageIn.width,
                             &imageIn.height, 
                             &imageIn.channelCount, 
                             STB_COLOR_CHANNELS);
    
    if (imageIn.imgC == NULL)
    {
        printf("Error: Couldn't load image\n");
        exit(EXIT_FAILURE);
    }
    printf("Loaded image %s of size %dx%d.\n", imageIn.fPath, imageIn.width, imageIn.height);

    imageOut.height = imageIn.height; // Height stays the same

    if (imageOut.width > imageIn.width) return EXIT_FAILURE;  // Output width should be smaller than input width

    //printMatrix(&imageIn.image, imageIn.width * image.channelCount, image.height);

    const int numPixelsIn = imageIn.width * imageIn.height;

    // Process image //////////////////////////////////////////////////////////////////////////

    /// Energy Calculation - Assign energy value for every pixel
    double startEnergy = omp_get_wtime();

    energyImage.imgI = (unsigned int *) malloc(sizeof(unsigned int) * numPixelsIn);
    energyImage.width = imageIn.width;
    energyImage.height = imageIn.height;
    calcEnergy(&imageIn, &energyImage);

    double stopEnergy = omp_get_wtime();
    printf(" -> time to assign energy value: %f s\n", stopEnergy - startEnergy);

    /// Vertical Seam Identification - Find an 8-connected path of the pixels with the least energy
    
    // Copying unnecessary outside testing environment, so it is not timed
    seamIdImage.imgI = (unsigned int *) malloc(sizeof(unsigned int) * numPixelsIn);
    memcpy(seamIdImage.imgI, energyImage.imgI, sizeof(unsigned int) * numPixelsIn);
    seamIdImage.width = energyImage.width;
    seamIdImage.height = energyImage.height;

    double startSeamId = omp_get_wtime();

    seamIdentification(&seamIdImage);

    double stopSeamId = omp_get_wtime();
    printf(" -> time to identify vertical seam: %f s\n", stopSeamId - startSeamId);

    /// Vertical Seam Removal - Follow the cheapest path to remove one pixel from each row or column to resize the image

    // Copying unnecessary outside testing environment, so it is not timed
    seamMaskImage.imgI = (unsigned int *) malloc(sizeof(unsigned int) * numPixelsIn);
    memcpy(seamMaskImage.imgI, seamIdImage.imgI, sizeof(unsigned int) * numPixelsIn);
    seamMaskImage.width = seamIdImage.width;
    seamMaskImage.height = seamIdImage.height;

    double startSeamRemove = omp_get_wtime();

    seamRemovalAll(&seamMaskImage, imageOut.width);

    double stopSeamRemove = omp_get_wtime();
    printf(" -> time to remove vertical seam: %f s\n", stopSeamRemove - startSeamRemove);

    // Output image
    double startOutput = omp_get_wtime();

    const int numPixelsOutput = imageOut.width * imageOut.height;
    imageOut.channelCount = imageIn.channelCount;
    imageOut.imgC = (unsigned char *) malloc(
        sizeof(unsigned char *) * numPixelsOutput * imageOut.channelCount);

    copyToOutput(&imageIn, &imageOut, &seamMaskImage);

    double stopOutput = omp_get_wtime();
    printf(" -> time to copy to output image: %f s\n", stopOutput - startOutput);

    // Write the output image to file
    stbi_write_png(imageOut.fPath, imageOut.width, imageOut.height, 
        imageOut.channelCount, imageOut.imgC, 
        imageOut.width * imageOut.channelCount);
    
    printf(" -> total time: %f s\n", stopEnergy - startEnergy +
         stopSeamId - startSeamId + 
         stopSeamRemove - startSeamRemove + 
         stopOutput - startOutput);


    // FREE ///////////////////////////////////////////////////////////////////////////////
    free(seamIdImage.imgI);
    free(seamMaskImage.imgI);

    free(energyImage.imgI);
    free(imageOut.imgC);
    stbi_image_free(imageIn.imgC);

    return EXIT_SUCCESS;
}