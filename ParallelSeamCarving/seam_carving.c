// LOCAL RUNNING
// gcc -lm --openmp -g3 -O0 seam_carving.c -o seam_carving.out; ./seam_carving.out ./test_images/720x480.png ./output_images/720x480.png 720 480

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

Image imageIn, imageOut, energyImage, seamIdImage;


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
    // Pixels outside the bounds and should be ignored during min function
    if (x >= image->width || y >= image->height || x < 0 || y < 0) {
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

unsigned int _calcEnergyPixel(Image* image, int x, int y)
{
    int energy = 0;
    for (int rgbChannel = 0; rgbChannel < image->channelCount; rgbChannel++)
    {
        int Gx = - getPixelE(image, x - 1, y - 1)[rgbChannel] -
                 2 * getPixelE(image, x, y - 1)[rgbChannel] -
                 getPixelE(image, x + 1, y - 1)[rgbChannel] +
                 getPixelE(image, x - 1, y + 1)[rgbChannel] +
                 2 * getPixelE(image, x, y + 1)[rgbChannel] + 
                 getPixelE(image, x + 1, y + 1)[rgbChannel];

        int Gy = getPixelE(image, x - 1, y - 1)[rgbChannel] +
                 2 * getPixelE(image, x - 1, y)[rgbChannel] +
                 getPixelE(image, x - 1, y + 1)[rgbChannel] -
                 getPixelE(image, x + 1, y - 1)[rgbChannel] -
                 2 * getPixelE(image, x + 1, y)[rgbChannel] -
                 getPixelE(image, x + 1, y + 1)[rgbChannel];

        energy += sqrt(pow(Gx, 2) + pow(Gy, 2));
    }

    return energy / image->channelCount;
}

void _calcEnergy(Image* image, Image* energyImage)
{
    for (int y = 0; y < image->height; y++)
    {
        for (int x = 0; x < image->width; x++)
        {
            int pixelPos = y * (image->width) + x;
            energyImage->imgI[pixelPos] = _calcEnergyPixel(image, x, y);
        }
    }
}

unsigned int seamIdentificationPixel(Image* image, int x, int y)
{   
    unsigned int currEnergy = getPixelEnergy(image, x, y);
    // Minimum between three neighbouring values
    unsigned int minEnergy = min(getPixelEnergy(image, x + 1, y - 1), 
                                min(getPixelEnergy(image, x + 1, y), 
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


int main(int argc, char *args[]) 
{
    // Read arguments
    if (argc != 5)
    {
        printf("Error: Invalid amount of arguments. [%d]\n", argc);
        exit(EXIT_FAILURE);
    }

    imageIn.fPath = args[1];
    imageOut.fPath = args[2];
    imageOut.width = atoi(args[3]);
    imageOut.height = atoi(args[4]);

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

    //printMatrix(&imageIn.image, imageIn.width * image.channelCount, image.height);

    const int numPixelsIn = imageIn.width * imageIn.height;

    // Process image //////////////////////////////////////////////////////////////////////////
    
    /// Energy Calculation - Assign energy value for every pixel
    double startEnergy = omp_get_wtime();

    energyImage.imgI = (unsigned int *) malloc(sizeof(unsigned int) * numPixelsIn);
    energyImage.width = imageIn.width;
    energyImage.height = imageIn.height;
    _calcEnergy(&imageIn, &energyImage);

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


    // Output image
    const int numPixelsOutput = imageOut.width * imageOut.height;
    imageOut.channelCount = imageIn.channelCount;
    unsigned char *outputImage = (unsigned char *) malloc(
        sizeof(unsigned char *) * numPixelsOutput * imageOut.channelCount);


    // FREE ///////////////////////////////////////////////////////////////////////////////
    free(energyImage.imgI);
    free(imageOut.imgC);
    stbi_image_free(imageIn.imgC);

    return EXIT_SUCCESS;
}