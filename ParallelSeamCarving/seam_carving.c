// SYSTEM LIBS //////////////////////////////////////////////////////////////////////////
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <limits.h>


// IMPORTED LIBS //////////////////////////////////////////////////////////////////////////
#define STB_IMAGE_IMPLEMENTATION
#include "lib/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "lib/stb_image_write.h"


// CONSTANTS //////////////////////////////////////////////////////////////////////////////
#define COLOR_CHANNELS 0 // 0 for dynamic color channels
#define REMOVED_SEAMS 128


unsigned char *getPixel(unsigned char *image, int x, int y, int channel, int width, int height) {
    if (x >= width || y >= height) {
        return NULL;
    }

    int pixelCount = width * height;
    int pixelPos = y * width + x + channel;
    if (pixelPos >= pixelCount) {
        return NULL;
    }

    return &image[pixelPos];
}

int main(int argc, char *args[]) {
    // Read arguments
    if (argc != 5)
    {
        printf("Error: Invalid amount of arguments. [%d]\n", argc);
        exit(EXIT_FAILURE);
    }

    char *imageInPath = args[1];
    char *imageOutPath = args[2];
    int outWidth = atoi(args[3]);
    int outHeight = atoi(args[4]);

    // Load image
    int imageWidth, imageHeight, imageChannelCount, imageDataSizeBytes;
    unsigned char *imageIn = stbi_load(imageInPath, &imageWidth, &imageHeight, &imageChannelCount, COLOR_CHANNELS);
    if (imageIn == NULL)
    {
        printf("Error: Couldn't load image\n");
        exit(EXIT_FAILURE);
    }
    printf("Loaded image %s of size %dx%d.\n", imageInPath, imageWidth, imageHeight);
    
    const int numPixels = imageWidth * imageHeight;

    double start = omp_get_wtime();
    // Process image
    
    /// Assign energy value for every pixel
    unsigned int *energyImage = (unsigned int *) malloc(sizeof(unsigned int) * numPixels);

    

    double stopEnergy = omp_get_wtime();
    printf(" -> time to assign energy value: %f s\n", stopEnergy - start);

    /// Find an 8-connected path of the pixels with the least energy


    /// Follow the cheapest path to remove one pixel from each row or column to resize the image


    // Output image
    const int numPixelsOutput = outWidth * outHeight;
    unsigned char *outputImage = (unsigned char *) malloc(sizeof(unsigned char *) * numPixelsOutput * imageChannelCount);


    // FREE ///////////////////////////////////////////////////////////////////////////////
    free(energyImage);
    free(outputImage);

    stbi_image_free(imageIn);
    free(imageOutPath);
    free(imageInPath);

    return EXIT_SUCCESS;
}