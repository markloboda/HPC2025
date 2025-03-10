#define STB_IMAGE_IMPLEMENTATION
#include <lib/stb_image.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <lib/stb_image_write.h>

// 0 for dynamic color channels
#define COLOR_CHANNELS 0

unsigned char *getPixel(unsigned char *image, int x, int y, int channel, int width, int height) {
    if (x >= width || y >= height) {
        return NULL;
    }

    int pixelCount = width * height;
    int pixelPos = y * width + x + channel;
    if (pixelPos >= pixelCount) {
        return NULL;
    }

    return image[pixelPos];
}

int main(int argc, char *args[]) {
    // Read arguments
    if (argc != 5)
    {
        printf("Error: Invalid amount of arguments. [%d]\n", argc);
        exit(1);
    }

    char *imageInPath = args[1];
    char *imageOutPath = args[2];
    int outWidth = args[3];
    int outHeight = args[4];

    // Load image
    int imageWidth, imageHeight, imageChannelCount, imageDataSizeBytes;
    unsigned char *imageIn = stbi_load(imageInPath, &imageWidth, &imageHeight, &imageChannelCount, COLOR_CHANNELS);
    if (imageIn == NULL)
    {
        printf("Error: Couldn't load image\n");
        exit(1);
    }

    // Process image
    
    /// Assign energy value for every pixel
    unsigned int *energyImage = (unsigned int *)malloc(sizeof(unsigned int) * imageWidth * imageHeight);


    /// Find an 8-connected path of the pixels with the least energy


    /// Follow the cheapest path to remove one pixel from each row or column to resize the image


    // Output image
    unsigned char *outputImage = (unsigned char *)malloc(sizeof(unsigned char *) * imageWidth * imageHeight * imageChannelCount);


    free(outputImage);

    free(energyImage);

    stbi_image_free(imageIn);
    free(imageOutPath);
    free(imageInPath);
}