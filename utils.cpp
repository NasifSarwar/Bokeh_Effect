#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include "utils.h"
#include <iostream>

unsigned char* load_image(const std::string& path,
                          int& width,
                          int& height,
                          int& channels,
                          bool force_grayscale)
{
    int desired_channels = force_grayscale ? 1 : 3;

    unsigned char* img = stbi_load(path.c_str(),
                                   &width,
                                   &height,
                                   &channels,
                                   desired_channels);

    if (!img) {
        std::cerr << "Error loading image: " << path << std::endl;
    }

    channels = desired_channels;
    return img;
}


void save_image(const std::string& path, unsigned char* data, int width, int height, int channels) {
    stbi_write_png(path.c_str(), width, height, channels, data, width * channels);
}
