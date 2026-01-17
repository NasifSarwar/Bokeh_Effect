#ifndef UTILS_H
#define UTILS_H

#include <string>

unsigned char* load_image(const std::string& path, int& width, int& height, int& channels, bool force_grayscale=false);
void save_image(const std::string& path, unsigned char* data, int width, int height, int channels);

#endif
