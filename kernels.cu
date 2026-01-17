#include <cuda_runtime.h>

__global__ void blur5x5_naive(
    unsigned char* input,
    unsigned char* output,
    int width, int height, int channels)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int half = 15;  // 5x5 kernel radius
    int count = 0;

    float r = 0.0f, g = 0.0f, b = 0.0f;

    for (int dy = -half; dy <= half; dy++) {
        for (int dx = -half; dx <= half; dx++) {

            int nx = x + dx;
            int ny = y + dy;

            if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                int idx = (ny * width + nx) * channels;

                r += input[idx + 0];
                g += input[idx + 1];
                b += input[idx + 2];

                count++;
            }
        }
    }

    int outIdx = (y * width + x) * channels;

    output[outIdx + 0] = (unsigned char)(r / count);
    output[outIdx + 1] = (unsigned char)(g / count);
    output[outIdx + 2] = (unsigned char)(b / count);
}


__global__ void merge_mask(
    unsigned char* original,
    unsigned char* blurred,
    unsigned char* mask,
    unsigned char* output,
    int width, int height, int channels,
    unsigned char threshold)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int idx_rgb = (y * width + x) * channels;
    int idx_mask = y * width + x;

    if (mask[idx_mask] > threshold) {
        // Foreground: keep original sharp
        output[idx_rgb + 0] = original[idx_rgb + 0];
        output[idx_rgb + 1] = original[idx_rgb + 1];
        output[idx_rgb + 2] = original[idx_rgb + 2];
    } else {
        // Background: use blurred pixel
        output[idx_rgb + 0] = blurred[idx_rgb + 0];
        output[idx_rgb + 1] = blurred[idx_rgb + 1];
        output[idx_rgb + 2] = blurred[idx_rgb + 2];
    }
}

