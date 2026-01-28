#include <cuda_runtime.h>
#define TILE_W (16 + 2*RADIUS)
#define TILE_H (16 + 2*RADIUS)
#define RADIUS 15
#define KSIZE (2*RADIUS + 1)
#define SEP_TILE_W 128
#define SEP_TILE_H 16

__global__ void blur_naive_31(
    unsigned char* input,
    unsigned char* output,
    int w, int h, int c)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= w || y >= h) return;

    float r = 0, g = 0, b = 0;
    int count = 0;

    for (int dy = -RADIUS; dy <= RADIUS; dy++) {
        for (int dx = -RADIUS; dx <= RADIUS; dx++) {

            int nx = x + dx;
            int ny = y + dy;

            if (nx >= 0 && nx < w && ny >= 0 && ny < h) {
                int idx = (ny * w + nx) * c;
                r += input[idx];
                g += input[idx + 1];
                b += input[idx + 2];
                count++;
            }
        }
    }

    int out = (y * w + x) * c;
    output[out]     = (unsigned char)(r / count);
    output[out + 1] = (unsigned char)(g / count);
    output[out + 2] = (unsigned char)(b / count);
}


__global__ void blur_shared_31(
    unsigned char* input,
    unsigned char* output,
    int w, int h, int c)
{


    __shared__ unsigned char tile[TILE_W * TILE_H * 3];

    int bx = blockIdx.x * 16;
    int by = blockIdx.y * 16;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    for (int yy = ty; yy < TILE_H; yy += blockDim.y) {
        for (int xx = tx; xx < TILE_W; xx += blockDim.x) {

            int gx = bx + xx - RADIUS;
            int gy = by + yy - RADIUS;

            if (gx < 0) gx = 0;
            if (gy < 0) gy = 0;
            if (gx >= w) gx = w - 1;
            if (gy >= h) gy = h - 1;

            int in_idx = (gy * w + gx) * 3;
            int t_idx  = (yy * TILE_W + xx) * 3;

            tile[t_idx]     = input[in_idx];
            tile[t_idx + 1] = input[in_idx + 1];
            tile[t_idx + 2] = input[in_idx + 2];
        }
    }

    __syncthreads();

    int x = bx + tx;
    int y = by + ty;

    if (x >= w || y >= h) return;

    float r = 0, g = 0, b = 0;

    for (int dy = -RADIUS; dy <= RADIUS; dy++) {
        for (int dx = -RADIUS; dx <= RADIUS; dx++) {

            int nx = tx + dx + RADIUS;
            int ny = ty + dy + RADIUS;

            int t_idx = (ny * TILE_W + nx) * 3;

            r += tile[t_idx];
            g += tile[t_idx + 1];
            b += tile[t_idx + 2];
        }
    }

    int out_idx = (y * w + x) * 3;
    output[out_idx]     = (unsigned char)(r / (KSIZE * KSIZE));  // R
    output[out_idx + 1] = (unsigned char)(g / (KSIZE * KSIZE));  // G
    output[out_idx + 2] = (unsigned char)(b / (KSIZE * KSIZE));  // B

}


__global__ void merge_mask(
    unsigned char* orig,
    unsigned char* blur,
    unsigned char* mask,
    unsigned char* out,
    int w, int h, int c,
    unsigned char threshold)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= w || y >= h) return;

    int id = (y * w + x) * c;
    int mid = (y * w + x);

    if (mask[mid] > threshold) {
        out[id]     = orig[id];
        out[id + 1] = orig[id + 1];
        out[id + 2] = orig[id + 2];
    } else {
        out[id]     = blur[id];
        out[id + 1] = blur[id + 1];
        out[id + 2] = blur[id + 2];
    }
}


// ============ Separable Blur with Shared Memory ============

// Horizontal pass
__global__ void blur_separable_h(
    unsigned char* input,
    float* temp,
    int w, int h, int c)
{
    __shared__ unsigned char tile[(SEP_TILE_W + 2 * RADIUS) * 3];

    int x = blockIdx.x * SEP_TILE_W + threadIdx.x;
    int y = blockIdx.y;

    if (y >= h) return;

    int tile_start = blockIdx.x * SEP_TILE_W - RADIUS;

    // Load tile with halo
    for (int i = threadIdx.x; i < SEP_TILE_W + 2 * RADIUS; i += blockDim.x) {
        int gx = tile_start + i;

        if (gx < 0) gx = 0;
        if (gx >= w) gx = w - 1;

        int in_idx = (y * w + gx) * 3;
        int t_idx = i * 3;

        tile[t_idx]     = input[in_idx];
        tile[t_idx + 1] = input[in_idx + 1];
        tile[t_idx + 2] = input[in_idx + 2];
    }

    __syncthreads();

    if (x >= w) return;

    float r = 0, g = 0, b = 0;

    for (int dx = -RADIUS; dx <= RADIUS; dx++) {
        int t_idx = (threadIdx.x + RADIUS + dx) * 3;
        r += tile[t_idx];
        g += tile[t_idx + 1];
        b += tile[t_idx + 2];
    }

    int out_idx = (y * w + x) * 3;
    temp[out_idx]     = r / KSIZE;
    temp[out_idx + 1] = g / KSIZE;
    temp[out_idx + 2] = b / KSIZE;
}

// Vertical pass
__global__ void blur_separable_v(
    float* temp,
    unsigned char* output,
    int w, int h, int c)
{
    __shared__ float tile[(SEP_TILE_H + 2 * RADIUS) * 3];

    int x = blockIdx.x;
    int y = blockIdx.y * SEP_TILE_H + threadIdx.x;

    if (x >= w) return;

    int tile_start = blockIdx.y * SEP_TILE_H - RADIUS;

    // Load tile with halo
    for (int i = threadIdx.x; i < SEP_TILE_H + 2 * RADIUS; i += blockDim.x) {
        int gy = tile_start + i;

        if (gy < 0) gy = 0;
        if (gy >= h) gy = h - 1;

        int in_idx = (gy * w + x) * 3;
        int t_idx = i * 3;

        tile[t_idx]     = temp[in_idx];
        tile[t_idx + 1] = temp[in_idx + 1];
        tile[t_idx + 2] = temp[in_idx + 2];
    }

    __syncthreads();

    if (y >= h) return;

    float r = 0, g = 0, b = 0;

    for (int dy = -RADIUS; dy <= RADIUS; dy++) {
        int t_idx = (threadIdx.x + RADIUS + dy) * 3;
        r += tile[t_idx];
        g += tile[t_idx + 1];
        b += tile[t_idx + 2];
    }

    int out_idx = (y * w + x) * 3;
    output[out_idx]     = (unsigned char)(r / KSIZE);
    output[out_idx + 1] = (unsigned char)(g / KSIZE);
    output[out_idx + 2] = (unsigned char)(b / KSIZE);
}
