#include <iostream>
#include <cuda_runtime.h>
#include "utils.h"

__global__ void blur5x5_naive(unsigned char*, unsigned char*, int, int, int);
__global__ void merge_mask(unsigned char*, unsigned char*, unsigned char*, unsigned char*, int, int, int, unsigned char);

int main() {
    int w, h, c;
    int wm, hm, cm;

    // We will process frames 00000 → 00004 (test set)
    for (int frame_id = 0; frame_id < 90; frame_id++) {

        char frame_path[64];
        char mask_path[64];
        char out_path[64];

        snprintf(frame_path, sizeof(frame_path), "frames/%05d.jpg", frame_id);
        snprintf(mask_path, sizeof(mask_path),   "masks/%05d.png", frame_id);
        snprintf(out_path, sizeof(out_path),     "output_frames/%05d.png", frame_id);

        std::cout << "Processing frame " << frame_id << "...\n";

        // Load image and mask
        unsigned char* h_input = load_image(frame_path, w, h, c, false);
        unsigned char* h_mask  = load_image(mask_path,  wm, hm, cm, true);

        if (!h_input || !h_mask) {
            std::cout << "Skipping missing frame " << frame_id << "\n";
            continue;
        }

        if (w != wm || h != hm) {
            std::cout << "Error: size mismatch in frame " << frame_id << "\n";
            continue;
        }

        size_t img_size  = w * h * c;
        size_t mask_size = w * h;

        // Allocate host buffers
        unsigned char* h_blurred = (unsigned char*)malloc(img_size);
        unsigned char* h_output  = (unsigned char*)malloc(img_size);

        // Allocate device memory
        unsigned char *d_input, *d_blurred, *d_mask, *d_output;
        cudaMalloc(&d_input,   img_size);
        cudaMalloc(&d_blurred, img_size);
        cudaMalloc(&d_output,  img_size);
        cudaMalloc(&d_mask,    mask_size);

        // Copy data to GPU
        cudaMemcpy(d_input, h_input, img_size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_mask,  h_mask,  mask_size, cudaMemcpyHostToDevice);

        // Kernel launch config
        dim3 block(16, 16);
        dim3 grid((w + block.x - 1) / block.x,
                  (h + block.y - 1) / block.y);

        // Blur kernel
        blur5x5_naive<<<grid, block>>>(d_input, d_blurred, w, h, c);
        cudaDeviceSynchronize();

        // Merge kernel
        unsigned char threshold = 50;  // adjust if needed
        merge_mask<<<grid, block>>>(d_input, d_blurred, d_mask, d_output, w, h, c, threshold);
        cudaDeviceSynchronize();

        // Copy result back
        cudaMemcpy(h_output, d_output, img_size, cudaMemcpyDeviceToHost);

        // Save to file
        save_image(out_path, h_output, w, h, c);
        std::cout << "Saved " << out_path << "\n";

        // Cleanup for this frame
        cudaFree(d_input);
        cudaFree(d_blurred);
        cudaFree(d_output);
        cudaFree(d_mask);

        free(h_input);
        free(h_mask);
        free(h_blurred);
        free(h_output);
    }

        std::cout << "Stitching frames into video...\n";

    int ret = system("~/Project/ffmpeg -y -framerate 30 -i output_frames/%05d.png -pix_fmt yuv420p output_video.mp4");

    if (ret == 0) {
        std::cout << "Video created successfully: output_video.mp4\n";
    } else {
        std::cout << "FFmpeg failed to stitch video.\n";
    }


    return 0;
}
