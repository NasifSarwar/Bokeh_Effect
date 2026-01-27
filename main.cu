#include <iostream>
#include <cuda_runtime.h>
#include "utils.h"
#include <filesystem>
namespace fs = std::filesystem;


__global__ void blur_naive_31(unsigned char*, unsigned char*, int, int, int);
__global__ void blur_shared_31(unsigned char*, unsigned char*, int, int, int);
__global__ void merge_mask(unsigned char*, unsigned char*, unsigned char*, unsigned char*, int, int, int, unsigned char);

int main() {
    bool useOptimized = true;   // switch naive/optimized

    float total_ms = 0.0f;
    int frame_count = 0;
    int frames = 0;
    for (const auto& entry : fs::directory_iterator("frames")) {
        if (entry.path().extension() == ".jpg")
            frames++;
    }

    for (int frame = 0; frame < frames; frame++) {
        char fpath[64], mpath[64], outpath[64];
        snprintf(fpath, 64, "frames/%05d.jpg", frame);
        snprintf(mpath, 64, "masks/%05d.png", frame);
        snprintf(outpath, 64, "output_frames/%05d.png", frame);

        int w, h, c;
        int wm, hm, cm;

        unsigned char* h_in  = load_image(fpath, w, h, c, false);
        unsigned char* h_mask = load_image(mpath, wm, hm, cm, true);

        if (!h_in || !h_mask) continue;
        if (w != wm || h != hm) continue;

        size_t img_size  = w * h * c;
        size_t mask_size = w * h;

        unsigned char* h_blur  = (unsigned char*)malloc(img_size);
        unsigned char* h_out   = (unsigned char*)malloc(img_size);

        unsigned char *d_in, *d_blur, *d_mask, *d_out;
        cudaMalloc(&d_in,   img_size);
        cudaMalloc(&d_blur, img_size);
        cudaMalloc(&d_mask, mask_size);
        cudaMalloc(&d_out,  img_size);

        cudaMemcpy(d_in,   h_in,   img_size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_mask, h_mask, mask_size, cudaMemcpyHostToDevice);

        dim3 block(16,16);
        dim3 grid((w + 15)/16, (h + 15)/16);


        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        // Start timing
        cudaEventRecord(start);

        if (useOptimized)
            blur_shared_31<<<grid, block>>>(d_in, d_blur, w, h, c);
        else
            blur_naive_31<<<grid, block>>>(d_in, d_blur, w, h, c);

        cudaDeviceSynchronize();

        merge_mask<<<grid, block>>>(d_in, d_blur, d_mask, d_out, w, h, c, 50);

        cudaDeviceSynchronize();

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float ms = 0;
        cudaEventElapsedTime(&ms, start, stop);
        printf("Frame %d GPU time: %.2f ms\n", frame+1, ms);
        total_ms += ms;
        frame_count++;

        cudaMemcpy(h_out, d_out, img_size, cudaMemcpyDeviceToHost);

        save_image(outpath, h_out, w, h, c);

        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        cudaFree(d_in);
        cudaFree(d_blur);
        cudaFree(d_mask);
        cudaFree(d_out);
        free(h_in);
        free(h_mask);
        free(h_blur);
        free(h_out);
    }

    float avg_ms = total_ms / frame_count;
    printf("Average GPU time per frame: %.2f ms\n", avg_ms);

    system("~/Project/ffmpeg -y -framerate 30 -i frames/%05d.jpg -pix_fmt yuv420p original_video.mp4");

    system("~/Project/ffmpeg -y -framerate 30 -i output_frames/%05d.png -pix_fmt yuv420p blurred_video.mp4");

    return 0;
}
