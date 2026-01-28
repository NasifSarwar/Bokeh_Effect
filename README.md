# GPU-Based Depth-Aware Background Blur (CUDA)

This project implements a **DSLR-style portrait background blur** using CUDA.  
The pipeline processes a sequence of video frames, blurs the background using a
**31×31 Gaussian-like box blur**, and preserves the subject using a binary mask.

Two blur kernels are provided:

- **Naive Kernel** (`blur_naive_31`)  
  - Direct global-memory sampling  
  - Very slow (O(31²) per pixel)
- **Optimized Kernel** (`blur_shared_31`)  
  - Uses shared-memory tiling (16×16 blocks + halo)  
  - ~6–10× faster and GPU-friendly

The project can process a full batch of frames and also measures **per-frame GPU time**.

---

## 📂 Project Structure

├── main.cu                 # Main pipeline: load → blur → merge → save
├── kernels.cu              # Naive + shared-memory optimized blur kernels
├── utils.cpp               # Image loading/writing (stb_image)
├── utils.h
├── stb_image.h
├── stb_image_write.h
├── frames/                 # Input frames (added by user)
├── masks/                  # Binary masks (added by user)
├── output_frames/          # Output frames written here
└── final_project           # Compiled binary


---

## 🔧 Dependencies

No external libraries besides:


- stb_image.h (already included)
- stb_image_write.h (already included)
- FFmpeg *(For stitching video)*

Everything required is in this repository. (except FFmpeg)

---

## 🚀 How to Compile

From inside the `Project` folder:

```bash
module load cuda
nvcc main.cu utils.cpp kernels.cu -o final_project
```
This produces:
./final_project

▶️ How to Run

Default run:

./final_project



It will:

Load frames from frames/
Load masks from masks/
Blur background
Merge subject + blurred background
Save results into output_frames/
Print timing per frame

Example output:

Frame 0 GPU time: 5.12 ms
Frame 1 GPU time: 5.09 ms
...
Average GPU time per frame: 5.11 ms


# Switching Between Naive & Optimized Kernels

Inside main.cu there is a flag:
bool useOptimized = true;

Change to: 
bool useOptimized = false;


# GPU Timing

The project measures:
blur kernel time
merge kernel time
full GPU pipeline per frame
Timing uses CUDA events:

cudaEventRecord(start);
// blur + merge kernels
cudaEventRecord(stop);
cudaEventElapsedTime(&ms, start, stop);

# The output video will appear as:

blurred_output.mp4



