#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__host__ void Rgb2Lab(uchar3* image_gpu, float* image_gpu_double, int nPixels);
__host__ void Lab2Rgb(uchar3* image_gpu, float* image_gpu_double, int nPixels);

__global__ void rgb_to_lab(uchar3* image_gpu, float* image_gpu_double, int nPixels);
__global__ void lab_to_rgb( uchar3* image_gpu, float* image_gpu_double, int nPixels);