#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#ifndef MY_SP_STRUCT
#define MY_SP_STRUCT
#include "my_sp_struct.h"
#endif


__global__ void InitSpParams(superpixel_params*  sp_params, const int s_std, const int i_std, const int nSPs, int nSPs_buffer, int nPixels);
__host__ void CudaInitSpParams(superpixel_params*  sp_params, const int s_std, const int i_std, const int nSPs, int nSPs_buffer, int nPixels);

__host__ void CUDA_get_image_overlaid(uchar3* image, const bool* border, const int nPixels, int xdim);
__global__ void GetImageOverlaid(uchar3* image, const bool* border, const int nPixels, int xdim);

__host__ void CUDA_get_image_cartoon(uchar3* image_mean_gpu, const int* seg, const superpixel_params*  sp_params, const int nPixels);
__global__ void GetImageCartoon(uchar3* image_mean_gpu, const int* seg, const superpixel_params*  sp_params, const int nPixels);