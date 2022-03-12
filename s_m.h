#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#ifndef MY_SP_STRUCT
#define MY_SP_STRUCT
#include "my_sp_struct.h"
#endif

__host__ void CudaCalcMergeCandidate(const float* image_gpu_double, int* split_merge_pairs, int* seg, bool* border,  superpixel_params* sp_params, superpixel_GPU_helper* sp_gpu_helper, superpixel_GPU_helper_sm* sp_gpu_helper_sm, const int nPixels, const int xdim, const int ydim, const int nSPs_buffer, const int change, float i_std, float alpha);
__global__  void calc_merge_candidate(int* seg, bool* border, int* split_merge_pairs, const int nPixels, const int xdim, const int ydim, const int change);  
__global__ void sum_by_label_sm(const float* image_gpu_double, const int* seg_gpu, superpixel_params* sp_params, superpixel_GPU_helper_sm* sp_gpu_helper_sm, const int nPixels, const int xdim);
__global__ void calc_bn(int* seg, int* split_merge_pairs, superpixel_params* sp_params, superpixel_GPU_helper* sp_gpu_helper, superpixel_GPU_helper_sm* sp_gpu_helper_sm, const int nPixels, const int xdim, const int nsuperpixel_buffer, float b_0);
__global__ void calc_marginal_liklelyhoood_of_sp(const float* image_gpu_double, int* split_merge_pairs, superpixel_params* sp_params, superpixel_GPU_helper* sp_gpu_helper, superpixel_GPU_helper_sm* sp_gpu_helper_sm, const int nPixels, const int xdim, const int nsuperpixel_buffer,  float a_0, float b_0);
__global__ void calc_hasting_ratio(const float* image_gpu_double,int* split_merge_pairs, superpixel_params* sp_params, superpixel_GPU_helper* sp_gpu_helper, superpixel_GPU_helper_sm* sp_gpu_helper_sm, const int nPixels, const int xdim, const int nsuperpixel_buffer, float a0, float b0, float alpha_hasting_ratio, int* mutex );
__global__ void calc_hasting_ratio2(const float* image_gpu_double,int* split_merge_pairs, superpixel_params* sp_params, superpixel_GPU_helper* sp_gpu_helper, superpixel_GPU_helper_sm* sp_gpu_helper_sm, const int nPixels, const int xdim, const int nsuperpixel_buffer, float a0, float b0, float alpha_hasting_ratio, int* mutex );
__global__  void merge_sp(int* seg, bool* border, int* split_merge_pairs, superpixel_params* sp_params, superpixel_GPU_helper_sm* sp_gpu_helper_sm, const int nPixels, const int xdim, const int ydim);  
__global__ void init_sm(const float* image_gpu_double, const int* seg_gpu, superpixel_params* sp_params, superpixel_GPU_helper_sm* sp_gpu_helper_sm, const int nsuperpixel_buffer, const int xdim, int* split_merge_pairs);
__global__ void remove_sp( int* split_merge_pairs, superpixel_params* sp_params, superpixel_GPU_helper_sm* sp_gpu_helper_sm, const int nsuperpixel_buffer);
__global__  void calc_split_candidate(int* seg, bool* border,int distance, int* mutex, const int nPixels, const int xdim, const int ydim);
__global__ void init_split(const bool* border, int* seg_gpu, superpixel_params* sp_params, superpixel_GPU_helper_sm* sp_gpu_helper_sm, const int nsuperpixel_buffer, const int xdim,  const int ydim, const int offset, const int* seg, int* max_sp, int max_SP);
__host__ int CudaCalcSplitCandidate(const float* image_gpu_double, int* split_merge_pairs, int* seg, bool* border,  superpixel_params* sp_params, superpixel_GPU_helper* sp_gpu_helper, superpixel_GPU_helper_sm* sp_gpu_helper_sm, const int nPixels, const int xdim, const int ydim, const int nSPs_buffer, int* seg_split1, int* seg_split2, int* split3, int max_SP, int count, float i_std, float alpha);
__global__ void calc_seg_split(int* seg_split1, int* seg_split2,int* seg, int* seg_split3, const int nPixels, int max_SP);
__global__ void calc_bn_split(int* seg, int* split_merge_pairs, superpixel_params* sp_params, superpixel_GPU_helper* sp_gpu_helper, superpixel_GPU_helper_sm* sp_gpu_helper_sm, const int nPixels, const int xdim, const int nsuperpixel_buffer, float b_0, int max_SP);
__global__ void calc_marginal_liklelyhoood_of_sp_split(const float* image_gpu_double, int* split_merge_pairs, superpixel_params* sp_params, superpixel_GPU_helper* sp_gpu_helper, superpixel_GPU_helper_sm* sp_gpu_helper_sm, const int nPixels, const int xdim, const int nsuperpixel_buffer,  float a_0, float b_0, int max_SP);
__global__ void calc_hasting_ratio_split(const float* image_gpu_double,int* split_merge_pairs, superpixel_params* sp_params, superpixel_GPU_helper* sp_gpu_helper, superpixel_GPU_helper_sm* sp_gpu_helper_sm, const int nPixels, const int xdim, const int nsuperpixel_buffer, float a0, float b0, float alpha_hasting_ratio, int* mutex, int max_SP,int* max_sp );
__global__ void sum_by_label_split(const float* image_gpu_double, const int* seg, superpixel_params* sp_params, superpixel_GPU_helper_sm* sp_gpu_helper_sm, const int nPixels, const int xdim, int max_SP);
__global__  void split_sp(int* seg, int* seg_split1, int* split_merge_pairs, superpixel_params* sp_params, superpixel_GPU_helper_sm* sp_gpu_helper_sm, const int nPixels, const int xdim, const int ydim,int max_SP);  
