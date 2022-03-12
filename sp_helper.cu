#include "sp_helper.h"
#define THREADS_PER_BLOCK 512
#include <stdio.h>

__host__ void CudaInitSpParams(superpixel_params* sp_params, const int s_std, const int i_std, const int nSPs, int nSPs_buffer, int nPixels){
  int num_block = ceil( double(nSPs) / double(THREADS_PER_BLOCK) ); //Roy- TO Change
  dim3 ThreadPerBlock(THREADS_PER_BLOCK,1);
  dim3 BlockPerGrid(num_block,1);
  InitSpParams<<<BlockPerGrid,ThreadPerBlock>>>(sp_params,s_std,i_std,nSPs,nSPs_buffer, nPixels);

}


__global__ void InitSpParams(superpixel_params* sp_params, const int s_std, const int i_std, const int nSPs, int nSPs_buffer, int nPixels)
{
  // the label
  int k = threadIdx.x + blockIdx.x * blockDim.x;  
  if (k>=nSPs_buffer) return;


  double s_std_square = double(s_std) * double(s_std); 

  // calculate the inverse of covariance
  double3 sigma_s_local;
  sigma_s_local.x = 1.0/s_std_square;
  sigma_s_local.y = 0.0;
  sigma_s_local.z = 1.0/s_std_square;
  sp_params[k].sigma_s = sigma_s_local;
  sp_params[k].prior_count = nPixels/nSPs;
  if(k>=nSPs) 
  {
    sp_params[k].count = 0;
    float3 mu_i;
    mu_i.x = -999;
    mu_i.y = -999;
    mu_i.z = -999;
    sp_params[k].mu_i = mu_i;
    double2 mu_s;
    mu_s.x = -999;
    mu_s.y = -999;
    sp_params[k].mu_s = mu_s;
    sp_params[k].valid = 0;
  }
  else sp_params[k].valid = 1;


  

  // calculate the log of the determinant of covariance
  sp_params[k].logdet_Sigma_s = log(s_std_square * s_std_square);  

}




__host__ void CUDA_get_image_overlaid(uchar3* image, const bool* border, const int nPixels, const int xdim){
  int num_block = ceil( double(nPixels) / double(THREADS_PER_BLOCK) ); 
  dim3 ThreadPerBlock(THREADS_PER_BLOCK,1);
  dim3 BlockPerGrid(num_block,1);
  GetImageOverlaid<<<BlockPerGrid,ThreadPerBlock>>>(image, border, nPixels, xdim);
}

__global__ void GetImageOverlaid(uchar3* image, const bool* border, const int nPixels, const int xdim){
  int t = threadIdx.x + blockIdx.x * blockDim.x;  
  if (t>=nPixels) return;

  if (border[t]){
    //change the color value to red
    uchar3 p;   
    p.x =  0;
    p.y =  0;
    p.z =  255.0;
    image[t] = p;   
  }
}



__host__ void CUDA_get_image_cartoon(uchar3* image_mean_gpu, const int* seg, const superpixel_params* sp_params, const int nPixels){
  int num_block = ceil( double(nPixels) / double(THREADS_PER_BLOCK) ); 
  dim3 ThreadPerBlock(THREADS_PER_BLOCK,1);
  dim3 BlockPerGrid(num_block,1);

  GetImageCartoon<<<BlockPerGrid,ThreadPerBlock>>>(image_mean_gpu, seg, sp_params, nPixels); 
}


__global__ void GetImageCartoon(uchar3* image_mean_gpu, const int* seg, const superpixel_params* sp_params, const int nPixels){
  //each each pixel, find the mu_i_lab;
  int t = threadIdx.x + blockIdx.x * blockDim.x;  
  if (t>=nPixels) return;
  int k = seg[t];
  // convert mu_i_lab to mu_i_rgb

  double L = double(sp_params[k].mu_i.x * (-100));
  double La = double(sp_params[k].mu_i.y * 100);
  double Lb = double(sp_params[k].mu_i.z * 100);

  if (L!=L || La!=La || Lb!=Lb) return;

  //convert from LAB to XYZ
  double fy = (L+16) / 116;
  double fx = La/500 + fy;
  double fz = fy-Lb/200;

  double x,y,z;
  double xcube = powf(fx,3); 
  double ycube = powf(fy,3); 
  double zcube = powf(fz,3); 
  if (ycube>0.008856) y = ycube;
  else        y = (fy-16.0/116.0)/7.787;
  if (xcube>0.008856) x = xcube;
  else        x = (fx - 16.0/116.0)/7.787;
  if (zcube>0.008856) z = zcube;
  else        z = (fz - 16.0/116.0)/7.787;

  double X = 0.950456 * x;
  double Y = 1.000 * y;
  double Z = 1.088754 * z;

  //convert from XYZ to rgb
  double R = X *  3.2406 + Y * -1.5372 + Z * -0.4986;
  double G = X * -0.9689 + Y *  1.8758 + Z *  0.0415;
  double B = X *  0.0557 + Y * -0.2040 + Z *  1.0570;

  double r,g,b;
  if (R>0.0031308) r = 1.055 * (powf(R,(1.0/2.4))) - 0.055;
  else             r = 12.92 * R;
  if (G>0.0031308) g = 1.055 * ( powf(G,(1.0/2.4))) - 0.055;
  else             g= 12.92 * G;
  if (B>0.0031308) b = 1.055 * (powf(B, (1.0/2.4))) - 0.055;
  else             b = 12.92 * B;

  uchar3 p;
  
  p.x =  min(255.0, b * 255.0);
  p.y =  min(255.0, g * 255.0);
  p.z =  min(255.0, r * 255.0);

  p.x =  max(0.0, double(p.x));
  p.y =  max(0.0, double(p.y));
  p.z =  max(0.0, double(p.z));

  image_mean_gpu[t] = p;
  return;
}