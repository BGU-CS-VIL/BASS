#ifndef OUT_OF_BOUNDS_LABEL
#define OUT_OF_BOUNDS_LABEL -1
#endif

#ifndef BAD_TOPOLOGY_LABEL 
#define BAD_TOPOLOGY_LABEL -2
#endif

#ifndef NUM_OF_CHANNELS 
#define NUM_OF_CHANNELS 3
#endif


#ifndef USE_COUNTS
#define USE_COUNTS 1
#endif

#define THREADS_PER_BLOCK 1024

#include "update_param.h"

#include <stdio.h>
#include <math.h>



__host__ void update_param(const float* image_gpu_double, const int* seg_gpu, superpixel_params* sp_params, superpixel_GPU_helper* sp_gpu_helper, const int nPixels, const int nSps, const int nSps_buffer, const int xdim, const int ydim, const int prior_sigma_s, const int prior_count){
  	dim3 ThreadPerBlock(THREADS_PER_BLOCK,1);

    int num_block1 = ceil( double(nPixels) / double(THREADS_PER_BLOCK) ); 
  	//int num_block2 = ceil( double(nSps) / double(THREADS_PER_BLOCK) );
	int num_block2 = ceil( double(nSps_buffer) / double(THREADS_PER_BLOCK) );
    dim3 BlockPerGrid1(num_block1,1);
    dim3 BlockPerGrid2(num_block2,1);
	
    clear_fields<<<BlockPerGrid2,ThreadPerBlock>>>(sp_params,sp_gpu_helper,nSps, nSps_buffer);
	cudaMemset(sp_gpu_helper, 0, nSps_buffer*sizeof(superpixel_GPU_helper));

    sum_by_label<<<BlockPerGrid1,ThreadPerBlock>>>(image_gpu_double, seg_gpu, sp_params,sp_gpu_helper,nPixels, xdim);
    //reduce0<<<totalBlocks, threadsPerBlock, threadsPerBlock*sizeof(float)>>>(image_gpu_double, seg_gpu, sp_params,sp_gpu_helper,nPixels, xdim);

	calculate_mu_and_sigma<<<BlockPerGrid2,ThreadPerBlock>>>(sp_params, sp_gpu_helper, nSps, nSps_buffer, prior_sigma_s, prior_count); 
}

__global__ void clear_fields(superpixel_params* sp_params, superpixel_GPU_helper* sp_gpu_helper, const int nsuperpixel, const int nsuperpixel_buffer){
	int k = threadIdx.x + blockIdx.x * blockDim.x;  // the label

	if (k>=nsuperpixel_buffer) return;
	if (sp_params[k].valid == 0) return;

	sp_params[k].count = 0;
	sp_params[k].log_count = 0.1;
	
	/*float3 mu_i_sum;
	mu_i_sum.x = 0;
	mu_i_sum.y = 0;
	mu_i_sum.z = 0;
	sp_gpu_helper[k].mu_i_sum =  mu_i_sum;
*/
	float3 mu_i;
	mu_i.x = 0;
	mu_i.y = 0;
	mu_i.z = 0;
	sp_params[k].mu_i = mu_i;

/*
	int2 mu_s_sum;
	mu_s_sum.x = 0;
	mu_s_sum.y = 0;
	sp_gpu_helper[k].mu_s_sum = mu_s_sum;
*/
	double2 mu_s;
	mu_s.x = 0;
	mu_s.y = 0;
	sp_params[k].mu_s = mu_s;

/*
	longlong3 sigma_s_sum;
	sigma_s_sum.x = 0;
	sigma_s_sum.y = 0;
	sigma_s_sum.z = 0;
	sp_gpu_helper[k].sigma_s_sum = sigma_s_sum;
*/
}



__global__ void sum_by_label(const float* image_gpu_double, const int* seg_gpu, superpixel_params* sp_params, superpixel_GPU_helper* sp_gpu_helper, const int nPixels, const int xdim) {
	// getting the index of the pixel
  int t = threadIdx.x + blockIdx.x * blockDim.x;
	if (t>=nPixels) return;

	//get the label
	int k = seg_gpu[t];

	atomicAdd(&sp_params[k].count, 1);

	atomicAdd(&sp_gpu_helper[k].mu_i_sum.x, image_gpu_double[3*t]);
	atomicAdd(&sp_gpu_helper[k].mu_i_sum.y, image_gpu_double[3*t+1]);
	atomicAdd(&sp_gpu_helper[k].mu_i_sum.z, image_gpu_double[3*t+2]);


	int x = t % xdim;
	int y = t / xdim; 
	int xx = x * x;
	int xy = x * y;
	int yy = y * y;

	atomicAdd(&sp_gpu_helper[k].mu_s_sum.x, x);
	atomicAdd(&sp_gpu_helper[k].mu_s_sum.y, y);
    atomicAdd((unsigned long long *)&sp_gpu_helper[k].sigma_s_sum.x, xx);
	atomicAdd((unsigned long long *)&sp_gpu_helper[k].sigma_s_sum.y, xy);
	atomicAdd((unsigned long long *)&sp_gpu_helper[k].sigma_s_sum.z, yy);

	
}



__global__ void calculate_mu_and_sigma(superpixel_params*  sp_params, superpixel_GPU_helper* sp_gpu_helper, const int nsuperpixel, const int nsuperpixel_buffer, const int prior_sigma_s, const int prior_count) {

	int k = threadIdx.x + blockIdx.x * blockDim.x; // the label

	if (k>=nsuperpixel_buffer) return;
	if (sp_params[k].valid == 0) return;
	//printf("Roy: %d",nsuperpixel_buffer);

	int count_int = sp_params[k].count;
	float a_prior = sp_params[k].prior_count;
	float prior_sigma_s_2 = a_prior * a_prior;
	double count = count_int * 1.0;
	double mu_x = 0.0;

	double mu_y = 0.0;

	//calculate the mean
	if (count_int>0){

		sp_params[k].log_count = log(count);
	    mu_x = sp_gpu_helper[k].mu_s_sum.x / count;   
	    mu_y = sp_gpu_helper[k].mu_s_sum.y / count;  
		sp_params[k].mu_s.x = mu_x; 
	    sp_params[k].mu_s.y = mu_y;

	    sp_params[k].mu_i.x = sp_gpu_helper[k].mu_i_sum.x / count;
		sp_params[k].mu_i.y = sp_gpu_helper[k].mu_i_sum.y / count;
  		sp_params[k].mu_i.z = sp_gpu_helper[k].mu_i_sum.z / count;

	   /* sp_params[k].sigma_s.x = sp_gpu_helper[k].mu_i_sum.x / count *0;
		sp_params[k].sigma_s.y = sp_gpu_helper[k].mu_i_sum.y / count *0;
  		sp_params[k].sigma_s.z = sp_gpu_helper[k].mu_i_sum.z / count * 0;	
*/
		//printf(" k is %d , %f,  %f, %f\n",k,sp_gpu_helper[k].mu_i_sum.x, sp_gpu_helper[k].mu_i_sum.y,sp_gpu_helper[k].mu_i_sum.z);
	}

	//calculate the covariance
	
	double C00 = sp_gpu_helper[k].sigma_s_sum.x ;
	double C01 =  sp_gpu_helper[k].sigma_s_sum.y ;
	double C11 = sp_gpu_helper[k].sigma_s_sum.z; 
	//double total_count = (double) sp_params[k].count + prior_count; 
	double total_count = (double) sp_params[k].count + a_prior*50;
	if (count_int > 3){	    
	    //update cumulative count and covariance
	    C00 = C00 - mu_x * mu_x * count;
	    C01 = C01 - mu_x * mu_y * count;
	    C11 = C11 - mu_y * mu_y * count;
	}

  C00 = (prior_sigma_s_2 + C00) / (total_count - 3.0);
  C01 = C01 / (total_count - 3);
  C11 = (prior_sigma_s_2 + C11) / (total_count - 3.0);

  double detC = C00 * C11 - C01 * C01;
  if (detC <= 0){
      C00 = C00 + 0.00001;
      C11 = C11 + 0.00001;
      detC = C00*C11-C01*C01;
      if(detC <=0) detC = 0.0001;//hack
  }

  //Take the inverse of sigma_space to get J_space
  sp_params[k].sigma_s.x = C11 / detC;     
  sp_params[k].sigma_s.y = - C01 / detC; 
  sp_params[k].sigma_s.z = C00 / detC; 
  sp_params[k].logdet_Sigma_s = log(detC);

}


