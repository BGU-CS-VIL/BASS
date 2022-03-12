
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <float.h>
#include <math.h>


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


#ifndef OUT_OF_BOUNDS_LABEL
#define OUT_OF_BOUNDS_LABEL -1
#endif

#define THREADS_PER_BLOCK 512


#include "s_m.h"
#include "sp.h"

#include <stdio.h>
#ifndef WIN32
#include <unistd.h>
#endif

int tresh = -2;
__device__ volatile int sem = 0;

__device__ void acquire_semaphore(volatile int *lock){
  while (atomicCAS((int *)lock, 0, 1) != 0);
  }

__device__ void release_semaphore(volatile int *lock){
  *lock = 0;
  __threadfence();
  }


__device__ __forceinline__ float atomicMaxFloat (float * addr, float value) {
    float old;
    old = (value >= 0) ? __int_as_float(atomicMax((int *)addr, __float_as_int(value))) :
         __uint_as_float(atomicMin((unsigned int *)addr, __float_as_uint(value)));

    return old;
}

__device__ int mLock=0;

__host__ void CudaCalcMergeCandidate(const float* image_gpu_double, int* split_merge_pairs, int* seg, bool* border,  superpixel_params* sp_params, superpixel_GPU_helper* sp_gpu_helper, superpixel_GPU_helper_sm* sp_gpu_helper_sm, const int nPixels, const int xdim, const int ydim, const int nSPs_buffer, const int change, float i_std, float alpha){
    int num_block = ceil( double(nPixels) / double(THREADS_PER_BLOCK) ); 
    int num_block2 = ceil( double(nSPs_buffer) / double(THREADS_PER_BLOCK) );
    dim3 BlockPerGrid2(num_block2,1);
    dim3 ThreadPerBlock(THREADS_PER_BLOCK,1);
    dim3 BlockPerGrid(num_block,1);
    float a0 = 10000;
    float b0 = i_std * (a0) ;
    //b0 = 0.05*0.05*a0;
    int* mutex ;
    float alpha_hasting_ratio = alpha;
    cudaMalloc((void **)&mutex, sizeof(int));
    cudaMemset(mutex, 0, sizeof(int));

    init_sm<<<BlockPerGrid2,ThreadPerBlock>>>(image_gpu_double,seg,sp_params,sp_gpu_helper_sm, nSPs_buffer, xdim,split_merge_pairs);
    calc_merge_candidate<<<BlockPerGrid,ThreadPerBlock>>>(seg,border,split_merge_pairs,nPixels, xdim, ydim, change); 
    sum_by_label_sm<<<BlockPerGrid,ThreadPerBlock>>>(image_gpu_double,seg,sp_params,sp_gpu_helper_sm, nPixels, xdim);
    calc_bn<<<BlockPerGrid2,ThreadPerBlock>>>(seg, split_merge_pairs, sp_params, sp_gpu_helper, sp_gpu_helper_sm, nPixels, xdim, nSPs_buffer, b0);
    calc_marginal_liklelyhoood_of_sp<<<BlockPerGrid2,ThreadPerBlock>>>(image_gpu_double,  split_merge_pairs,  sp_params,  sp_gpu_helper, sp_gpu_helper_sm,  nPixels, xdim, nSPs_buffer , a0, b0);
    calc_hasting_ratio<<<BlockPerGrid2,ThreadPerBlock>>>(image_gpu_double,  split_merge_pairs, sp_params, sp_gpu_helper, sp_gpu_helper_sm, nPixels,  xdim, nSPs_buffer, a0,  b0, alpha_hasting_ratio, mutex);
    calc_hasting_ratio2<<<BlockPerGrid2,ThreadPerBlock>>>(image_gpu_double,  split_merge_pairs, sp_params, sp_gpu_helper, sp_gpu_helper_sm, nPixels,  xdim, nSPs_buffer, a0,  b0, alpha_hasting_ratio, mutex);
    remove_sp<<<BlockPerGrid2,ThreadPerBlock>>>(split_merge_pairs,sp_params,sp_gpu_helper_sm,nSPs_buffer);
    merge_sp<<<BlockPerGrid,ThreadPerBlock>>>(seg,border, split_merge_pairs, sp_params, sp_gpu_helper_sm, nPixels, xdim, ydim);  

}





__host__ int CudaCalcSplitCandidate(const float* image_gpu_double, int* split_merge_pairs, int* seg, bool* border,  superpixel_params* sp_params, superpixel_GPU_helper* sp_gpu_helper, superpixel_GPU_helper_sm* sp_gpu_helper_sm, const int nPixels, const int xdim, const int ydim, const int nSPs_buffer, int* seg_split1 ,int* seg_split2, int* seg_split3, int max_SP, int count, float i_std, float alpha){
    int num_block = ceil( double(nPixels) / double(THREADS_PER_BLOCK) ); 
    int num_block2 = ceil( double(nSPs_buffer) / double(THREADS_PER_BLOCK) );
    dim3 BlockPerGrid2(num_block2,1);
    dim3 ThreadPerBlock(THREADS_PER_BLOCK,1);
    dim3 BlockPerGrid(num_block,1);
    float a0 = 10000;
    float b0 = i_std * (a0) ;
    float alpha_hasting_ratio =  alpha;
    int* mutex_2;
    int done = 1;
    int* max_sp;
    cudaMalloc((void **)&max_sp, sizeof(int));
    cudaMalloc((void **)&mutex_2, sizeof(int));  // malloc of single value is also important

    int distance = 1;
    int offset = count%2+1;
    cudaMemset(seg_split1, 0, nPixels*sizeof(int));
    cudaMemset(seg_split2, 0, nPixels*sizeof(int));
    init_sm<<<BlockPerGrid2,ThreadPerBlock>>>(image_gpu_double,seg,sp_params,sp_gpu_helper_sm, nSPs_buffer, xdim,split_merge_pairs);

    init_split<<<BlockPerGrid2,ThreadPerBlock>>>(border,seg_split1,sp_params,sp_gpu_helper_sm, nSPs_buffer, xdim, ydim, offset , seg, max_sp, max_SP);
    init_split<<<BlockPerGrid2,ThreadPerBlock>>>(border,seg_split2,sp_params,sp_gpu_helper_sm, nSPs_buffer, xdim,ydim, -offset, seg,max_sp, max_SP);
    split_sp<<<BlockPerGrid,ThreadPerBlock>>>(seg,seg_split1, split_merge_pairs, sp_params, sp_gpu_helper_sm, nPixels, xdim, ydim, max_SP);

    
    while(done)
    {

    
		cudaMemset(mutex_2, 0, sizeof(int));
        cudaMemcpy(&done, mutex_2, sizeof(int), cudaMemcpyDeviceToHost);

        calc_split_candidate<<<BlockPerGrid,ThreadPerBlock>>>(seg_split1,border,distance, mutex_2, nPixels, xdim, ydim); 
        distance++;
        cudaMemcpy(&done, mutex_2, sizeof(int), cudaMemcpyDeviceToHost);


    }
    done =1;
    distance = 1;
    while(done)
    {
		cudaMemset(mutex_2, 0, sizeof(int));
        cudaMemcpy(&done, mutex_2, sizeof(int), cudaMemcpyDeviceToHost);
        calc_split_candidate<<<BlockPerGrid,ThreadPerBlock>>>(seg_split2 ,border,distance, mutex_2, nPixels, xdim, ydim); 
        distance++;
        cudaMemcpy(&done, mutex_2, sizeof(int), cudaMemcpyDeviceToHost);
    }

    calc_seg_split<<<BlockPerGrid,ThreadPerBlock>>>(seg_split1,seg_split2, seg, seg_split3, nPixels, max_SP);
    sum_by_label_split<<<BlockPerGrid,ThreadPerBlock>>>(image_gpu_double,seg_split1,sp_params,sp_gpu_helper_sm, nPixels, xdim,max_SP);
    calc_bn_split<<<BlockPerGrid2,ThreadPerBlock>>>(seg_split3, split_merge_pairs, sp_params, sp_gpu_helper, sp_gpu_helper_sm, nPixels, xdim, nSPs_buffer, b0, max_SP);

    calc_marginal_liklelyhoood_of_sp_split<<<BlockPerGrid2,ThreadPerBlock>>>(image_gpu_double,  split_merge_pairs,  sp_params,  sp_gpu_helper, sp_gpu_helper_sm,  nPixels, xdim, nSPs_buffer , a0, b0, max_SP);

    calc_hasting_ratio_split<<<BlockPerGrid2,ThreadPerBlock>>>(image_gpu_double,  split_merge_pairs, sp_params, sp_gpu_helper, sp_gpu_helper_sm, nPixels,  xdim, nSPs_buffer, a0,  b0, alpha_hasting_ratio, 0,max_SP, max_sp);

    split_sp<<<BlockPerGrid,ThreadPerBlock>>>(seg,seg_split1, split_merge_pairs, sp_params, sp_gpu_helper_sm, nPixels, xdim, ydim, max_SP);

    cudaMemcpy(&max_SP, max_sp, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(max_sp);
    cudaFree(mutex_2);

    return max_SP;
}



__global__ void init_sm(const float* image_gpu_double, const int* seg_gpu, superpixel_params* sp_params, superpixel_GPU_helper_sm* sp_gpu_helper_sm, const int nsuperpixel_buffer, const int xdim,int* split_merge_pairs) {
	int k = threadIdx.x + blockIdx.x * blockDim.x;  // the label
	if (k>=nsuperpixel_buffer) return;
	//if (sp_params[k].valid == 0) return;
    sp_gpu_helper_sm[k].b_n.x = 0;
    sp_gpu_helper_sm[k].b_n.y = 0;
    sp_gpu_helper_sm[k].b_n.z = 0;

	sp_gpu_helper_sm[k].squares_i.x = 0;
	sp_gpu_helper_sm[k].squares_i.y = 0;
	sp_gpu_helper_sm[k].squares_i.z = 0;
    sp_gpu_helper_sm[k].mu_i_sum.x = 0;
	sp_gpu_helper_sm[k].mu_i_sum.y = 0;
	sp_gpu_helper_sm[k].mu_i_sum.z = 0;
    sp_gpu_helper_sm[k].count_f = 0;
    sp_gpu_helper_sm[k].count = 0;
    sp_gpu_helper_sm[k].hasting = -999999;
    //sp_params[k].count = 0;

    sp_gpu_helper_sm[k].merge = false;
    sp_gpu_helper_sm[k].remove = false;
    split_merge_pairs[k*2+1] = 0;
    split_merge_pairs[k*2] = 0;

   

}

__global__  void calc_merge_candidate(int* seg, bool* border, int* split_merge_pairs, const int nPixels, const int xdim, const int ydim, const int change){   
    int idx = threadIdx.x + blockIdx.x * blockDim.x;  
    if (idx>=nPixels) return; 
    if(!border[idx]) return;
    int x = idx % xdim;
    int y = idx / xdim;

    int C = seg[idx]; // center 
    int W; // north, south, east,west            
    W = OUT_OF_BOUNDS_LABEL; // init 

        if(change==1)
            {
                if ((y>1) && (y< ydim-2))
                {
                    W = __ldg(&seg[idx+ydim]);  // left
                }
            }

            else
            {
                if ((x>1) && (x< xdim-2))
                {
                    W = __ldg(&seg[idx-1]);  // left
                }
            }
        
        // If the nbr is different from the central pixel and is not out-of-bounds,
        // then it is a border pixel.
        if (W>0 && C!=W)
        {
            atomicMax(&split_merge_pairs[C*2+1],W);
    
        }

   

    return;        
}

__global__  void calc_split_candidate(int* seg, bool* border,int distance, int* mutex, const int nPixels, const int xdim, const int ydim){   
    int idx = threadIdx.x + blockIdx.x * blockDim.x;  
    
    if (idx>=nPixels) return; 
    if(border[idx]) return; 
    int x = idx % xdim;
    int y = idx / xdim;

    int C = seg[idx]; // center 

    if(C!=distance) return;

    if ((y>0)&&(idx-xdim>=0)){
        if(!seg[idx-xdim])
        {
            seg[idx-xdim] = distance +1 ;
            mutex[0] = 1;
        }
    }          
    if ((x>0)&&(idx-1>=0)){

        if(!seg[idx-1])
        {
            seg[idx-1] = distance +1 ;
            mutex[0] = 1;
        }
    }
    if ((y<ydim-1)&&(idx+xdim<nPixels)){

        if(!seg[idx+xdim])
        {
            seg[idx+xdim] = distance +1 ;
            mutex[0] = 1;
        }
    }   
    if ((x<xdim-1)&&(idx+1<nPixels)){
        
        if(!seg[idx+1])
        {
            seg[idx+1] = distance +1 ;
            mutex[0] = 1;
        }

    }       

    
    return;        
}


__global__ void init_split(const bool* border, int* seg_gpu, superpixel_params* sp_params, superpixel_GPU_helper_sm* sp_gpu_helper_sm, const int nsuperpixel_buffer, const int xdim, const int ydim, const int offset, const int* seg, int* max_sp, int max_SP) {
	int k = threadIdx.x + blockIdx.x * blockDim.x;  // the label
    *max_sp = max_SP+1;
	if (k>=nsuperpixel_buffer) return;
	if (sp_params[k].valid == 0) return;
    int x;
    int y;
    if((offset==1)||(offset==-1))
    {
        x = int(sp_params[k].mu_s.x)+offset;
        y = int(sp_params[k].mu_s.y);
    }
    else
    {
        x = int(sp_params[k].mu_s.x);
        y = int(sp_params[k].mu_s.y)+offset;

    }
    
    int ind = y*xdim+x;
    if((ind<0)||(ind>xdim*ydim-1)) return;
    
    if(border[ind]) return;
    if (seg[ind]!=k) return;
    seg_gpu[ind] = 1;

}


__global__ void calc_seg_split(int* seg_split1, int* seg_split2,int* seg, int* seg_split3, const int nPixels, int max_SP) {
    int t = threadIdx.x + blockIdx.x * blockDim.x;
	if (t>=nPixels) return;
    int seg_val = __ldg(&seg[t]);

    if(seg_split1[t]>__ldg(&seg_split2[t])) seg_val += max_SP; 
    seg_split1[t] = seg_val;

    return;
    
}


__global__ void sum_by_label_sm(const float* image_gpu_double, const int* seg_gpu, superpixel_params* sp_params, superpixel_GPU_helper_sm* sp_gpu_helper_sm, const int nPixels, const int xdim) {
	// getting the index of the pixel
    int t = threadIdx.x + blockIdx.x * blockDim.x;
	if (t>=nPixels) return;

	//get the label
	int k = __ldg(&seg_gpu[t]);
    float l = __ldg(& image_gpu_double[3*t]);
    float a = __ldg(& image_gpu_double[3*t+1]);
    float b = __ldg(& image_gpu_double[3*t+2]);
	//atomicAdd(&sp_params[k].count, 1); //TODO: Time it
	atomicAdd(&sp_gpu_helper_sm[k].squares_i.x, l*l);
	atomicAdd(&sp_gpu_helper_sm[k].squares_i.y, a*a);
	atomicAdd(&sp_gpu_helper_sm[k].squares_i.z,b*b);
}

__global__ void sum_by_label_split(const float* image_gpu_double, const int* seg, superpixel_params* sp_params, superpixel_GPU_helper_sm* sp_gpu_helper_sm, const int nPixels, const int xdim, int max_SP) {
	// getting the index of the pixel
    int t = threadIdx.x + blockIdx.x * blockDim.x;
	if (t>=nPixels) return;

	//get the label
    
	int k = __ldg(&seg[t]);
    float l = __ldg(& image_gpu_double[3*t]);
    float a = __ldg(& image_gpu_double[3*t+1]);
    float b = __ldg(& image_gpu_double[3*t+2]);
	atomicAdd(&sp_gpu_helper_sm[k].count, 1); //TODO: Time it

	atomicAdd(&sp_gpu_helper_sm[k].squares_i.x, l*l);
	atomicAdd(&sp_gpu_helper_sm[k].squares_i.y, a*a);
	atomicAdd(&sp_gpu_helper_sm[k].squares_i.z,b*b);
    atomicAdd(&sp_gpu_helper_sm[k].mu_i_sum.x, l);
	atomicAdd(&sp_gpu_helper_sm[k].mu_i_sum.y, a);
	atomicAdd(&sp_gpu_helper_sm[k].mu_i_sum.z, b);
    return;
}

__global__ void calc_bn(int* seg, int* split_merge_pairs, superpixel_params* sp_params, superpixel_GPU_helper* sp_gpu_helper, superpixel_GPU_helper_sm* sp_gpu_helper_sm, const int nPixels, const int xdim, const int nsuperpixel_buffer, float b_0) {
	// getting the index of the pixel
	int k = threadIdx.x + blockIdx.x * blockDim.x;  // the label
	if (k>=nsuperpixel_buffer) return;
	if (sp_params[k].valid == 0) return;

    // TODO: check if there is no neigh
    //get the label of neigh
    int f = split_merge_pairs[2*k+1];
	//if (sp_params[f].valid == 0) return;
    //if (f<=0) return;

    float count_f = __ldg(&sp_params[f].count);
    float count_k = __ldg(&sp_params[k].count);

    float squares_f_x = __ldg(&sp_gpu_helper_sm[f].squares_i.x);
    float squares_f_y = __ldg(&sp_gpu_helper_sm[f].squares_i.y);
    float squares_f_z = __ldg(&sp_gpu_helper_sm[f].squares_i.z);
   
    float squares_k_x = __ldg(&sp_gpu_helper_sm[k].squares_i.x);
    float squares_k_y = __ldg(&sp_gpu_helper_sm[k].squares_i.y);
    float squares_k_z = __ldg(&sp_gpu_helper_sm[k].squares_i.z);
   
    float mu_f_x = __ldg(&sp_gpu_helper[f].mu_i_sum.x);
    float mu_f_y = __ldg(&sp_gpu_helper[f].mu_i_sum.y);
    float mu_f_z = __ldg(&sp_gpu_helper[f].mu_i_sum.z);
   
    float mu_k_x = __ldg(&sp_gpu_helper[k].mu_i_sum.x);
    float mu_k_y = __ldg(&sp_gpu_helper[k].mu_i_sum.y);
    float mu_k_z = __ldg(&sp_gpu_helper[k].mu_i_sum.z);
    //if ((k==105)||(k==42)) printf("Merger:  %d, %d ,sq_x: %f , sq_y: %f , sq_z: %f\n", k, f,squares_k_x, squares_k_y, squares_k_z) ;   


    int count_fk = count_f + count_k;
    sp_gpu_helper_sm[k].count_f = count_fk;
    //sp_gpu_helper_sm[k].count_f = sp_params[k].count + sp_params[f].count;
    sp_gpu_helper_sm[k].b_n.x = b_0 + 0.5 * ((squares_k_x) -
                                ( mu_k_x*mu_k_x/
                                count_k));
    
    sp_gpu_helper_sm[k].b_n_f.x = b_0 + 0.5 *( (squares_k_x+squares_f_x) -
                                ( (mu_f_x + mu_k_x ) * (mu_f_x + mu_k_x ) /
                                (count_fk)));



    sp_gpu_helper_sm[k].b_n.y = b_0 + 0.5 * ((squares_k_y) -
                                ( mu_k_y*mu_k_y/
                                count_k));
    
    sp_gpu_helper_sm[k].b_n_f.y = b_0 + 0.5 *( (squares_k_y+squares_f_y) -
                                ( (mu_f_y + mu_k_y ) * (mu_f_y + mu_k_y ) /
                                (count_fk)));


    sp_gpu_helper_sm[k].b_n.z = b_0 + 0.5 * ((squares_k_z) -
                                ( mu_k_z*mu_k_z/
                                count_k));
    
    sp_gpu_helper_sm[k].b_n_f.z = b_0 + 0.5 *( (squares_k_z+squares_f_z) -
                                ( (mu_f_z + mu_k_z ) * (mu_f_z + mu_k_z ) /
                                (count_fk)));



    if(  sp_gpu_helper_sm[k].b_n.x<0)   sp_gpu_helper_sm[k].b_n.x = 0.1;
    if(  sp_gpu_helper_sm[k].b_n.y<0)   sp_gpu_helper_sm[k].b_n.y = 0.1;
    if(  sp_gpu_helper_sm[k].b_n.z<0)   sp_gpu_helper_sm[k].b_n.z = 0.1;

    if(  sp_gpu_helper_sm[k].b_n_f.x<0)   sp_gpu_helper_sm[k].b_n_f.x = 0.1;
    if(  sp_gpu_helper_sm[k].b_n_f.y<0)   sp_gpu_helper_sm[k].b_n_f.y = 0.1;
    if(  sp_gpu_helper_sm[k].b_n_f.z<0)   sp_gpu_helper_sm[k].b_n_f.z = 0.1;

     
}



__global__ void calc_bn_split(int* seg, int* split_merge_pairs, superpixel_params* sp_params, superpixel_GPU_helper* sp_gpu_helper, superpixel_GPU_helper_sm* sp_gpu_helper_sm, const int nPixels, const int xdim, const int nsuperpixel_buffer, float b_0, int max_SP) {
	// getting the index of the pixel
	int k = threadIdx.x + blockIdx.x * blockDim.x;  // the label
	if (k>=nsuperpixel_buffer) return;
	if (sp_params[k].valid == 0) return;
    // TODO: check if there is no neigh
    //get the label of neigh
    int s = k + max_SP;
	if (s>=nsuperpixel_buffer) return;
    float count_f = __ldg(&sp_params[k].count);
    float count_k= __ldg(&sp_gpu_helper_sm[k].count);
    float count_s = __ldg(&sp_gpu_helper_sm[s].count);
    if((count_f<1)||( count_k<1)||(count_s<1)) return;


    float squares_s_x = __ldg(&sp_gpu_helper_sm[s].squares_i.x);
    float squares_s_y = __ldg(&sp_gpu_helper_sm[s].squares_i.y);
    float squares_s_z = __ldg(&sp_gpu_helper_sm[s].squares_i.z);
   
    float squares_k_x = __ldg(&sp_gpu_helper_sm[k].squares_i.x);
    float squares_k_y = __ldg(&sp_gpu_helper_sm[k].squares_i.y);
    float squares_k_z = __ldg(&sp_gpu_helper_sm[k].squares_i.z);
   
    float mu_s_x = __ldg(&sp_gpu_helper_sm[s].mu_i_sum.x);
    float mu_s_y = __ldg(&sp_gpu_helper_sm[s].mu_i_sum.y);
    float mu_s_z = __ldg(&sp_gpu_helper_sm[s].mu_i_sum.z);

    float mu_k_x = __ldg(&sp_gpu_helper_sm[k].mu_i_sum.x);
    float mu_k_y = __ldg(&sp_gpu_helper_sm[k].mu_i_sum.y);
    float mu_k_z = __ldg(&sp_gpu_helper_sm[k].mu_i_sum.z);

    float mu_f_x =__ldg(&sp_gpu_helper[k].mu_i_sum.x);
    float mu_f_y = __ldg(&sp_gpu_helper[k].mu_i_sum.y);
    float mu_f_z = __ldg(&sp_gpu_helper[k].mu_i_sum.z);



    
    sp_gpu_helper_sm[k].b_n.x = b_0 + 0.5 * ((squares_k_x) -
                                ( (mu_k_x*mu_k_x)/ (count_k)));

   //sp_gpu_helper_sm[k].b_n.x = b_0 + (squares_k_x)+(mu_k_x*mu_k_x)/(count_k*count_k)-2*(mu_k_x*mu_k_x)/(count_k)+(mu_k_x*mu_k_x)/(count_k*count_k);
                            
    sp_gpu_helper_sm[k].b_n.y = b_0 + 0.5 * ((squares_k_y) -
                                ( mu_k_y*mu_k_y/ count_k));
 
    //sp_gpu_helper_sm[k].b_n.y = b_0 + (squares_k_y)+(mu_k_y*mu_k_y)/(count_k*count_k)-2*(mu_k_y*mu_k_y)/(count_k)+(mu_k_y*mu_k_y)/(count_k*count_k);

    sp_gpu_helper_sm[k].b_n.z = b_0 + 0.5 * ((squares_k_z) -
                                ( mu_k_z*mu_k_z/ count_k));
 
   // sp_gpu_helper_sm[k].b_n.z = b_0 + (squares_k_z)+(mu_k_z*mu_k_z)/(count_k*count_k)-2*(mu_k_z*mu_k_z)/(count_k)+(mu_k_z*mu_k_z)/(count_k*count_k);


    sp_gpu_helper_sm[s].b_n.x = b_0 + 0.5 * ((squares_s_x) -
                                ( mu_s_x*mu_s_x/ count_s));


    sp_gpu_helper_sm[s].b_n.y = b_0 + 0.5 * ((squares_s_y) -
                                ( mu_s_y*mu_s_y/ count_s));
 
    sp_gpu_helper_sm[s].b_n.z = b_0 + 0.5 * ((squares_s_z) -
                                ( mu_s_z*mu_s_z/ count_s));

/*
     sp_gpu_helper_sm[s].b_n.x = b_0 + (squares_s_x)+(mu_s_x*mu_s_x)/(count_s*count_s)-2*(mu_s_x*mu_s_x)/(count_s)+(mu_s_x*mu_s_x)/(count_k*count_k);
    sp_gpu_helper_sm[s].b_n.y = b_0 + (squares_s_y)+(mu_s_y*mu_s_y)/(count_s*count_s)-2*(mu_s_y*mu_s_y)/(count_s)+(mu_s_y*mu_s_y)/(count_k*count_k);
                   
    sp_gpu_helper_sm[s].b_n.z = b_0 + (squares_s_z)+(mu_s_z*mu_s_z)/(count_s*count_s)-2*(mu_s_z*mu_s_z)/(count_s)+(mu_s_z*mu_s_z)/(count_k*count_k);                             
*/

    sp_gpu_helper_sm[k].b_n_f.x = b_0 + 0.5 * ((squares_k_x+squares_s_x) -
                                ( mu_f_x*mu_f_x/ count_f));
 
    sp_gpu_helper_sm[k].b_n_f.y = b_0 + 0.5 * ((squares_k_y+squares_s_y) -
                                ( mu_f_y*mu_f_y/ count_f));
 
    sp_gpu_helper_sm[k].b_n_f.z = b_0 + 0.5 * ((squares_k_z+squares_s_z) -
                                ( mu_f_z*mu_f_z/ count_f));
                       
}




__global__ void calc_marginal_liklelyhoood_of_sp_split(const float* image_gpu_double, int* split_merge_pairs, superpixel_params* sp_params, superpixel_GPU_helper* sp_gpu_helper, superpixel_GPU_helper_sm* sp_gpu_helper_sm, const int nPixels, const int xdim, const int nsuperpixel_buffer,  float a_0, float b_0, int max_SP) {
	// getting the index of the pixel
	int k = threadIdx.x + blockIdx.x * blockDim.x;  // the label
    if (k>=nsuperpixel_buffer) return;
	if (sp_params[k].valid == 0) return;


    int s = k + max_SP;
    if (s>=nsuperpixel_buffer) return;
    float count_f = __ldg(&sp_params[k].count);
    float count_k= __ldg(&sp_gpu_helper_sm[k].count);
    float count_s = __ldg(&sp_gpu_helper_sm[s].count);

    if((count_f<1)||( count_k<1)||(count_s<1)) return;
    if (count_f!=count_k+count_s) return;
    // TODO: check if there is no neigh
    // TODO: check if num is the same
	//get the label
    //a_0 = 1100*(count_f);

    float a_n_k = a_0 + float(count_k)/2;
    float a_n_s = a_0+ float(count_s)/2;
    float a_n_f = a_0+ float(count_f)/2;
    


    float v_n_k = 1/float(count_k);
    float v_n_s = 1/float(count_s);
    float v_n_f = 1/float(count_f);
   /* v_n_k = 1;
    v_n_f =1;
    v_n_s=1;*/

    float b_n_k_x = __ldg(&sp_gpu_helper_sm[k].b_n.x);
    float b_n_k_y = __ldg(&sp_gpu_helper_sm[k].b_n.y);
    float b_n_k_z = __ldg(&sp_gpu_helper_sm[k].b_n.z);

    float b_n_s_x = __ldg(&sp_gpu_helper_sm[s].b_n.x);
    float b_n_s_y = __ldg(&sp_gpu_helper_sm[s].b_n.y);
    float b_n_s_z = __ldg(&sp_gpu_helper_sm[s].b_n.z);

    float b_n_f_x = __ldg(&sp_gpu_helper_sm[k].b_n_f.x);
    float b_n_f_y = __ldg(&sp_gpu_helper_sm[k].b_n_f.y);
    float b_n_f_z = __ldg(&sp_gpu_helper_sm[k].b_n_f.z);

    a_0 =a_n_k;

    
    sp_gpu_helper_sm[k].numerator.x = a_0 * __logf(b_0) + lgammaf(a_n_k)+ 0.5*__logf(v_n_k);
    sp_gpu_helper_sm[k].denominator.x = a_n_k * __logf (b_n_k_x) + 0.5 * count_k * __logf (M_PI) + \
                                        count_k * __logf (2) + lgammaf(a_0);

    sp_gpu_helper_sm[k].denominator.y = a_n_k * __logf (b_n_k_y) + 0.5 * count_k * __logf (M_PI) + \
                                        count_k * __logf (2) + lgammaf(a_0);

    sp_gpu_helper_sm[k].denominator.z = a_n_k * __logf (b_n_k_z) + 0.5 * count_k * __logf (M_PI) + \
                                        count_k * __logf (2) + lgammaf(a_0);

    


    a_0 =a_n_s;

    sp_gpu_helper_sm[s].numerator.x = a_0 * __logf(b_0) + lgammaf(a_n_s)+0.5*__logf(v_n_s);
    sp_gpu_helper_sm[s].denominator.x = a_n_s * __logf (b_n_s_x) + 0.5 * count_s * __logf (M_PI) + \
                                        count_s * __logf (2) + lgammaf(a_0);

    sp_gpu_helper_sm[s].denominator.y = a_n_s * __logf (b_n_s_y) + 0.5 * count_s * __logf (M_PI) + \
                                        count_s * __logf (2) + lgammaf(a_0);

    sp_gpu_helper_sm[s].denominator.z = a_n_s * __logf (b_n_s_z) + 0.5 * count_s * __logf (M_PI) + \
                                        count_s * __logf (2) + lgammaf(a_0);      

    a_0 =a_n_f;

    sp_gpu_helper_sm[k].numerator_f.x = a_0 * __logf(b_0) + lgammaf(a_n_f)+0.5*__logf(v_n_f);
    sp_gpu_helper_sm[k].denominator_f.x = a_n_f * __logf (b_n_f_x) + 0.5 * count_f * __logf (M_PI) + \
                                        count_f * __logf (2) + lgammaf(a_0);

    sp_gpu_helper_sm[k].denominator_f.y = a_n_f * __logf (b_n_f_y) + 0.5 * count_f * __logf (M_PI) + \
                                        count_f * __logf (2) + lgammaf(a_0);

    sp_gpu_helper_sm[k].denominator_f.z = a_n_f * __logf (b_n_f_z) + 0.5 * count_f * __logf (M_PI) + \
                                        count_f * __logf (2) + lgammaf(a_0);        


}   


__global__ void calc_marginal_liklelyhoood_of_sp(const float* image_gpu_double, int* split_merge_pairs, superpixel_params* sp_params, superpixel_GPU_helper* sp_gpu_helper, superpixel_GPU_helper_sm* sp_gpu_helper_sm, const int nPixels, const int xdim, const int nsuperpixel_buffer,  float a_0, float b_0) {
	// getting the index of the pixel
	int k = threadIdx.x + blockIdx.x * blockDim.x;  // the label
    if (k>=nsuperpixel_buffer) return;

	if (sp_params[k].valid == 0) return;

    // TODO: check if there is no neigh
    // TODO: check if num is the same
	//get the label
    
    float count_k = __ldg(&sp_params[k].count);
    float count_f = __ldg(&sp_gpu_helper_sm[k].count_f);
    //if ((count_k==0)||(count_f==0)) return;

	//if (sp_params[k].valid == 0) return;
    //if (f==-1) return;

    float a_n = a_0 + float(count_k) / 2;
    float a_n_f = a_0+ float(count_f) / 2;
    // float v_n = 1 / float(num_pixels_in_sp);
    float v_n = 1/float(count_k);
    float v_n_f = 1/float(count_f);

    //printf("Merge: %f,%d, \n", sp_gpu_helper_sm[k].b_n.x, count_k);

    a_0 = a_n;
    sp_gpu_helper_sm[k].numerator.x = a_0 * __logf(b_0) + lgammaf(a_n)+0.5*__logf(v_n);


    sp_gpu_helper_sm[k].denominator.x = a_n* __logf ( __ldg(&sp_gpu_helper_sm[k].b_n.x)) + 0.5 * count_k * __logf (M_PI) + \
                                        count_k * __logf (2) + lgammaf(a_0);


    //sp_gpu_helper_sm[k].numerator.y = a_0 * __logf (b_0) + lgammaf(a_0)+0.5*v_n;
    sp_gpu_helper_sm[k].denominator.y = a_n* __logf ( __ldg(&sp_gpu_helper_sm[k].b_n.y)) + 0.5 * count_k * __logf (M_PI) + \
                                        count_k * __logf (2) + lgamma(a_0);

    //sp_gpu_helper_sm[k].numerator.z = a_0 * __logf(b_0) + lgammaf(a_0)+0.5*v_n;
    sp_gpu_helper_sm[k].denominator.z = a_n* __logf(__ldg(&sp_gpu_helper_sm[k].b_n.z)) + 0.5 * count_k * __logf (M_PI) + \
                                        count_k * __logf (2) + lgammaf(a_0);                                        


    a_0 = a_n_f;
    sp_gpu_helper_sm[k].numerator_f.x = a_0 * __logf (b_0) + lgammaf(a_n_f)+0.5*__logf(v_n_f);
    sp_gpu_helper_sm[k].denominator_f.x = a_n_f* __logf (__ldg(&sp_gpu_helper_sm[k].b_n_f.x)) + 0.5 * count_f * __logf (M_PI) + \
                                        count_f * __logf (2) + lgammaf(a_0);



    //sp_gpu_helper_sm[k].numerator_f.y = a_0 * __logf (b_0) + lgammaf(a_0)+0.5*v_n_f;
    sp_gpu_helper_sm[k].denominator_f.y = a_n_f* __logf (__ldg(&sp_gpu_helper_sm[k].b_n_f.y)) + 0.5 * count_f * __logf (M_PI) + \
                                        count_f * __logf (2) + lgammaf(a_0);

    //sp_gpu_helper_sm[k].numerator_f.z = a_0 * __logf (b_0) + lgammaf(a_0)+0.5*v_n_f;
    sp_gpu_helper_sm[k].denominator_f.z = a_n_f* __logf (__ldg(&sp_gpu_helper_sm[k].b_n_f.z)) + 0.5 * count_f* __logf (M_PI) + \
                                        count_f * __logf (2) + lgammaf(a_0);         


}   




__global__ void calc_hasting_ratio(const float* image_gpu_double,int* split_merge_pairs, superpixel_params* sp_params, superpixel_GPU_helper* sp_gpu_helper, superpixel_GPU_helper_sm* sp_gpu_helper_sm, const int nPixels, const int xdim, const int nsuperpixel_buffer, float a0, float b0, float alpha_hasting_ratio, int* mutex ) {
	// getting the index of the pixel
	int k = threadIdx.x + blockIdx.x * blockDim.x;  // the label

	if (k>=nsuperpixel_buffer) return;
	if (sp_params[k].valid == 0) return;

    int f = split_merge_pairs[2*k+1];
	if (sp_params[f].valid == 0) return;
    if(f<=0) return;

    float count_k = __ldg(&sp_params[k].count);
    float count_f = __ldg(&sp_gpu_helper_sm[k].count_f);
    
    if ((count_k<1)||(count_f<1)) return;

    sp_gpu_helper_sm[k].merge = false;
    float num_k = __ldg(&sp_gpu_helper_sm[k].numerator.x);

    float total_marginal_1 = (num_k - __ldg(&sp_gpu_helper_sm[k].denominator.x)) +  
                         (num_k - __ldg(&sp_gpu_helper_sm[k].denominator.y)) + 
                         (num_k - __ldg(&sp_gpu_helper_sm[k].denominator.z)); 

    float num_f = __ldg(&sp_gpu_helper_sm[f].numerator.x);

    float total_marginal_2 = (num_f - __ldg(&sp_gpu_helper_sm[f].denominator.x)) +   
                         (num_f - __ldg(&sp_gpu_helper_sm[f].denominator.y)) + 
                         (num_f - __ldg(&sp_gpu_helper_sm[f].denominator.z));

    float num_kf = __ldg(&sp_gpu_helper_sm[k].numerator_f.x);

    float total_marginal_f = (num_kf - __ldg(&sp_gpu_helper_sm[k].denominator_f.x)) +   
                         (num_kf - __ldg(&sp_gpu_helper_sm[k].denominator_f.y)) + 
                         (num_kf - __ldg(&sp_gpu_helper_sm[k].denominator_f.z));


    float log_nominator = lgammaf(count_f) + total_marginal_f + lgammaf(alpha_hasting_ratio) + 
        lgammaf(alpha_hasting_ratio / 2 + count_k) + lgammaf(alpha_hasting_ratio / 2 + count_f -  count_k);

   float log_denominator = __logf(alpha_hasting_ratio) + lgammaf(count_k) + lgammaf(count_f -  count_k) + total_marginal_1 + 
        total_marginal_2 + lgammaf(alpha_hasting_ratio + count_f) + lgammaf(alpha_hasting_ratio / 2) + 
        lgammaf(alpha_hasting_ratio / 2);

    log_denominator = __logf(alpha_hasting_ratio) + total_marginal_1 + total_marginal_2;
    log_nominator = total_marginal_f ;


    sp_gpu_helper_sm[k].hasting = log_nominator - log_denominator;



    return;
}


__global__ void calc_hasting_ratio2(const float* image_gpu_double,int* split_merge_pairs, superpixel_params* sp_params, superpixel_GPU_helper* sp_gpu_helper, superpixel_GPU_helper_sm* sp_gpu_helper_sm, const int nPixels, const int xdim, const int nsuperpixel_buffer, float a0, float b0, float alpha_hasting_ratio, int* mutex ) {
	// getting the index of the pixel
	int k = threadIdx.x + blockIdx.x * blockDim.x;  // the label

	if (k>=nsuperpixel_buffer) return;
	if (sp_params[k].valid == 0) return;

    int f = split_merge_pairs[2*k+1];
	if (sp_params[f].valid == 0) return;
    if(f<=0) return;
    if((sp_gpu_helper_sm[k].hasting ) > -2)
    {
            //printf("Want to merge k: %d, f: %d, splitmerge k %d, splitmerge  f %d, %d\n", k, f, split_merge_pairs[2*k], split_merge_pairs[2*f], split_merge_pairs[2*f+1] );
            if( k > atomicMax(&split_merge_pairs[2*f],k))
            {
                //printf("Merge: %f \n",sp_gpu_helper_sm[k].hasting );

                sp_gpu_helper_sm[k].merge = true;
            }   

        }
         
    
    
    return;

}

__global__ void calc_hasting_ratio_split(const float* image_gpu_double,int* split_merge_pairs, superpixel_params* sp_params, superpixel_GPU_helper* sp_gpu_helper, superpixel_GPU_helper_sm* sp_gpu_helper_sm, const int nPixels, const int xdim, const int nsuperpixel_buffer, float a0, float b0, float alpha_hasting_ratio, int* mutex, int max_SP, int* max_sp ) {
	// getting the index of the pixel
	int k = threadIdx.x + blockIdx.x * blockDim.x;  // the label

	if (k>=nsuperpixel_buffer) return;
	if (sp_params[k].valid == 0) return;
    
    int s = k + max_SP;
    if(s>=nsuperpixel_buffer) return;
    float count_f = __ldg(&sp_params[k].count);
    float count_k= __ldg(&sp_gpu_helper_sm[k].count);
    float count_s = __ldg(&sp_gpu_helper_sm[s].count);

    if((count_f<1)||( count_k<1)||(count_s<1)) return;

    float num_k = __ldg(&sp_gpu_helper_sm[k].numerator.x);
    float num_s = __ldg(&sp_gpu_helper_sm[s].numerator.x);
    float num_f = __ldg(&sp_gpu_helper_sm[k].numerator_f.x);
    
    float total_marginal_k = (num_k - __ldg(&sp_gpu_helper_sm[k].denominator.x)) +  
                         (num_k - __ldg(&sp_gpu_helper_sm[k].denominator.y)) + 
                         (num_k - __ldg(&sp_gpu_helper_sm[k].denominator.z)); 

    float total_marginal_s = (num_s - __ldg(&sp_gpu_helper_sm[s].denominator.x)) +  
                         (num_s - __ldg(&sp_gpu_helper_sm[s].denominator.y)) + 
                         (num_s - __ldg(&sp_gpu_helper_sm[s].denominator.z)); 

    float total_marginal_f = (num_f - __ldg(&sp_gpu_helper_sm[k].denominator_f.x)) +  
                         (num_f - __ldg(&sp_gpu_helper_sm[k].denominator_f.y)) + 
                         (num_f - __ldg(&sp_gpu_helper_sm[k].denominator_f.z)); 



 
     //printf("hasating:x k: %d, count: %f, den: %f, %f, %f, b_n: %f, %f, %f, num: %f \n",k, count_k,  sp_gpu_helper_sm[k].denominator.x, sp_gpu_helper_sm[k].denominator.y,  sp_gpu_helper_sm[k].denominator.z,   __logf (sp_gpu_helper_sm[k].b_n.x) ,  __logf (sp_gpu_helper_sm[k].b_n.y),   __logf (sp_gpu_helper_sm[k].b_n.z), sp_gpu_helper_sm[k].numerator.x);
    float log_nominator = __logf(alpha_hasting_ratio)+ lgammaf(count_k)+ total_marginal_k + lgammaf(count_s) + total_marginal_s ;
    log_nominator = total_marginal_k + total_marginal_s ;

    float log_denominator = lgammaf(count_f) + total_marginal_f;
    log_denominator =total_marginal_f;
    sp_gpu_helper_sm[k].hasting = log_nominator - log_denominator;


    sp_gpu_helper_sm[k].merge = (sp_gpu_helper_sm[k].hasting > -2);
    sp_gpu_helper_sm[s].merge = (sp_gpu_helper_sm[k].hasting > -2);

   if((sp_gpu_helper_sm[k].merge))
    {

        s = atomicAdd(max_sp,1) +1;
        split_merge_pairs[2*k] = s;

        //atomicMax(max_sp,s);
        sp_params[k].prior_count/=2;
        sp_params[s].prior_count=  sp_params[k].prior_count; //#check why.


    }


}




__global__  void merge_sp(int* seg, bool* border, int* split_merge_pairs, superpixel_params* sp_params, superpixel_GPU_helper_sm* sp_gpu_helper_sm, const int nPixels, const int xdim, const int ydim){   
    int idx = threadIdx.x + blockIdx.x * blockDim.x;  
    if (idx>=nPixels) return; 
    int k = seg[idx]; // center 
    //if (sp_params[k].valid == 0) return;
    int f = split_merge_pairs[2*k+1];
    if(sp_gpu_helper_sm[k].remove)
    seg[idx] =  f;

    return;  
      
}

__global__  void split_sp(int* seg, int* seg_split1, int* split_merge_pairs, superpixel_params* sp_params, superpixel_GPU_helper_sm* sp_gpu_helper_sm, const int nPixels, const int xdim, const int ydim,int max_SP){   
    int idx = threadIdx.x + blockIdx.x * blockDim.x;  
    if (idx>=nPixels) return; 
    int k = seg[idx]; // center 
    int k2 = k + max_SP;
    if ((sp_gpu_helper_sm[k].merge == false)||sp_gpu_helper_sm[k2].merge == false ) return; 
    if(seg_split1[idx]==k2) seg[idx] = split_merge_pairs[2*k];
    //seg[idx] = seg_split1[idx];
    //printf("Add the following: %d - %d'\n", k,split_merge_pairs[2*k]);
    sp_params[split_merge_pairs[2*k]].valid = 1;


    return;  
}




__global__ void remove_sp(int* split_merge_pairs, superpixel_params* sp_params, superpixel_GPU_helper_sm* sp_gpu_helper_sm, const int nsuperpixel_buffer) {
	// getting the index of the pixel
	int k = threadIdx.x + blockIdx.x * blockDim.x;  // the label
	if (k>=nsuperpixel_buffer) return;
    int f = split_merge_pairs[2*k+1];
    if ((sp_params[k].valid == 0)||(sp_params[f].valid == 0)) return;    
    if(f<=0) return;
    if ((sp_gpu_helper_sm[k].merge == true) && (sp_gpu_helper_sm[f].merge == false) && (split_merge_pairs[2*f]==k) )
    {

    sp_gpu_helper_sm[k].remove=true;
    sp_params[k].valid =0;
    sp_params[f].prior_count =sp_params[k].prior_count+sp_params[f].prior_count;
    }
    else
    {
        sp_gpu_helper_sm[k].remove=false;
    }
    
    return;
    
}


        
