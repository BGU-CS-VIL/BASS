#include <stdio.h>

#ifndef MY_SP_STRUCT
#define MY_SP_STRUCT
#include "my_sp_struct.h"
#endif

__device__ __forceinline__ float atomicMaxFloat2 (float * addr, float value) {
    float old;
    old = (value >= 0) ? __int_as_float(atomicMax((int *)addr, __float_as_int(value))) :
         __uint_as_float(atomicMin((unsigned int *)addr, __float_as_uint(value)));

    return old;
}
/*bool nums_bool[256];
int nums_indices[48] = {2, 3, 6, 7, 8,  9,  11,  15,  16,  20,  22,  23,  31,
                40,  41,  43,  47,  63,  64,  96, 104, 105, 107, 111, 144, 148,
              150, 151, 159, 191, 192, 208, 212, 214, 215, 224, 232, 233, 235,
              239, 240, 244, 246, 247, 248, 249, 252, 253};

for (int i = 0; i<256; i++){
  nums_bool[i] = 0;
  for(int j = 0; j< 48; j++){
    if(nums_indices[i]==j) nums_bool[i] =1;
}
*/

//bass table
__device__ const bool bass_lut[256] = {0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0,
	       1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1,
	       0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0,
	       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	       0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0,
	       0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0,
	       0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0,
	       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0,
	       0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0,
	       0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0};

   /*
__device__ inline int ischangbale_by_nbrs(bool* nbrs){
  // This function does the following:
  // 1) converts the arrray of binary labels of the 8 nbrs into an integer  btwn 0 and 255
  // 2) does a lookup check and count number of different labels, than set the last cell of nbrs array as
  // 0 or 1 which means if it is valid or not.

    int num,count_diff = 0;
#pragma unroll
   for (int i=7; i>=0; i--){
      num <<= 1;
      if (nbrs[i]) num++;
      else count_diff++;
   }
  if (num == 0)
    return 0;
  else {
	nbrs[8] = 0;

	// indexes of valid configuration
	int nums [48] = {2, 3, 6, 7, 8,  9,  11,  15,  16,  20,  22,  23,  31,
				         40,  41,  43,  47,  63,  64,  96, 104, 105, 107, 111, 144, 148,
				        150, 151, 159, 191, 192, 208, 212, 214, 215, 224, 232, 233, 235,
				        239, 240, 244, 246, 247, 248, 249, 252, 253};
	for(int j = 0; j< 48; j++){
		if(num == nums[j]){nbrs[8] = 1;}
	}
	//nbrs[8] = lut[num];
	return count_diff;
  }

    //return ischangbale_by_num(num);
    }
  */

__device__ inline int ischangbale_by_nbrs(bool* nbrs){

  int num,count_diff = 0;
#pragma unroll
   for (int i=7; i>=0; i--)
   {
      num <<= 1;
      if (nbrs[i]) num++;
      else count_diff++;
   }
	nbrs[8] = bass_lut[num];
	return count_diff;
  
}



   
/*
* Set the elements in nbrs "array" to 1 if corresponding neighbor pixel has the same superpixel as "label"
*/
__device__ inline void  set_nbrs(int NW,
                                 int N, 
                                 int NE,
                                int W,
                                int E,
                                int SW,
                                int S,
                                int SE,
                                int label, bool* nbrs)
                                {
 
        nbrs[0] = (label ==NW);
     
        nbrs[1] = (label == N);
 
        nbrs[2] = (label == NE);

        nbrs[3] = (label == W);

        nbrs[4] = (label == E);

        nbrs[5] = (label == SW);

        nbrs[6] = (label == S);

        nbrs[7] = (label == SE);
 

    return;
}   

__device__ inline float2 cal_posterior_new(
    float* img, int* seg,
    int x, int y,
    superpixel_params* sp_params,  
    int idx,
    int seg_idx,
    float3 J_i, float logdet_Sigma_i, float i_std, int s_std,
    post_changes_helper* post_changes, float potts, float beta, float2 res_max)

{
    float res = -1000; // some large negative number
  
    float* imgC = img + idx * 3;
    //if (idx>154064)
    //    printf("%d ,%d , %d\n",idx_inside, idx, seg_idx);
    const float x0 = __ldg(&imgC[0])-__ldg(&sp_params[seg_idx].mu_i.x);
    const float x1 = __ldg(&imgC[1])-__ldg(&sp_params[seg_idx].mu_i.y);
    const float x2 = __ldg(&imgC[2])-__ldg(&sp_params[seg_idx].mu_i.z);

    const int d0 = x - __ldg(&sp_params[seg_idx].mu_s.x);
    const int d1 = y - __ldg(&sp_params[seg_idx].mu_s.y);
    //color component
    const float J_i_x = J_i.x;
    const float J_i_y = J_i.y;
    const float J_i_z = J_i.z;
    const float sigma_s_x = __ldg(&sp_params[seg_idx].sigma_s.x);
    const float sigma_s_y = __ldg(&sp_params[seg_idx].sigma_s.y);
    const float sigma_s_z = __ldg(&sp_params[seg_idx].sigma_s.z);
    const float logdet_sigma_s = __ldg(&sp_params[seg_idx].logdet_Sigma_s);


    res = res - (x0*x0*J_i_x + x1*x1*J_i_y + x2*x2*J_i_z);   //res = -calc_squared_mahal_3d(imgC,mu_i,J_i);
    res = res -logdet_Sigma_i;
    //space component
    res = res - d0*d0*sigma_s_x;
    res = res - d1*d1*sigma_s_z;
    res = res -  2*d0*d1*sigma_s_y;            // res -= calc_squared_mahal_2d(pt,mu_s,J_s);
    res = res -  logdet_sigma_s;
    res = res -beta*potts;
    /*if (res > atomicMaxFloat2(&post_changes[idx].post[4],res))
    {
      seg[idx] = seg_idx;

    }*/
    if( res>res_max.x)
    {
      res_max.x = res;
      res_max.y = seg_idx;

    }
    


    return res_max;
    
}



/*__device__ inline float cal_posterior(
    bool isValid,
    float* imgC,
    int x, int y, double* pt,
    superpixel_params* sp_params,  
    int seg,
    float3 J_i, float logdet_Sigma_i, 
    bool cal_cov, float i_std, int s_std)
{
    
      float res = -1000; // some large negative number
      if (isValid){
        const float3 mu_i = sp_params[seg].mu_i;
        const double3 sigma_s = sp_params[seg].sigma_s;
        const double2 mu_s = sp_params[seg].mu_s;
        
        float x0 = imgC[0]-mu_i.x;
        float x1 = imgC[1]-mu_i.y;
        float x2 = imgC[2]-mu_i.z;

        double d0 = x - mu_s.x;
        double d1 = y - mu_s.y;

        const double logdet_Sigma_s = sp_params[seg].logdet_Sigma_s;
        const double log_count = sp_params[seg].log_count;


		//color component
		res = - (x0*x0*J_i.x + x1*x1*J_i.y + x2*x2*J_i.z);   //res = -calc_squared_mahal_3d(imgC,mu_i,J_i);
		res -= logdet_Sigma_i;
		//space component
		res -= d0*d0*sigma_s.x;
		res -= d1*d1*sigma_s.z;
		res -= 2*d0*d1*sigma_s.y;            // res -= calc_squared_mahal_2d(pt,mu_s,J_s);
		res -= logdet_Sigma_s;
		//res += potts_res;


        //add in prior prob
#if USE_COUNTS 
        const double prior_weight = 0.5;
        res *= (1-prior_weight);
        double prior = prior_weight * log_count;
        res += prior;
#endif    
      }
      return res;
}

*/

__device__ inline float calc_total(float posterior, float potts_term){
		float total = 0;

		total = posterior +potts_term;
		return total;
	}

__device__ inline float calc_potts(float beta, int count_diff){
	float potts;
	potts = -beta*count_diff;
	return potts;
}

  
