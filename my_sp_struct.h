#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"


struct alignas(16) superpixel_params
{
    float3 mu_i;
    double3 sigma_s;
    double2 mu_s;
    double logdet_Sigma_s;
    int count;
    double log_count;
    int valid;
    float prior_count;
};

struct alignas(16) superpixel_GPU_helper{
    float3 mu_i_sum;  // with respect to nSps
    int2 mu_s_sum;
    longlong3 sigma_s_sum;
};


struct alignas(16) superpixel_GPU_helper_sm {
    float3 squares_i;
    int count_f;
    float3 b_n;
    float3 b_n_f;
    float3 numerator;
    float3 denominator; 
    float3 numerator_f;
    float3 denominator_f; 
    float hasting;
    bool merge;
    bool remove;
    bool stop_bfs;
    float3 mu_i_sum;
    int count;
    int max_sp;
};

struct alignas(16) post_changes_helper{
    int changes[4];
    float post[5];
    bool skip_post[5]; 
    bool skip_post_calc[4];
};

struct alignas(16)  superpixel_options{
    int nPixels_in_square_side,area; 
    float i_std, s_std, prior_count;
    bool permute_seg, calc_cov, use_hex;
    int prior_sigma_s_sum;
    int nEMIters, nInnerIters;
    float beta_potts_term;
    float alpha_hasting;
};

