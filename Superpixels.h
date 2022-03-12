#pragma once 
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include "./include/optionparser.h"

#ifndef MY_SP_STRUCT
#define MY_SP_STRUCT
#include "my_sp_struct.h"
#endif

 
#include "RgbLab.h"
#include "init_seg.h"
#include "sp_helper.h"
#include "update_param.h"
#include "update_seg.h"
#include "s_m.h"

using namespace cv;
using namespace std;


class Superpixels {
    int dim_i, dim_s;
    int dim_x, dim_y;
    int nPixels_in_square_side;

    int nInnerIters;
    bool init_sp;
    superpixel_options sp_options;

    //superpixel_params* sp_params;
    //superpixel_params* sp_params_cpu;
    //superpixel_GPU_helper* sp_gpu_helper;
    
    // since we fix covariance for color component
    float3 J_i; //fixed
    float logdet_Sigma_i; //fixed

    float* image_cpu;
    uchar3* image_gpu;




  public:
    Superpixels(int img_dimx, int img_dimy, superpixel_options spoptions);
    ~Superpixels();

    superpixel_params* sp_params;
    superpixel_params* sp_params_cpu;
    superpixel_GPU_helper* sp_gpu_helper;
    post_changes_helper* post_changes;
    float* image_gpu_double;
    int nSPs, nSPs_buffer, nPixels;
    int max_SP;
    /*
    Roy - Change
    */
    bool* border_cpu;
    bool* border_gpu;
    int* seg_cpu;
    int* seg_gpu;
    int* seg_split1;
    int* seg_split2;
    int* seg_split3;
    int* seg_potts_label;
    int* split_merge_pairs;
    int* split_merge_unique;
    superpixel_GPU_helper_sm* sp_gpu_helper_sm;
    int get_dim_x();
    int get_dim_y();
    bool* get_border_cpu();
    int* get_seg_cpu();
    superpixel_params* get_sp_params() ;
    int get_nSPs();
    void set_nSPs(int new_nSPs);
    void load_img(float* imgP);
    void calc_seg();
    void gpu2cpu();
    Mat get_img_overlaid();
    Mat get_img_cartoon();
    float* get_image_cpu();

    void convert_lab_to_rgb();
    vector<int> get_superpixel_indexes(int sp_index);
    vector<unsigned char> get_superpixel_by_channel(int sp_index, int channel);
    void cpu_merge_superpixels_pair(int sp_idx_1, int sp_idx_2);
    void cpu_merge_all_sp_pairs(vector<short> first, vector<short> second);
    void cpu_split_superpixel(int original_sp_idx, int new_sp_idx, int* proposed_seg_map);
    void cpu_split_superpixels(vector<int> first, vector<int> second, int* proposed_seg_map);
    vector<int> get_unique_sp_idxs();
};

