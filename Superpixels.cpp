

#include "Superpixels.h"
#define THREADS_PER_BLOCK 128
#include "utils.h"

//#include <ppl.h>
#include <thrust/system_error.h>
#include <thrust/system/cuda/error.h>
#include <iostream>
#include <fstream>

using namespace std;
//using namespace concurrency;
void throw_on_cuda_error(cudaError_t code)
{
  if(code != cudaSuccess){
    throw thrust::system_error(code, thrust::cuda_category());
  }
}



// constructor
// init the superpixels with dim_x, dim_y, dim_i and options
Superpixels::Superpixels(int img_dimx, int img_dimy, superpixel_options spoptions){
    init_sp = false;
    dim_x = img_dimx;
    dim_y = img_dimy;
    nPixels = dim_x * dim_y;

    dim_i = 3; // RGB/BGR/LAB
    dim_s = 2;

    sp_options = spoptions;
    float i_std  = float(sp_options.i_std);

    float half_i_std_square = float(i_std/2) * float(i_std/2);
    float i_std_square = float(i_std) * float(i_std);

    logdet_Sigma_i = log(half_i_std_square * i_std_square * i_std_square);
    logdet_Sigma_i = log(half_i_std_square * half_i_std_square * half_i_std_square);

    J_i.x = 1.0/half_i_std_square;

    J_i.y = 1.0/i_std_square;
    J_i.z = 1.0/i_std_square;

    J_i.y = 1.0/half_i_std_square;
    J_i.z = 1.0/half_i_std_square;

    //allocate memory for the cpu variables: image_cpu, seg_cpu and border_cpu
    const int sizeofint = sizeof(int);
    const int sizeofbool = sizeof(bool);
    const int sizeofuchar = sizeof(unsigned char);
    const int sizeofd = sizeof(double);
    const int sizeofuchar3 = sizeof(uchar3);

    image_cpu = (float*) malloc(dim_i*nPixels*sizeofuchar);
    seg_cpu = (int*) malloc(nPixels*sizeofint);
    border_cpu = (bool*) malloc(nPixels * sizeofbool);

     // allocate memory for the cuda variables
    try{
        throw_on_cuda_error( cudaMalloc((void**) &image_gpu, nPixels*sizeofuchar3));
        throw_on_cuda_error( cudaMalloc((void**) &image_gpu_double, dim_i*nPixels*sizeofd));
        throw_on_cuda_error( cudaMalloc((void**) &seg_gpu, nPixels * sizeofint));
        throw_on_cuda_error( cudaMalloc((void**) &seg_split1, nPixels * sizeofint));
        throw_on_cuda_error( cudaMalloc((void**) &seg_split2, nPixels * sizeofint));
        throw_on_cuda_error( cudaMalloc((void**) &seg_split3, nPixels * sizeofint));

        throw_on_cuda_error( cudaMalloc((void**) &seg_potts_label, nPixels * sizeofint));
        throw_on_cuda_error( cudaMalloc((void**) &border_gpu, nPixels*sizeofbool));
        throw_on_cuda_error( cudaMalloc((void**) &split_merge_pairs, 2*nPixels*sizeofint));
        throw_on_cuda_error( cudaMalloc((void**) &split_merge_unique, 2*nPixels*sizeofint));


    }
    catch(thrust::system_error &e){
        std::cerr << "CUDA error after cudaMalloc: " << e.what() << std::endl;
        cudaSetDevice(0);
    }

    if (dim_x>0){
        cout << "dim_x:" << dim_x << endl;
        cout << "dim_y:" << dim_y << endl;
        cout << "i_std:" << sp_options.i_std << endl;
        cout << "nPixels_in_square_side:" << sp_options.nPixels_in_square_side << endl;

    }
    // initialize the gpu variables: seg_gpu, border_gpu, sp_params
    nSPs = CudaInitSeg(seg_cpu, seg_gpu, split_merge_pairs, nPixels, sp_options.nPixels_in_square_side, dim_x, dim_y, sp_options.use_hex);
    if (dim_x > 0) {
        cout << "nSPs:" << nSPs << endl;
    }
    max_SP = nSPs;
    int nSPs_buffer = nSPs * 50;

    const int sofsparams = sizeof(superpixel_params);
    const int sofsphelper = sizeof(superpixel_GPU_helper);
    const int sofsphelper_sm = sizeof(superpixel_GPU_helper_sm);
    const int sofpost_changes = sizeof(post_changes_helper);

    sp_params_cpu = (superpixel_params*)malloc(nSPs_buffer * sofsparams);

    try {
        throw_on_cuda_error(cudaMalloc((void**)&sp_params, nSPs_buffer * sofsparams));
        throw_on_cuda_error(cudaMalloc((void**)&sp_gpu_helper, nSPs_buffer * sofsphelper));
        throw_on_cuda_error(cudaMalloc((void**)&sp_gpu_helper_sm, nSPs_buffer * sofsphelper_sm));
        throw_on_cuda_error(cudaMalloc((void**)&post_changes, nPixels * sofpost_changes));
    }
    catch (thrust::system_error& e) {
        std::cerr << "CUDA error after cudaMalloc: " << e.what() << std::endl;
        cudaSetDevice(0);
    }

    CudaFindBorderPixels(seg_gpu, border_gpu, nPixels, dim_x, dim_y, 0);
    CudaInitSpParams(sp_params, sp_options.s_std, sp_options.i_std, nSPs, nSPs_buffer, nPixels);

    init_sp = true;

   //warm_up();


}

int Superpixels::get_dim_y()
{
    return dim_y;
}

int Superpixels::get_dim_x()
{
    return dim_x;
}

bool* Superpixels::get_border_cpu() {
    return border_cpu;
}

int* Superpixels::get_seg_cpu() {
    cudaMemcpy(seg_cpu, seg_gpu, nPixels * sizeof(int), cudaMemcpyDeviceToHost);
    return seg_cpu;
}

superpixel_params* Superpixels::get_sp_params() {
    cudaMemcpy(sp_params_cpu, sp_params, nSPs_buffer * sizeof(superpixel_params), cudaMemcpyDeviceToHost);
    return sp_params_cpu;
}


int Superpixels::get_nSPs() {
    return nSPs;
}


void Superpixels::set_nSPs(int new_nSPs) {
    nSPs = new_nSPs;
}
//read an rgb image, set the gpu copy, set the float_gpu to be the lab image
void Superpixels::load_img(float* imgP) {
    memcpy(image_cpu, imgP, dim_i * nPixels * sizeof(unsigned char));
    cudaMemcpy(image_gpu, image_cpu, dim_i * nPixels * sizeof(unsigned char), cudaMemcpyHostToDevice);
    Rgb2Lab(image_gpu, image_gpu_double, nPixels);
}


// update seg_gpu, sp_gpu_helper and sp_params
void Superpixels::calc_seg() {
    //sp_gpu_helper_sm[0].max_sp = max_SP;
    int prior_sigma_s = sp_options.area * sp_options.area;
    int prior_count = sp_options.area;
    const int sizeofint = sizeof(int);
    const int sizeoffloat = sizeof(float);
    bool cal_cov = sp_options.calc_cov;
    float i_std = sp_options.i_std;
    float alpha = sp_options.alpha_hasting;
    int s_std = sp_options.s_std;
    int nInnerIters = sp_options.nInnerIters;
    nSPs_buffer = nSPs * 45 ;
    int count = 1;
    int count_split =0;
    for (int i = 0; i < sp_options.nEMIters*1; i++) {
    //printf("%d \n",i);
    //for (int i = 0; i < 3; i++) {
        // "M step"
        update_param(image_gpu_double, seg_gpu, sp_params, sp_gpu_helper, nPixels, nSPs, nSPs_buffer, dim_x, dim_y, prior_sigma_s, prior_count);
        if((i<sp_options.nEMIters*20) && (i>-1) )
        {
            if(i>sp_options.nEMIters*0){
            if((i%4==0)&&(count<100))
            {
                count+=1;

                max_SP = CudaCalcSplitCandidate(image_gpu_double, split_merge_pairs, seg_gpu, border_gpu, sp_params ,sp_gpu_helper,sp_gpu_helper_sm, nPixels,dim_x,dim_y,nSPs_buffer,seg_split1,seg_split2, seg_split3,max_SP, count, i_std, alpha);
                update_param(image_gpu_double, seg_gpu, sp_params, sp_gpu_helper, nPixels, nSPs, nSPs_buffer, dim_x, dim_y, prior_sigma_s, prior_count);
            }

            if((i%4==2)&&(count<100)){

                for(int j=0; j<1; j++){
                    CudaCalcMergeCandidate(image_gpu_double, split_merge_pairs, seg_gpu, border_gpu, sp_params ,sp_gpu_helper,sp_gpu_helper_sm, nPixels,dim_x,dim_y,nSPs_buffer,count%2,i_std, alpha);
                    update_param(image_gpu_double, seg_gpu, sp_params, sp_gpu_helper, nPixels, nSPs, nSPs_buffer, dim_x, dim_y, prior_sigma_s, prior_count);
                }
            }
            }
        }
        //"(Hard) E step" - find only the max value after potts term to get the best label
        update_seg(image_gpu_double, seg_gpu, seg_potts_label, border_gpu, sp_params, J_i, logdet_Sigma_i, cal_cov, i_std, s_std, nInnerIters, nPixels, nSPs, nSPs_buffer, dim_x, dim_y, sp_options.beta_potts_term,post_changes);

      cudaError_t err_t = cudaDeviceSynchronize();
        if (err_t) {
            std::cerr << "CUDA error after cudaDeviceSynchronize. " << err_t << std::endl;
            cudaError_t err = cudaGetLastError();
        }
    }
   CudaFindBorderPixels_end(seg_gpu, border_gpu, nPixels, dim_x, dim_y, 1);

}
//Set the pixels on the superpixel boundary to red:
Mat Superpixels::get_img_overlaid() {
    float* image_border_cpu = (float*)malloc(dim_i * nPixels * sizeof(unsigned char));
    CUDA_get_image_overlaid(image_gpu, border_gpu, nPixels, dim_x);
    cudaMemcpy(image_border_cpu, image_gpu, dim_i * nPixels * sizeof(unsigned char), cudaMemcpyDeviceToHost);
    Mat img_border(dim_y, dim_x, CV_8UC3, image_border_cpu);
    return img_border;
}

//replace pixel color by superpixel mean
Mat Superpixels::get_img_cartoon() {
    // fill in image_mean_gpu with superpixel mean in
    uchar3* image_mean_gpu;
    float* image_mean_cpu = (float*)malloc(dim_i * nPixels * sizeof(unsigned char));
    cudaMalloc((void**)&image_mean_gpu, dim_i * nPixels * sizeof(uchar3));
    CUDA_get_image_cartoon(image_mean_gpu, seg_gpu, sp_params, nPixels);
    cudaMemcpy(image_mean_cpu, image_mean_gpu, dim_i * nPixels * sizeof(unsigned char), cudaMemcpyDeviceToHost);
    Mat img_mean(dim_y, dim_x, CV_8UC3, image_mean_cpu);
    return img_mean;
}

float* Superpixels::get_image_cpu() {
    return image_cpu;
}
void Superpixels::convert_lab_to_rgb() {
    Lab2Rgb(image_gpu, image_gpu_double, nPixels);
}

void Superpixels::gpu2cpu() {
    cudaMemcpy(seg_cpu, seg_gpu, nPixels * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(border_cpu, border_gpu, nPixels * sizeof(bool), cudaMemcpyDeviceToHost);
    cudaMemcpy(sp_params_cpu, sp_params, nSPs_buffer * sizeof(superpixel_params), cudaMemcpyDeviceToHost);
    get_unique_sp_idxs();
}

vector<int> Superpixels::get_superpixel_indexes(int sp_index) {
    std::vector<int> sp_index_arr;

    for (int i = 0; i < dim_y * dim_x; i++)
    {
        if (seg_cpu[i] == sp_index) {
            sp_index_arr.push_back(i);
        }
}

    return sp_index_arr;
}

vector<unsigned char> Superpixels::get_superpixel_by_channel(int sp_index, int channel) {
    std::vector<unsigned char> sp_pixels;

    //calc start point by channel
    float* start = image_cpu + (channel * nPixels);

    for (int i = 0; i < nPixels; i++)
    {
        if (seg_cpu[i] == sp_index) {
            sp_pixels.push_back(*(start + i));
        }
    }


    return sp_pixels;
}

void Superpixels::cpu_merge_superpixels_pair(int sp_idx_1, int sp_idx_2) {
    // update seg map cpu values
    for (int i = 0; i < dim_y * dim_x; i++)
    {
    if (seg_cpu[i] == sp_idx_2) {
            seg_cpu[i] = sp_idx_1;
            sp_params_cpu[sp_idx_2].valid = 0;
       }
       /*printf("%d, %d \n",seg_gpu[i],sp_idx_2);
        if (seg_gpu[i] == sp_idx_2) {
            seg_gpu[i] = sp_idx_1;
            sp_params[sp_idx_2].valid = 0;
       }
*/
    }

}

//input: vectors of sp_idx pairs to merge(size of vectors as the size of the pairs)
void Superpixels::cpu_merge_all_sp_pairs(vector<short> first, vector<short> second) {
    cudaMemcpy(sp_params_cpu, sp_params, nSPs_buffer * sizeof(superpixel_params), cudaMemcpyDeviceToHost);

    for (size_t i = 0; i < first.size(); i++)
    {
        short idx_sp_1 = first[i];
        short idx_sp_2 = second[i];
        cpu_merge_superpixels_pair(idx_sp_1, idx_sp_2);
    }

    // copy or update cpu_seg_map to gpu
    cudaMemcpy(seg_gpu, seg_cpu, nPixels * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(sp_params, sp_params_cpu, nSPs_buffer * sizeof(superpixel_params), cudaMemcpyHostToDevice);

    //TODO: remove?
    CudaFindBorderPixels(seg_gpu, border_gpu, nPixels, dim_x, dim_y, 1);
    //TODO: copy or update sp_params to gpu
}

/// <summary>
/// This function updates cpu_seg map to seg map after split with new superpixels
/// </summary>
/// <param name="original_sp_idx"></param>
/// <param name="new_sp_idx"></param>
void Superpixels::cpu_split_superpixel(int original_sp_idx, int new_sp_idx, int* proposed_seg_map)
{
    // update seg map cpu values by passing over the indexes of proposed seg map split and update
    for (int i = 0; i < dim_y * dim_x; i++)
    {
        if (proposed_seg_map[i] == new_sp_idx)
        {
            seg_cpu[i] = new_sp_idx;
            sp_params_cpu[new_sp_idx].valid = 1;
            //TODO: update sp_params also..
        }
    }

        //TODO: remove sp_params of sp_idx_2
    //TODO: remove sp_params of sp_idx_2
}



void Superpixels::cpu_split_superpixels(vector<int> first, vector<int> second, int* proposed_seg_map)
{
    cudaMemcpy(sp_params_cpu, sp_params, nSPs_buffer * sizeof(superpixel_params), cudaMemcpyDeviceToHost);

    for (int i = 0; i < first.size(); i++)
    //for (size_t(0) i=0; first.size(); [&](size_t i))
        {
            short idx_sp_1 = first[i];
            short idx_sp_2 = second[i];
            if (idx_sp_2 > 0)
            {
                printf("idx_sp_2, %d\n", idx_sp_2);
                cpu_split_superpixel(idx_sp_1, idx_sp_2, proposed_seg_map);
            }
        }

    cudaMemcpy(seg_gpu, seg_cpu, nPixels * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(sp_params, sp_params_cpu, nSPs_buffer * sizeof(superpixel_params), cudaMemcpyHostToDevice);
    CudaFindBorderPixels(seg_gpu, border_gpu, nPixels, dim_x, dim_y, 1);

}



vector<int> Superpixels::get_unique_sp_idxs() {
    get_sp_params();
    int n = get_dim_x() * get_dim_y() * sizeof(int) / sizeof(seg_cpu[0]);
    vector<int> sp_idxs(seg_cpu, seg_cpu + n);//(first elem, last elem)
    vector<int>::iterator it;
    it = std::unique(sp_idxs.begin(), sp_idxs.end());
    sp_idxs.erase(it, sp_idxs.end());
    std::sort(sp_idxs.begin(), sp_idxs.end());
    int uniqueCount = std::unique(sp_idxs.begin(), sp_idxs.end()) - sp_idxs.begin();
    printf("ToTal number of SP: %d ", uniqueCount);
    ////it = std::unique(sp_idxs.begin(), sp_idxs.end());
   // sp_idxs.erase(it, sp_idxs.end());

    return sp_idxs;
}

Superpixels::~Superpixels()
{
    if (init_sp){
        free(sp_params_cpu);
        cudaFree(sp_params);
        cudaFree(sp_gpu_helper);
        cudaFree(sp_gpu_helper_sm);
        cudaFree(post_changes);

        //cout << "free sp_params..." << endl;
        cudaFree(seg_split1);
        cudaFree(seg_split2);
        cudaFree(seg_split3);
        cudaFree(split_merge_pairs);
        cudaFree(split_merge_unique);
        free(image_cpu);
        //cout << "free image_cpu" << endl;
        cudaFree(image_gpu);
        //cout << "free image_gpu" << endl;
        cudaFree(image_gpu_double);
        //cout << "free image_gpu_double" << endl;

        free(border_cpu);
        cudaFree(border_gpu);
        //cout << "free border..." << endl;

        free(seg_cpu);
        cudaFree(seg_gpu);
        //cout << "free seg..." << endl;
        cudaFree(seg_potts_label);
    }else{
        cout << "init_sp = false" << endl;
    }
    init_sp = false;

   // cout << "Object is being deleted" << endl;
}
