#include "split_merge.h"
#include <opencv2/imgproc.hpp>
#include <utility>
#include <set>
#include <algorithm>
#include <random> 
# include <queue>
#include <fstream>
using namespace cv;
using namespace std;

#include <thrust/system_error.h>
#include <thrust/system/cuda/error.h>
#include <iostream>
#include <fstream>



float SplitMerge::calc_bn(std::vector<unsigned char>  pixels) {
    float res = 0.0;

    //TODO: change to vector multiplication 
    float sum_of_pixels = std::accumulate(pixels.begin(), pixels.end(),
       0.0);
    float sum_of_squares = std::inner_product(pixels.begin(), pixels.end(), pixels.begin(), 0);
    
    //res=b_0 + 0.5 * (sum_of_squares  * pow(sum_of_pixels, 2)* pixels.size());
    res = b_0 + 0.5 * (sum_of_squares - pow(sum_of_pixels, 2) / pixels.size());
    return res;
}

float calc_vn(int sp_index) {
    float res = 1;
    return res;
}


SplitMerge::SplitMerge(Superpixels* sp, float desired_samples_mean_IG, float desired_samples_var_IG, bool is_split_horizontal)
{
    // Members

    m_sp = sp;
    
    //x
    m_desired_samples_mean_IG = desired_samples_mean_IG;
    // variance = x^2/y
    m_desired_samples_var_IG = desired_samples_var_IG;
    float y = pow(m_desired_samples_mean_IG,2) / m_desired_samples_var_IG;

    a_0 = y + 2;
    b_0 = m_desired_samples_mean_IG * a_0 - desired_samples_mean_IG;
    k_0 = 0.00001;
    V_0 = 1 / k_0;
    alpha_hasting_ratio = 6000;
    variance_normal = V_0;
    mean_normal = 0;
    bfs = new BfsSplit(m_sp->get_dim_y(), m_sp->get_dim_x(), is_split_horizontal);
    
    proposed_seg_map = new int[sp->get_dim_x() * sp->get_dim_y()];
    
    max_sp_idx = 0;   
}

SplitMerge::~SplitMerge()
{
    delete bfs;
    if (proposed_seg_map)
        delete[] proposed_seg_map;
}

// TODO:  maybe it shouldn't be public and part of the class?
float SplitMerge::calc_marginal_liklelyhoood_of_sp(std::vector<unsigned char>  pixels)
{
    float res_log = 0.0;

    int num_pixels_in_sp = pixels.size();

    float a_n = a_0 + num_pixels_in_sp / 2;
    
    float b_n = calc_bn(pixels);

    float v_n = 1 / float(num_pixels_in_sp);
    v_n = float(num_pixels_in_sp);

    float res_numerator = a_0 * log(b_0) + calc_gamma_function(a_0)+0.5*v_n;
    float res_denominator = a_n* log(b_n) + 0.5 * num_pixels_in_sp * log(M_PI) + num_pixels_in_sp * log(2) + calc_gamma_function(a_n);



    return res_log;
}

// add documentation that is one channel only
float calc_merge_hasting_ratio(std::vector<unsigned char>  pix_sp1, std::vector<unsigned char> pix_sp_2) {
    return 0.0;

}


// this func look for  0 1 0 on border (horizontally and vertically) and if we found exits,check at the same location in segmentation map for sp
// indexes pairs. return the number of pairs to be merged.
int SplitMerge::cpu_calc_merge_candidates(vector<short> &first, vector<short> &second)
{
    //short* border_p = (short*)(m_sp->get_border_cpu());
    bool* border_p = m_sp->get_border_cpu();
    int* seg_p = (int*)(m_sp->get_seg_cpu());

    int rows = m_sp->get_dim_y();
    int cols = m_sp->get_dim_x();
    // TODO: may be it will be more efficient to use pointers and calc every time locations on the neighbours pixels than use mat
        
    int up = -cols;
    int down = cols;
    int right = 1;
    int left = -1;

/*
    set<pair<short, short>> unique_pairs;

    // we want only inter pixels so there will be exist right and left neighbours
    for (int idx = cols + 1; idx < (rows*cols) - cols - 1; idx++)
    {
        int x = idx % cols;
        int y = idx % rows;
        if (border_p[idx]) {
            
            // ensure that left and right exist(we are not at the start or end of line) and that they are border
            if (x > 0 && x < cols-1 && border_p[idx +right] == 0 && border_p[idx + left] == 0) {
                short first = short(seg_p[idx + right]);
                short second = short(seg_p[idx + left]);
                if (first != second)
                    unique_pairs.insert(make_pair(first, second));
            }

            // ensure that up and down exist(we are not at the start or end of column) and that they are border
            if (y > 0 && y < rows-1 && short(border_p[idx + up]) == 0 && short(border_p[idx + down] == 0)) {

                short first = short(seg_p[idx + up]);
                short second = short(seg_p[idx + down]);

                if(first != second)
                    unique_pairs.insert(make_pair(first, second));
            }
        }

    }
    
    cout << "finish pairs" << endl;
    
    //TODO: seperate to another function, it is just for now to check logic of merge
    // PERMUTE

    // obtain a time-based seed:
    //unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    //shuffle(unique_pairs.begin(), unique_pairs.end(), std::default_random_engine(1));
    

    // Check if pairs should be merged by hasting ratio
    // TODO: seperate to different method
    std::set<pair<short, short>>::iterator it = unique_pairs.begin();
    while (it != unique_pairs.end())
    {
        pair <short, short> curr_p = *it;
        //TODO: check also if there not exist common index of superpixels
        if(is_merge_needed(int(curr_p.first), int(curr_p.second)))
            it++;
        
        else {
            it = unique_pairs.erase(it);
        }

    }

    // AGGREGATION :remove common sp by make an array in size of superpixels and for each index write 1 if it already has pair and 0 if not and update it

    std::vector<bool> sp_has_pairs(m_sp->get_nSPs(), 0);
    //TODO: when there will be two arrays and not set unstead of remove we can update them to 0 , 0 or other meaningful value
    std::set<pair<short,short>>::iterator it_common = unique_pairs.begin();
    while(it_common != unique_pairs.end())
    {
        pair <short, short> curr_p = *it_common;

        if (sp_has_pairs[curr_p.first] == 0 && sp_has_pairs[curr_p.second] == 0) {
            sp_has_pairs[curr_p.first] = true;
            sp_has_pairs[curr_p.second] = true;
            it_common++;

        }
        else {
            it_common = unique_pairs.erase(it_common);
        }

    }

    cout << "pairs are ready" << endl;


    // Copy set to vector 
   it = unique_pairs.begin();
    int idx = 0;
    while (it != unique_pairs.end())
    {
        pair <short, short> curr_p = *it;
        first[idx] = curr_p.first;
        second[idx] = curr_p.second;

        idx++;
        it++;
    }

*/
    int idx =3;
    border_p = m_sp->border_gpu;
    seg_p = m_sp->seg_gpu;
    int* split_merge_p = m_sp->split_merge_pairs;
    superpixel_params* sp_params_p = m_sp->sp_params;
    superpixel_GPU_helper* sp_helper_p = m_sp->sp_gpu_helper;
    superpixel_GPU_helper_sm* sp_helper_sm_p = m_sp->sp_gpu_helper_sm;
    float* image_gpu_double_p = m_sp->image_gpu_double;

    clock_t start,finish;

    rows = m_sp->get_dim_y();
    cols = m_sp->get_dim_x();
    start = clock();

    //CudaCalcMergeCandidate(image_gpu_double_p, split_merge_p, seg_p, border_p, sp_params_p,sp_helper_p,sp_helper_sm_p, rows*cols,rows,cols,m_sp->nSPs_buffer);
    finish = clock();

    cout<< "Merge CUDA takes " << ((double)(finish-start)/CLOCKS_PER_SEC) << " sec" << endl;

    return idx;




}


void SplitMerge::calc_merge_candidates()
{
    bool* border_p = m_sp->border_gpu;
    int* seg_p = m_sp->seg_gpu;

    int rows = m_sp->get_dim_y();
    int cols = m_sp->get_dim_x();

    // TODO: may be it will be more efficient to use pointers and calc every time locations on the neighbours pixels than use mat

    int up = -cols;
    int down = cols;
    int right = 1;
    int left = -1;

    //TODO: remove set, instead use 2 vectors/arrays a and b and call to kernel instead of for loop
    set<pair<short, short>> unique_pairs;

    for (size_t idx = 1; idx < rows * cols; idx++)
    {
        int x = idx % cols;
        int y = idx % rows;

        if (border_p[idx] == 1) {

            if (x > 0 && border_p[idx + right] == 0 && border_p[idx + left] == 0) {
                short first = short(seg_p[idx + right]);
                short second = short(seg_p[idx + left]);

                if (first != second)
                    unique_pairs.insert(make_pair(first, second));
            }
            if (y > 0 && border_p[idx + up] == 0 && border_p[idx + down] == 0) {

                short first = short(seg_p[idx + up]);
                short second = short(seg_p[idx + down]);

                if (first != second)
                    unique_pairs.insert(make_pair(first, second));
            }
        }

    }

    cout << "finish pairs" << endl;

}


// threshold means that if the hasting raio > threshold num them merge is needed and return true.
bool SplitMerge::is_merge_needed(int first_sp_idx, int second_sp_idx, int threshold)
{
    // get all pixels of first sp by all channels
    std::vector<unsigned char>  pxls_1_c0 = m_sp->get_superpixel_by_channel(first_sp_idx, 0);
    std::vector<unsigned char>  pxls_1_c1 = m_sp->get_superpixel_by_channel(first_sp_idx, 1);
    std::vector<unsigned char>  pxls_1_c2 = m_sp->get_superpixel_by_channel(first_sp_idx, 2);


    // get all pixels of second sp by all channels
    std::vector<unsigned char>  pxls_2_c0 = m_sp->get_superpixel_by_channel(second_sp_idx, 0);
    std::vector<unsigned char>  pxls_2_c1 = m_sp->get_superpixel_by_channel(second_sp_idx, 1);
    std::vector<unsigned char>  pxls_2_c2 = m_sp->get_superpixel_by_channel(second_sp_idx, 2);

    float f_1_c0 = calc_marginal_liklelyhoood_of_sp(pxls_1_c0);
    float f_1_c1 = calc_marginal_liklelyhoood_of_sp(pxls_1_c1);
    float f_1_c2 = calc_marginal_liklelyhoood_of_sp(pxls_1_c2);

    float f_2_c0 = calc_marginal_liklelyhoood_of_sp(pxls_2_c0);
    float f_2_c1 = calc_marginal_liklelyhoood_of_sp(pxls_2_c1);
    float f_2_c2 = calc_marginal_liklelyhoood_of_sp(pxls_2_c2);

    // channels are i.i.d + log
    float total_f_sp1 = f_1_c0 + f_1_c1 + f_1_c2;
    float total_f_sp2 = f_2_c0 + f_2_c1 + f_2_c2;

    float num_pxls_1 = pxls_1_c0.size();
    float num_pxls_2 = pxls_2_c0.size();

    // create merged new super pixel

    pxls_1_c0.insert(
        pxls_1_c0.end(),
        std::make_move_iterator(pxls_2_c0.begin()),
        std::make_move_iterator(pxls_2_c0.end())
    );

    pxls_1_c1.insert(
        pxls_1_c1.end(),
        std::make_move_iterator(pxls_2_c1.begin()),
        std::make_move_iterator(pxls_2_c1.end())
    );


    pxls_1_c2.insert(
        pxls_1_c2.end(),
        std::make_move_iterator(pxls_2_c2.begin()),
        std::make_move_iterator(pxls_2_c2.end())
    );

    float f_12_c0 = calc_marginal_liklelyhoood_of_sp(pxls_1_c0);
    float f_12_c1 = calc_marginal_liklelyhoood_of_sp(pxls_1_c1);
    float f_12_c2 = calc_marginal_liklelyhoood_of_sp(pxls_1_c2);

    // calc total f of new superpixel 
    //channels are i.i.d + log
    float total_f_sp12 = f_12_c0 + f_12_c1 + f_12_c2;

   /* float nominator = calc_gamma_function(num_pxls_1 + num_pxls_2) * total_f_sp12 * calc_gamma_function(alpha_hasting_ratio) * \
        calc_gamma_function(alpha_hasting_ratio / 2 + num_pxls_1) * calc_gamma_function(alpha_hasting_ratio / 2 + num_pxls_2); */

    float log_nominator = calc_gamma_function(num_pxls_1 + num_pxls_2) + total_f_sp12 + calc_gamma_function(alpha_hasting_ratio) + \
        calc_gamma_function(alpha_hasting_ratio / 2 + num_pxls_1) + calc_gamma_function(alpha_hasting_ratio / 2 + num_pxls_2);

    /*float denominator = alpha_hasting_ratio * calc_gamma_function(num_pxls_1)*calc_gamma_function(num_pxls_2)* total_f_sp1 * \
        total_f_sp2* calc_gamma_function(alpha_hasting_ratio + num_pxls_1 + num_pxls_2)* calc_gamma_function(alpha_hasting_ratio/2) * \
        calc_gamma_function(alpha_hasting_ratio/2); */
    
    float log_denominator = log(alpha_hasting_ratio) + calc_gamma_function(num_pxls_1) + calc_gamma_function(num_pxls_2) + total_f_sp1 + \
        total_f_sp2 + calc_gamma_function(alpha_hasting_ratio + num_pxls_1 + num_pxls_2) + calc_gamma_function(alpha_hasting_ratio / 2) + \
        calc_gamma_function(alpha_hasting_ratio / 2);

    float lroy1 = log(alpha_hasting_ratio) + calc_gamma_function(num_pxls_1) + calc_gamma_function(num_pxls_2) + total_f_sp1;
    float lroy12 = total_f_sp2 + calc_gamma_function(alpha_hasting_ratio + num_pxls_1 + num_pxls_2) + calc_gamma_function(alpha_hasting_ratio / 2);
    float lroy23 = calc_gamma_function(alpha_hasting_ratio / 2);
   //float hasting_ratio = nominator / denominator;

   float log_hasting_ratio = log_nominator - log_denominator;

   bool is_merge_needed = log_hasting_ratio > -1000;
   //if(is_merge_needed) 
   //printf("%f \n",log_hasting_ratio);
   return is_merge_needed;
}

bool SplitMerge::is_split_needed(int orig_sp_idx, int new_sp_idx, int threshold, int* proposed_seg_map)
{
    // get all pixels of original sp by all channels
    std::vector<unsigned char>  pxls_c0 = m_sp->get_superpixel_by_channel(orig_sp_idx, 0);
    std::vector<unsigned char>  pxls_c1 = m_sp->get_superpixel_by_channel(orig_sp_idx, 1);
    std::vector<unsigned char>  pxls_c2 = m_sp->get_superpixel_by_channel(orig_sp_idx, 2);

// TODO: the function get_superpixel_by_channel should calc it by the new seg_map..
    // get all pixels of sub right pixel
    std::vector<unsigned char>  pxls_11_c0 = get_split_seg_map_superpixel_by_channel(orig_sp_idx, 0);
    std::vector<unsigned char>  pxls_11_c1 = get_split_seg_map_superpixel_by_channel(orig_sp_idx, 1);
    std::vector<unsigned char>  pxls_11_c2 = get_split_seg_map_superpixel_by_channel(orig_sp_idx, 2);

    // get all pixels of sub left pixel 
    std::vector<unsigned char>  pxls_12_c0 = get_split_seg_map_superpixel_by_channel(new_sp_idx, 0);
    std::vector<unsigned char>  pxls_12_c1 = get_split_seg_map_superpixel_by_channel(new_sp_idx, 1);
    std::vector<unsigned char>  pxls_12_c2 = get_split_seg_map_superpixel_by_channel(new_sp_idx, 2);

    // calc marginal likelyhood of original, right and left 
    float f_c0 = calc_marginal_liklelyhoood_of_sp(pxls_c0);
    float f_c1 = calc_marginal_liklelyhoood_of_sp(pxls_c1);
    float f_c2 = calc_marginal_liklelyhoood_of_sp(pxls_c2);

    float f_11_c0 = calc_marginal_liklelyhoood_of_sp(pxls_11_c0);
    float f_11_c1 = calc_marginal_liklelyhoood_of_sp(pxls_11_c1);
    float f_11_c2 = calc_marginal_liklelyhoood_of_sp(pxls_11_c2);

    float f_12_c0 = calc_marginal_liklelyhoood_of_sp(pxls_12_c0);
    float f_12_c1 = calc_marginal_liklelyhoood_of_sp(pxls_12_c1);
    float f_12_c2 = calc_marginal_liklelyhoood_of_sp(pxls_12_c2);

    // channels are i.i.d + log
    float total_f_orig_sp = f_c0 + f_c1 + f_c2;
    float total_f_sp_11 = f_11_c0 + f_11_c1 + f_11_c2;
    float total_f_sp_12 = f_12_c0 + f_12_c1 + f_12_c2;

    float num_pxls_orig = pxls_c0.size();
    float num_pxls_11 = pxls_11_c0.size();
    float num_pxls_12 = pxls_12_c0.size();
    bool is_split_needed = false;
    // only if there exist two sub superpixels, we want to calc there hasting ratio, otherwise it is not relevant 
    if (num_pxls_11 > 0 && num_pxls_12 > 0)
    {
        /* float nominator = calc_gamma_function(num_pxls_1 + num_pxls_2) * total_f_sp12 * calc_gamma_function(alpha_hasting_ratio) * \
             calc_gamma_function(alpha_hasting_ratio / 2 + num_pxls_1) * calc_gamma_function(alpha_hasting_ratio / 2 + num_pxls_2); */

        float log_nominator = log(alpha_hasting_ratio) + calc_gamma_function(num_pxls_11) + total_f_sp_11 + calc_gamma_function(num_pxls_12) + \
            total_f_sp_12;

        /*float denominator = alpha_hasting_ratio * calc_gamma_function(num_pxls_1)*calc_gamma_function(num_pxls_2)* total_f_sp1 * \
            total_f_sp2* calc_gamma_function(alpha_hasting_ratio + num_pxls_1 + num_pxls_2)* calc_gamma_function(alpha_hasting_ratio/2) * \
            calc_gamma_function(alpha_hasting_ratio/2); */

        float log_denominator = +calc_gamma_function(num_pxls_orig) + total_f_orig_sp;


        //float hasting_ratio = nominator / denominator;

        float log_hasting_ratio = log_nominator - log_denominator;
        printf("%f\n",log_hasting_ratio);
        is_split_needed = log_hasting_ratio > -999;
    }

    return is_split_needed;
}

vector<unsigned char>SplitMerge::get_split_seg_map_superpixel_by_channel(int sp_index, int channel) {
    std::vector<unsigned char> sp_pixels;

    //calc start point by channel
    int nPixels = m_sp->get_dim_x() * m_sp->get_dim_y();
    float* start = m_sp->get_image_cpu() + (channel * (nPixels));
    //int* proposed_seg_map = bfs->get_proposed_seg_mat();
    for (int i = 0; i < nPixels; i++)
    {
        if (proposed_seg_map[i] == sp_index) {
            sp_pixels.push_back(*(start + i));
        }
    }

    return sp_pixels;
}

/// <summary>
/// CONTINUE: This function updates two vectors: the first is the original sp_idx and the second is the corresponding new_sp_idx.
/// if in the second vector there is value less than zero, it means that no split is needed.
/// </summary>
void SplitMerge::calc_split_candidates(vector<int>& original_sp_idxs, vector<int>& new_sp_idxs) {
    vector<vector<int>> grid;

    int cols_dim = m_sp->get_dim_x();
    int rows_dim = m_sp->get_dim_y();

    for (int i = 0; i < rows_dim; i++)
    {
        // construct a vector of int
        vector<int> v;
        for (int j = 0; j < cols_dim; j++) {
            v.push_back(i * rows_dim + j);
        }

        // push back above one-dimensional vector
        grid.push_back(v);
    }
    int* seg_cpu = (int*) m_sp->get_seg_cpu();
    int max_sp_idx = get_max_sp_idx();
    vector<int> sp_idxs = m_sp->get_unique_sp_idxs();

    bfs->BfsParallelTwoCenters(grid, seg_cpu,
    m_sp->get_dim_x() * m_sp->get_dim_y(), m_sp->sp_params_cpu, sp_idxs, max_sp_idx);
    //bfs->UpdateProposedSegMapBySplit(seg_cpu);
    
    UpdateProposedSegMapBySplit(seg_cpu);

    std::ofstream file3;
    file3.open("test_array_3.csv");

    for (int i = 0; i < 480; i++) 
    {
        for (int j = 0; j < 320; j++) {
            file3 << proposed_seg_map[i*320+j] <<" , ";
        }
        file3 << '\n';

    }
    file3.close();

    int count = 0;
    //int max_sp_idx = get_max_sp_idx();
    for (int ip = 0; ip<int(max_sp_idx); ip++){
        int idx_sp_1 = ip;
        int idx_sp_2 = ip + int(max_sp_idx);
        if (is_split_needed(idx_sp_1, idx_sp_2, -100, proposed_seg_map))
             new_sp_idxs[count] = idx_sp_2;
             count++;
    }

}

int SplitMerge::get_max_sp_idx() {
    //TODO: change to max element of unique sp_idxs
    vector<int> sp_idxs = m_sp->get_unique_sp_idxs();
    max_sp_idx = *max_element(sp_idxs.begin(), sp_idxs.end());

    return max_sp_idx;
}

void SplitMerge::UpdateProposedSegMapBySplit(int* seg_map) {
    int m_cols_dim = m_sp->get_dim_x();
    int m_rows_dim = m_sp->get_dim_y();
    max_sp_idx = get_max_sp_idx();
    for (int i = 0; i < m_cols_dim* m_rows_dim; i++)
    {
        int row_idx = i / m_cols_dim;
        int col_idx = i % m_cols_dim;

        //cout << row_idx << " , " << col_idx << endl;
        //proposed_seg_map[i] = std::min_element(distance_map[row_idx][col_idx], distance_map_2[row_idx][col_idx]);
        
        if (bfs->distance_map_2[row_idx][col_idx] <= bfs->distance_map[row_idx][col_idx]) {
            proposed_seg_map[i] = seg_map[i];
            //proposed_seg_map[i] = 1;

        }
        else
        {
            if(bfs->distance_map_2[row_idx][col_idx] > bfs->distance_map[row_idx][col_idx])

                proposed_seg_map[i] = int(seg_map[i] + max_sp_idx);
                //proposed_seg_map[i] = 1;
        }


    }



    std::ofstream file3;
    file3.open("test_array_4.csv");

    for (int i = 0; i < 480; i++) 
    {
        for (int j = 0; j < 320; j++) {
            file3 << proposed_seg_map[i*320+j] <<" , ";
        }
        file3 << '\n';

    }
    file3.close();

    Mat seg_mat = Mat::zeros(m_rows_dim, m_cols_dim, CV_32SC1);;
    int idx_of_seg = 0;
    for (int i = 0; i < m_rows_dim; i++)
    {
        for (int j = 0; j < m_cols_dim; j++)
        {
            int sp_idx = int(proposed_seg_map[idx_of_seg]);
         
            seg_mat.at<int>(i, j) = sp_idx;
            idx_of_seg++;
        }

    }
    string work_dir = get_curr_work_dir();
    String seg_idx_path = "/home/uzielr/Roy-CUDA/result/isegMap_split.txt";
    writeMatToFile(seg_mat, seg_idx_path.c_str());
}

int* SplitMerge::get_proposed_seg_map() {
    return proposed_seg_map;
}