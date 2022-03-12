#define _USE_MATH_DEFINES
#include <cmath>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include "Superpixels.h"
#include <stdio.h>
#include <random>
#include <numeric>
#include "utils.h"
#include "bfs_split.h"

using namespace cv;
using namespace std;

class SplitMerge {

	//Data Members
	Superpixels* m_sp;
	float m_desired_samples_mean_IG;
	float m_desired_samples_var_IG;
	
	// IG params
	float a_0;
	float b_0;
	float k_0;
	float V_0;
	float alpha_hasting_ratio;

	// Normal params
	float mean_normal;
	float variance_normal;
	
	//Split 
	BfsSplit* bfs;
	int* proposed_seg_map;

	float calc_bn(std::vector<unsigned char>  pixels);
	float calc_vn(int sp_index);
	float calc_merge_hasting_ratio(std::vector<unsigned char>  pix_sp1, std::vector<unsigned char> pix_sp_2);
	void UpdateProposedSegMapBySplit(int* seg_map); //TODO: maybe add proposed segmap pointer to params?
	int max_sp_idx;

public:
	SplitMerge();
	SplitMerge(Superpixels* sp, float desired_samples_mean_IG, float desired_samples_var_IG, bool is_split_horizontal = false);
	~SplitMerge();
	float calc_marginal_liklelyhoood_of_sp(std::vector<unsigned char>  pixels);
	bool is_merge_needed(int first_sp_idx, int second_sp_idx, int threshold = 1);
	int cpu_calc_merge_candidates(vector<short>& first, vector<short>& second);
	void calc_merge_candidates();
	//Split
	bool is_split_needed(int orig_sp_idx, int new_sp_idx, int threshold , int* proposed_seg_map);
	vector<unsigned char> get_split_seg_map_superpixel_by_channel(int sp_index, int channel);
	void calc_split_candidates(vector<int>& original_sp_idxs, vector<int>& new_sp_idxs);
	int* get_proposed_seg_map();
	int get_max_sp_idx();
	// for super pixel and current direction, calc his smallest adjacent superpixel
	//void choose_merge_candidate_by_direction(int sp_idx, int direction);

	// calc for all superpixels and add to dictionary
	//void calc_candidates_to_merge();

	// go over candidates dictionary and remove unnessecary candidates
	//void calc_valid_merges();

	// merge candidates, return new K and clean dictionary values(not  keys
	//int merge_valid_candidates();
};
