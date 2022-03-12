#pragma once

/* this class used for splitting superpixels into subsuperpixels. 
 It is done simultaneously by all superpixels tpgethere. 
 input is the segmentation map and output is the distance map 
from each startpoint whithin each superpixeks for all superpixels
Class methods currently are implemented on cpu*/

#include <cmath>
#include <iostream>
#include "Superpixels.h"
#include <stdio.h>
#include <opencv2/imgproc.hpp>
#include <utility>
#include <set>
#include <algorithm>
#include <random> 
# include <queue>
//#include <ppl.h>

using namespace cv;
using namespace std;

class BfsSplit {
	//TODO:REmove
	int count_bfs;

	// Data Members
	int m_rows_dim;
	int m_cols_dim;

	// Direction vectors
	const int* dRow;
	const int* dCol;

	//not sure : may be it should be pass as parameter in split func int* seg_map
	// Not initialized from outside
	// not sure: may be it shpuld be pass as parameter in split methodvector<vector<int>> grid;
	vector<vector<bool>> vis;
	// not sure maybe it shpuld returned by split method vector<vector<int>> distance_map;
	
	

	vector<vector<bool>> vis_2;
	

	bool is_horizontal;
	int count_split_iteration;
	int max_sp_idx;

public:
	//BfsSplit();
	//holds the distance from the center fo reach pixel in the grid
	vector<vector<int>> distance_map;
	vector<vector<int>> distance_map_2;

	BfsSplit(int rows_dim, int cols_dim, bool is_split_horizontal = false);
	~BfsSplit();

	//Methods 
	//split(startinpoints?, sp_idx?)
	bool isValidBFS(int row, int col);
	void BFS(vector<vector<int>> grid,
		int row, int col, int* seg_map, int sp_idx);
	void BFS_1(vector<vector<int>> grid,
		int row, int col, int* seg_map, int sp_idx);
	void BFS_2(vector<vector<int>> grid,
		int row, int col, int* seg_map, int sp_idx);
	bool isValidBFS_1(int row, int col);
	bool isValidBFS_2(int row, int col);
	void BfsAllSuperpixels(vector<vector<int>> grid,
		 int* seg_map, int n_pxls, superpixel_params* sp_params_cpu);
	void BfsParallelTwoCenters(vector<vector<int>> grid,
		int* seg_map, int n_pxls, superpixel_params* sp_params_cpu, vector<int> sp_idxs, int max_sp_idx);
	void setIsHorizontal(bool value);
	//clearfields() decide if we want to clear field or just to remove the instances of the class..
};
