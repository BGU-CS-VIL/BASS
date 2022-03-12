# include "bfs_split.h"
#include "utils.h"
#include <fstream>

using namespace cv;
using namespace std;
//using namespace concurrency;

BfsSplit::BfsSplit(int rows_dim, int cols_dim, bool is_split_horizontal) {
    //TODO:Remove
    count_bfs = 0;
    m_rows_dim = rows_dim;
    m_cols_dim = cols_dim;
    vector<bool> r_vis(cols_dim, false);
    
    // initialize visited array with falses
    vector<vector<bool>> vis1(rows_dim, r_vis);
    vis = vis1;
 
    // init direction vectors
    // Direction vectors
     dRow = new const int[4]{ -1, 0, 1, 0 };
     dCol = new const int[4]{ 0, 1, 0, -1 };

     vector<vector<int>> d_map(m_rows_dim, vector<int>(m_cols_dim, 0));
     distance_map = d_map;

     vector<bool> r_vis_2(cols_dim, false);

     // initialize visited array with falses
     vector<vector<bool>> vis2(rows_dim, r_vis_2);
     vis_2 = vis2;

     vector<vector<int>> d_map_2(m_rows_dim, vector<int>(m_cols_dim, 0));
     distance_map_2 = d_map_2;

     setIsHorizontal(is_split_horizontal);
}
//
//BfsSplit::BfsSplit() {
//    cout << "regular constructor" << endl;
//}

BfsSplit::~BfsSplit() {
    if (dRow) // True if drow is not a null pointer
        delete[] dRow;
    if (dCol)
        delete[] dCol;
    
}


//TODO: change to for with few starting points and few sp_indexes
void BfsSplit::BFS(vector<vector<int>> grid,
    int row, int col, int* seg_map, int sp_idx) {
    count_bfs++;

    // Stores indices of the matrix cells
    queue<pair<int, int> > q;

    // Mark the starting cell as visited
    // and push it into the queue
    q.push({ row, col });
    vis[row][col] = true;

    // Iterate while the queue
    // is not empty
    int distance = 0;

    while (!q.empty()) {

        pair<int, int> cell = q.front();
        int x = cell.first;
        int y = cell.second;

        //cout << grid[x][y] << " ";
        if (distance == 0 )
            distance_map[x][y] = distance;
        distance++;

        q.pop();

        // Go to the adjacent cells
        for (int i = 0; i < 4; i++) {

            int adjx = x + dRow[i];
            int adjy = y + dCol[i];

            if (isValidBFS(adjx, adjy)) {
                if (seg_map[int(adjx*m_cols_dim+adjy)] == sp_idx) {
                    q.push({ adjx, adjy });
                    vis[adjx][adjy] = true;
                    distance_map[adjx][adjy] = distance;
                    
                }
            }
        }
    }
}


void BfsSplit::BFS_1(vector<vector<int>> grid,
    int row, int col, int* seg_map, int sp_idx) {
    // Stores indices of the matrix cells
    queue<pair<int, int> > q;


    // Mark the starting cell as visited
    // and push it into the queue
    q.push({ row, col });
    vis[row][col] = true;


    // Iterate while the queue
    // is not empty
    int distance = 0;

    while (!q.empty()) {

        pair<int, int> cell = q.front();
        int x = cell.first;
        int y = cell.second;

        //cout << grid[x][y] << " ";
        if (distance == 0 )
            distance_map[x][y] = distance;

        distance = distance_map[x][y];
        distance++;
        
        if(distance==3)
                    int a=0;
        q.pop();

        // Go to the adjacent cells
        for (int i = 0; i < 4; i++) {

            int adjx = x + dRow[i];
            int adjy = y + dCol[i];
            //printf("ttttttt : %d\n",seg_map[(int)grid[adjx][adjy]] );
            //printf("ttttttt : %d\n",seg_map[(int)(adjx*m_cols_dim+adjy)] );

            if (isValidBFS_1(adjx, adjy)) {

                if (seg_map[(int)(adjx*m_cols_dim+adjy)] == sp_idx) {

                    q.push({ adjx, adjy });
                    vis[adjx][adjy] = true;
                    distance_map[adjx][adjy] = distance;
                }
            }
        }
    }
}
void BfsSplit::BFS_2(vector<vector<int>> grid,
    int row, int col, int* seg_map, int sp_idx) {
    // Stores indices of the matrix cells
    queue<pair<int, int> > q;
    // Mark the starting cell as visited
    // and push it into the queue
    q.push({ row, col });
    vis_2[row][col] = true;
    
    // Iterate while the queue
    // is not empty
    int distance = 0;

    while (!q.empty()) {

        pair<int, int> cell = q.front();
        int x = cell.first;
        int y = cell.second;

        //cout << grid[x][y] << " ";
        if (distance == 0 )
            distance_map_2[x][y] = distance;
        distance = distance_map_2[x][y];

        distance++;
        q.pop();

        // Go to the adjacent cells
        for (int i = 0; i < 4; i++) {

            int adjx = x + dRow[i];
            int adjy = y + dCol[i];

            if (isValidBFS_2(adjx, adjy)) {
                if (seg_map[(int)(adjx*m_cols_dim+adjy)] == sp_idx) {
                    q.push({ adjx, adjy });
                    vis_2[adjx][adjy] = true;
                    distance_map_2[adjx][adjy] = distance;

                }
            }
        }
    }

}

void BfsSplit::setIsHorizontal(bool value) {
    is_horizontal = value;
}
bool BfsSplit::isValidBFS(int row, int col)
{

    // If cell lies out of bounds
    if (row < 0 || col < 0
        || row >= m_rows_dim || col >= m_cols_dim)
        return false;

    // If cell is already visited
    if (vis[row][col])
        return false;

    // Otherwise
    return true;
}

bool BfsSplit::isValidBFS_1(int row, int col) {

    // If cell lies out of bounds
    if (row < 0 || col < 0
        || row >= m_rows_dim || col >= m_cols_dim)
        return false;

    // If cell is already visited
    if (vis[row][col])
        return false;

    // Otherwise
    return true;
}
bool BfsSplit::isValidBFS_2(int row, int col) {

    // If cell lies out of bounds
    if (row < 0 || col < 0
        || row >= m_rows_dim || col >= m_cols_dim)
        return false;

    // If cell is already visited
    if (vis_2[row][col])
        return false;

    // Otherwise
    return true;
}


/*
// This function parallelize bfs search for each sp_idx in seg_map, for one center
void BfsSplit::BfsAllSuperpixels(vector<vector<int>> grid,
    int* seg_map, int n_pxls, superpixel_params* sp_params_cpu) {

    int n = n_pxls* sizeof(int) / sizeof(seg_map[0]);
    vector<int> sp_idxs(seg_map, seg_map + n);//(first elem, last elem)
    vector<int>::iterator it;
    it = std::unique(sp_idxs.begin(), sp_idxs.end());
    sp_idxs.erase(it, sp_idxs.end());
    std::sort(sp_idxs.begin(), sp_idxs.end());
    it = std::unique(sp_idxs.begin(), sp_idxs.end());
    sp_idxs.erase(it, sp_idxs.end());
    
    //for (int i = 0; i < sp_idxs.size(); i++)
    //{
    //    //cout <<"for superpixel idx " << sp_idxs[i] << "center coords are " << x_coords[i] << "," << y_coords[i] << endl;
    //    cout << "for superpixel idx " << sp_idxs[i] << "center coords are column" << ceil(sp_params_cpu[i].mu_s.x) << ", row" << ceil(sp_params_cpu[i].mu_s.y) << endl;
    //}
    

    // parallel for
    for(size_t(0),sp_idxs.size(), [&](size_t value) {
    //parallel_for_each(begin(sp_idxs), end(sp_idxs), [&](int value) {
        int row = ceil(sp_params_cpu[value].mu_s.y);
        int col = ceil(sp_params_cpu[value].mu_s.x);
        int sp_idx = sp_idxs[value];
        //BFS(grid, row, col, seg_map, sp_idx);
        BFS(grid, row, col, seg_map, sp_idx);

        };

    cout << "count_bfs" << count_bfs << endl;
}

*/
void BfsSplit::BfsParallelTwoCenters(vector<vector<int>> grid,
    int* seg_map, int n_pxls, superpixel_params* sp_params_cpu, vector<int> sp_idxs, int max_sp_idx)
{
    int offset_row_1 = 0;
    int offset_row_2 = 0;

    int offset_col_1 = 0;
    int offset_col_2 = 0;

    // the diviation to two superpixels is hprizonntally
    if(is_horizontal) {
        offset_row_1 = +1;
        offset_row_2 = -1;
    }
    else {
        offset_col_1 = +1;
        offset_col_2 = -1;
    }

    
    //parallel_for(size_t(0), sp_idxs.size()*2 , [&](size_t value) {
    //    if (value < sp_idxs.size())
    //    {
    //        //first center coords
    //        int row = ceil(sp_params_cpu[value].mu_s.y) +offset_row_1;
    //        int col = ceil(sp_params_cpu[value].mu_s.x) + offset_col_1;
    //        int sp_idx = sp_idxs[value];
    //        BFS_1(grid, row, col, seg_map, sp_idx);
    //    }
    //    else {
    //        // second center coords
    //        int new_value = value - int(sp_idxs.size());
    //        int row = ceil(sp_params_cpu[new_value].mu_s.y);// +offset_row_2;
    //        int col = ceil(sp_params_cpu[new_value].mu_s.x);// +offset_col_2;
    //        int sp_idx = sp_idxs[new_value];
    //        BFS_2(grid, row, col, seg_map, sp_idx);
    //    }
    //    });
    int count1=0;
    int count2=0;
    for (int value = 0; value < max_sp_idx *2 ; value++)
    {
        printf("This is: %d :  ",value);
        if (value < max_sp_idx)
        {

            //first center coords
            if(sp_params_cpu[value].valid==1){
            printf(" %d\n",value);

            int row = ceil(sp_params_cpu[value].mu_s.y) + offset_row_1;
            int col = ceil(sp_params_cpu[value].mu_s.x) + offset_col_1;
            int sp_idx = value;
            BFS_1(grid, row, col, seg_map, sp_idx);
            count1++;
            }
        }
        else {
            // second center coords
            int new_value = value - int(max_sp_idx);
            if(sp_params_cpu[new_value].valid==1){
                printf(" %d\n",new_value);

                int row = ceil(sp_params_cpu[new_value].mu_s.y);// +offset_row_2;
                int col = ceil(sp_params_cpu[new_value].mu_s.x);// +offset_col_2;


                int sp_idx = new_value;
                BFS_2(grid, row, col, seg_map, sp_idx);
                count2++;
            }
        }
    };
    int *test = new int[m_rows_dim * m_cols_dim]();
        std::ofstream file1;
        std::ofstream file2;

        file1.open("test_array_1.csv");
        file2.open("test_array_2.csv");

        for (int i = 0; i < m_rows_dim; i++) {
            for (int j = 0; j < m_cols_dim; j++) {

                file1 << distance_map[i][j] <<" , ";
                file2 << distance_map_2[i][j] <<" , ";

            }
            file1 << '\n';
            file2 << '\n';


    }
    file1.close();
    file2.close();

    // parallel for over center one and center two
    //TODO: in this way the second parallel for take too much time, understand why ut happens..
        //parallel_for(size_t(0), sp_idxs.size(), [&](size_t value) {
        //parallel_for_each(begin(sp_idxs), end(sp_idxs), [&](int value) {
        //int row = ceil(sp_params_cpu[value].mu_s.y) + offset_row_1;
        //int col = ceil(sp_params_cpu[value].mu_s.x) + offset_col_1;
        //int sp_idx = sp_idxs[value];
        //BFS_1(grid, row, col, seg_map, sp_idx);

    //parallel_for(size_t(0), sp_idxs.size(), [&](size_t value) {
    //    //parallel_for_each(begin(sp_idxs), end(sp_idxs), [&](int value) {
    //    int row = ceil(sp_params_cpu[value].mu_s.y) + offset_row_2;
    //    int col = ceil(sp_params_cpu[value].mu_s.x) + offset_col_2;
    //    int sp_idx = sp_idxs[value];
    //    BFS_2(grid, row, col, seg_map, sp_idx);
    //    });
    //TODO: if i 
}
