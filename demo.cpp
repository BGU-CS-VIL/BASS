#include "Superpixels.h"
#include "split_merge.h"
#include <opencv2/opencv.hpp>
#include "utils.h"
#include <iostream>
#include "bfs_split.h"

static void show_usage(std::string name)
{
    std::cerr << "Usage of " << name << ":\n"
              << "\t-h, --help\t\tShow this help message\n"
              << "\t-i, --img_filename \tSpecify the path of the image to work on\n"
              << "\t-n, --nPixels_on_side \tthe desired number of pixels on the side of a superpixel\n"
              << "\t--i_std \tstd dev for color Gaussians, should be 5<= value <=40. A smaller value leads to more irregular superpixels\n"
              << std::endl;
}


// Set Configuration
static superpixel_options get_sp_options(int nPixels_in_square_side, float i_std){
    superpixel_options opt;
    opt.nPixels_in_square_side = nPixels_in_square_side; 
    opt.i_std = i_std;

    opt.area = opt.nPixels_in_square_side*opt.nPixels_in_square_side;   
    opt.s_std = opt.nPixels_in_square_side;
    opt.prior_count = opt.area*opt.area ;
    opt.calc_cov = true;
    opt.use_hex = false;

    opt.nEMIters = opt.nPixels_in_square_side;
    //opt.nEMIters = 15;
    opt.nInnerIters = 4;
    opt.beta_potts_term = 0;
    return opt;
}


//./Sp_demo -i <img_filename> -n <nPixels_on_side> --i_std <i_std>
int main( int argc, char** argv )
{
    
    // get the image filename, nPixels_in_square_side and i_std
    // the defaults
    string work_dir = get_curr_work_dir();
    String img_path = "/home/uzielr/Roy-CUDA/images/60079.jpg";

    // Control number of superpixels, smaller value will lead to bigger number of superpixels
    int nPixels_in_square_side = 15;

    // ?control color and location weighting
    float i_std = 0.02;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if ((arg == "-h") || (arg == "--help")) {
            show_usage(argv[0]);
            return 0;
        } 
        else if ((arg == "-i") || (arg == "--img_path")) {
            if (i + 1 < argc) { 
                i++;
                img_path = argv[i];
            } else {
                std::cerr << "--img_filename option requires one argument." << std::endl;
                return 1;
            }  
        } 
        else if ((arg == "-n") || (arg == "--nPixels_on_side")) {
            if (i + 1 < argc) { 
                i++;
                nPixels_in_square_side = atoi (argv[i]);
                if (nPixels_in_square_side<3) {
                    std::cerr << "--nPixels_in_square_side option requires nPixels_in_square_side >= 3." << std::endl;
                    return 1;
                }

            } else {
                std::cerr << "--nPixels_on_side option requires one argument." << std::endl;
                return 1;
            }  
        }
        else if (arg == "--i_std") {
            if (i + 1 < argc) { 
                i++;
                i_std = atoi (argv[i]);
                
                if (i_std<5 || i_std>40) {
                    std::cerr << "--i_std option requires 5<= value <=40." << std::endl;
                    return 1;
                }
            } else {
                std::cerr << "--i_std option requires a number." << std::endl;
                return 1;
            }  
        }else{  

        }
    }
    cout << "finish reading the arguments" << endl;
    //TODO: CONFIGURE 
    bool is_merge = true;
    cout << "String is  : " << img_path << endl ;

    Mat image = imread(img_path, IMREAD_COLOR);
    
    // Check for invalid input
    if(! image.data ){
        cout <<  "Could not open or find the image" << std::endl ;
        return -1;
    }

    cout << "finish reading the image" << endl;
    //Part 1: Specify the parameters:
    superpixel_options spoptions = get_sp_options(nPixels_in_square_side, i_std);

    // Part 2 : prepare for segmentation
    int dimy = image.rows;
    int dimx = image.cols;
    Superpixels sp = Superpixels(dimx, dimy, spoptions);
    cout << "finish init sp" << endl;
    cudaDeviceSynchronize();
    sp.load_img((float*)(image.data));

   cout << "finish loading the image" << endl;
   // TODO: not sure about that
   //cudaDeviceSynchronize();

    // Part 3: Do the superpixel segmentation
    clock_t start,finish;
    start = clock();
    sp.calc_seg();
    //cudaDeviceSynchronize();

    cudaError_t err_t = cudaDeviceSynchronize();
    if (err_t) {
        std::cerr << "CUDA error after cudaDeviceSynchronize. " << err_t <<std::endl;
        cudaError_t err = cudaGetLastError();
        return 0;
    }

    // shut down when using merge demo
    /*if (!is_merge) {
        sp.convert_lab_to_rgb();
    }*/

    sp.gpu2cpu();
    
    // Merge demo

    for (int i = 1; i < 1; ++i){
        if (is_merge) {
            SplitMerge merge = SplitMerge(&sp, i_std, 0.1);
            //float hasting_needed = merge.is_merge_needed(0, 1, 0);
            int max_num_of_pairs = sp.get_nSPs();
            vector<short> first(max_num_of_pairs, -1);
            vector<short> second(max_num_of_pairs, -1);
            // modify first and second vectors so that there will be pairs
            // int num_pairs = merge.cpu_calc_merge_candidates(first, second);

            // earase not necessary elements from vectors
            //first.erase(first.begin() + num_pairs, first.end());
            //second.erase(second.begin() + num_pairs, second.end());

            // sp.cpu_merge_all_sp_pairs(first, second); /Roy Remove!

            //put in comment if you want to see the merge result without EM again
            sp.calc_seg();
            // cudaDeviceSynchronize();

            cudaError_t err_t_merge = cudaDeviceSynchronize();
            if (err_t_merge) {
                std::cerr << "CUDA error after cudaDeviceSynchronize megre. " << std::endl;
                cudaError_t err = cudaGetLastError();
                return 0;
            }
            
            // sp.convert_lab_to_rgb();
            //sp.gpu2cpu();
        }
    };
     finish = clock();
    cout<< "Segmentation takes " << ((double)(finish-start)/CLOCKS_PER_SEC) << " sec" << endl;

    // Split Demo
    start = clock();
    SplitMerge split = SplitMerge(&sp, i_std, 0.0001, true);

    int rows_dim = sp.get_dim_y();
    int cols_dim = sp.get_dim_x();

    vector<vector<int>> grid;

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

    // vector<bool> r_vis(cols_dim, false);
    // vector<vector<bool>> vis(rows_dim, r_vis);

    // //split.BFS(grid, vis, 0,0, sp.get_seg_cpu(), 15, rows_dim, cols_dim);
    // //cout << " " <<  endl;
    //// cout << "from another center" << endl;
    // //memset(vis, false, sizeof vis);
    // 
    // vector<vector<bool>> vis_1(rows_dim, r_vis);
    // //split.BFS(grid, vis_1, 13, 6, sp.get_seg_cpu(), 15, rows_dim, cols_dim);

    // //TDO: this section shpuld be in split_merge class in new func
    //BfsSplit bfs_s_1 = BfsSplit(rows_dim, cols_dim);
    // BfsSplit bfs_s_2 = BfsSplit(rows_dim, cols_dim);
    // 
    //// cout << endl;
    // //cout << "try with class" << endl;

    //// bfs_s_1.BFS(grid, 0, 0, sp.get_seg_cpu(), 15);
    // 
    // //cout << endl; 

    //// cout << "try with class 2" << endl;

    // //bfs_s_2.BFS(grid, 13, 6, sp.get_seg_cpu(), 15);
    // //bfs_s_1.BfsAllSuperpixels(grid, sp.get_seg_cpu(),sp.get_dim_x()*sp.get_dim_y(), sp.sp_params_cpu);
     //bfs_s_1.BfsParallelTwoCenters(grid, sp.get_seg_cpu(), sp.get_dim_x() * sp.get_dim_y(), sp.sp_params_cpu);
     //bfs_s_1.UpdateSegMapBySplit(sp.get_seg_cpu());

    //TODO: instead of that think aboout passing on cpu_sp_params, where shpuld be idx for each superpixel
    // Build list of candidates to split and check which are relevant
    int* seg_cpu = sp.get_seg_cpu();
    vector<int> sp_1_list(seg_cpu, seg_cpu + sp.get_dim_x() * sp.get_dim_y());
    vector<int>::iterator ip;
    // Sorting the array
    std::sort(sp_1_list.begin(), sp_1_list.end());

    // Using std::unique
    ip = std::unique(sp_1_list.begin(), sp_1_list.begin() + sp.get_dim_y() * sp.get_dim_x());

    // Resizing the vector so as to remove the undefined terms in the end of the vector
    sp_1_list.resize(std::distance(sp_1_list.begin(), ip));

    // clac hasting ratio results for splitting
    vector<int> sp_2_list(sp_1_list.size(), -1);

    //split.calc_split_candidates(sp_1_list, sp_2_list);
    //sp.cpu_split_superpixels(sp_1_list, sp_2_list, split.get_proposed_seg_map());
     
    //sp.calc_seg(); //NEWWWW

    finish = clock();
     cout << "Split takes " << ((double)(finish - start) / CLOCKS_PER_SEC) << " sec" << endl;

    // Part 4: Save the mean/boundary image 
    
    Mat border_img = sp.get_img_overlaid();
    String fname_res_border = "/home/uzielr/Roy-CUDA/result/img_border.png";
    imwrite(fname_res_border, border_img);
    cout << "saved " << fname_res_border << endl;
    

    Mat mean_img = sp.get_img_cartoon();
    String fname_res_mean = "/home/uzielr/Roy-CUDA/result/img_mean.png";
    imwrite(fname_res_mean, mean_img);
    cout << "saving " << fname_res_mean << endl;

    //// For dubugging - write image with cluster numbers on it and
    Mat seg_mat = Mat::zeros(image.rows, image.cols, CV_32SC1);;
    bool DRAW_CLUSTERS_ON_IMAGE = true;
    Mat img_with_idx = image.clone();
    int* seg_map = sp.get_seg_cpu();
    //bool* seg_map = sp.get_border_cpu();
    int idx_of_seg = 0;
    int last_draw_sp = 0;

    Mat seg_mat_2 = Mat::zeros(image.rows, image.cols, CV_32SC1);;
    int idx_of_seg_2 = 0;
    for (int i = 0; i < image.rows; i++)
    {
        for (int j = 0; j < image.cols; j++)
        {
            int sp_idx = int(seg_map[idx_of_seg_2]);
            seg_mat_2.at<int>(i, j) = sp_idx;
            idx_of_seg_2++;
        }

    }
    String seg_idx_path = "/home/uzielr/Roy-CUDA/result/test_mew.csv";
    writeMatToFile(seg_mat_2, seg_idx_path.c_str());

/*
    for (int i = 0; i < img_with_idx.rows; i++)
    {
        for (int j = 0; j < img_with_idx.cols; j++)
        {
            int sp_idx = int(seg_map[idx_of_seg]);
            seg_mat.at<int>(i, j) = sp_idx;
            
            if (DRAW_CLUSTERS_ON_IMAGE && i % 20 == 0 && j % 20 == 0)
            {
                std::string text = " " + std::to_string(sp_idx) + " ";
                putText(img_with_idx, text, cv::Point(j, i + 10), cv::FONT_HERSHEY_SIMPLEX,
                    0.3,
                    CV_RGB(255, 0, 0), 0.005);
                last_draw_sp = sp_idx;
            }

            idx_of_seg++;
        }
        
    }

    //// Writing segmentation map to file
    String seg_idx_path = work_dir + "/Roy-CUDA/result/segMap.txt";
    writeMatToFile(seg_mat, seg_idx_path.c_str());

    //// save img with clusters
    
    String path_to_save = "/home/uzielr/Roy-CUDA/result/img_with_idx.png";
    imwrite(path_to_save, img_with_idx);
*/

    return 0;
}
