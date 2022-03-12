#include <dirent.h>

#include "Superpixels.h"
#include <opencv2/opencv.hpp>
#include "utils.h"
#include <iostream>
#include <fstream>


static void show_usage(std::string name){
    std::cerr << "Usage of " << name << ":\n"
              << "\t-h, --help \tShow this help message\n"
              << "\t-d, --image_direc \tthe directory to work on\n"
              << "\t-n, --nPixels_on_side \tthe desired number of pixels on the side of a superpixel\n"
              << "\t--i_std std dev for color Gaussians, should be 0.01<= value <=0.05. A smaller value leads to more irregular superpixels\n"
              << "\t--im_size resizing input images \n"
              << "\t--beta beta value (Potts) \n"
              << "\t--alpha alpha value (Hasting) \n"
              << std::endl;
}

// Set Configuration
static superpixel_options get_sp_options(int nPixels_in_square_side, float i_std,float beta, float alpha_hasting){
    superpixel_options opt;
    opt.nPixels_in_square_side = nPixels_in_square_side;
    opt.i_std = i_std;
    opt.beta_potts_term = beta;
    opt.area = opt.nPixels_in_square_side*opt.nPixels_in_square_side;
    opt.s_std = opt.nPixels_in_square_side;
    opt.prior_count = opt.area*opt.area ;
    opt.calc_cov = true;
    opt.use_hex = false;
    opt.alpha_hasting = alpha_hasting;

    opt.nEMIters = opt.nPixels_in_square_side;
    //opt.nEMIters = 15;
    opt.nInnerIters = 4;
    return opt;
}

// ./Sp_demo_for_direc.py -d <directory_name> -i_std ... -n... --imgs_of_the_same_size
int main( int argc, char** argv )
{

    // get the image filename, nPixels_in_square_side and i_std
    // the defaults
    const char* direc = "image/";
    int nPixels_in_square_side = 15;
    int im_size =0;
    float i_std = 0.018;
    bool same_size = false;
    float beta = 0.5;
    float alpha = 0.5;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if ((arg == "-h") || (arg == "--help")) {
            show_usage(argv[0]);
            return 0;
        }
        else if ((arg == "-d") || (arg == "--image_direc")) {
            if (i + 1 < argc) {
                i++;
                direc = argv[i];
            } else {
                std::cerr << "--img_filename option requires one argument." << std::endl;
                return 1;
            }
        }
        else if ((arg == "-n") || (arg == "--nPixels_on_side")) {
            if (i + 1 < argc) {
                i++;
                nPixels_in_square_side = atoi (argv[i]);
                if (nPixels_in_square_side<1) {
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
                i_std = atof (argv[i]);

            } else {
                std::cerr << "--i_std option requires a number." << std::endl;
                return 1;
            }
             }
        else if (arg == "--im_size") {
            if (i + 1 < argc) {
                i++;
                im_size = atof (argv[i]);

            } else {
                std::cerr << "--im_size option requires a number." << std::endl;
                return 1;
            }
             }
        else if (arg == "--beta") {
            if (i + 1 < argc) {
                i++;
                beta = atof (argv[i]);
                printf("Beta: %f", beta);
            } else {
                std::cerr << "--beta option requires a number." << std::endl;
                return 1;
            }
    }
            else if (arg == "--alpha") {
            if (i + 1 < argc) {
                i++;
                alpha = atof (argv[i]);
            } else {
                std::cerr << "--alpha option requires a number." << std::endl;
                return 1;
            }
    }
  }

    DIR *dpdf;
    struct dirent *epdf;

    superpixel_options spoptions = get_sp_options(nPixels_in_square_side,i_std,beta, alpha);
    int count = 0;
    double timer=0;
    //Superpixels sp = Superpixels(0,0,spoptions);
    dpdf = opendir(direc);
    if (dpdf != NULL){

        while (epdf = readdir(dpdf)){
            String img_name =  epdf->d_name;

            String filename = string(direc) + img_name;


            Mat image1 = imread(filename, IMREAD_COLOR);
            if(! image1.data ) continue;
            Mat image;
            if (im_size==0)
            {
                image = image1;
            }
            else
            {
                resize(image1, image, cv::Size(im_size,im_size));
            }
            cout << "Filename: " << filename <<endl;
            Superpixels sp = Superpixels(image.cols, image.rows, spoptions);
            cudaDeviceSynchronize();

            //cout << "finish init sp" << endl;
            sp.load_img((float*)(image.data));

            //cout << "finish loading the image" << endl;

            // Part 3: Do the superpixel segmentation
            clock_t start,finish;
            start = clock();
            sp.calc_seg();
            //cudaDeviceSynchronize();
            sp.convert_lab_to_rgb();
            sp.gpu2cpu();

            finish = clock();
            cout<< "Segmentation takes " << ((double)(finish-start)/CLOCKS_PER_SEC) << " sec" << endl;
            timer += (double)(finish - start)/CLOCKS_PER_SEC;
            // Part 4: Save the mean/boundary image
            cudaError_t err_t = cudaDeviceSynchronize();
            if (err_t){
                std::cerr << "CUDA error after cudaDeviceSynchronize." << err_t << std::endl;
                return 0;
            }

            String img_number =  img_name.substr (0, img_name.find("."));


            Mat border_img = sp.get_img_overlaid();
            String fname_res_border = "../result/border_"+img_number+".png";
            imwrite(fname_res_border, border_img);
            cout << "saved " << fname_res_border << endl;


            Mat mean_img = sp.get_img_cartoon();
            String fname_res_mean = "../result/mean_"+img_number+".png";
            imwrite(fname_res_mean, mean_img);
            cout << "saving " << fname_res_mean << endl;

            count++;
            String seg_idx_path = "../result/"+img_number+".csv";

            int* seg_map = sp.get_seg_cpu();
            int idx_of_seg = 0;
            int last_draw_sp = 0;
            int m_rows_dim = image.rows;
            int m_cols_dim = image.cols;
            int *test = new int[m_rows_dim * m_cols_dim]();
            std::ofstream file1;
	    cout << seg_idx_path ;
            file1.open(seg_idx_path);
            int idx = 0;
            for (int i = 0; i < m_rows_dim; i++)
            {

                for (int j = 0; j < m_cols_dim; j++)
                {

                    if (j==m_cols_dim-1)
                            file1 << seg_map[idx];
		    else
                    	file1 << seg_map[idx] <<",";
                    idx++;
                }
                file1 << '\n';


            }
            file1.close();
            cudaDeviceReset();
        }
        cudaDeviceReset();
        cout << "MeanTime " << timer/count << endl;
    }
    }
