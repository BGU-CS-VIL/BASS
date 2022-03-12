#ifdef WIN32
#include <direct.h>
#define GetCurrentDir _getcwd
#else
#include <unistd.h>
#define GetCurrentDir getcwd
#endif

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <cmath>
using namespace cv;
using namespace std;

void save_image(cv::Mat img, String img_name, String img_path);
float calc_gamma_function(float x);
std::string get_current_dir();
std::string get_curr_work_dir();
void writeMatToFile(cv::Mat& m, const char* filename);