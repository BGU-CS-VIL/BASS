#include "utils.h"
#include <fstream>

using namespace cv;
using namespace std;

// write mat of type int
// TODO: check if there is any option to discover by runtime the type f matrix 
void writeMatToFile(cv::Mat& m, const char* filename)
{
	ofstream fout(filename);

	if (!fout)
	{
		cout << "File Not Opened" << endl;  return;
	}

	for (int i = 0; i < m.rows; i++)
	{
		for (int j = 0; j < m.cols; j++)
		{
			fout << m.at<int>(i, j) << ",";
		}
		fout << endl;
	}

	fout.close();
}

void save_image(Mat img, String img_name, String img_path = "Debug") {
	String full_path = img_path + img_name;
	imwrite(full_path, img);
}

// return log of gamma function
float calc_gamma_function(float x) {
	float result;
	result = lgamma(x);
	//printf("gamma(%f) = %f\n", x, result);
	return result;
}

std::string get_current_dir() {
	char buff[FILENAME_MAX]; //create string buffer to hold path
	GetCurrentDir(buff, FILENAME_MAX);
	string current_working_dir(buff);
	return current_working_dir;
}

std::string get_curr_work_dir() {
	string full_curr_path = get_current_dir();
	std::size_t pos = full_curr_path.find("./out");
	std::string work_dir = full_curr_path.substr(0, pos);

	return work_dir;
}