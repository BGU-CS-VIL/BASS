using namespace std;

#define BLCK_SIZE 6
#define MAX_BLOCK_SIZE 256
#define THREADS_PER_BLOCK 512

#include "RgbLab.h"
#include <stdio.h>
#include <math.h>



__host__ void Rgb2Lab(uchar3* image_gpu, float* image_gpu_double, int nPixels){
	int num_block = ceil( double(nPixels) / double(THREADS_PER_BLOCK) ); 
	dim3 ThreadPerBlock(THREADS_PER_BLOCK,1);
	dim3 BlockPerGrid(num_block,1);
   
	rgb_to_lab<<<BlockPerGrid,ThreadPerBlock>>>(image_gpu,image_gpu_double,nPixels);
}

__global__ void rgb_to_lab(uchar3* image_gpu, float* image_gpu_double, int nPts) {
	int t = threadIdx.x + blockIdx.x * blockDim.x;  
	if (t>=nPts) return;

	uchar3 p = image_gpu[t];

	double sB = (double)p.x;
	double sG = (double)p.y;
	double sR = (double)p.z;	

	if (sR!=sR || sG!=sG || sB!=sB) return;

	//RGB (D65 illuninant assumption) to XYZ conversion
	double R = sR/255.0;
	double G = sG/255.0;
	double B = sB/255.0;

	double r, g, b;
	if(R <= 0.04045)	r = R/12.92;
	else				r = powf((R+0.055)/1.055,2.4);
	if(G <= 0.04045)	g = G/12.92;
	else				g = powf((G+0.055)/1.055,2.4);
	if(B <= 0.04045)	b = B/12.92;
	else				b = powf((B+0.055)/1.055,2.4);

	double X = r*0.4124564 + g*0.3575761 + b*0.1804375;
	double Y = r*0.2126729 + g*0.7151522 + b*0.0721750;
	double Z = r*0.0193339 + g*0.1191920 + b*0.9503041;

	
	//convert from XYZ to LAB

	double epsilon = 0.008856;	//actual CIE standard
	double kappa   = 903.3;		//actual CIE standard

	double Xr = 0.950456;	//reference white
	double Yr = 1.0;		//reference white
	double Zr = 1.088754;	//reference white

	double xr = X/Xr;
	double yr = Y/Yr;
	double zr = Z/Zr;

	double fx, fy, fz;
	if(xr > epsilon)	fx = powf(xr, 1.0/3.0);
	else				fx = (kappa*xr + 16.0)/116.0;
	if(yr > epsilon)	fy = powf(yr, 1.0/3.0);
	else				fy = (kappa*yr + 16.0)/116.0;
	if(zr > epsilon)	fz = powf(zr, 1.0/3.0);
	else				fz = (kappa*zr + 16.0)/116.0;

	float lval = 116.0*fy-16.0;
	float aval = 500.0*(fx-fy);
	float bval = 200.0*(fy-fz);

	image_gpu_double[3*t] = lval/(-100);
	image_gpu_double[3*t+1] = aval/100;
	image_gpu_double[3*t+2] = bval/100;

	//image_gpu_double[3*t] = lval/100;
	//image_gpu_double[3*t+1] = aval/100;
	//image_gpu_double[3*t+2] = bval/100;
}


__host__ void Lab2Rgb(uchar3* image_gpu, float* image_gpu_double, int nPixels){
	int num_block = ceil( double(nPixels) / double(THREADS_PER_BLOCK) ); 
	dim3 ThreadPerBlock(THREADS_PER_BLOCK,1);
	dim3 BlockPerGrid(num_block,1);

	lab_to_rgb<<<BlockPerGrid,ThreadPerBlock>>>(image_gpu,image_gpu_double,nPixels);
}

__global__ void lab_to_rgb( uchar3* image_gpu, float* image_gpu_double, int nPixels) {
	int t = threadIdx.x + blockIdx.x * blockDim.x;  
	if (t>=nPixels) return;

    double L = image_gpu_double[3*t] *(-100);//* 100;
	double La = image_gpu_double[3*t+1]*100;//* 100;
	double Lb = image_gpu_double[3*t+2]*100; //*100 ;
	
	if (L!=L || La!=La || Lb!=Lb) return;

    //convert from LAB to XYZ
    double fy = (L+16) / 116;
	double fx = La/500 + fy;
	double fz = fy-Lb/200;

	double x,y,z;
	double xcube = powf(fx,3); 
	double ycube = powf(fy,3); 
	double zcube = powf(fz,3); 
	if (ycube>0.008856)	y = ycube;
	else				y = (fy-16.0/116.0)/7.787;
	if (xcube>0.008856)	x = xcube;
	else				x = (fx - 16.0/116.0)/7.787;
	if (zcube>0.008856)	z = zcube;
	else				z = (fz - 16.0/116.0)/7.787;

	double X = 0.950456 * x;
	double Y = 1.000 * y;
	double Z = 1.088754 * z;

	//convert from XYZ to rgb
	double R = X *  3.2406 + Y * (-1.5372) + Z * (-0.4986);
	double G = X * -0.9689 + Y * 1.8758 + Z *  0.0415;
	double B = X *  0.0557 + Y * (-0.2040) + Z * 1.0570;

	double r,g,b;
	if (R>0.0031308) r = 1.055 * (powf(R,(1.0/2.4))) - 0.055;
	else             r = 12.92 * R;
	if (G>0.0031308) g = 1.055 * ( powf(G,(1.0/2.4))) - 0.055;
	else             g = 12.92 * G;
	if (B>0.0031308) b = 1.055 * (powf(B, (1.0/2.4))) - 0.055;
	else             b = 12.92 * B;

	uchar3 p;
	
	p.x =  min(255.0, b * 255.0);
	p.y =  min(255.0, g * 255.0);
	p.z =  min(255.0, r * 255.0);
	p.x =  max(0.0, double(p.x));
    p.y =  max(0.0, double(p.y));
    p.z =  max(0.0, double(p.z));
   
    image_gpu[t] = p;
}