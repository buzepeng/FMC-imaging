// #include <iostream>
// #include <stdio.h>
// #include <fstream>  
// #include <sstream>
// #include <vector>
// #include <math.h>
// #include <iomanip>
// #include <cuda.h>
// #include <cuda_runtime.h>
#include <cufft.h> 
// #include <device_launch_parameters.h>
// #include <thrust/extrema.h>
// #include <thrust/execution_policy.h>
// #include <thrust/transform.h>
// #include <thrust/iterator/counting_iterator.h>
// #include <chrono>
// #include <cstdlib>

// #define TILEW 16
// #define TILEH 16

// #define PI acos(-1)
// std::vector<float> sinc(std::vector<float> x)
// {
//    std::vector<float> y;
//    for (int i = 0; i < x.size(); i++)
//    {
//    	float temp = PI * x[i];
//    	if (temp == 0) {
//    		y.push_back(0.0);
//    	}
//    	else {
//    		y.push_back(sin(temp) / temp);
//    	}
//    }
//    return y;
// }

// std::vector <float> fir1(int n, std::vector<float> Wn)
// {

//    /*
 
//    	参数输入介绍：
//    		n：  对应matlab的fir1里的阶数n
//    		Wn:  对应matlab的fir1里的阶数Wn，但应注意传进
//    				 来的数据应存在一个vector的float数组里。

//    	参数输出介绍：
//    				vector <float>的一个数组，里面存的是长度
//    				为n的滤波器系数。
//    */
   
//    //在截止点的左端插入0（在右边插入1也行）
//    //使得截止点的长度为偶数，并且每一对截止点对应于通带。
//    if (Wn.size() == 1 || Wn.size() % 2 != 0) {
//    	Wn.insert(Wn.begin(), 0.0);
//    }

//    /*
//    	‘ bands’是一个二维数组，每行给出一个 passband 的左右边缘。
//    	（即每2个元素组成一个区间）
//    */
//    std::vector<std::vector <float>> bands;
//    for (int i = 0; i < Wn.size();) {
//    	std::vector<float> temp = { Wn[i], Wn[i + 1] };
//    	bands.push_back(temp);
//    	i = i + 2;
//    }

//    // 建立系数
//    /*
//    	m = [0-(n-1)/2,
//    		 1-(n-1)/2,
//    		 2-(n-1)/2,
//    		 ......
//    		 255-(n-1)/2]
//    	h = [0,0,0......,0]
//    */
//    float alpha = 0.5 * (n - 1);
//    std::vector<float> m;
//    std::vector<float> h;
//    for (int i = 0; i < n; i++) {
//    	m.push_back(i - alpha);
//    	h.push_back(0);
//    }
//    /*
//    	对于一组区间的h计算
//    	left:	一组区间的左边界
//    	right:  一组区间的右边界
//    */
//    for (int i = 0; i < Wn.size();) {
//    	float left = Wn[i];
//    	float right = Wn[i+1];
//    	std::vector<float> R_sin, L_sin;
//    	for (int j = 0; j < m.size(); j++) {
//    		R_sin.push_back(right * m[j]);
//    		L_sin.push_back(left * m[j]);
//    	}
//    	for (int j = 0; j < R_sin.size(); j++) {
//    		h[j] += right * sinc(R_sin)[j];
//    		h[j] -= left * sinc(L_sin)[j];
//    	}

//    	i = i + 2;
//    }

//    // 应用窗口函数，这里和matlab一样
//    // 默认使用hamming，要用别的窗可以去matlab查对应窗的公式。
//    std::vector <float> Win;
//    for (int i = 0; i < n; i++)
//    {
//    	Win.push_back(0.54 - 0.46*cos(2.0 * PI * i / (n - 1)));	//hamming窗系数计算公式
//    	h[i] *= Win[i];
//    }

//    bool scale = true;
//    // 如果需要，现在可以处理缩放.
//    if (scale) {
//    	float left = bands[0][0];
//    	float right = bands[0][1];
//    	float scale_frequency = 0.0;
//    	if (left == 0)
//    		scale_frequency = 0.0;
//    	else if (right == 1)
//    		scale_frequency = 1.0;
//    	else
//    		scale_frequency = 0.5 * (left + right);

//    	std::vector<float> c;
//    	for (int i = 0; i < m.size(); i++) {
//    		c.push_back(cos(PI * m[i] * scale_frequency));
//    	}
//    	float s = 0.0;
//    	for (int i = 0; i < h.size(); i++) {
//    		s += h[i] * c[i];
//    	}
//    	for (int i = 0; i < h.size(); i++) {
//    		h[i] /= s;
//    	}
//    }
//    return h;
// }

// __device__ cufftComplex operator * (cufftComplex a, cufftComplex b){
//     cufftComplex res;
//     res.x = (a.x*b.x - a.y*b.y);
//     res.y = (a.x*b.y + a.y*b.x);
//     return res;
// }
// __device__ cufftComplex operator * (cufftComplex a, float b){
//     cufftComplex res;
//     res.x = a.x*b;
//     res.y = a.y*b;
//     return res;
// }
// __device__ cufftComplex operator / (cufftComplex a, float b){
//     cufftComplex res;
//     res.x = a.x/b;
//     res.y = a.y/b;
//     return res;
// }
// __device__ cufftComplex operator + (cufftComplex a, cufftComplex b){
//     cufftComplex res;
//     res.x = a.x+b.x;
//     res.y = a.y+b.y;
//     return res;
// }

// __global__ void frequency_filtering_kernel(cufftComplex* signals, cufftComplex* filter, cufftComplex *HilbertMat, int WaveNum, int WaveLength){
//     int ix = blockDim.x*blockIdx.x+threadIdx.x;
//     int iy = blockDim.y*blockIdx.y+threadIdx.y;
//     int idx = iy*WaveLength+ix;

//     if(iy<WaveNum && ix<WaveLength){
//         HilbertMat[idx] = signals[idx]*filter[ix];
//     }
// }

// __global__ void transposeCoalesced(short *idata, short *odata, int row, int col)
// {
//     __shared__ int block[TILEH][TILEW];

//     int i = blockIdx.y * blockDim.y + threadIdx.y;
//     int j = blockIdx.x * blockDim.x + threadIdx.x;
//     if (i < row && j < col) {
//             block[threadIdx.y][threadIdx.x] = idata[i * col + j];
//             __syncthreads();
//             odata[j * row + i] = block[threadIdx.y][threadIdx.x];
//     }
// }

// struct thrust_imaging{
//     float *offLineFmc;
//     short *Tof;
//     cufftComplex *FmcMatHilbert;
//     int WaveNum, WaveLength, iWaveLength, col_tof;

//     thrust_imaging(float *_offLineFmc, short *_Tof, cufftComplex *_FmcMatHilbert, int _WaveNum, int _WaveLength,int _iWaveLength, int _col_tof):offLineFmc(_offLineFmc), Tof(_Tof), FmcMatHilbert(_FmcMatHilbert), WaveNum(_WaveNum), WaveLength(_WaveLength), iWaveLength(_iWaveLength), col_tof(_col_tof){};
//     __device__
//     float operator()(int i){
//         cufftComplex complex_sum;
//         complex_sum.x = 0;
//         complex_sum.y = 0;
//         for(int s = 0;s<WaveNum;s++){
//             int tIndex = int(offLineFmc[iWaveLength*s]-1), rIndex = int(offLineFmc[iWaveLength*s+1]-1);
//             int trTofIndex = int(Tof[tIndex*col_tof+i]+Tof[rIndex*col_tof+i]), fmc_ind = s*WaveLength+trTofIndex-1;
//             cufftComplex temp = FmcMatHilbert[fmc_ind];
//             complex_sum = complex_sum + temp;
//         }
//         return cuCabsf(complex_sum);
//     }
// };

class FmcImaging{

    public:
        short *h_iTof, *d_iTof;
        float *h_offLineFmc, *d_offLineFmc,  *d_TfmImage;
        cufftComplex *d_Hilbert, *d_f_offLineFmc, *d_f_filter;
        cufftHandle planForward, planInverse;

        int WaveNum, WaveLength, iWaveLength, row_tof, col_tof, NZ, NX, taps;

        FmcImaging(int _WaveNum, int _WaveLength, int _iWaveLength, int _row_tof, int _col_tof, int _NZ, int _NX, int _taps);
        ~FmcImaging();
        void transpose(short *iTof, int row_tof, int col_tof);
        void get_filter(float f1, float f2, cufftComplex* filter);
        void filtering(float* offLineFmc, cufftComplex* f_filter, cufftComplex* f_offLineFmc);
        void hilbert_transform(cufftComplex *f_offLineFmc, cufftComplex* HilbertFilter);
        void imaging(float* offLineFmc_ind, short* Tof, cufftComplex* FmcMatHilbert, float* TfmImage, float MindB);
        void read_FMC(std::string filepath, float* input, int row, int col);
        void read_TOF(std::string filepath, short* input, int row, int col);
        //save TfmImage to txt
        void save_result_to_txt(std::string filepath, float* d_result);
};
