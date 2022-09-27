#include <iostream>
#include <stdio.h>
#include <fstream>  
#include <sstream>
#include <vector>
#include <math.h>
#include <chrono>
#include <iomanip>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cufft.h> 
#include <chrono>
#include <device_launch_parameters.h>
#include <thrust/extrema.h>
#include <thrust/execution_policy.h>
#include <thrust/transform.h>
#include <chrono>
#include <cstdlib>

#define TILEW 16
#define TILEH 16

#define PI acos(-1)
std::vector<float> sinc(std::vector<float> x)
{
   std::vector<float> y;
   for (int i = 0; i < x.size(); i++)
   {
   	float temp = PI * x[i];
   	if (temp == 0) {
   		y.push_back(0.0);
   	}
   	else {
   		y.push_back(sin(temp) / temp);
   	}
   }
   return y;
}

std::vector <float> fir1(int n, std::vector<float> Wn)
{

   /*
   	未写排错  检查输入有需要自己进行完善
   	原matlab函数fir(n, wn)	【函数默认使用hamming】

   	参数输入介绍：
   		n：  对应matlab的fir1里的阶数n
   		Wn:  对应matlab的fir1里的阶数Wn，但应注意传进
   				 来的数据应存在一个vector的float数组里。

   	参数输出介绍：
   				vector <float>的一个数组，里面存的是长度
   				为n的滤波器系数。
   */
   
   //在截止点的左端插入0（在右边插入1也行）
   //使得截止点的长度为偶数，并且每一对截止点对应于通带。
   if (Wn.size() == 1 || Wn.size() % 2 != 0) {
   	Wn.insert(Wn.begin(), 0.0);
   }

   /*
   	‘ bands’是一个二维数组，每行给出一个 passband 的左右边缘。
   	（即每2个元素组成一个区间）
   */
   std::vector<std::vector <float>> bands;
   for (int i = 0; i < Wn.size();) {
   	std::vector<float> temp = { Wn[i], Wn[i + 1] };
   	bands.push_back(temp);
   	i = i + 2;
   }

   // 建立系数
   /*
   	m = [0-(n-1)/2,
   		 1-(n-1)/2,
   		 2-(n-1)/2,
   		 ......
   		 255-(n-1)/2]
   	h = [0,0,0......,0]
   */
   float alpha = 0.5 * (n - 1);
   std::vector<float> m;
   std::vector<float> h;
   for (int i = 0; i < n; i++) {
   	m.push_back(i - alpha);
   	h.push_back(0);
   }
   /*
   	对于一组区间的h计算
   	left:	一组区间的左边界
   	right:  一组区间的右边界
   */
   for (int i = 0; i < Wn.size();) {
   	float left = Wn[i];
   	float right = Wn[i+1];
   	std::vector<float> R_sin, L_sin;
   	for (int j = 0; j < m.size(); j++) {
   		R_sin.push_back(right * m[j]);
   		L_sin.push_back(left * m[j]);
   	}
   	for (int j = 0; j < R_sin.size(); j++) {
   		h[j] += right * sinc(R_sin)[j];
   		h[j] -= left * sinc(L_sin)[j];
   	}

   	i = i + 2;
   }

   // 应用窗口函数，这里和matlab一样
   // 默认使用hamming，要用别的窗可以去matlab查对应窗的公式。
   std::vector <float> Win;
   for (int i = 0; i < n; i++)
   {
   	Win.push_back(0.54 - 0.46*cos(2.0 * PI * i / (n - 1)));	//hamming窗系数计算公式
   	h[i] *= Win[i];
   }

   bool scale = true;
   // 如果需要，现在可以处理缩放.
   if (scale) {
   	float left = bands[0][0];
   	float right = bands[0][1];
   	float scale_frequency = 0.0;
   	if (left == 0)
   		scale_frequency = 0.0;
   	else if (right == 1)
   		scale_frequency = 1.0;
   	else
   		scale_frequency = 0.5 * (left + right);

   	std::vector<float> c;
   	for (int i = 0; i < m.size(); i++) {
   		c.push_back(cos(PI * m[i] * scale_frequency));
   	}
   	float s = 0.0;
   	for (int i = 0; i < h.size(); i++) {
   		s += h[i] * c[i];
   	}
   	for (int i = 0; i < h.size(); i++) {
   		h[i] /= s;
   	}
   }
   return h;
}

__device__ cufftComplex operator * (cufftComplex a, cufftComplex b){
    cufftComplex res;
    res.x = (a.x*b.x - a.y*b.y);
    res.y = (a.x*b.y + a.y*b.x);
    return res;
}
__device__ cufftComplex operator + (cufftComplex a, cufftComplex b){
    cufftComplex res;
    res.x = a.x+b.x;
    res.y = a.x+b.y;
    return res;
}

__global__ void FreqDomainMul(cufftComplex* signals, cufftComplex* filter, int WaveNum, int WaveLength){
    int ix = blockDim.x*blockIdx.x+threadIdx.x;
    int iy = blockDim.y*blockIdx.y+threadIdx.y;
    int idx = iy*WaveLength+ix;

    if(iy<WaveNum && ix<WaveLength){
        signals[idx] = signals[idx]*filter[ix];
    }
}

__global__ void transposeCoalesced(short *idata, short *odata, int row, int col)
{
    __shared__ int block[TILEH][TILEW];

    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < row && j < col) {
            block[threadIdx.y][threadIdx.x] = idata[i * col + j];
            __syncthreads();
            odata[j * row + i] = block[threadIdx.y][threadIdx.x];
    }
}

struct thrust_imaging{
    float *offLineFmc;
    short *Tof;
    cufftComplex *FmcMatHilbert;
    int WaveNum, iWaveLength, WaveLength, col_tof;

    thrust_imaging(float *_offLineFmc, short *_Tof, cufftComplex *_FmcMatHilbert, int _WaveNum, int _iWaveLength, int _WaveLength, int _col_tof):offLineFmc(_offLineFmc), Tof(_Tof), FmcMatHilbert(_FmcMatHilbert), WaveNum(_WaveNum), iWaveLength(_iWaveLength), WaveLength(_WaveLength), col_tof(_col_tof){};
    __device__
    float operator()(int i){
        cufftComplex complex_sum;
        complex_sum.x = 0;
        complex_sum.y = 0;
        for(int s = 0;s<WaveNum;s++){
            int tIndex = int(offLineFmc[s*iWaveLength]-1), rIndex = int(offLineFmc[s*iWaveLength+1]-1);
            int trTofIndex = int(Tof[tIndex*col_tof+i]+Tof[rIndex*col_tof+i]), fmc_ind = s*WaveLength+trTofIndex-1;
            cufftComplex temp = FmcMatHilbert[fmc_ind];
            complex_sum = complex_sum + temp;
        }
        return cuCabsf(complex_sum);
    }
};

struct zero_phase_filt{
    cufftComplex *filter, *signals;
    int WaveLength;
    zeros_phase_filt(cufftComplex* _signals, cufftComplex* _filter, int _WaveLength)signals(_signals),filter(_filter),WaveLength(_WaveLength){};
    __device__ 
    cufftComplex operator()(int i){
        return signals[i]*pow(cuCabsf(filter[i%WaveLength]));
    }
}

static float sinc(const float x)
{
    if (x == 0)
        return 1;

    return sin(M_PI * x) / (M_PI * x);
}


class FmcImaging{

    public:
        short *h_iTof, *d_iTof;
        float *h_offLineFmc, *d_offLineFmc, *d_TfmImage;
        cufftComplex *d_H, *d_Hilbert, *d_f_offLineFmc, d_FmcMatHilbert;
        cufftHandle planForward, planInverse;

        int WaveNum, WaveLength, iWaveLength, row_tof, col_tof, NZ, NX;

        FmcImaging(int _WaveNum, int _WaveLength, int _iWaveLength, int _row_tof, int _col_tof, int _NZ, int _NX);
        ~FmcImaging();
        void transpose(short *iTof, int row_tof, int col_tof);
        void get_freq_filter(float low, float high, int taps, cufftComplex* H);
        void freq_domain_filtering(float* offLineFmc, cufftComplex* f_offLineFmc, cufftComplex* H);
        void hilbert_transform(cufftComplex* FmcMatHilbert, cufftComplex* HilbertFilter);
        void imaging(float* offLineFmc, short* Tof, cufftComplex* FmcMatHilbert, float* TfmImage, float MindB);
        //read offLineFmcMat or iTof to gpu
        template<typename T>
        void read_data(std::string filepath, T* gpu_addr, int row, int col);
        //save TfmImage to txt
        void save_result_to_txt(std::string filepath, float* d_result);
};

FmcImaging::FmcImaging(int _WaveNum, int _WaveLength, int _iWaveLength, int _row_tof, int _col_tof, int _NZ, int _NX):WaveNum(_WaveNum),WaveLength(_WaveLength),iWaveLength(_iWaveLength),row_tof(_row_tof),col_tof(_col_tof),NZ(_NZ),NX(_NX)
{
    //alloc memory
    h_iTof = (short*)malloc(row_tof*col_tof*sizeof(short));

    cudaMallocHost((void **)&h_offLineFmc, WaveNum*iWaveLength*sizeof(float));

    cudaMalloc((void**) &d_iTof, row_tof * col_tof * sizeof(short));

    cudaMalloc((void**) &d_offLineFmc, iWaveLength * WaveNum *sizeof(float));
    cudaMalloc((void**) &d_TfmImage, NZ * NX * sizeof(float));

    cudaMalloc((void**) &d_H, WaveLength * sizeof(cufftComplex));
    cudaMalloc((void**) &d_Hilbert, WaveLength * sizeof(cufftComplex));
    cudaMalloc((void**) &d_f_offLineFmc, WaveLength * WaveNum * sizeof(cufftComplex));

    //init cufft plan
    int rank=1;
    int n[1];
    n[0]=WaveLength;
    int istride=1;
    int idist = WaveLength;
    int ostride=1;
    int odist = WaveLength;
    int inembed[2];
    int onembed[2];
    inembed[0]= WaveLength;  onembed[0]=WaveLength;
    inembed[1] = WaveNum; onembed[0] = WaveNum;

    cufftPlanMany(&planForward,rank,n,inembed, istride ,idist , onembed, ostride,odist, CUFFT_R2C, WaveNum);

    idist = WaveLength;
    inembed[0]=WaveLength;
    cufftPlanMany(&planInverse,rank,n,inembed, istride ,idist , onembed, ostride,odist, CUFFT_C2C, WaveNum);

    //generate hilbert filter
    cufftComplex* Hilbert = (cufftComplex*)malloc(WaveLength*sizeof(cufftComplex));
    for(int i = 0;i<WaveLength;i++){
        if(i == 0 || (float)i == WaveLength/2.0){
            Hilbert[i].x = 1;
            Hilbert[i].y = 0;
        }else if(i<WaveLength/2){
            Hilbert[i].x = 2;
            Hilbert[i].y = 0;
        }else{
            Hilbert[i].x = 0;
            Hilbert[i].y = 0;
        }
    }
    cudaMemcpy(d_Hilbert, Hilbert, WaveLength * sizeof(cufftComplex), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    free(Hilbert);
}

FmcImaging::~FmcImaging(){
    cudaFreeHost(h_offLineFmc);

    cudaFree(d_iTof);
    cudaFree(d_offLineFmc);
    cudaFree(d_TfmImage);
    cudaFree(d_H);
    cudaFree(d_Hilbert);
    cudaFree(d_f_offLineFmc);

    cufftDestroy(planForward);
    cufftDestroy(planInverse);
}

void FmcImaging::imaging(float* offLineFmc, short* Tof, cufftComplex* FmcMatHilbert, float* TfmImage, float MindB){
    thrust::transform(thrust::device, thrust::counting_iterator<int>(0), thrust::counting_iterator<int>(NX*NZ), TfmImage, thrust_imaging(offLineFmc, Tof, FmcMatHilbert, WaveNum, iWaveLength, WaveLength, col_tof));
    std::cout<<"test"<<std::endl;
    float h_MaxInTfmImage, *d_MaxInTfmImage = thrust::max_element(thrust::device, TfmImage, TfmImage+NX*NZ);
    cudaMemcpy(&h_MaxInTfmImage, d_MaxInTfmImage, sizeof(float), cudaMemcpyDeviceToHost);
    thrust::transform(thrust::device, TfmImage, TfmImage+NX*NZ, TfmImage, [=]__device__(float val)->float{
        return fmaxf(MindB, 20*log10f(val/h_MaxInTfmImage));
    });
}

void FmcImaging::transpose(short *iTof, int row_tof, int col_tof){
    short* iTof_trans;
    cudaMalloc((void**) &iTof_trans, row_tof * col_tof * sizeof(short));
    dim3 block(TILEW, TILEH, 1);
    dim3 grid(ceil(row_tof/block.x)+1, ceil(col_tof/block.y)+1, 1);
    transposeCoalesced<<<grid, block>>>(iTof, iTof_trans, col_tof, row_tof);
    cudaMemcpy(iTof, iTof_trans, row_tof * col_tof * sizeof(short), cudaMemcpyDeviceToDevice);
    cudaDeviceSynchronize();
    cudaFree(iTof_trans);
}

void FmcImaging::hilbert_transform(cufftComplex* f_offLineFmc, cufftComplex* HilbertFilter){
    dim3 block(TILEW, TILEH, 1);
    dim3 grid(ceil(WaveLength/block.x)+1, ceil(WaveNum/block.y)+1, 1);
    FreqDomainMul<<<grid, block>>>(f_offLineFmc, HilbertFilter, WaveNum, WaveLength);
    cudaDeviceSynchronize();
    cufftExecC2C(planInverse, f_offLineFmc, f_offLineFmc, CUFFT_INVERSE);
}

void FmcImaging::freq_domain_filtering(float* offLineFmc, cufftComplex* f_offLineFmc, cufftComplex* H){
    cufftReal *temp;
    cudaMalloc((void **)&temp, WaveNum*WaveLength*sizeof(cufftReal));
    for(int i = 0;i<WaveNum;i++){
        cudaMemcpy(temp+i*WaveLength, (cufftReal*)offLineFmc+i*iWaveLength+2, WaveLength*sizeof(cufftReal), cudaMemcpyDeviceToDevice);
        cudaDeviceSynchronize();
    }

    cufftExecR2C(planForward, temp, f_offLineFmc);
    thrust::transform(thrust::device, thrust::counting_iterator<int>(0), thrust::counting_iterator(WaveNum*WaveLength), f_offLineFmc, zero_phase_filt(f_offLineFmc, filter,WaveLength));
    // dim3 block(TILEW, TILEH, 1);
    // dim3 grid(ceil(WaveLength/block.x)+1, ceil(WaveNum/block.y)+1, 1);
    // FreqDomainMul<<<grid, block>>>(f_offLineFmc, H, WaveNum, WaveLength);
    // cudaDeviceSynchronize();
    // FreqDomainMul<<<grid, block>>>(f_offLineFmc, H, WaveNum, WaveLength);
    // cudaDeviceSynchronize();
    cudaFree(temp);
}

void FmcImaging::get_freq_filter(float f1, float f2, int taps, cufftComplex* H){
    //generate bandpass filter
    float *h;
    cufftReal *d_h;
    cufftHandle planFilter;
    h = (float*)malloc(sizeof(float)*WaveLength);
    memset(h, 0, sizeof(float)*WaveLength);
    cudaMalloc((void**) &d_h, WaveLength * sizeof(cufftReal));
    std::vector<float> filter = fir1(taps, {f1, f2});
    memcpy(h, filter.data(), taps*sizeof(float));
    cudaMemcpy(d_h, (cufftReal*)h, WaveLength * sizeof(cufftReal), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    cufftPlan1d(&planFilter, WaveLength, CUFFT_R2C, 1);
    cufftExecR2C(planFilter, d_h, H);
    cudaDeviceSynchronize();
    
    free(h);
    cudaFree(d_h);
    cufftDestroy(planFilter);
}

template<typename T>
void FmcImaging::read_data(std::string filepath, T *input, int row, int col){

    std::ifstream fp_input;
    fp_input.open(filepath, std::ios::in);
    if (!fp_input) { //打开失败
        std::cout << "error opening source file." << std::endl;
        std::exit(0);
    }
    std::string line;
    unsigned long element_num = 0;
    bool exit = false;
    while(getline(fp_input, line) && !exit){
        std::string number;
        std::istringstream readstr(line);
        while(getline(readstr, number, ',')){
            if(typeid(T) == typeid(float)){
                input[element_num++] = std::stof(number);
            }
            else if(typeid(T) == typeid(short)){
                input[element_num++] = std::stoi(number);
            }
            else{
                std::cout<<"invalid type"<<std::endl;
                std::exit(1);
            }
            if(element_num>=row*col){
                exit = true;
                break;
            }
        }
    }
    fp_input.close();
}

void FmcImaging::save_result_to_txt(std::string filepath, float* d_result){
    float *result;
    result = (float*)malloc(NX*NZ*sizeof(float));
    cudaMemcpy(result, d_result, NX*NZ*sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    std::ofstream fp_output;
    fp_output.open(filepath, std::ios::out);
    if (!fp_output) {
        std::cout << "error opening destination file." << std::endl;
        return;
    }
    for(int i = 0;i<NZ;i++){
        for(int j = 0;j<NX;j++){
            // std::cout<<result[i*NX+j]<<" "<<std::endl;
            if(j!=NX-1){
                fp_output<<result[i*NX+j]<<' ';
            }else{
                fp_output<<result[i*NX+j]<<'\n';
            } 
        }
    }
    fp_output.close();
    free(result);
}
