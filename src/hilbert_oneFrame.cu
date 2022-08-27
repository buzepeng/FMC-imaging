#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <iostream>
#include <fstream>  
#include <chrono>
#include <iomanip>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cufft.h> 
#include <device_launch_parameters.h>
#include <device_functions.h>
#define NX 1500
#define NY 1000
#define NXY NX*NY
#define TILEW 16
#define TILEH 16

using namespace std;

__global__ void hilbertFreqFilter(cufftComplex* signal){
    int ix = threadIdx.x + blockIdx.x*blockDim.x;
    int iy = threadIdx.y + blockIdx.y*blockDim.y;
    int idx = ix + iy * NX;
    if(ix<NX && iy<NY){
        signal[idx].x = ((ix==0)+2*(ix>0 && ix < NX/2)+0*(ix>=NX/2 && ix < NX))*signal[idx].x;      //对判断条件进行代数运算，避免分支
        signal[idx].y = ((ix==0)+2*(ix>0 && ix < NX/2)+0*(ix>=NX/2 && ix < NX))*signal[idx].y;
        __syncthreads();
    }
}

__global__ void computEnvelope(float* origin, cufftComplex* filtered){
    int ix = threadIdx.x + blockIdx.x*blockDim.x;
    int iy = threadIdx.y + blockIdx.y*blockDim.y;
    int idx = ix + iy * NX;

    if(ix<NX && iy<NY){
        float yi = origin[idx] - filtered[idx].x/NX;    //cufft没有考虑信号的尺度变换
        origin[idx] = abs(yi);
        __syncthreads();
    }
}

cv::Mat hilbert(   float* signals,
                cufftReal* d_inputData,
                cufftComplex* d_fftOutput,
                cufftHandle planForward,
                cufftHandle planInverse
){
    //将数据拷贝到显存，执行FFT正变换
    cudaMemcpy(d_inputData, (cufftReal*)signals, NXY * sizeof(cufftReal), cudaMemcpyHostToDevice);
    cufftExecR2C(planForward, d_inputData, d_fftOutput);
    dim3 block(TILEH, TILEW, 1);
    dim3 grid((NX-block.x)/block.x+1, (NY-block.y)/block.y+1, 1);
    //希尔伯特频域滤波
    hilbertFreqFilter<<<grid, block>>>(d_fftOutput);
    cudaDeviceSynchronize();
    //执行FFT反变换
    cufftExecC2C(planInverse, d_fftOutput, d_fftOutput, CUFFT_INVERSE);
    cudaDeviceSynchronize();
    //计算信号包络
    computEnvelope<<<grid, block>>>(d_inputData, d_fftOutput);
    cudaDeviceSynchronize();
    //将显存中的数据拷贝至内存，并进行设备主机同步
    cv::Mat result = cv::Mat::zeros(NY, NX, CV_32FC1);
    cudaMemcpy(signals, d_inputData, sizeof(float)*NXY, cudaMemcpyDeviceToHost);
    memcpy(result.ptr<float>(0), signals, sizeof(float)*NXY);
    cudaDeviceSynchronize();
    return result;
}

int main(){
    //申请锁页内存和显存
    float *input;
    cudaMallocHost((void**)&input, (size_t)sizeof(float)*NXY);
    cufftReal *d_inputData;
	cufftComplex *d_fftOutput;
	cudaMalloc((void**) &d_inputData, NXY * sizeof(cufftReal));
	cudaMalloc((void**) &d_fftOutput, NXY * sizeof(cufftComplex));
    //打开文件，将txt中的数据读取到input数组中
    ifstream fp_input;
	ofstream fp_output;
    fp_input.open("./data/data.txt", ios::in);
	if (!fp_input) { //打开失败
        cout << "error opening source file." << endl;
        return 0;
    }
	fp_output.open("./data/output_gpu.txt", ios::out);
	if (!fp_output) {
        fp_input.close(); //程序结束前不能忘记关闭以前打开过的文件
        cout << "error opening destination file." << endl;
        return 0;
    }
    for(int i = 0;i<NY;i++){
        for(int j = 0;j<NX; j++){
            fp_input>>input[i*NX+j];
        }
    }
    //定义两次FFT的类型
    cufftHandle planForward, planInverse;
    int rank=1;
	int n[1];
	n[0]=NX;
	int istride=1;
	int idist = NX;
	int ostride=1;
	int odist = NX;
	int inembed[2];
	int onembed[2];
	inembed[0]=NX;  onembed[0]=NX;
	inembed[1] = NY; onembed[1] = NY;
    cufftPlanMany(&planForward,rank,n,inembed, istride ,idist , onembed, ostride,odist, CUFFT_R2C, NY);
    cufftPlanMany(&planInverse,rank,n,inembed, istride ,idist , onembed, ostride,odist, CUFFT_C2C, NY);
    //进行希尔伯特变换并计时
    auto startTime = chrono::system_clock::now();
    cv::Mat result = hilbert(input, d_inputData, d_fftOutput, planForward, planInverse);
	auto endTime = chrono::system_clock::now();
    cout << "gpu time:" << chrono::duration_cast<chrono::milliseconds>(endTime - startTime).count() <<"ms"<< endl;
    for(int i = 0;i<NY;i++){
        for(int j = 0;j<NX;j++){
            // if(j!=NX-1) fp_output<<input[i*NX+j]<<'\t';
            // else fp_output<<input[i*NX+j];
            if(j!=NX-1) fp_output<<result.at<float>(i,j)<<'\t';
            else fp_output<<result.at<float>(i,j);
        }
        fp_output<<'\n';
    }
	//释放内存，关闭文件
	fp_input.close();
	fp_output.close();
    cufftDestroy(planForward);
    cufftDestroy(planInverse);
    cudaFreeHost(input);
    cudaFree(d_inputData);
    cudaFree(d_fftOutput);
	return 0;
}