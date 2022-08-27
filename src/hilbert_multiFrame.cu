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
#define NX 1500
#define NY 1000
#define NXY NX*NY
#define TILEW 16
#define TILEH 16
#define BATCH 64

using namespace std;

__global__ void hilbertFreqFilter(cufftComplex* signal){
    int ix = threadIdx.x + blockIdx.x*blockDim.x;
    int iy = threadIdx.y + blockIdx.y*blockDim.y;
    int iz = blockIdx.z;
    int idx = ix + iy * NX + iz*NXY;
    if(ix<NX && iy<NY){
        signal[idx].x = ((ix==0)+2*(ix>0 && ix < NX/2)+0*(ix>=NX/2 && ix < NX))*signal[idx].x;
        signal[idx].y = ((ix==0)+2*(ix>0 && ix < NX/2)+0*(ix>=NX/2 && ix < NX))*signal[idx].y;
        __syncthreads();
    }
}

__global__ void computEnvelope(float* origin, cufftComplex* filtered){
    int ix = threadIdx.x + blockIdx.x*blockDim.x;
    int iy = threadIdx.y + blockIdx.y*blockDim.y;
    int iz = blockIdx.z;
    int idx = ix + iy * NX + iz*NXY;

    if(ix<NX && iy<NY){
        // float xr = origin[idx];
        // float yr = filtered[idx].y;
        // float yi = xr - filtered[idx].x;
        float yi = origin[idx] - filtered[idx].x/NX;
        // float amp = sqrtf(yr*yr*yi*yi);
        origin[idx] = abs(yi);
        __syncthreads();
    }
}

void hilbert(float* signals){

    cufftReal *d_inputData;
	cufftComplex *d_fftOutput;
	cudaMalloc((void**) &d_inputData, BATCH*NXY * sizeof(cufftReal));
	cudaMalloc((void**) &d_fftOutput, BATCH*NXY * sizeof(cufftComplex));
	cudaMemcpy(d_inputData, (cufftReal*)signals, BATCH*NXY * sizeof(cufftReal), cudaMemcpyHostToDevice);
	cufftHandle plan;

    int rank=1;
	int n[1];
	n[0]=NX;
	int istride=1;
	int idist = NX;
	int ostride=1;
	int odist = NX;
	int inembed[3];
	int onembed[3];
	inembed[0]=NX;  onembed[0]=NX;
	inembed[1] = NY; onembed[1] = NY;
    inembed[2] = BATCH; onembed[2] = BATCH;

    cufftPlanMany(&plan,rank,n,inembed, istride ,idist , onembed, ostride,odist, CUFFT_R2C, NY*BATCH);
    cufftExecR2C(plan, d_inputData, d_fftOutput);
    dim3 block(TILEH, TILEW, 1);
    dim3 grid((NX-block.x)/block.x+1, (NY-block.y)/block.y+1, BATCH);
    hilbertFreqFilter<<<grid, block>>>(d_fftOutput);
    cudaDeviceSynchronize();

    cufftPlanMany(&plan,rank,n,inembed, istride ,idist , onembed, ostride,odist, CUFFT_C2C, NY*BATCH);
    cufftExecC2C(plan, d_fftOutput, d_fftOutput, CUFFT_INVERSE);
    cudaDeviceSynchronize();

    computEnvelope<<<grid, block>>>(d_inputData, d_fftOutput);
    cudaDeviceSynchronize();

    cudaMemcpy(signals, d_inputData, sizeof(float)*BATCH*NXY, cudaMemcpyDeviceToHost);
    cudaFree(d_inputData);
    cudaFree(d_fftOutput);
}

int main(){
    ifstream fp_input;
    fp_input.open("./data/data.txt", ios::in);
	if (!fp_input) { //打开失败
        cout << "error opening source file." << endl;
        return 0;
    }
    cv::Mat src_data = cv::Mat::zeros(NY, NX, CV_32FC1);
    for(int i = 0;i<NY;i++){
        for(int j = 0;j<NX; j++){
            fp_input >> src_data.at<float>(i, j);
        }
    }
    // float *input = new float[NXY]();
    float *input;
    cudaMallocHost((void**)&input, (size_t)sizeof(float)*NXY*BATCH);
    for(int i = 0;i<BATCH;i++){
        memcpy(input+i*NXY, src_data.ptr<float>(0, 0), sizeof(float)*NXY);
    }
    auto startTime = chrono::system_clock::now();
    hilbert(input);
	auto endTime = chrono::system_clock::now();
	cout << "gpu time:" << chrono::duration_cast<chrono::milliseconds>(endTime - startTime).count()<<"ms"<< endl;
	
	fp_input.close();
    cudaFreeHost(input);
	return 0;
}