#include <iostream>
#include <stdio.h>
#include <fstream>  
#include <vector>
#include <math.h>
#include <chrono>
#include <iomanip>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cufft.h> 
#include <chrono>
#include <device_launch_parameters.h>
#define NX 4000
#define NY 4096
#define TILEW 16
#define TILEH 16
#define TAPS 128

__constant__ cufftComplex H[NX];

__device__ cufftComplex operator * (cufftComplex a, cufftComplex b){
    cufftComplex res;
    res.x = (a.x*b.x - a.y*b.y);
    res.y = (a.x*b.y + a.y*b.x);
    return res;
}

__global__ void FreqDomainMul(cufftComplex* signals){
    int ix = blockDim.x*blockIdx.x+threadIdx.x;
    int iy = blockDim.y*blockIdx.y+threadIdx.y;
    int idx = iy*NX+ix;
    int tile_idx = threadIdx.y*TILEW+threadIdx.x;

    __shared__ cufftComplex filter_s[TILEW], signals_s[TILEW*TILEH];
    filter_s[threadIdx.x] = H[ix];
    signals_s[tile_idx] = signals[idx];
    __syncthreads();

    signals_s[tile_idx] = signals_s[tile_idx] * filter_s[threadIdx.x];
    signals[idx] = signals_s[tile_idx];
    __syncthreads();
}

__global__ void Real2Complex(cufftReal* in, cufftComplex* out){
    int ix = blockDim.x*blockIdx.x+threadIdx.x;
    int iy = blockDim.y*blockIdx.y+threadIdx.y;
    int idx = iy*NX+ix;

    out[idx].x = in[idx];
    out[idx].y = 0;
    __syncthreads();
}

__global__ void normalization(cufftReal* in){
    int ix = blockDim.x*blockIdx.x+threadIdx.x;
    int iy = blockDim.y*blockIdx.y+threadIdx.y;
    int idx = iy*NX+ix;

    in[idx] = in[idx]/(2*NX);
    __syncthreads();
}

__global__ void warmup(){

}

static float sinc(const float x)
{
    if (x == 0)
        return 1;

    return sin(M_PI * x) / (M_PI * x);
}

class HilbertBandPassFilter{
    private:
        float *input, *h;
        cufftReal *d_inputData, *d_h_in;
        cufftComplex *d_fftOutput, *d_h_out;
        cufftHandle planForward, planInverse, planFilter;
    public:
        HilbertBandPassFilter(float f1, float f2){

            // cudaFree(0);
            warmup<<<1,1>>>();
            cudaMallocHost((void**)&input, (size_t)sizeof(float)*NX*NY);
            cudaMallocHost((void**)&h, (size_t)sizeof(float)*NX);
            cudaMalloc((void**) &d_h_in, NX * sizeof(cufftReal));
	        cudaMalloc((void**) &d_h_out, NX * sizeof(cufftComplex));
            cudaMalloc((void**) &d_inputData, NX * NY * sizeof(cufftReal));
	        cudaMalloc((void**) &d_fftOutput, NX * NY * sizeof(cufftComplex));

            // int rank=1, istride=1, idist = NX, ostride=1, odist = NX;
            // int n[1] = {NX}, inembed[2] = {NY, NX}, onembed[2] = {NY, NX};
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
            inembed[1] = NY; onembed[0] = NY;

            // cufftPlanMany(&planForward,rank,n,inembed, istride ,idist , onembed, ostride,odist, CUFFT_R2C, NY);
            cufftPlanMany(&planForward,rank,n,inembed, istride ,idist , onembed, ostride,odist, CUFFT_R2C, NY);
            cufftPlanMany(&planInverse,rank,n,inembed, istride ,idist , onembed, ostride,odist, CUFFT_C2R, NY);
            cufftPlan1d(&planFilter, NX, CUFFT_R2C, 1);

            //generate bandpas filter
            memset(h, 0, NX);
            for(int i = 0; i < TAPS; i++) {
                int n = i - int(TAPS/2);
                float w =  sin(((float) M_PI * i) / (TAPS - 1)) *
                        sin(((float) M_PI * i) / (TAPS - 1));
                h[i] = 2.0*f1*sinc(2.0*f1*n) - 2.0*f2*sinc(2.0*f2*n);
                h[i] = w * h[i];
            }

            cudaMemcpy(d_h_in, (cufftReal*)h, NX * sizeof(cufftReal), cudaMemcpyHostToDevice);
            cufftExecR2C(planFilter, d_h_in, d_h_out);
            cudaMemcpyToSymbol( H, d_h_out, sizeof(cufftComplex)*NX, 0, cudaMemcpyDeviceToDevice );

            //generate hilbert filter
            for(int i = 0;i<NX;i++){
                if(i == 0 || (float)i == NX/2.0){
                    h[i] = 1;
                }else if(i<NX/2){
                    h[i] = 2;
                }else{
                    h[i] = 0;
                }
            }
            cudaMemcpy(d_h_in, (cufftReal*)h, NX * sizeof(cufftReal), cudaMemcpyHostToDevice);
            dim3 block(TILEW, 1, 1);
            dim3 grid(ceil(NX/block.x), 1, 1);
            Real2Complex<<<grid, block>>>(d_h_in, d_h_out);
            FreqDomainMul<<<grid, block>>>(d_h_out);
            cudaMemcpyToSymbol( H, d_h_out, sizeof(cufftComplex)*NX, 0, cudaMemcpyDeviceToDevice );
        }
        float* read_data(std::string file_path){
            std::ifstream fp_input;
            fp_input.open(file_path, std::ios::in);
            if (!fp_input) { //打开失败
                std::cout << "error opening source file." << std::endl;
                return nullptr;
            }
            for(int i = 0;i<NY;i++){
                for(int j = 0;j<NX; j++){
                    fp_input>>input[i*NX+j];
                }
            }
            fp_input.close();
            return input;
        }

        float* filter(float* signals){
            cudaMemcpy(d_inputData, (cufftReal*)signals, NX * NY * sizeof(cufftReal), cudaMemcpyHostToDevice);
            cufftExecR2C(planForward, d_inputData, d_fftOutput);
            dim3 block(TILEW, TILEH, 1);
            dim3 grid(ceil(NX/block.x), ceil(NY/block.y), 1);
            FreqDomainMul<<<grid, block>>>(d_fftOutput);
            //执行FFT反变换
            cufftExecC2R(planInverse, d_fftOutput, d_inputData);
            normalization<<<grid, block>>>(d_inputData);
            cudaMemcpy(input, (float*)d_inputData, (size_t)sizeof(float)*NX*NY, cudaMemcpyDeviceToHost);
            cudaDeviceSynchronize();
            return input;
        }

        void save_result(float* result, std::string filepath){
            std::ofstream fp_output;
            fp_output.open(filepath, std::ios::out);
            if (!fp_output) {
                std::cout << "error opening destination file." << std::endl;
                return;
            }
            for(int i = 0;i<NY;i++){
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
        }
        ~HilbertBandPassFilter(){
            cufftDestroy(planForward);
            cufftDestroy(planInverse);
            cufftDestroy(planFilter);
            cudaFreeHost(input);
            cudaFreeHost(h);
            cudaFree(d_inputData);
            cudaFree(d_fftOutput);
            cudaFree(d_h_out);
            cudaFree(d_h_in);
        }
};


int main(){
    float *signals, *results;
    float fs = 1000, fl = 15.0/fs, fh = 45.0/fs;

    HilbertBandPassFilter hf(fl, fh);
    
    signals = hf.read_data("/home/hilbert/data/input_data.txt");
    std::cout<<"Signal read complete!"<<std::endl;

    auto start = std::chrono::high_resolution_clock::now();
    results = hf.filter(signals);
    auto end = std::chrono::high_resolution_clock::now();
    std::cout<<"Result compute complete! Time spend: " << std::chrono::duration<float>(end - start).count()<<std::endl;
    hf.save_result(results, "/home/hilbert/data/output_data.txt");

	return 0;
}