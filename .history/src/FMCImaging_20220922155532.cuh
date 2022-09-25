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
#include <cstdlib>

#define TILEW 16
#define TILEH 16


__device__ cufftComplex operator * (cufftComplex a, cufftComplex b){
    cufftComplex res;
    res.x = (a.x*b.x - a.y*b.y);
    res.y = (a.x*b.y + a.y*b.x);
    return res;
}

__global__ void FreqDomainMul(cufftComplex* signals, cufftComplex* filter, int WaveLength){
    int ix = blockDim.x*blockIdx.x+threadIdx.x;
    int iy = blockDim.y*blockIdx.y+threadIdx.y;
    int idx = iy*WaveLength+ix;
    int tile_idx = threadIdx.y*TILEW+threadIdx.x;

    __shared__ cufftComplex filter_s[TILEW], signals_s[TILEW*TILEH];
    filter_s[threadIdx.x] = filter[ix];
    signals_s[tile_idx] = signals[idx];
    __syncthreads();

    signals_s[tile_idx] = signals_s[tile_idx] * filter_s[threadIdx.x];
    signals[idx] = signals_s[tile_idx];
    __syncthreads();
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

__global__ void Imaging(float *offLineFmcMat, short *Tof, cufftComplex *FmcMatHilbert, float* TfmImage, float MindB, int WaveNum, int iWaveLength, int  WaveLength, int row_tof, int col_tof, int NZ, int NX){
    int ix = blockDim.x*blockIdx.x+threadIdx.x;
    int iy = blockDim.y*blockIdx.y+threadIdx.y;
    int tid = threadIdx.y*TILEW+threadIdx.x;
    int idx = iy*NX+ix;
    extern __shared__ short s_buffer[];
    short *s_offLineFmc0 = &s_buffer[0], *s_offLineFmc1 = &s_buffer[WaveNum];
    for(int i = 0;i<WaveNum;i = i+TILEW*TILEH){
        int s = tid+i;
        if(s<WaveNum){
            s_offLineFmc0[s] = short(offLineFmcMat[s*iWaveLength])-1;
            s_offLineFmc1[s] = short(offLineFmcMat[s*iWaveLength+1])-1;
        }
    }
    __syncthreads();
    
    if(iy<NZ && ix<NX){
        float real = 0, imag = 0;
        for(int s = 0; s < WaveNum; s++){

            short tIndex = s_offLineFmc0[s], rIndex = s_offLineFmc1[s];
            int trTofIndex = int(Tof[tIndex*col_tof+idx]+Tof[rIndex*col_tof+idx]);

            real += FmcMatHilbert[s*WaveLength+trTofIndex-1].x;
            imag += FmcMatHilbert[s*WaveLength+trTofIndex-1].y;

        }
        __syncthreads();
        
        TfmImage[idx] = sqrtf(powf(real, 2)+powf(imag, 2));
        // MaxInTfmImage = *thrust::max_element(thrust::device, TfmImage, TfmImage+NX*NZ);
        // TfmImage[idx] = fmaxf(MindB, 20*log10f(TfmImage[idx]/MaxInTfmImage));
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
        short *d_iTof;
        float *d_offLineFmc, *d_TfmImage;
        cufftComplex *d_H, *d_Hilbert, *d_f_offLineFmc;
        cufftHandle planForward;

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
        void read_data_to_gpu(std::string filepath, T* gpu_addr, int row, int col);
        //save TfmImage to txt
        void save_result_to_txt(std::string filepath, float* d_result);
};

FmcImaging::FmcImaging(int _WaveNum, int _WaveLength, int _iWaveLength, int _row_tof, int _col_tof, int _NZ, int _NX):WaveNum(_WaveNum),WaveLength(_WaveLength),iWaveLength(_iWaveLength),row_tof(_row_tof),col_tof(_col_tof),NZ(_NZ),NX(_NX)
{
    //alloc memory
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
    int idist = iWaveLength;
    int ostride=1;
    int odist = WaveLength;
    int inembed[2];
    int onembed[2];
    inembed[0]=iWaveLength;  onembed[0]=WaveLength;
    inembed[1] = WaveNum; onembed[0] = WaveNum;

    cufftPlanMany(&planForward,rank,n,inembed, istride ,idist , onembed, ostride,odist, CUFFT_R2C, WaveNum);

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
    cudaFree(d_iTof);
    cudaFree(d_offLineFmc);
    cudaFree(d_TfmImage);
    cudaFree(d_H);
    cudaFree(d_Hilbert);
    cudaFree(d_f_offLineFmc);

    cufftDestroy(planForward);
}

void FmcImaging::imaging(float* offLineFmc, short* Tof, cufftComplex* FmcMatHilbert, float* TfmImage, float MindB){
    dim3 block(TILEW, TILEH, 1);
    dim3 grid(ceil(NX/block.x), ceil(NZ/block.y), 1);
    Imaging<<<grid, block, 2*WaveNum*sizeof(short)>>>(offLineFmc, Tof, FmcMatHilbert, TfmImage, MindB, WaveNum, iWaveLength, WaveLength, row_tof, col_tof, NZ, NX);
    cudaDeviceSynchronize();
    float MaxInTfmImage = 0;
    MaxInTfmImage = *thrust::max_element(thrust::device, TfmImage, TfmImage+NX*NZ);
    thrust::transform
}

void FmcImaging::transpose(short *iTof, int row_tof, int col_tof){
    short* iTof_trans;
    cudaMalloc((void**) &iTof_trans, row_tof * col_tof * sizeof(short));
    dim3 block(TILEW, TILEH, 1);
    dim3 grid(ceil(row_tof/block.x), ceil(col_tof/block.y), 1);
    transposeCoalesced<<<grid, block>>>(iTof, iTof_trans, col_tof, row_tof);
    cudaMemcpy(iTof, iTof_trans, row_tof * col_tof * sizeof(short), cudaMemcpyDeviceToDevice);
    cudaDeviceSynchronize();
    cudaFree(iTof_trans);
}

void FmcImaging::hilbert_transform(cufftComplex* FmcMatHilbert, cufftComplex* HilbertFilter){
    dim3 block(TILEW, TILEH, 1);
    dim3 grid(ceil(WaveLength/block.x), ceil(WaveNum/block.y), 1);
    FreqDomainMul<<<grid, block>>>(FmcMatHilbert, HilbertFilter, WaveLength);
    cudaDeviceSynchronize();
}

void FmcImaging::freq_domain_filtering(float* offLineFmc, cufftComplex* f_offLineFmc, cufftComplex* H){
    cufftExecR2C(planForward, offLineFmc, f_offLineFmc);
    dim3 block(TILEW, TILEH, 1);
    dim3 grid(ceil(WaveLength/block.x), ceil(WaveNum/block.y), 1);
    FreqDomainMul<<<grid, block>>>(f_offLineFmc, H, WaveLength);
    cudaDeviceSynchronize();
}

void FmcImaging::get_freq_filter(float f1, float f2, int taps, cufftComplex* H){
    //generate bandpass filter
    float *h;
    cufftReal *d_h;
    cufftHandle planFilter;
    h = (float*)malloc(sizeof(float)*WaveLength);
    cudaMalloc((void**) &d_h, WaveLength * sizeof(cufftReal));
    for(int i = 0; i < taps; i++) {
        int n = i - int(taps/2);
        float w =  sin(((float) M_PI * i) / (taps - 1)) *
                sin(((float) M_PI * i) / (taps - 1));
        h[i] = 2.0*f1*sinc(2.0*f1*n) - 2.0*f2*sinc(2.0*f2*n);
        h[i] = w * h[i];
    }
    cudaMemcpy(d_h, (cufftReal*)h, WaveLength * sizeof(cufftReal), cudaMemcpyHostToDevice);
    cufftPlan1d(&planFilter, WaveLength, CUFFT_R2C, 1);
    cufftExecR2C(planFilter, d_h, H);
    cudaDeviceSynchronize();
    
    free(h);
    cudaFree(d_h);
    cufftDestroy(planFilter);
}

template<typename T>
void FmcImaging::read_data_to_gpu(std::string filepath, T *gpu_addr, int row, int col){
    T* input;
    input = (T*)malloc(row*col*sizeof(T));
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
    cudaMemcpy(gpu_addr, input, row*col*sizeof(T), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    free(input);
}

void FmcImaging::save_result_to_txt(std::string filepath, float* d_result){
    float *result;
    result = (float*)malloc(NX*NZ*sizeof(float));
    cudaMemcpy(result, d_result, NX*NZ*sizeof(float), cudaMemcpyDeviceToHost);
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