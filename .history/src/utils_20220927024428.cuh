

__device__ cufftComplex operator * (cufftComplex a, cufftComplex b){
    cufftComplex res;
    res.x = (a.x*b.x - a.y*b.y);
    res.y = (a.x*b.y + a.y*b.x);
    return res;
}
__device__ cufftComplex operator * (cufftComplex a, float b){
    cufftComplex res;
    res.x = a.x*b;
    res.y = a.y*b;
    return res;
}
__device__ cufftComplex operator / (cufftComplex a, float b){
    cufftComplex res;
    res.x = a.x/b;
    res.y = a.y/b;
    return res;
}
__device__ cufftComplex operator + (cufftComplex a, cufftComplex b){
    cufftComplex res;
    res.x = a.x+b.x;
    res.y = a.y+b.y;
    return res;
}

__global__ void frequency_filtering_kernel(cufftComplex* signals, cufftComplex* filter, cufftComplex *HilbertMat, int WaveNum, int WaveLength){
    int ix = blockDim.x*blockIdx.x+threadIdx.x;
    int iy = blockDim.y*blockIdx.y+threadIdx.y;
    int idx = iy*WaveLength+ix;

    if(iy<WaveNum && ix<WaveLength){
        HilbertMat[idx] = signals[idx]*filter[ix];
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
    int WaveNum, WaveLength, iWaveLength, col_tof;

    thrust_imaging(float *_offLineFmc, short *_Tof, cufftComplex *_FmcMatHilbert, int _WaveNum, int _WaveLength,int _iWaveLength, int _col_tof):offLineFmc(_offLineFmc), Tof(_Tof), FmcMatHilbert(_FmcMatHilbert), WaveNum(_WaveNum), WaveLength(_WaveLength), iWaveLength(_iWaveLength), col_tof(_col_tof){};
    __device__
    float operator()(int i){
        cufftComplex complex_sum;
        complex_sum.x = 0;
        complex_sum.y = 0;
        for(int s = 0;s<WaveNum;s++){
            int tIndex = int(offLineFmc[iWaveLength*s]-1), rIndex = int(offLineFmc[iWaveLength*s+1]-1);
            int trTofIndex = int(Tof[tIndex*col_tof+i]+Tof[rIndex*col_tof+i]), fmc_ind = s*WaveLength+trTofIndex-1;
            cufftComplex temp = FmcMatHilbert[fmc_ind];
            complex_sum = complex_sum + temp;
        }
        return cuCabsf(complex_sum);
    }
};