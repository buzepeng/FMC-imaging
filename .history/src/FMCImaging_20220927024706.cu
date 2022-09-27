#include 

FmcImaging::FmcImaging(int _WaveNum, int _WaveLength, int _iWaveLength, int _row_tof, int _col_tof, int _NZ, int _NX, int _taps):WaveNum(_WaveNum),WaveLength(_WaveLength),iWaveLength(_iWaveLength),row_tof(_row_tof),col_tof(_col_tof),NZ(_NZ),NX(_NX),taps(_taps)
{
    //alloc memory
    h_iTof = (short*)malloc(row_tof*col_tof*sizeof(short));

    cudaMallocHost((void **)&h_offLineFmc, WaveNum*iWaveLength*sizeof(float));

    cudaMalloc((void**) &d_iTof, row_tof * col_tof * sizeof(short));

    cudaMalloc((void **)&d_f_filter, WaveLength*sizeof(cufftComplex));
    cudaMalloc((void**) &d_offLineFmc, iWaveLength * WaveNum *sizeof(float));
    // cudaMalloc((void**) &d_offLineFmc_filted, iWaveLength * WaveNum *sizeof(float));
    cudaMalloc((void**) &d_TfmImage, NZ * NX * sizeof(float));

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
    int inembed[1];
    int onembed[1];
    inembed[0]= iWaveLength;  onembed[0]=WaveLength;
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
    free(Hilbert);
}

FmcImaging::~FmcImaging(){
    cudaFreeHost(h_offLineFmc);

    cudaFree(d_iTof);
    cudaFree(d_offLineFmc);
    // cudaFree(d_offLineFmc_filted);
    cudaFree(d_TfmImage);
    cudaFree(d_f_filter);
    cudaFree(d_Hilbert);
    cudaFree(d_f_offLineFmc);

    cufftDestroy(planForward);
    cufftDestroy(planInverse);
}

void FmcImaging::imaging(float* offLineFmc, short* Tof, cufftComplex* FmcMatHilbert, float* TfmImage, float MindB){
    thrust::transform(thrust::device, thrust::counting_iterator<int>(0), thrust::counting_iterator<int>(NX*NZ), TfmImage, thrust_imaging(offLineFmc, Tof, FmcMatHilbert, WaveNum, WaveLength, iWaveLength, col_tof));
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

void FmcImaging::hilbert_transform(cufftComplex *f_offLineFmc, cufftComplex* HilbertFilter){
    
    dim3 block(TILEW, TILEH, 1);
    dim3 grid(ceil(WaveLength/block.x)+1, ceil(WaveNum/block.y)+1, 1);
    frequency_filtering_kernel<<<grid, block>>>(f_offLineFmc, HilbertFilter, f_offLineFmc, WaveNum, WaveLength);
    cudaDeviceSynchronize();
    cufftExecC2C(planInverse, f_offLineFmc, f_offLineFmc, CUFFT_INVERSE);
}

void FmcImaging::filtering(float* offLineFmc, cufftComplex* f_filter, cufftComplex* f_offLineFmc){
    cufftExecR2C(planForward, &offLineFmc[2], f_offLineFmc);
    dim3 block(TILEW, TILEH, 1);
    dim3 grid(ceil(WaveLength/block.x)+1, ceil(WaveNum/block.y)+1, 1);
    frequency_filtering_kernel<<<grid, block>>>(f_offLineFmc, f_filter, f_offLineFmc, WaveNum, WaveLength);
    cudaDeviceSynchronize();
}

void FmcImaging::get_filter(float f1, float f2, cufftComplex* f_filter){
    //generate bandpass filter
    float* d_filter;
    cufftHandle planFilter;
    cufftPlan1d(&planFilter, WaveLength, CUFFT_R2C, 1);
    cudaMalloc((void **)&d_filter, WaveLength*sizeof(cufftComplex));
    cudaMemset(d_filter, 0, WaveLength*sizeof(float));
    std::vector<float> bandpass = fir1(taps, {f1, f2});
    cudaMemcpy(d_filter, bandpass.data(), taps * sizeof(float), cudaMemcpyHostToDevice);
    cufftExecR2C(planFilter, d_filter, f_filter);
    cudaFree(d_filter);
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
