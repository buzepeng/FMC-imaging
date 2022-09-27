#include "FMCImaging.cuh"

int main(){
    int NX=1001, NZ = 1001, WaveNum=2048, iWaveLength = 4002, WaveLength = 4000, row_tof=64, col_tof=1002001;
    int taps = 100;
    float sample_freq = 1e8, low_cut = (4e6)/sample_freq, high_cut = (16e6)/sample_freq, MindB = -30;
    std::cout<<"Initialing..."<<std::endl;
    FmcImaging fmc_img(WaveNum, WaveLength, iWaveLength, row_tof, col_tof, NZ, NX, taps);
    std::cout<<"Reading FMC..."<<std::endl;
    fmc_img.read_data("/home/hilbert/data/YJBL_5L64_0p5_100MHz_4000.csv", fmc_img.h_offLineFmc, WaveNum, iWaveLength);
    std::cout<<"Reading TOF..."<<std::endl;
    fmc_img.read_data("/home/hilbert/data/TOF_Data_40_40_0p04Plane.csv", fmc_img.h_iTof, col_tof, row_tof);
    cudaMemcpy(fmc_img.d_iTof, fmc_img.h_iTof, fmc_img.row_tof*fmc_img.col_tof*sizeof(short), cudaMemcpyHostToDevice);

    std::cout<<"Transposing TOF..."<<std::endl;
    fmc_img.transpose(fmc_img.d_iTof, row_tof, col_tof);
    std::cout<<"Generating bandpass filter..."<<std::endl;
    fmc_img.get_filter(low_cut, high_cut, fmc_img.d_filter);

    auto start = std::chrono::high_resolution_clock::now();
    // for(int i = 0;i<WaveNum;i++){
    //     cudaMemcpy(fmc_img.d_offLineFmc_ind+2*i, fmc_img.h_offLineFmc+i*iWaveLength, 2*sizeof(float), cudaMemcpyHostToDevice);
    //     cudaMemcpy(fmc_img.d_offLineFmc_data+WaveLength*i, fmc_img.h_offLineFmc+i*iWaveLength+2, WaveLength*sizeof(float), cudaMemcpyHostToDevice);
    // }
    cudaMemcpy(fmc_img.d_offLineFmc, fmc_img.h_offLineFmc, iWaveLength*WaveNum*sizeof(float), cuda);
    std::cout<<"Filtering..."<<std::endl;
    fmc_img.filtering(fmc_img.d_offLineFmc, fmc_img.d_filter, fmc_img.d_offLineFmc_filted);
    std::cout<<"Hilbert Transforming..."<<std::endl;
    fmc_img.hilbert_transform(fmc_img.d_offLineFmc_filted, fmc_img.d_Hilbert, fmc_img.d_f_offLineFmc);

    std::cout<<"Imaging..."<<std::endl;
    fmc_img.imaging(fmc_img.d_offLineFmc, fmc_img.d_iTof, fmc_img.d_f_offLineFmc, fmc_img.d_TfmImage, MindB);
    auto end   = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout<<"time spend:"<< double(duration.count())/1000<< "ms" << std::endl;

    std::cout<<"Saving result..."<<std::endl;
    fmc_img.save_result_to_txt("/home/hilbert/data/output_data.txt", fmc_img.d_TfmImage);
}