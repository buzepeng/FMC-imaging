#include "FMCImaging.cuh"

int main(){
    int NX=1001, NZ = 1001, WaveNum=2048, iWaveLength = 4002, WaveLength = 4000, row_tof=64, col_tof=1002001;
    int taps = 128;
    float sample_freq = 100, element_freq = 5, low_cut = ((element_freq - 3)*2)/sample_freq, high_cut = ((element_freq+3)*2)/sample_freq, MindB = -30;
    std::cout<<"Initialing"<<std::endl;
    FmcImaging fmc_img(WaveNum, WaveLength, iWaveLength, row_tof, col_tof, NZ, NX);
    std::cout<<"Reading FMC"<<std::endl;
    fmc_img.read_data_to_gpu("../data/FMC_Data_YJBL_5L64_0p5_100MHz_4000.csv", fmc_img.d_offLineFmc, WaveNum, iWaveLength);
    std::cout<<""
    fmc_img.read_data_to_gpu("../data/TOF_Data_40_40_0p04Plane.csv", fmc_img.d_iTof, col_tof, row_tof);
    
    fmc_img.transpose(fmc_img.d_iTof, row_tof, col_tof);
    fmc_img.get_freq_filter(low_cut, high_cut, taps, fmc_img.d_H);

    fmc_img.freq_domain_filtering(fmc_img.d_offLineFmc, fmc_img.d_f_offLineFmc, fmc_img.d_H);
    fmc_img.hilbert_transform(fmc_img.d_f_offLineFmc, fmc_img.d_Hilbert);
    fmc_img.imaging(fmc_img.d_offLineFmc, fmc_img.d_iTof, fmc_img.d_f_offLineFmc, fmc_img.d_TfmImage, MindB);

    fmc_img.save_result_to_txt("../data/output_data.txt", fmc_img.d_TfmImage);
}