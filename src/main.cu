#include <chrono>
#include <iostream>

#include "FMCImaging.cuh"

#define WAVENUM 2048
#define IWAVELENGTH 4002
#define WAVELENGTH 4000
#define ROW_TOF 64
#define COL_TOF 1002001

int main(){
    int NX=1001, NZ = 1001;
    int taps = 100;
    float sample_freq = 1e8, low_cut = (4e6)/sample_freq, high_cut = (16e6)/sample_freq, MindB = -30;

    static short h_iTof[ROW_TOF*COL_TOF];
    static char h_offLineFmc[WAVENUM][IWAVELENGTH];

    std::cout<<"Initialing..."<<std::endl;
    FmcImaging fmc_img(WAVENUM, WAVELENGTH, IWAVELENGTH, ROW_TOF, COL_TOF, NZ, NX, taps);
    std::cout<<"Reading FMC..."<<std::endl;
    fmc_img.read_FMC("/home/hilbert/data/YJBL_5L64_0p5_100MHz_4000.csv", &h_offLineFmc[0][0], WAVENUM, IWAVELENGTH);
    std::cout<<"Reading TOF..."<<std::endl;
    fmc_img.read_TOF("/home/hilbert/data/TOF_Data_40_40_0p04Plane.csv", h_iTof, COL_TOF, ROW_TOF);
    cudaMemcpy(fmc_img.d_iTof, h_iTof, ROW_TOF*COL_TOF*sizeof(short), cudaMemcpyHostToDevice);

    std::cout<<"Transposing TOF..."<<std::endl;
    fmc_img.transpose(fmc_img.d_iTof, ROW_TOF, COL_TOF);
    std::cout<<"Generating bandpass filter..."<<std::endl;
    fmc_img.get_filter(low_cut, high_cut, fmc_img.d_f_filter);

    auto start = std::chrono::high_resolution_clock::now();
    std::cout<<"Copying FMC..."<<std::endl;
    fmc_img.copy_FMC(&h_offLineFmc[0][0], fmc_img.d_offLineFmc_char, fmc_img.d_offLineFmc_float);
    std::cout<<"Filtering..."<<std::endl;
    fmc_img.filtering(fmc_img.d_offLineFmc_float, fmc_img.d_f_filter, fmc_img.d_f_offLineFmc);

    std::cout<<"Hilbert Transforming..."<<std::endl;
    fmc_img.hilbert_transform(fmc_img.d_f_offLineFmc, fmc_img.d_Hilbert);

    std::cout<<"Imaging..."<<std::endl;
    fmc_img.imaging(fmc_img.d_offLineFmc_char, fmc_img.d_iTof, fmc_img.d_f_offLineFmc, fmc_img.d_TfmImage, MindB);
    auto end   = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout<<"time spend:"<< double(duration.count())/1000<< "ms" << std::endl;

    std::cout<<"Saving result..."<<std::endl;
    fmc_img.save_result_to_txt("/home/hilbert/data/output_data.txt", fmc_img.d_TfmImage);
}