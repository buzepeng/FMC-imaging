#include <cufft.h> 

class FmcImaging{

    public:
        short *d_iTof;
        char *d_offLineFmc_char;
        float *d_TfmImage, *d_offLineFmc_float;
        cufftComplex *d_Hilbert, *d_f_offLineFmc, *d_f_filter;
        cufftHandle planForward, planInverse;

        int WaveNum, WaveLength, iWaveLength, row_tof, col_tof, NZ, NX, taps;

        FmcImaging(int _WaveNum, int _WaveLength, int _iWaveLength, int _row_tof, int _col_tof, int _NZ, int _NX, int _taps);
        ~FmcImaging();
        void copy_FMC(char* h_offLineFmc, char* d_offLineFmc_char, float* d_offLineFmc_float);
        void transpose(short *iTof, int row_tof, int col_tof);
        void get_filter(float f1, float f2, cufftComplex* filter);
        void filtering(float* offLineFmc, cufftComplex* f_filter, cufftComplex* f_offLineFmc);
        void hilbert_transform(cufftComplex *f_offLineFmc, cufftComplex* HilbertFilter);
        void imaging(char* offLineFmc, short* Tof, cufftComplex* FmcMatHilbert, float* TfmImage, float MindB);
        void read_FMC(std::string filepath, char *input, int row, int col);
        void read_TOF(std::string filepath, short *input, int row, int col);
        void save_result_to_txt(std::string filepath, float* d_result);
};
