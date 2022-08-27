#include <opencv2/core/core.hpp>
#include <iostream>
#include <fstream>  
#include <chrono>
#include <iomanip>
#define NY 1000
#define NX 1500

using namespace std;

void hilbert(float* data, const int length)//data为输入一维数组，length为数组长度
{
	vector<cv::Complex<float>> dft_one(length, cv::Complex<float>(0, 0));
	vector<cv::Complex<float>> dft_out(length, cv::Complex<float>(0, 0));
	float* x_r = new float[length]();
	float* x_i = new float[length]();
	float* z_r = new float[length]();
	float* z_i = new float[length]();
	float* y_r = new float[length]();
	float* y_i = new float[length]();
	float* s_r = new float[length]();
	float* s_i = new float[length]();
	float* m_r = new float[length]();
	float* m_i = new float[length]();
	float* XK = new float[length]();
	float* xr = new float[length]();
	float* xi = new float[length]();
	float* amp = new float[length]();
	memset(m_i, 0, sizeof(float)*length);
	memset(m_r, 0, sizeof(float)*length);
	// for (int k = 0; k < length; ++k)
	// {
	// 	for (int n = 0; n < length; ++n)
	// 	{
	// 		dft_one[n].re = data[n] * cos(2 * CV_PI / length * n * k);//实部信号
	// 		dft_one[n].im = data[n] * sin(2 * CV_PI / length * n * k);//虚部信号
	// 		dft_out[k].re += dft_one[n].re;
	// 		dft_out[k].im += dft_one[n].im;//DFT后的实部，虚部相加
	// 	}
	// }

	for (int k = 0; k < length; k++)//DFT
	{
		for (int n = 0; n < length; n++)
		{
			dft_one[n].re = data[n] * cos(2 * CV_PI / length * n * k);//实部信号
			dft_one[n].im = -data[n] * sin(2 * CV_PI / length * n * k);//虚部信号
			dft_out[k].re += dft_one[n].re;
			dft_out[k].im += dft_one[n].im;//DFT后的实部，虚部相加
		}
		x_r[k] = dft_out[k].re;
		x_i[k] = dft_out[k].im;
	}

	for (int n = 0; n < length; n++)//得到Z(K)
	{
		if (n == 0)
		{
			z_r[n] = x_r[n];
			z_i[n] = x_i[n];
		}
		if (0 < n && n < length / 2)
		{
			z_r[n] = 2 * x_r[n];
			z_i[n] = 2 * x_i[n];
		}
		if (length / 2 <= n && n < length)
		{
			z_r[n] = 0;
			z_i[n] = 0;
		}
	}

	for (int n = 0; n < length; n++) //idft
	{
		for (int k = 0; k < length; k++)
		{
			m_r[n] += z_r[k] * cos((2.0 * CV_PI) / length * n * k) - z_i[k] * sin((2.0 * CV_PI) / length * n * k);
			m_i[n] += z_i[k] * cos((2.0 * CV_PI) / length * n * k) + z_r[k] * sin((2.0 * CV_PI) / length * n * k);
		}
		s_r[n] = 1.0 / length * m_r[n];
		s_i[n] = 1.0 / length * m_i[n];
	}

	for (int n = 0; n < length; n++)//输出
	{
		xr[n] = data[n];//原始信号
		y_r[n] = s_i[n];
		y_i[n] = xr[n] - s_r[n];
		amp[n] = sqrt(y_r[n] * y_r[n] + y_i[n] * y_i[n]);
		data[n] = abs(y_i[n]);
	}
	
	delete[] x_r;
	delete[] x_i;
	delete[] z_r;
	delete[] z_i;
	delete[] y_r;
	delete[] y_i;
	delete[] s_r;
	delete[] s_i;
	delete[] m_r;
	delete[] m_i;
	delete[] XK;
	delete[] xr;
	delete[] xi;
	delete[] amp;
}

int main(){
    ifstream fp_input;
	ofstream fp_output;
    fp_input.open("./data/data.txt", ios::in);
	if (!fp_input) { //打开失败
        cout << "error opening source file." << endl;
        return 0;
    }
	fp_output.open("./data/output_cpu.txt", ios::out);
	if (!fp_output) {
        fp_input.close(); //程序结束前不能忘记关闭以前打开过的文件
        cout << "error opening destination file." << endl;
        return 0;
    }
    cv::Mat src_data = cv::Mat::zeros(NY, NX, CV_32FC1);
    for(int i = 0;i<NY;i++){
        for(int j = 0;j<NX; j++){
            fp_input >> src_data.at<float>(i, j);
        }
    }
	auto startTime = chrono::system_clock::now();
	for(int i = 0; i<NY; i++){
		hilbert(src_data.ptr<float>(i, 0), NX);
	}
	auto endTime = chrono::system_clock::now();
	cout << "cpu time:" << chrono::duration_cast<chrono::seconds>(endTime - startTime).count() << "s" << endl;
	for(int i = 0;i<NY;i++){
        for(int j = 0;j<NX;j++){
            if(j!=NX-1) fp_output<<src_data.at<float>(i,j)<<'\t';
            else fp_output<<src_data.at<float>(i,j);
        }
        fp_output<<'\n';
    }
	fp_input.close();
	fp_output.close();
	return 0;
}