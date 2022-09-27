#include <vector>
#include <iostream>
#include <math.h>
using namespace std;
const struct comDouble
{
   vector<double> real;
   vector<double> imag;
};
comDouble filter(vector<double> b, comDouble x)
{
   /*
   	注意：该函数仅实现了a为标量且为1时的函数滤波！！！
   	参数介绍：
   	b：		滤波器系数
   	a：		分母系数
   	x:		复数
   	x.real: 复数实部
   	x.imag: 复数虚部

   	Y：		复数滤波结果
   	Y.real：复数实部滤波结果
   	Y.imag：复数虚部滤波结果

   	公式：(a = 1时)
   	当 i < 滤波器阶数 时有:
   		Y[i] = ∑b[j]*x[i-j] (下限j=0, 上限j<i) 
   	当 i > 滤波器阶数 时有:
   		Y[i] = ∑b[j]*x[i-j] (下限j=0, 上限j<滤波器阶数)
   */
   comDouble Y;
   for (int i = 0; i < b.size(); i++)
   {
   		double real = 0.0;
   		double imag = 0.0;
   
   		for (int j = 0; j <= i; j++) {
   			real += b[j] * x.real[i - j];
   			imag += b[j] * x.imag[i - j];
   		}
   		Y.real.push_back(real);
   		Y.imag.push_back(imag);
   }
   for (int i = b.size(); i < x.real.size(); i++) {
   		double real = 0.0;
   		double imag = 0.0;
   		
   		for (int j = 0; j < b.size(); j++) {
   			real += b[j] * x.real[i - j];
   			imag += b[j] * x.imag[i - j];
   		}
   		
   		Y.real.push_back(real);
   		Y.imag.push_back(imag);
   }
   return Y;
}

int main(){
	double fs = 150, f1 = 10, f2 = 20, f3 = 30;
	vector<double> signal(151);
	for(int i = 0;i<151;i++){
		double t = i/fs;
		signal[i] = 
	}
}