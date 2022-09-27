// #include <stdio.h>
// #include <math.h>
// #include <vector>
// #include <iostream>
// using namespace std;

// // static double sinc(const double x)
// // {
// //     if (x == 0)
// //         return 1;

// //     return sin(M_PI * x) / (M_PI * x);
// // }

// // std::vector<double> window_hammig(int taps)
// // {
// //     std::vector<int>    n(taps, 0);
// //     std::vector<double> w(taps, 0);

// //     double alpha   = 0.54;
// //     double beta    = 0.46;

// //     for(int i = 0; i < taps; i++) {
// //         w[i] = alpha - beta * cos(2.0 * M_PI * i / (taps - 1));
// //     }

// //     return w;
// // }

// // std::vector<double> bandPass_coefficient(int taps, double f1, double f2)
// // {
// //     std::vector<int>    n(taps, 0);
// //     std::vector<double> h(taps, 0);

// //     for(int i = 0; i < taps; i++) {
// //         n[i] = i - int(taps/2);
// //     }

// //     for(int i = 0; i < taps; i++) {
// //         h[i] = f2*sinc(f2*n[i]) - f1*sinc(f1*n[i]);
// //     }

// //     return h;
// // }

// // int main(){
// //     int NN = 100;
// //     double fs = 1e8;
// //     std::vector<double> wn = {4e6/fs, 16e6/fs};
// //     std::vector<double> h = bandPass_coefficient(NN-1, wn[0], wn[1]), w = window_hammig(NN-1);
// //     for(int i = 0;i<99;i++){
// //         h[i] = h[i]*w[i];
// //     }
// //     double f = (wn[0]+wn[1])/2, s = 0;
// //     for(int i = 0;i<99;i++){
// //         int m = i - int(100)
// //         s += cos(M_PI*f)
// //     }

// //     for(int i = 0;i<99;i++){
// //         std::cout<<h[i]<<" ";
// //     }
// // }

//    #include <stdio.h>
//    #include <math.h>
//    #include <vector>
//    #include <iostream>
//    using namespace std;
//    #define PI acos(-1)
   
//    double sincEasy(double *x, int len, int index) {
//        double temp = PI * x[index];
//        double y;
//        if (temp == 0) {
//            y = 0.0;
//        }
//        else {
//            y = sin(temp) / temp;
//        }
//        return y;
//    }

//    //滤波器系数
//    double *fir1(
//            int lbflen,
//            double Wn[],
//            double lbf[])
//    {

//        /*
//            未写排错  检查输入有需要自己进行完善
//            原matlab函数fir(j, wn)	【函数默认使用hamming】

//            参数输入介绍：
//                j：  对应matlab的fir1里的阶数j
//                Wn:  对应matlab的fir1里的阶数Wn，但应注意传进
//                         来的数据应存在一个vector的double数组里。

//            参数输出介绍：
//                        vector <double>的一个数组，里面存的是长度
//                        为j的滤波器系数。
//        */

//        //在截止点的左端插入0（在右边插入1也行）
//        //使得截止点的长度为偶数，并且每一对截止点对应于通带。
//        //if (Wn.size() == 1 || Wn.size() % 2 != 0) {
//        //	Wn.insert(Wn.begin(), 0.0);
//        //}

//        double alpha = 0.5 * (lbflen - 1);
//        double *m = new double[lbflen];
//        for (int i = 0; i < lbflen; i++) {
//            m[i] = i - alpha;
//            lbf[i] = 0;
//        }

//        double *R_sin = new double[lbflen];
//        double *L_sin = new double[lbflen];
//        for (int i = 0; i < 2;) {
//            double left = Wn[i];
//            double right = Wn[i + 1];
//            for (int j = 0; j < lbflen; j++) {
//                R_sin[j] = right * m[j];
//                L_sin[j] = left * m[j];
//            }
//            for (int j = 0; j < lbflen; j++) {
//                lbf[j] += right * sincEasy(R_sin, lbflen, j);
//                lbf[j] -= left * sincEasy(L_sin, lbflen, j);
//            }

//            i = i + 2;
//        }

//        // 应用窗口函数，这里和matlab一样
//        // 默认使用hamming，要用别的窗可以去matlab查对应窗的公式。
//        for (int i = 0; i < lbflen; i++)
//        {
//            double Win = 0.54 - 0.46*cos(2.0 * PI * i / (lbflen - 1));	//hamming窗系数计算公式
//            lbf[i] *= Win;
//        }

//        // 如果需要，现在可以处理缩放.
//        if (true) {
//            double left = Wn[0];
//            double right = Wn[1];
//            double scale_frequency = 0.0;
//            if (left == 0)
//                scale_frequency = 0.0;
//            else if (right == 1)
//                scale_frequency = 1.0;
//            else
//                scale_frequency = 0.5 * (left + right);

//            double s = 0.0;
//            for (int i = 0; i < lbflen; i++) {
//                double c = cos(PI * m[i] * scale_frequency);
//                s += lbf[i] * c;
//            }
//            for (int i = 0; i < lbflen; i++) {
//                lbf[i] /= s;
//            }
//        }
//        delete[] m;
//        delete[] R_sin, L_sin;
//        return lbf;
//    }

// int main(){
//     int NN = 100;
//     double fs = 1e8;
//     std::vector<double> wn = {4e6/fs, 16e6/fs};
// }
#include <stdio.h>
#include <math.h>
#include <vector>
#include <iostream>
using namespace std;
#define PI acos(-1)
vector<double> sinc(vector<double> x)
{
   vector<double> y;
   for (int i = 0; i < x.size(); i++)
   {
   	double temp = PI * x[i];
   	if (temp == 0) {
   		y.push_back(0.0);
   	}
   	else {
   		y.push_back(sin(temp) / temp);
   	}
   }
   return y;
}

vector <double> fir1(int n, vector<double> Wn)
{

   /*
   	未写排错  检查输入有需要自己进行完善
   	原matlab函数fir(n, wn)	【函数默认使用hamming】

   	参数输入介绍：
   		n：  对应matlab的fir1里的阶数n
   		Wn:  对应matlab的fir1里的阶数Wn，但应注意传进
   				 来的数据应存在一个vector的double数组里。

   	参数输出介绍：
   				vector <double>的一个数组，里面存的是长度
   				为n的滤波器系数。
   */
   
   //在截止点的左端插入0（在右边插入1也行）
   //使得截止点的长度为偶数，并且每一对截止点对应于通带。
   if (Wn.size() == 1 || Wn.size() % 2 != 0) {
   	Wn.insert(Wn.begin(), 0.0);
   }

   /*
   	‘ bands’是一个二维数组，每行给出一个 passband 的左右边缘。
   	（即每2个元素组成一个区间）
   */
   vector<vector <double>> bands;
   for (int i = 0; i < Wn.size();) {
   	vector<double> temp = { Wn[i], Wn[i + 1] };
   	bands.push_back(temp);
   	i = i + 2;
   }

   // 建立系数
   /*
   	m = [0-(n-1)/2,
   		 1-(n-1)/2,
   		 2-(n-1)/2,
   		 ......
   		 255-(n-1)/2]
   	h = [0,0,0......,0]
   */
   double alpha = 0.5 * (n - 1);
   vector<double> m;
   vector<double> h;
   for (int i = 0; i < n; i++) {
   	m.push_back(i - alpha);
   	h.push_back(0);
   }
   /*
   	对于一组区间的h计算
   	left:	一组区间的左边界
   	right:  一组区间的右边界
   */
   for (int i = 0; i < Wn.size();) {
   	double left = Wn[i];
   	double right = Wn[i+1];
   	vector<double> R_sin, L_sin;
   	for (int j = 0; j < m.size(); j++) {
   		R_sin.push_back(right * m[j]);
   		L_sin.push_back(left * m[j]);
   	}
   	for (int j = 0; j < R_sin.size(); j++) {
   		h[j] += right * sinc(R_sin)[j];
   		h[j] -= left * sinc(L_sin)[j];
   	}

   	i = i + 2;
   }

   // 应用窗口函数，这里和matlab一样
   // 默认使用hamming，要用别的窗可以去matlab查对应窗的公式。
   vector <double> Win;
   for (int i = 0; i < n; i++)
   {
   	Win.push_back(0.54 - 0.46*cos(2.0 * PI * i / (n - 1)));	//hamming窗系数计算公式
   	h[i] *= Win[i];
   }

   bool scale = true;
   // 如果需要，现在可以处理缩放.
   if (scale) {
   	double left = bands[0][0];
   	double right = bands[0][1];
   	double scale_frequency = 0.0;
   	if (left == 0)
   		scale_frequency = 0.0;
   	else if (right == 1)
   		scale_frequency = 1.0;
   	else
   		scale_frequency = 0.5 * (left + right);

   	vector<double> c;
   	for (int i = 0; i < m.size(); i++) {
   		c.push_back(cos(PI * m[i] * scale_frequency));
   	}
   	double s = 0.0;
   	for (int i = 0; i < h.size(); i++) {
   		s += h[i] * c[i];
   	}
   	for (int i = 0; i < h.size(); i++) {
   		h[i] /= s;
   	}
   }
   return h;
}

int main(){
    int NN = 100;
    double fs = 1e8;
    std::vector<double> wn = {4e6/fs, 16e6/fs};
    std::vector<double> filter = fir1(NN, wn);
    for(int i = 0;i<100;i++){
        cout<<filter[i]<<" ";
    }
}