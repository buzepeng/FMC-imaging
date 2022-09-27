#include <vector>
#include 

#define PI acos(-1)
std::vector<float> sinc(std::vector<float> x)
{
   std::vector<float> y;
   for (int i = 0; i < x.size(); i++)
   {
   	float temp = PI * x[i];
   	if (temp == 0) {
   		y.push_back(0.0);
   	}
   	else {
   		y.push_back(sin(temp) / temp);
   	}
   }
   return y;
}

std::vector <float> fir1(int n, std::vector<float> Wn)
{

   /*
 
   	参数输入介绍：
   		n：  对应matlab的fir1里的阶数n
   		Wn:  对应matlab的fir1里的阶数Wn，但应注意传进
   				 来的数据应存在一个vector的float数组里。

   	参数输出介绍：
   				vector <float>的一个数组，里面存的是长度
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
   std::vector<std::vector <float>> bands;
   for (int i = 0; i < Wn.size();) {
   	std::vector<float> temp = { Wn[i], Wn[i + 1] };
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
   float alpha = 0.5 * (n - 1);
   std::vector<float> m;
   std::vector<float> h;
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
   	float left = Wn[i];
   	float right = Wn[i+1];
   	std::vector<float> R_sin, L_sin;
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
   std::vector <float> Win;
   for (int i = 0; i < n; i++)
   {
   	Win.push_back(0.54 - 0.46*cos(2.0 * PI * i / (n - 1)));	//hamming窗系数计算公式
   	h[i] *= Win[i];
   }

   bool scale = true;
   // 如果需要，现在可以处理缩放.
   if (scale) {
   	float left = bands[0][0];
   	float right = bands[0][1];
   	float scale_frequency = 0.0;
   	if (left == 0)
   		scale_frequency = 0.0;
   	else if (right == 1)
   		scale_frequency = 1.0;
   	else
   		scale_frequency = 0.5 * (left + right);

   	std::vector<float> c;
   	for (int i = 0; i < m.size(); i++) {
   		c.push_back(cos(PI * m[i] * scale_frequency));
   	}
   	float s = 0.0;
   	for (int i = 0; i < h.size(); i++) {
   		s += h[i] * c[i];
   	}
   	for (int i = 0; i < h.size(); i++) {
   		h[i] /= s;
   	}
   }
   return h;
}