# 希尔伯特滤波
用cuda实现信号的希尔伯特变换然后进行带通滤波，在jupyter notebook中验证滤波的结果

# cuda程序编译及运行
新建build目录，用cmake进行编译
```
mkdir build && cd build && cmake .. && make

./hilbert
```