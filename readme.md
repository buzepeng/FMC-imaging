# 希尔伯特滤波
用cuda实现信号的带通滤波并进行希尔伯特变换，在jupyter notebook中验证滤波的结果

# cuda程序编译及运行
运行jupyter notebook中的生成测试数据代码块生成测试数据

新建build目录，用cmake进行编译
```
mkdir build && cd build && cmake .. && make

./hilbert
```