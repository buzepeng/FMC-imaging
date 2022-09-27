# 希尔伯特滤波
用cuda实现2048*4000维信号的频域带通滤波，并进行了希尔伯特变换，最后用全聚焦成像算法生成图像，在jupyter notebook中验证成像的结果。单帧成像cpu用时40s左右，gpu用时28ms左右。

# cuda计算结果
利用cuda计算得到的成像结果如图：

![FMC](img/result.png)