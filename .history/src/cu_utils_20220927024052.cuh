__device__ cufftComplex operator * (cufftComplex a, cufftComplex b){
    cufftComplex res;
    res.x = (a.x*b.x - a.y*b.y);
    res.y = (a.x*b.y + a.y*b.x);
    return res;
}
__device__ cufftComplex operator * (cufftComplex a, float b){
    cufftComplex res;
    res.x = a.x*b;
    res.y = a.y*b;
    return res;
}
__device__ cufftComplex operator / (cufftComplex a, float b){
    cufftComplex res;
    res.x = a.x/b;
    res.y = a.y/b;
    return res;
}
__device__ cufftComplex operator + (cufftComplex a, cufftComplex b){
    cufftComplex res;
    res.x = a.x+b.x;
    res.y = a.y+b.y;
    return res;
}