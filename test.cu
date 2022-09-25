#include <thrust/extrema.h>
#include <thrust/execution_policy.h>
int main(){
    int data[6] = {1, 0, 2, 2, 1, 3};
    int *d_data;
    cudaMalloc((void **)&d_data, 6*sizeof(int));
    cudaMemcpy(d_data, data, 6*sizeof(int), cudaMemcpyHostToDevice);
    int h_result, *d_result = thrust::max_element(thrust::device, d_data, d_data + 6);
    cudaMemcpy(&h_result, d_result, sizeof(int), cudaMemcpyDeviceToHost);
    std::cout<<h_result<<std::endl;
    cudaFree(d_data);
}