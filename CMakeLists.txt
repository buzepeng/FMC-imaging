cmake_minimum_required(VERSION 3.10)

project(hilbert LANGUAGES CUDA CXX)

find_package(CUDA REQUIRED)

cuda_add_executable(hilbert src/main.cu src/FMCImaging.cu OPTIONS -O2 --std=c++11 --expt-extended-lambda)
target_include_directories(hilbert PRIVATE src)
target_link_libraries(hilbert cufft)
set_target_properties(
    hilbert 
    PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES 75
)