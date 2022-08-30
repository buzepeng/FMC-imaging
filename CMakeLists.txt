cmake_minimum_required(VERSION 3.10)

project(hilbert LANGUAGES CUDA CXX)

find_package(CUDA REQUIRED)

set(CMAKE_BUILD_TYPE Release)

cuda_add_executable(hilbert src/hilbert_oneFrame.cu)
target_link_libraries(hilbert PRIVATE ${CUDA_LIBRARY})
CUDA_ADD_CUFFT_TO_TARGET(hilbert)