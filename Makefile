CC=g++
NVCC=nvcc
CUDAFLAGS= -arch=sm_61 -cudart=shared -rdc=true -g -G -O0 -std=c++11 
INCLUDES=-I/usr/local/include/opencv4
LIBDIRS = -L/usr/local/lib
LIBS = -lopencv_core -lcudart -lcufft
SRCPATH=./src
CPU=cpu
SRC1=$(SRCPATH)/hilbert.cpp
GPU=gpu
SRC2=$(SRCPATH)/hilbert_oneFrame.cu $(SRCPATH)/hilbert_multiFrame.cu

all: $(GPU)

$(CPU):
	$(NVCC) $(CUDAFLAGS) $(FLAGS) ${INCLUDES} ${LIBDIRS} $(LIBS) $(SRC1) -o $(CPU) 

$(GPU):
	$(NVCC) $(CUDAFLAGS) $(FLAGS) ${INCLUDES} ${LIBDIRS} $(LIBS) $(SRCPATH)/hilbert_oneFrame.cu -o gpu_oneFrame 
	# $(NVCC) $(CUDAFLAGS) $(FLAGS) ${INCLUDES} ${LIBDIRS} $(LIBS) $(SRCPATH)/hilbert_multiFrame.cu -o gpu_multiFrame

clean:
	$(RM) cpu gpu_oneFrame gpu_multiFrame
