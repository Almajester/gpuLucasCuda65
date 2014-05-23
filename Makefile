#!/usr/bin/make -f
# Makefile for static build using cuda 6.5 (pre-release) on Ubuntu 14
#   --static builds are necessary for using pre and post fft callback functions
# A. Thall
# 5/22/2014

NVIDIA_SDK = /usr/local/cuda-6.5/samples
NVLIBDIR = /usr/local/cuda-6.5/lib64
NVLIBS = -L$(NVLIBDIR) -lcudart_static -lcufft_static -lcuos -lculibos
NVCC_ARCHES += -gencode arch=compute_35,code=sm_35
#NVCC_ARCHES += -gencode arch=compute_20,code=sm_21

CFLAGS = -O3 -Wall
NVCC_FLAGS =  -use_fast_math $(NVCC_ARCHES) --compiler-options="$(CFLAGS) -fno-strict-aliasing"

all: gpuLucas

gpuLucas.o: gpuLucas.cu IrrBaseBalanced.cu
	nvcc -dc -o $@ gpuLucas.cu -O3 -I$(NVIDIA_SDK)/common/inc $(NVCC_FLAGS) -w

gpuLucasLinked.o: gpuLucas.o
	nvcc $(NVCC_FLAGS) -dlink gpuLucas.o $(NVLIBS) -o gpuLucasLinked.o

gpuLucas: gpuLucasLinked.o
	g++ -fPIC $(CFLAGS) -o $@ $^  -Wl,-O1 -Wl,--as-needed gpuLucas.o $(NVLIBS) -lqd -ldl -lpthread -lrt

clean:
	-rm *.o 
	-rm gpuLucas
