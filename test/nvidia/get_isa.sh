#!/bin/sh

#nvcc -O3 -arch=sm_35 -I ./ --cubin -o lulesh_temp.cubin lulesh.cu
cuobjdump -sass axcuda_v1.cubin > axcuda_v1.isa
rm axcuda_v1.cubin
