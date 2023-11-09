# Nearest Neighboring Particles Searching (NNPS) algorithms for Smoothed Particle Hydrodynamics method

## Source code 1: CPU code of all-list NNPS algorithm
All-list NNPS algorithm searches neighbors within a certain searching radius from a target particle by explicitly checking all the particles in domain.
Two independent source codes, one takes float-point-64 or FP64 computation while the other one takes FP32.
- AllList_CPU_FP64.f90
- AllList_CPU_FP32.f90

To compile the FORTRAN code on CPUs, using: gfortran AllList_FPU_FP*.f90 -o AL_CPU_FP* -O3

To run the computation, using: ./AL_CPU_FP*

Note that, the '*' above can be chosen as '64' or '32' to test the two independent source code.

To make a more accurate performance comparason of the various precisions, it is suggested to use the exact same particles distribution stored in 'coordinate*.txt', rather than generting particles randomly.

## Source code 2: GPU code of all-list NNPS algorithm
Three independent source codes taking FP64, FP32, and FP16.
- AllList_GPU_FP64.cu
- AllList_GPU_FP32.cu
- AllList_GPU_FP16.cu

To compile the CUDA code on GPUs, using: nvcc AllList_GPU_FP*.cu -o AL_GPU_FP* -arc=sm_80 -use_fast_math

To run it, using: ./AL_GPU_FP*

Note that, the '*' above can be chosen as '64', '32', or '16' to test the three independent source code.

If memory segmentation fault is reported, try to resovle it by seting unlimited memory using: limit -s unlimited

To make a more accurate performance comparason of the various precisions, it is suggested to use the exact same particles distribution stored in 'coordinate*.txt', rather than generting particles randomly.

## Source code 3: GPU code of Relative Coordinate-based Link List (RCLL) algorithm
RCLL algorithm searches neighbors within a certain searching radius from a target particle by checking only surrounding particles. Particles are assigned into background cells. Only those particles locating in the same or adjecent cells are subjective to be checked. To achieve the best accuracy of FP16, the coordinates of particles are expressed in terms of the relative coordinate within the cell and the cell center's index.

Three independent source codes taking FP64, FP32, and FP16.
- RCLL_GPU_FP64.cu
- RCLL_GPU_FP32.cu
- RCLL_GPU_FP16.cu

To compile the CUDA code on GPUs, using: nvcc RCLL_GPU_FP*.cu -o RCLL_GPU_FP* -arc=sm_80 -use_fast_math 

To run it, using: ./RCLL_GPU_FP*

Note that, the '*' above can be chosen as '64', '32', or '16' to test the three independent source code.

If memory segmentation fault is reported, try to resovle it by seting unlimited memory using: limit -s unlimited

To make a more accurate performance comparason of the various precisions, it is suggested to use the exact same particles distribution stored in 'coordinate*.txt', rather than generting particles randomly.

## Source code 4: GPU code of RCLL algorithm applied to gradient approximation with SPH method
The SPH method approximates the gradient of a polynomial function based on the neighbors list obtained from RCLL algorithm.
Four independent source codes taking FP64, FP32, and FP16.
- SPH_RCLL_GPU_FP64.cu
- SPH_RCLL_GPU_FP32.cu
- SPH_RCLL_GPU_FP16.cu
- SPH_RCLL_GPU_FP16_sort.cu

The source code 'SPH_RCLL_GPU_FP16_sort.cu' is a optimized version of the 'SPH_RCLL_GPU_FP16.cu'. It sorts the particles based on their spatial distribution. 

In this way, the GPU memory bandwidth can be utilized more effectively in GPU parallel computation.

To compile the CUDA code on GPUs, using: nvcc SPH_RCLL_GPU_FP*.cu -o SPH_RCLL_GPU_FP* -arc=sm_80 -use_fast_math

TO compile the optimized FP16 code, using: ncvv --extended-lambda -std=c++14 -arch=sm_80  SPH_RCLL_GPU_FP16_sort.cu -o SPH_RCLL_GPU_FP16

To run it, using: ./SPH_RCLL_GPU_FP*

Note that, the '*' above can be chosen as '64', '32', or '16' to test the three independent source code.

If memory segmentation fault is reported, try to resovle it by seting unlimited memory using: limit -s unlimited

To make a more accurate performance comparason of the various precisions, it is suggested to use the exact same particles distribution stored in 'coordinate*.txt', rather than generting particles randomly.

# Performance comparison of different versions of code
- The efficiency advantage of lower precision computation can be demonstrated by comparing *_FP16.cu to *_FP32.cu and *_FP64.cu.
- The great efficiency advantage of GPU parallel computation over CPU series computation can be highlighted by comparing the source code 2 to source code 1.
- The great efficiency advantage of RCLL over all-list algorithm on GPUs can be evidienced by comparing source code 3 to source code 2.
- The efficency enhancement obtained from properly managing GPU memory bandwidth utilization can be demonstrated by comparing SPH_RCLL_GPU_FP16_sort.cu to SPH_RCLL_GPU_FP16.cu.


## Authors
Zirui Mao (PNNL), Ang Li (PNNL), Xinyi Li (PNNL and Utah)

## Citation
Please cite our [Paper](https://arxiv.org/submit/5226355/preview).

## License
This project is licensed under the BSD license, see [license](https://github.com/pnnl/lpNNPS4SPH/blob/master/LICENSE.txt)  for details.

## Acknowledgements
**PNNL-IPID:32908-E**

This material is based upon work supported by the U.S. Department of Energy, Office of Science, Office of Advanced Scientific Computing Research, ComPort: Rigorous Testing Methods to Safeguard Software Porting, under Award Number 78284. This research used resources of the National Energy Research Scientific Computing Center (NERSC), a U.S. Department of Energy Office of Science User Facility located at Lawrence Berkeley National Laboratory, operated under Contract No. DE-AC02-05CH11231. This research used resources of the Oak Ridge Leadership Computing Facility, which is a DOE Office of Science User Facility supported under Contract DE-AC05-00OR22725. The Pacific Northwest National Laboratory is operated by Battelle for the U.S. Department of Energy under Contract DE-AC05-76RL01830.
