// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!
// !  This script is developed to search neighbors for SPH with all-list algorithm %
// !                                                                         %
// !  Author: Zirui Mao (Pacific Northwest National Laboratory)              %
// !  Date last modified: Sept. 05, 2023                                     %
// !                                                                         %
// ! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
// !  !!!!!!!!! Read me before using !!!!!!!!!!                              %
// !  By using this freeware, you are agree to the following:                %
// !  1. you are free to copy and redistribute the material in any format;   %
// !  2. you are free to remix, transform, and build upon the material for   %
// !     any purpose, even commercially;                                     %
// !  3. you must provide the name of the creator and attribution parties,   %
// !     a copyright notice, a license notice, a disclaimer notice, and a    % 
// !     link to the material [link];                                        %
// !  4. users are entirely at their own risk using this freeware.           %
// !                                                                         %
// !  Before use, please read the License carefully:                         %
// !  <a rel="license" href="http://creativecommons.org/licenses/by/4.0/">   %
// !  <img alt="Creative Commons License" style="border-width:0"             %
// !  src="https://i.creativecommons.org/l/by/4.0/88x31.png" /></a><br />    %
// !  This work is licensed under a                                          %
// !  <a rel="license" href="http://creativecommons.org/licenses/by/4.0/">   %
// !  Creative Commons Attribution 4.0 International License</a>.   

// //////// works with large data /////////
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define N 1000000
#define R_CUT 0.02
#define max_neighbor 100

__global__ void search_neighbors(float *x, float *y, float *z, int *neighbors, int *num_neighbors, int start_j, int end_j) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= N) return;

    for (int j = start_j; j < end_j; j++)
    {
        if (i != j)
        {
            float dx = x[i] - x[j];
            float dy = y[i] - y[j];
            float dz = z[i] - z[j];
            float r = sqrtf(dx*dx + dy*dy + dz*dz);
            if (r <= R_CUT)
            {
                int count = atomicAdd(&num_neighbors[i], 1);
                neighbors[i*max_neighbor + count] = j;
            }
        }
    }
}

int main() {
    float x[N], y[N], z[N];
    int neighbors[N*max_neighbor], num_neighbors[N];
    FILE *fp;

    fp = fopen("coordinates_1000000.dat", "r");

    for (int i = 0; i < N; i++) {
        fscanf(fp, "%f %f %f", &x[i], &y[i], &z[i]);
        x[i] = x[i];
        y[i] = y[i];
        z[i] = z[i];
    }

    fclose(fp);

    float *d_x, *d_y, *d_z;
    int *d_neighbors, *d_num_neighbors;

    cudaMalloc((void **)&d_x, N*sizeof(float));
    cudaMalloc((void **)&d_y, N*sizeof(float));
    cudaMalloc((void **)&d_z, N*sizeof(float));
    cudaMalloc((void **)&d_neighbors, N*max_neighbor*sizeof(int));
    cudaMalloc((void **)&d_num_neighbors, N*sizeof(int));

    cudaMemcpy(d_x, x, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_z, z, N*sizeof(float), cudaMemcpyHostToDevice);

    int block_size = 256;
    int grid_size_x = (N + block_size - 1) / block_size;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    cudaMemset(d_num_neighbors, 0, N*sizeof(int));

    int num_splits = 8;
    int j_per_split = N / num_splits;
    for (int i = 0; i < num_splits; i++) {
        int start_j = i * j_per_split;
        int end_j = start_j + j_per_split;
        dim3 grid_size(grid_size_x, 1, 1);
        search_neighbors<<<grid_size, block_size>>>(d_x, d_y, d_z, d_neighbors, d_num_neighbors, start_j, end_j);
    }
   
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    float elapsed_time_ms;
    cudaEventElapsedTime(&elapsed_time_ms, start, stop);
    
    cudaMemcpy(neighbors, d_neighbors, N*max_neighbor*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(num_neighbors, d_num_neighbors, N*sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = N-1; i < N; i++) {
        printf("Particle %d neighbors: ", i);
        for (int j = 0; j < num_neighbors[i]; j++) {
            printf("%d ", neighbors[i*max_neighbor + j]);
        }
        printf("\n");
    }

    printf("number of neighbors = %d \n", num_neighbors[N-1]);

    printf("Elapsed time: %.3f ms\n", elapsed_time_ms);

    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_z);
    cudaFree(d_neighbors);
    cudaFree(d_num_neighbors);

    return 0;
}
