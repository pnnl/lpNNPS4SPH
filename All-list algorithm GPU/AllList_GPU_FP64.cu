
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define N 1000000
#define R_CUT 0.02
#define max_neighbor 100

__global__ void search_neighbors(double *x, double *y, double *z, int *neighbors, int *num_neighbors, int start_j, int end_j) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= N) return;

    for (int j = start_j; j < end_j; j++)
    {
        if (i != j)
        {
            double dx = x[i] - x[j];
            double dy = y[i] - y[j];
            double dz = z[i] - z[j];
            double r = sqrt(dx*dx + dy*dy + dz*dz);
            if (r <= R_CUT)
            {
                int count = atomicAdd(&num_neighbors[i], 1);
                neighbors[i*max_neighbor + count] = j;
            }
        }
    }
}

int main() {
    double x[N], y[N], z[N];
    int neighbors[N*max_neighbor], num_neighbors[N];
    FILE *fp;

    fp = fopen("coordinates_1000000.dat", "r");

    for (int i = 0; i < N; i++) {
        fscanf(fp, "%lf %lf %lf", &x[i], &y[i], &z[i]);
    }

    fclose(fp);

    // for (int i = 0; i < N; i++) {
    //     x[i] = rand() / (double)RAND_MAX;
    //     y[i] = rand() / (double)RAND_MAX;
    //     z[i] = rand() / (double)RAND_MAX;
    // }

    double *d_x, *d_y, *d_z;
    int *d_neighbors, *d_num_neighbors;

    cudaMalloc((void **)&d_x, N*sizeof(double));
    cudaMalloc((void **)&d_y, N*sizeof(double));
    cudaMalloc((void **)&d_z, N*sizeof(double));
    cudaMalloc((void **)&d_neighbors, N*max_neighbor*sizeof(int));
    cudaMalloc((void **)&d_num_neighbors, N*sizeof(int));

    cudaMemcpy(d_x, x, N*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, N*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_z, z, N*sizeof(double), cudaMemcpyHostToDevice);

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

    printf("Elapsed time: %.3f ms\n", elapsed_time_ms);

    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_z);
    cudaFree(d_neighbors);
    cudaFree(d_num_neighbors);

    return 0;
}
