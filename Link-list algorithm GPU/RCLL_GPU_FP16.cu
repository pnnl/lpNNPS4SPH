// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!
// !  This script is developed to search neighbors for SPH with relative     % 
//    coordinate-based link list algorithm %
// !                                                                         %
// !  Author: Zirui Mao & Ang Li (Pacific Northwest National Laboratory)     %
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

//// REMEMBER: make NCX, NCY, NCZ consistent with the cell-size! and choose a sufficiently large max_particles_per_cell
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_fp16.h>

#define N 10000000
#define R_CUT 0.008
#define max_neighbor 100
#define cell_size 0.008 //// basically, this value is suggested consistent with R_CUT
#define NCX 125
#define NCY 125
#define NCZ 125
#define max_particles_per_cell 1000 // generally N/NCX/NCY/NCZ*10 is OK


__global__ void search_neighbors(int *cell_indices, int *cell_offsets, int *cell_particle_list, __half *relative_coordinates, int *neighbors, int *num_neighbors) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    int ix_cell = cell_indices[i*3];
    int iy_cell = cell_indices[i*3+1];
    int iz_cell = cell_indices[i*3+2];
    int ix_cellm = (ix_cell>0)?ix_cell-1:ix_cell;
    int ix_cellp = (ix_cell<NCX-1)?ix_cell+1:ix_cell;
    int iy_cellm = (iy_cell>0)?iy_cell-1:iy_cell;
    int iy_cellp = (iy_cell<NCY-1)?iy_cell+1:iy_cell;
    int iz_cellm = (iz_cell>0)?iz_cell-1:iz_cell;
    int iz_cellp = (iz_cell<NCZ-1)?iz_cell+1:iz_cell;
    
    for (int iix_cell=ix_cellm; iix_cell<=ix_cellp; iix_cell++)
    {
        for (int iiy_cell=iy_cellm; iiy_cell<=iy_cellp;iiy_cell++)
        {
            for(int iiz_cell=iz_cellm; iiz_cell<=iz_cellp;iiz_cell++)
            {
                for (int j = 0; j < cell_offsets[iix_cell*NCY*NCZ + iiy_cell*NCZ + iiz_cell]; j++)
                {
                    int ii = cell_particle_list[iix_cell*NCY*NCZ*max_particles_per_cell + iiy_cell*NCZ*max_particles_per_cell + iiz_cell*max_particles_per_cell + j];
                    if (ii != i)
                    {
                        __half dx = relative_coordinates[i*3] - relative_coordinates[ii*3] + __float2half((ix_cell - iix_cell)*cell_size);
                        __half dy = relative_coordinates[i*3+1] - relative_coordinates[ii*3+1] + __float2half((iy_cell - iiy_cell)*cell_size);
                        __half dz = relative_coordinates[i*3+2] - relative_coordinates[ii*3+2] + __float2half((iz_cell - iiz_cell)*cell_size);
                        __half r = (sqrtf(dx*dx + dy*dy + dz*dz));
                        
                        if (r <= __float2half(R_CUT))
                        {
                            int count = atomicAdd(&num_neighbors[i], 1);
                            neighbors[i*max_neighbor + count] = ii;
                        }
                    }
                }
            }
        }
    }

    
}


__global__ void assign_particles_to_cells(float *x, float *y, float *z, int *cell_indices, int *cell_offsets, int *cell_particle_list, __half *relative_coordinates)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    // Compute the cell indices for the current particle
    int cix = (int)(floorf(x[i] / (cell_size)));
    int ciy = (int)(floorf(y[i] / (cell_size)));
    int ciz = (int)(floorf(z[i] / (cell_size)));

    // if (cix>=NCX) cix = NCX - 1;
    // if (ciy>=NCY) ciy = NCY - 1;
    // if (ciz >= NCZ) ciz = NCZ - 1;
    // if (cix>=NCX || ciy>=NCY || ciz >= NCZ) printf("!!!!!! \n");
    // if (cix<0 || ciy<0 || ciz<0) printf(">>>>>>>> \n");
    

    // Compute the offset in the cell_particle_list for the current cell
    int cell_offset = atomicAdd(&cell_offsets[cix*NCY*NCZ + ciy*NCZ + ciz], 1);

    // Add the particle index to the cell_particle_list
    cell_particle_list[cix*NCY*NCZ*max_particles_per_cell + ciy*NCZ*max_particles_per_cell + ciz*max_particles_per_cell + cell_offset] = i;
    if (cell_offset>max_particles_per_cell) {
        printf("!!! The max_particles_per_cell is too small!!!! \n"); 
    }

    // Compute the relative coordinates of the particle in the cell
    relative_coordinates[i*3] = __float2half(x[i] - (cix+0.5f)*cell_size);
    relative_coordinates[i*3 + 1] = __float2half(y[i] - (ciy+0.5f)*cell_size);
    relative_coordinates[i*3 + 2] = __float2half(z[i] - (ciz+0.5f)*cell_size);

    // Store the cell indices for the current particle
    cell_indices[i*3] = cix;
    cell_indices[i*3 + 1] = ciy;
    cell_indices[i*3 + 2] = ciz;
}


int main() {
    float x[N], y[N], z[N];
    int neighbors[N*max_neighbor], num_neighbors[N];
    FILE *fp;

    fp = fopen("coordinates_10000000.dat", "r");

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

    int num_cells = NCX * NCY * NCZ;
    int *cell_indices = (int*)malloc(N * 3 * sizeof(int));
    int *cell_offsets = (int*)malloc(num_cells * sizeof(int));
    int *cell_particle_list = (int*)malloc(num_cells * max_particles_per_cell * sizeof(int));
    half *relative_coordinates = (half*)malloc(N * 3 * sizeof(half));
    int *d_cell_indices, *d_cell_offsets, *d_cell_particle_list;
    half *d_relative_coordinates;
    cudaMalloc(&d_cell_indices, N * 3 * sizeof(int));
    cudaMalloc(&d_cell_offsets, num_cells * sizeof(int));
    cudaMalloc(&d_cell_particle_list, num_cells * max_particles_per_cell * sizeof(int));
    cudaMalloc(&d_relative_coordinates, N * 3 * sizeof(half));

    // Define the grid and block dimensions
    int block_size =256;
    // int grid_size_x = (N + block_size - 1) / block_size;
    int num_blocks = (N + block_size - 1) / block_size;
    dim3 grid_dim(num_blocks, 1, 1);
    dim3 block_dim(block_size, 1, 1);
    

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);


    // Call the kernel
    cudaMemset(d_num_neighbors, 0, N*sizeof(int));
    assign_particles_to_cells<<<grid_dim, block_dim>>>(d_x, d_y, d_z, d_cell_indices, d_cell_offsets, d_cell_particle_list, d_relative_coordinates);
    search_neighbors<<<grid_dim, block_dim>>>(d_cell_indices, d_cell_offsets, d_cell_particle_list, d_relative_coordinates,d_neighbors, d_num_neighbors);
   
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float elapsed_time_ms;
    cudaEventElapsedTime(&elapsed_time_ms, start, stop);
    
    cudaMemcpy(neighbors, d_neighbors, N*max_neighbor*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(num_neighbors, d_num_neighbors, N*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(cell_indices, d_cell_indices, N * 3 * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(cell_offsets, d_cell_offsets, num_cells * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(cell_particle_list, d_cell_particle_list, num_cells * max_particles_per_cell * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(relative_coordinates, d_relative_coordinates, N * 3 * sizeof(half), cudaMemcpyDeviceToHost);

    for (int i = N-1; i < N; i++) {
        printf("Particle %d neighbors: ", i);
        for (int j = 0; j < num_neighbors[i]; j++) {
            printf("%d ", neighbors[i*max_neighbor + j]);
        }
        printf("\n");
    }

    // for (int i = N-10; i < N; i++) {
    //     printf("Particle %d neighbors: ", i);
    //     printf("%f %f %f  ", x[i],y[i],z[i]);
    //     printf("%d %d %d", cell_indices[i*3],cell_indices[i*3+1],cell_indices[i*3+2]);
    //     printf("\n");
    // }
    // for (int i = 0; i < cell_offsets[0*NCY*NCZ + 0*NCZ + 0]; i++) {
    //     int ii = cell_particle_list[0*NCY*NCZ*max_particles_per_cell + 0*NCZ*max_particles_per_cell + 0*max_particles_per_cell + i];
    //     printf("Cell %d has %d particles: ", 0, i);
    //     printf("%f %f %f  ", __half2float(x[ii]),__half2float(y[ii]),__half2float(z[ii]));
    //     printf("%f %f %f ", __half2float(relative_coordinates[ii*3]),__half2float(relative_coordinates[ii*3+1]),__half2float(relative_coordinates[ii*3+2]));
    //     printf("\n");
    // }

    printf("number of neighbors = %d \n", num_neighbors[N-1]);

    printf("Elapsed time: %.3f ms\n", elapsed_time_ms);

    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_z);
    cudaFree(d_neighbors);
    cudaFree(d_num_neighbors);
    cudaFree(d_cell_indices);
    cudaFree(d_cell_offsets);
    cudaFree(d_cell_particle_list);
    cudaFree(d_relative_coordinates);

    return 0;
}
