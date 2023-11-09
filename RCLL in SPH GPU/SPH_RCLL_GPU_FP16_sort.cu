// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!
// !  This script is developed to search neighbors for SPH with relative     % 
//    coordinate-based link list algorithm %
// !                                                                         %
// !  Author: Zirui Mao & Ang Li (Pacific Northwest National Laboratory)     %
// !          Xinyi Li (University of Utah)
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

#define N 1000000   //10000|40000|100000|400000|1000000
#define R_CUT 0.0024  //0.1114|0.0702|0.0517|0.03257 | 0.024
#define max_neighbor 200
#define cell_size 0.0025 //// basically, this value is suggested consistent with R_CUT
#define NCX 400
#define NCY 400
#define max_particles_per_cell 1000 // generally N/NCX/NCY/NCZ*10 is OK

__global__ void calc_gradient(float *x, float *y, float *func, float *fx, float *ddx, int *neighbors, int *num_neighbors) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= N) return;

    for (int j = 0; j < num_neighbors[i]; j++)
    {
        int ii = neighbors[i*max_neighbor + j];
        float df_x = func[ii] - func[i];
        float dx = x[i] - x[ii];
        float dy = y[i] - y[ii];
        float r = sqrt(dx*dx + dy*dy);
        float q = r/R_CUT*2.0;
        float factor = 60.0/(7.0*3.141593*R_CUT*R_CUT);
        float dwx = 0.0; 
        if (q >= 0.0 && q <= 1.0)
        {
            dwx = factor * (-8.0 + 6.0*q)*dx/R_CUT/R_CUT;
        }
        else if (q > 1.0 && q < 2.0)
        {
            dwx = -factor * (2 - q)*(2 - q) * dx / r / R_CUT;
        }
        fx[i] = fx[i] + df_x * dwx;
        ddx[i] = ddx[i] - dx * dwx;
    }

    fx[i] = fx[i]/ddx[i];
}

__global__ void search_neighbors(int *cell_indices, int *cell_offsets, int *cell_particle_list, half2 *relative_coordinates, int *neighbors, int *num_neighbors) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    int ix_cell = cell_indices[i*2];
    int iy_cell = cell_indices[i*2+1];
    int ix_cellm = (ix_cell>0)?ix_cell-1:ix_cell;
    int ix_cellp = (ix_cell<NCX-1)?ix_cell+1:ix_cell;
    int iy_cellm = (iy_cell>0)?iy_cell-1:iy_cell;
    int iy_cellp = (iy_cell<NCY-1)?iy_cell+1:iy_cell;
    
    for (int iix_cell=ix_cellm; iix_cell<=ix_cellp; iix_cell++)
    {
        for (int iiy_cell=iy_cellm; iiy_cell<=iy_cellp;iiy_cell++)
        {
            for (int j = 0; j < cell_offsets[iix_cell*NCY+ iiy_cell]; j++)
            {
                int ii = cell_particle_list[iix_cell*NCY*max_particles_per_cell + iiy_cell*max_particles_per_cell + j];
                if (ii != i)
                {
                    __half2 dx;
                    dx.x = relative_coordinates[i].x - relative_coordinates[ii].x + __float2half(((ix_cell - iix_cell)*cell_size));
                    dx.y = relative_coordinates[i].y - relative_coordinates[ii].y + __float2half(((iy_cell - iiy_cell)*cell_size));
                    float r = __half2float(__hadd(__hmul(dx.x, dx.x), __hmul(dx.y, dx.y)));
                    // printf("r = %f\n", r);
                    if (r <= R_CUT*R_CUT)
                    {
                        int count = atomicAdd(&num_neighbors[i], 1);
                        neighbors[i*max_neighbor + count] = ii;
                    }
                }
            }
        }
    }
}

__global__ void assign_particles_to_cells(float *xx, float *yy, int *cell_indices, int *cell_offsets, int *cell_particle_list, half2 *relative_coordinates)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return; 

    // Compute the cell indices for the current particle
    int cix = (int)(floorf(xx[i] / cell_size));
    int ciy = (int)(floorf(yy[i] / cell_size));

    // Compute the offset in the cell_particle_list for the current cell
    int cell_offset = atomicAdd(&cell_offsets[cix*NCY + ciy], 1);

    // Add the particle index to the cell_particle_list
    cell_particle_list[cix*NCY*max_particles_per_cell + ciy*max_particles_per_cell + cell_offset] = i;
    if (cell_offset>max_particles_per_cell) {
        printf("!!! The max_particles_per_cell is too small!!!! \n"); 
    }

    // Compute the relative coordinates of the particle in the cell
    relative_coordinates[i].x = xx[i] - (cix+0.5f)*cell_size;
    relative_coordinates[i].y = yy[i] - (ciy+0.5f)*cell_size;

    // Store the cell indices for the current particle
    cell_indices[i*2] = cix;
    cell_indices[i*2 + 1] = ciy;
}


int main() {
    double x[N], y[N];
    float func[N],fx[N],ddx[N];
    float xx[N],yy[N];
    int neighbors[N*max_neighbor], num_neighbors[N];
    FILE *fp;

    fp = fopen("coordinates_2D_10000000.dat", "r");

    for (int i = 0; i < N; i++) {
        fscanf(fp, "%lf %lf", &x[i], &y[i]);
        xx[i] = float(x[i]);
        yy[i] = float(y[i]);
    }

    fclose(fp);

    //// define the function x
    for (int i = 0; i < N; i++)
    {
        func[i] = std::pow(xx[i], 3);
        ddx[i] = 0.0;
        fx[i] = 0.0;
    }

    float *d_xx, *d_yy, *d_func, *d_fx, *d_ddx;
    double *d_x, *d_y;
    int *d_neighbors, *d_num_neighbors;

    cudaMalloc((void **)&d_xx, N*sizeof(float));
    cudaMalloc((void **)&d_yy, N*sizeof(float));
    cudaMalloc((void **)&d_x, N*sizeof(double));
    cudaMalloc((void **)&d_y, N*sizeof(double));
    cudaMalloc((void **)&d_func, N*sizeof(float));
    cudaMalloc((void **)&d_fx, N*sizeof(float));
    cudaMalloc((void **)&d_ddx, N*sizeof(float));
    cudaMalloc((void **)&d_neighbors, N*max_neighbor*sizeof(int));
    cudaMalloc((void **)&d_num_neighbors, N*sizeof(int));

    cudaMemcpy(d_xx, xx, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_yy, yy, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, x, N*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, N*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_func, func, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_fx, fx, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ddx, ddx, N*sizeof(float), cudaMemcpyHostToDevice);

    int num_cells = NCX * NCY;
    int *cell_indices = (int*)malloc(N * 2 * sizeof(int));
    int *cell_offsets = (int*)malloc(num_cells * sizeof(int));
    int *cell_particle_list = (int*)malloc(num_cells * max_particles_per_cell * sizeof(int));
    half2 *relative_coordinates = (half2*)malloc(N * sizeof(half2));
    int *d_cell_indices, *d_cell_offsets, *d_cell_particle_list;
    half2 *d_relative_coordinates;
    cudaMalloc(&d_cell_indices, N * 2 * sizeof(int));
    cudaMalloc(&d_cell_offsets, num_cells * sizeof(int));
    cudaMalloc(&d_cell_particle_list, num_cells * max_particles_per_cell * sizeof(int));
    cudaMalloc(&d_relative_coordinates, N * sizeof(half2));
    cudaMemset(d_cell_offsets, 0, num_cells*sizeof(int));

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
    assign_particles_to_cells<<<grid_dim, block_dim>>>(d_xx, d_yy, d_cell_indices, d_cell_offsets, d_cell_particle_list, d_relative_coordinates);
    cudaMemset(d_num_neighbors, 0, N*sizeof(int));
    search_neighbors<<<grid_dim, block_dim>>>(d_cell_indices, d_cell_offsets, d_cell_particle_list, d_relative_coordinates,d_neighbors, d_num_neighbors);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float elapsed_time_ms_NNPS;
    cudaEventElapsedTime(&elapsed_time_ms_NNPS, start, stop);

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    calc_gradient<<<grid_dim, block_dim>>>(d_xx, d_yy, d_func, d_fx, d_ddx, d_neighbors, d_num_neighbors);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float elapsed_time_ms_Grad;
    cudaEventElapsedTime(&elapsed_time_ms_Grad, start, stop);
    
    cudaMemcpy(neighbors, d_neighbors, N*max_neighbor*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(num_neighbors, d_num_neighbors, N*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(cell_indices, d_cell_indices, N * 2 * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(cell_offsets, d_cell_offsets, num_cells * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(cell_particle_list, d_cell_particle_list, num_cells * max_particles_per_cell * sizeof(int), cudaMemcpyDeviceToHost);
    // cudaMemcpy(relative_coordinates, d_relative_coordinates, N * 2 * sizeof(half2), cudaMemcpyDeviceToHost);
    cudaMemcpy(relative_coordinates, d_relative_coordinates, N * sizeof(half2), cudaMemcpyDeviceToHost);
    cudaMemcpy(fx, d_fx, N*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(ddx, d_ddx, N*sizeof(float), cudaMemcpyDeviceToHost);

    // for (int i = N-1; i < N; i++) {
    //     printf("Particle %d neighbors: ", i);
    //     for (int j = 0; j < num_neighbors[i]; j++) {
    //         printf("%d ", neighbors[i*max_neighbor + j]);
    //     }
    //     printf("\n");
    // }

    double error_x = 0.0;
    for (int i = 0; i < N; i++)
    {
        error_x = error_x + std::abs(3.0*x[i]*x[i] - fx[i])/N;
        // printf("%.5f  %.5f     \n", x[i], fx[i]/ddx[i]);
    }

    printf("error of SPH gradient approximation: %.6f \n", error_x);

    // for (int i = N-10; i < N; i++) {
    //     printf("Particle %d neighbors: ", i);
    //     printf("%f %f %f  ", x[i],y[i],z[i]);
    //     printf("%d %d %d", cell_indices[i*3],cell_indices[i*3+1],cell_indices[i*3+2]);
    //     printf("\n");
    // }
    // for (int i = 0; i < cell_offsets[0*NCY*NCZ + 0*NCZ + 0]; i++) {
    //     int ii = cell_particle_list[0*NCY*NCZ*max_particles_per_cell + 0*NCZ*max_particles_per_cell + 0*max_particles_per_cell + i];
    //     printf("Cell %d has %d particles: ", 0, i);
    //     printf("%f %f %f  ", x[ii],y[ii],z[ii]);
    //     printf("%f %f %f ", relative_coordinates[ii*3],relative_coordinates[ii*3+1],relative_coordinates[ii*3+2]);
    //     printf("\n");
    // }

    printf("number of neighbors = %d \n", num_neighbors[N-1]);

    printf("Elapsed time: %.3f %0.3f ms \n", elapsed_time_ms_NNPS, elapsed_time_ms_Grad);

    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_xx);
    cudaFree(d_yy);
    cudaFree(d_func);
    cudaFree(d_fx);
    cudaFree(d_ddx);
    cudaFree(d_neighbors);
    cudaFree(d_num_neighbors);
    cudaFree(d_cell_indices);
    cudaFree(d_cell_offsets);
    cudaFree(d_cell_particle_list);
    cudaFree(d_relative_coordinates);

    return 0;
}
