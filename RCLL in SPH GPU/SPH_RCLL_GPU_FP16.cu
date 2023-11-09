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
#include <iostream>
#include <cuda_fp16.h>

#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/tuple.h>
#include <thrust/copy.h>
#include <fstream>
#include <vector>

#define N 1000000   //10000|40000|100000|400000|1000000
#define R_CUT 0.0024  //0.1114|0.0702|0.0517|0.03257 | 0.024
#define max_neighbor 200
#define cell_size 0.0025 //// basically, this value is suggested consistent with R_CUT
#define NCX 400
#define NCY 400
#define max_particles_per_cell 1000 // generally N/NCX/NCY/NCZ*10 is OK

#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)
template <typename T>
void check(T err, const char* const func, const char* const file,
           int const line)
{
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line
                  << std::endl;
        std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

#define CHECK_LAST_CUDA_ERROR() checkLast(__FILE__, __LINE__)
void checkLast(const char* const file, int const line)
{
    cudaError_t err{cudaGetLastError()};
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line
                  << std::endl;
        std::cerr << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

struct Point {
    double x, y;
    int cix, ciy;
};

__global__ void compute_cell_indices(Point* points, int num_points) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < num_points) {
        points[idx].cix = static_cast<int>(floor(points[idx].x / cell_size));
        points[idx].ciy = static_cast<int>(floor(points[idx].y / cell_size));
    }
}
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
    // printf("here");
    int cell_x_relative = threadIdx.y - 1;
    int cell_y_relative = threadIdx.z - 1;
    // printf("cell_x_relative is: %d", cell_x_relative);
    // printf("cell_y_relative is: %d", cell_y_relative);
    if (i >= N) return;
    int ix_cell = cell_indices[i*2];
    int iy_cell = cell_indices[i*2+1];
    int iix_cell = (ix_cell > 0 || ix_cell < NCX-1) ? ix_cell + cell_x_relative:ix_cell;
    int iiy_cell = (iy_cell > 0 || iy_cell < NCY-1) ? iy_cell + cell_y_relative:iy_cell;
    // int ix_cellm = (ix_cell>0)?ix_cell-1:ix_cell;
    // int ix_cellp = (ix_cell<NCX-1)?ix_cell+1:ix_cell;
    // int iy_cellm = (iy_cell>0)?iy_cell-1:iy_cell;
    // int iy_cellp = (iy_cell<NCY-1)?iy_cell+1:iy_cell;
    // half2 i_corr = relative_coordinates[i];
    // for (int iix_cell=ix_cellm; iix_cell<=ix_cellp; iix_cell++)
    // {
    //     for (int iiy_cell=iy_cellm; iiy_cell<=iy_cellp;iiy_cell++)
    //     {
            // half2 cell_distance=__floats2half2_rn((ix_cell - iix_cell)*cell_size, (iy_cell-iiy_cell)*cell_size);
            // half2 di=relative_coordinates[i];
            for (int j = 0; j < cell_offsets[iix_cell*NCY+ iiy_cell]; j++)
            {
                int ii = cell_particle_list[iix_cell*NCY*max_particles_per_cell + iiy_cell*max_particles_per_cell + j];
                if (ii != i)
                {
                    __half2 dx;
                    // dx.x = i_corr.x - relative_coordinates[ii].x + __float2half(((ix_cell - iix_cell)*cell_size));
                    // dx.y = i_corr.y - relative_coordinates[ii].y + __float2half(((iy_cell - iiy_cell)*cell_size));
                    dx.x = relative_coordinates[i].x - relative_coordinates[ii].x + __float2half(((ix_cell - iix_cell)*cell_size));
                    dx.y = relative_coordinates[i].y - relative_coordinates[ii].y + __float2half(((iy_cell - iiy_cell)*cell_size));
                    float r = __half2float(__hadd(__hmul(dx.x, dx.x), __hmul(dx.y, dx.y)));
                    // dx = __hadd2(__hadd2(dx, -relative_coordinates[ii]), cell_distance);
                    // dx = __hadd2(__hadd2(di, -relative_coordinates[ii]), cell_distance);
                    // __half2 dx2 = __hmul2(dx, dx);
                    // float r = __half2float(__hadd(dx2.x,dx2.y));
                    if (r <= R_CUT*R_CUT)
                    {
                        int count = atomicAdd(&num_neighbors[i], 1);
                        neighbors[i*max_neighbor + count] = ii;
                    }
                }
            }
    //     }
    // }
}

__global__ void assign_particles_to_cells(float *xx, float *yy, int *cell_indices, int *cell_offsets, int *cell_particle_list, half2 *relative_coordinates)
// __global__ void assign_particles_to_cells(float *xx, float *yy, int *cell_indices, int *cell_offsets, int *particle_offset, half2 *coordinates)
// __global__ void assign_particles_to_cells(float *xx, float *yy, int *cell_indices, int *cell_offsets, half2 *relative_coordinates, int *particle_offsets)
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
    
    // int particle_index = cix*NCY*max_particles_per_cell + ciy*max_particles_per_cell + cell_offset;
    // coordinates[particle_index].x = xx[i] - (cix+0.5f)*cell_size;
    // coordinates[particle_index].y = yy[i] - (ciy+0.5f)*cell_size;
    if (cell_offset>max_particles_per_cell) {
        printf("!!! The max_particles_per_cell is too small!!!! \n"); 
    }
    // particle_offset[i] = cell_offset;
    // Compute the relative coordinates of the particle in the cell
    relative_coordinates[i].x = xx[i] - (cix+0.5f)*cell_size;
    relative_coordinates[i].y = yy[i] - (ciy+0.5f)*cell_size;

    // Store the cell indices for the current particle
    cell_indices[i*2] = cix;
    cell_indices[i*2 + 1] = ciy;
    // cell_indices[particle_index*2] = cix;
    // cell_indices[particle_index*2 + 1] = ciy;
    // particle_offset[particle_index] = cell_offset;
}

// __global__ void sort_particles(int *cell_indices, int *cell_offsets, int *particle_offset, half2 *relative_coordinates)
// {
//     int i = blockIdx.x * blockDim.x + threadIdx.x;
//     if (i >= N) return; 
//
//     // Compute the cell indices for the current particle
//     int cix = cell_indices[i*2];
//     int ciy = cell_indices[i*2+1];
//     // Compute the offset in the cell_particle_list for the current cell
//     int cell_offset = cell_offsets[cix*NCY + ciy];
//
//     int index = cix
//
//     // Add the particle index to the cell_particle_list
//     // cell_particle_list[cix*NCY*max_particles_per_cell + ciy*max_particles_per_cell + cell_offset] = i;
//     
//     // int particle_index = cix*NCY*max_particles_per_cell + ciy*max_particles_per_cell + cell_offset;
//     // coordinates[particle_index].x = xx[i] - (cix+0.5f)*cell_size;
//     // coordinates[particle_index].y = yy[i] - (ciy+0.5f)*cell_size;
//
//     // Compute the relative coordinates of the particle in the cell
//     // relative_coordinates[i].x = xx[i] - (cix+0.5f)*cell_size;
//     // relative_coordinates[i].y = yy[i] - (ciy+0.5f)*cell_size;
//
//     // Store the cell indices for the current particle
//     cell_indices[i*2] = cix;
//     cell_indices[i*2 + 1] = ciy;
//     // cell_indices[particle_index*2] = cix;
//     // cell_indices[particle_index*2 + 1] = ciy;
//     // particle_offset[particle_index] = cell_offset;
// }

int main() {
    // double x[N], y[N];
    float func[N],fx[N],ddx[N];
    // float xx[N],yy[N];
    int neighbors[N*max_neighbor], num_neighbors[N];
    // FILE *fp;
    //
    // fp = fopen("coordinates_2D_10000000.dat", "r");

    // for (int i = 0; i < N; i++) {
    //     fscanf(fp, "%lf %lf", &x[i], &y[i]);
    //     xx[i] = float(x[i]);
    //     yy[i] = float(y[i]);
    // }
    //
    // fclose(fp);


    std::vector<Point> h_points;
    std::ifstream file("./coordinates_2D_10000000.dat");

    if (file.is_open()) {
        double x, y;
        int count = 0; // To control the number of points read from file
        while (file >> x >> y && count < N) {
            h_points.push_back({x, y, 0, 0});
            count++;
        }
        file.close();
    }

    thrust::device_vector<Point> d_points = h_points;
    int num_points = d_points.size();

    int blockSize = 256;
    int numBlocks = (num_points + blockSize - 1) / blockSize;

    compute_cell_indices<<<numBlocks, blockSize>>>(thrust::raw_pointer_cast(d_points.data()), num_points);

    thrust::sort(d_points.begin(), d_points.end(), [] __device__ (const Point& a, const Point& b) {
        if (a.cix == b.cix)
            return a.ciy < b.ciy;
        else
            return a.cix < b.cix;
    });

    // Copy the sorted points back to host
    thrust::copy(d_points.begin(), d_points.end(), h_points.begin());

    // Separate the sorted coordinates into array x and array y
    float* xx = (float*) malloc(N * sizeof(float));
    float* yy = (float*) malloc(N * sizeof(float));

    for(int i = 0; i < N; ++i){
        xx[i] = float(h_points[i].x);
        yy[i] = float(h_points[i].y);
    }
    //// define the function x
    for (int i = 0; i < N; i++)
    {
        func[i] = std::pow(xx[i], 3);
        ddx[i] = 0.0;
        fx[i] = 0.0;
    }

    float *d_xx, *d_yy, *d_func, *d_fx, *d_ddx;
    // double *d_x, *d_y;
    int *d_neighbors, *d_num_neighbors;

    cudaMalloc((void **)&d_xx, N*sizeof(float));
    cudaMalloc((void **)&d_yy, N*sizeof(float));
    // cudaMalloc((void **)&d_x, N*sizeof(double));
    // cudaMalloc((void **)&d_y, N*sizeof(double));
    cudaMalloc((void **)&d_func, N*sizeof(float));
    cudaMalloc((void **)&d_fx, N*sizeof(float));
    cudaMalloc((void **)&d_ddx, N*sizeof(float));
    cudaMalloc((void **)&d_neighbors, N*max_neighbor*sizeof(int));
    cudaMalloc((void **)&d_num_neighbors, N*sizeof(int));

    cudaMemcpy(d_xx, xx, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_yy, yy, N*sizeof(float), cudaMemcpyHostToDevice);
    // cudaMemcpy(d_x, x, N*sizeof(double), cudaMemcpyHostToDevice);
    // cudaMemcpy(d_y, y, N*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_func, func, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_fx, fx, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ddx, ddx, N*sizeof(float), cudaMemcpyHostToDevice);

    int num_cells = NCX * NCY;
    // int *cell_indices = (int*)malloc(num_cells * max_particles_per_cell * 2 * sizeof(int));
    // int *cell_offsets = (int*)malloc(num_cells * sizeof(int));
    // int *particle_offset = (int*)malloc(num_cells * max_particles_per_cell * sizeof(int));
    // half2 *coordinates = (half2*)malloc( num_cells * max_particles_per_cell * sizeof(half2));
    // int *d_cell_indices, *d_cell_offsets, *d_particle_offset;
    // half2 *d_coordinates;
    // for(int i =0; i< num_cells*max_particles_per_cell;++i) {
    //     coordinates[i] = __floats2half2_rn(NAN, NAN);
    // }
    // cudaMalloc(&d_cell_indices,  num_cells * max_particles_per_cell * 2 * sizeof(int));
    // cudaMalloc(&d_cell_offsets, num_cells * sizeof(int));
    // cudaMalloc(&d_particle_offset, num_cells * max_particles_per_cell * sizeof(int));
    // // cudaMalloc(&d_relative_coordinates, N * 2 * sizeof(half2));
    // cudaMalloc(&d_coordinates, num_cells * max_particles_per_cell   * sizeof(half2));
    int *cell_indices = (int*)malloc(N * 2 * sizeof(int));
    int *cell_offsets = (int*)malloc(num_cells * sizeof(int));
    int *cell_particle_list = (int*)malloc(num_cells * max_particles_per_cell * sizeof(int));
    half2 *relative_coordinates = (half2*)malloc(N * sizeof(half2));
    int *d_cell_indices, *d_cell_offsets, *d_cell_particle_list;
    half2 *d_relative_coordinates;
    cudaMalloc(&d_cell_indices, N * 2 * sizeof(int));
    cudaMalloc(&d_cell_offsets, num_cells * sizeof(int));
    cudaMalloc(&d_cell_particle_list, num_cells * max_particles_per_cell * sizeof(int));
    // cudaMalloc(&d_relative_coordinates, N * 2 * sizeof(half2));
    cudaMalloc(&d_relative_coordinates, N  * sizeof(half2));
    cudaMemset(d_cell_offsets, 0, num_cells*sizeof(int));

    // Define the grid and block dimensions
    int block_size =256;
    int block_size2 =64;
    // int point_per_block =64;
    // int grid_size_x = (N + block_size - 1) / block_size;
    // int num_blocks = (N + block_size - 1) / block_size;
    int num_blocks = (N + block_size - 1) / block_size;
    int num_blocks2 = (N + block_size2 - 1) / block_size2;
    dim3 grid_dim(num_blocks, 1, 1);
    dim3 grid_dim2(num_blocks2, 1, 1);
    dim3 block_dim2(block_size2, 3, 3);
    dim3 block_dim(block_size, 1, 1);
    // dim3 block_dim(block_size, 1, 1);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    // Call the kernel
    assign_particles_to_cells<<<grid_dim, block_dim>>>(d_xx, d_yy, d_cell_indices, d_cell_offsets, d_cell_particle_list, d_relative_coordinates);
    // assign_particles_to_cells<<<grid_dim, block_dim>>>(d_xx, d_yy, d_cell_indices, d_cell_offsets, d_particle_offset, d_coordinates);
    // assign_particles_to_cells<<<grid_dim, block_dim>>>(d_xx, d_yy, d_cell_indices, d_cell_offsets);
    // sort_particles<<<grid_dim, block_dim>>>(d_cell_indices, d_cell_offsets, d_relative_coordinates);
    // CHECK_LAST_CUDA_ERROR();
    // for(int i =0; i<num_cells*max_particles_per_cell;++i){
    //     printf("d_coordinates[%d] = %f\n", i, __low2float(d_coordinates[i]) + __high2float(d_coordinates[i]));
    // }
    cudaMemset(d_num_neighbors, 0, N*sizeof(int));
    search_neighbors<<<grid_dim2, block_dim2>>>(d_cell_indices, d_cell_offsets, d_cell_particle_list, d_relative_coordinates,d_neighbors, d_num_neighbors);
    // CHECK_LAST_CUDA_ERROR();
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

    // for(int i = 1; i<N; i ++ ){
    //     printf("i is %d, fx is %f, ddx is %f\n", i, __half2float(__float2half(fx[i])), __half2float(__float2half(ddx[i])));
    // }
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
        // error_x = error_x + std::abs(3.0*x[i]*x[i] - fx[i])/N;
        error_x = error_x + std::abs(3.0*h_points[i].x*h_points[i].x - fx[i])/N;
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
    //
    // cudaFree(d_x);
    // cudaFree(d_y);
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
    // cudaFree(d_particle_offset);
    // cudaFree(d_coordinates);

    return 0;
}
