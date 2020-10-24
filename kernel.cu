#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/inner_product.h>

#include <ctime>
#include <iostream>

void transpose(int N, int M, double* src, double* dst) {

    for (int n = 0; n < N * M; n++) {
        int i = n / N;
        int j = n % N;
        dst[n] = src[M * j + i];
    }
}

void printMatrix(int N, int M, double* array) {
    for (int i = 0; i < N * M; i++) {
        std::cout << "\t" << array[i];

        if (!((i + 1) % M))
            std::cout << std::endl;
    }
    std::cout << std::endl;
}

void printMatrix(int N, int M, thrust::device_vector<double> array) {
    for (int i = 0; i < N * M; i++) {
        std::cout << "\t" << array[i];

        if (!((i + 1) % M))
            std::cout << std::endl;
    }
    std::cout << std::endl;
}

template<class T>
struct dp
{
    T* A, * B;
    int m, n, r;

    dp(T* _A, T* _B, int _m, int _n, int _r) : A(_A), B(_B), m(_m), n(_n), r(_r) {};

    __host__ __device__
        T operator()(size_t idx) {

        T sum = 0.0f;
        int row = idx / r;
        int col = idx - (row * r); // cheaper modulo

        for (int i = 0; i < m; i++)
            sum += A[row * m + i] * B[col * m + i];

        return sum;
    }
};

#define BLOCK_SIZE 16

__global__ void gpu_matrix_multiplication(double* a, double* b, double* c, const int m, const int n, const int k)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int sum = 0;

    if (col < k && row < m)
    {
        for (int i = 0; i < n; i++)
        {
            sum += a[row * n + i] * b[i * k + col];
        }

        c[row * k + col] = sum;
    }
}

int main()
{
    clock_t begin_time = clock();

    const int n = 2;
    const int m = 3;
    const int r = 2;

    double matrix1[n * m];
    double matrix2[m * r];
    double matrix2_T[m * r];
    double result[n * r];

    for (int i = 0; i < n * m; ++i) matrix1[i] = i;
    for (int i = 0; i < m * r; ++i) matrix2[i] = i;
    for (int i = 0; i < n * r; ++i) result[i] = 0;

    transpose(m, r, matrix2, matrix2_T);

    //////////////////////////////
    ///     CPU Nested loop     //
    //////////////////////////////

    begin_time = clock();
    std::cout << begin_time << " , CPU Nested loop" << std::endl;

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < r; ++j) {

            double acc = 0.0;

            for (int k = 0; k < m; ++k)
                acc += matrix1[k + i * m] * matrix2[k * r + j];

            result[i * n + j] = acc;
        }
    }

    std::cout << std::endl << "\tresults" << std::endl;
    printMatrix(n, r, result);

    std::cout << clock() << " , CPU Nested loop , " << double(clock() - begin_time) / CLOCKS_PER_SEC << std::endl;

    ////////////////////////////////////////
    ///     CPU Nested loop transpose     //
    ////////////////////////////////////////

    begin_time = clock();
    std::cout << std::endl << begin_time << " , CPU Nested loop transpose" << std::endl;

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < r; ++j) {
            double acc = 0.0;

            for (int k = 0; k < m; ++k)
                acc += matrix1[k + i * m] * matrix2_T[k + j * m];

            result[i * n + j] = acc;
        }
    }

    std::cout << std::endl << "\tresults" << std::endl;
    printMatrix(n, r, result);

    std::cout << clock() << " , CPU Nested loop transpose, " << double(clock() - begin_time) / CLOCKS_PER_SEC << std::endl;

    ////////////////////////////////////////
    ///     GPU thrust::inner_product     //
    ////////////////////////////////////////

    begin_time = clock();
    std::cout << std::endl << begin_time << " , GPU thrust::inner_product" << std::endl;

    thrust::device_vector<double> inner_matrix1(matrix1, matrix1 + n * m);
    thrust::device_vector<double> inner_matrix2(matrix2_T, matrix2_T + m * r);
    thrust::device_vector<double> inner_result(result, result + n * r);

    thrust::fill(inner_result.begin(), inner_result.end(), 0);
    cudaDeviceSynchronize();

    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < r; ++i)
            inner_result[j * n + i] = thrust::inner_product(inner_matrix1.begin() + j * m, inner_matrix1.begin() + j * m + m, inner_matrix2.begin() + i * m, 0.0f);
    }

    for (int i = 0; i < n * r; ++i)
        result[i] = inner_result[i];

    std::cout << std::endl << "\tresults" << std::endl;
    printMatrix(n, r, result);

    std::cout << clock() << " , GPU thrust::inner_product , " << double(clock() - begin_time) / CLOCKS_PER_SEC << std::endl;

    ////////////////////////////////////
    ///     GPU thrust::transform     //
    ////////////////////////////////////

    begin_time = clock();
    std::cout << std::endl << begin_time << " , GPU thrust::transform" << std::endl;

    thrust::device_vector<double> transform_matrix1(matrix1, matrix1 + n * m);
    thrust::device_vector<double> transform_matrix2(matrix2_T, matrix2_T + m * r);
    thrust::device_vector<double> transform_result(result, result + n * r);

    thrust::fill(inner_result.begin(), inner_result.end(), 0);
    cudaDeviceSynchronize();

    thrust::transform(thrust::counting_iterator<int>(0), thrust::counting_iterator<int>(n * r), transform_result.begin(), dp<double>(thrust::raw_pointer_cast(transform_matrix1.data()), thrust::raw_pointer_cast(transform_matrix2.data()), m, n, r));
    cudaDeviceSynchronize();

    for (int i = 0; i < n * r; ++i)
        result[i] = transform_result[i];

    std::cout << std::endl << "\tresults" << std::endl;
    printMatrix(n, r, result);

    std::cout << clock() << " , GPU thrust::transform , " << double(clock() - begin_time) / CLOCKS_PER_SEC << std::endl;

    //////////////////////////////
    ///     GPU CUDA kernel     //
    //////////////////////////////
    /*
    begin_time = clock();
    std::cout << std::endl << begin_time << " , GPU CUDA kernel" << std::endl;

    cudaError_t cudaStatus;

    double* dev_matrix1     = 0;
    double* dev_matrix2_T   = 0;
    double* dev_result      = 0;

    if (cudaMalloc((void**)dev_matrix1      , sizeof(double) * n * m) != cudaSuccess) std::cout << "cudaMalloc failed!" << std::endl;
    if (cudaMalloc((void**)dev_matrix2_T    , sizeof(double) * m * r) != cudaSuccess) std::cout << "cudaMalloc failed!" << std::endl;
    if (cudaMalloc((void**)dev_result       , sizeof(double) * n * r) != cudaSuccess) std::cout << "cudaMalloc failed!" << std::endl;

    if (cudaMemcpy(dev_matrix1  , matrix1   , n * m * sizeof(double), cudaMemcpyHostToDevice)) std::cout << "cudaMemcpy failed!" << std::endl;
    if (cudaMemcpy(dev_matrix2_T, matrix2_T , m * r * sizeof(double), cudaMemcpyHostToDevice)) std::cout << "cudaMemcpy failed!" << std::endl;

    unsigned int grid_rows = (m + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (r + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_cols, grid_rows);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

    //gpu_matrix_multiplication <<< dimGrid, dimBlock >>> (dev_matrix1, dev_matrix2_T, dev_result, m, n, r);

    cudaMemcpy(result, dev_result, sizeof(double) * n * r, cudaMemcpyDeviceToHost);
    cudaThreadSynchronize();

    cudaFree(dev_matrix1);
    cudaFree(dev_matrix2_T);
    cudaFree(dev_result);

    for (int i = 0; i < n * r; ++i)
        result[i] = transform_result[i];

    std::cout << std::endl << "\tresults" << std::endl;
    printMatrix(n, r, result);

    std::cout << clock() << " , GPU CUDA kernel , " << double(clock() - begin_time) / CLOCKS_PER_SEC << std::endl;
    */
    return 0;
}