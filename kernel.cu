
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/inner_product.h>


#include <cuda_runtime.h>
#include "cublas_v2.h"

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

    std::cout << std::endl << "matrix 1" << std::endl;
    printMatrix(n, m, matrix1);
    std::cout << std::endl << "matrix 2" << std::endl;
    printMatrix(m, r, matrix2);
    std::cout << std::endl << "matrix 2 transpose" << std::endl;
    printMatrix(n, m, matrix2_T);
    std::cout << std::endl << "result" << std::endl;
    printMatrix(n, r, result);

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

    std::cout << clock() << " , CPU Nested loop transpose , " << double(clock() - begin_time) / CLOCKS_PER_SEC << std::endl;

    ////////////////////////////////////////
    ///     GPU thrust::inner_product     //
    ////////////////////////////////////////

    begin_time = clock();
    std::cout << std::endl << begin_time << " , GPU thrust::inner_product" << std::endl;

    thrust::device_vector<double> inner_matrix1(matrix1, matrix1 + n * m);
    thrust::device_vector<double> inner_matrix2(matrix2_T, matrix2_T + m * r);

    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < r; ++i)
			result[j * n + i] = thrust::inner_product(inner_matrix1.begin() + j * m, inner_matrix1.begin() + j * m + m, inner_matrix2.begin() + i * m, 0.0f);            
    }

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
    thrust::device_vector<double> transform_result(n * r, 0);

    thrust::transform(thrust::counting_iterator<int>(0), thrust::counting_iterator<int>(n * r), transform_result.begin(), dp<double>(thrust::raw_pointer_cast(transform_matrix1.data()), thrust::raw_pointer_cast(transform_matrix2.data()), m, n, r));
    cudaDeviceSynchronize();

    thrust::copy(transform_result.begin(), transform_result.end(), result);

    std::cout << std::endl << "\tresults" << std::endl;
    printMatrix(n, r, result);

    std::cout << clock() << " , GPU thrust::transform , " << double(clock() - begin_time) / CLOCKS_PER_SEC << std::endl;

    //////////////////////////////
    ///     GPU CUDA CUBLAS     //
    //////////////////////////////

    begin_time = clock();
    std::cout << std::endl << begin_time << " , GPU CUDA CUBLAS" << std::endl;

    // Allocate device memory
    double* d_matrix1;
    double* d_matrix2;
    double *d_result;
    if (cudaMalloc(&d_matrix1, sizeof(double) * n * m) != cudaSuccess) std::cout << "cudaMalloc failed!" << std::endl;
    if (cudaMalloc(&d_matrix2, sizeof(double) * m * r) != cudaSuccess) std::cout << "cudaMalloc failed!" << std::endl;
    if (cudaMalloc(&d_result , sizeof(double) * n * r) != cudaSuccess) std::cout << "cudaMalloc failed!" << std::endl;

    if (cudaMemcpy(d_matrix1, matrix1, n * m * sizeof(double), cudaMemcpyHostToDevice)) std::cout << "cudaMemcpy failed!" << std::endl;
    if (cudaMemcpy(d_matrix2, matrix2, m * r * sizeof(double), cudaMemcpyHostToDevice)) std::cout << "cudaMemcpy failed!" << std::endl;

    // cuBLAS handle
    cublasHandle_t handle;

    if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) {
        std::cout << "CUBLAS initialization failed" << std::endl;
        return EXIT_FAILURE;
    }

    // Scalaing factors
    double alpha = 1.0;
    double beta = 0.0;

    // Calculate: c = (alpha*a) * b + (beta*c)
    // nxr = nxm * mxr
    // Signature: handle, operation, operation, n, r, m, alpha, A, lda, B, ldb,
    // beta, C, ldc
    cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, r, m, &alpha, d_matrix1, n, d_matrix2, m, &beta, d_result, r);
    
    // Copy back the three matrices
    cudaMemcpy(result, d_result, sizeof(double) * n * r, cudaMemcpyDeviceToHost);
    
    // Free our memory
    cudaFree(d_matrix1);
    cudaFree(d_matrix2);
    cudaFree(d_result);

    std::cout << std::endl << "\tresults" << std::endl;
    printMatrix(n, r, result);

    std::cout << clock() << " , CUDA CUBLAS , " << double(clock() - begin_time) / CLOCKS_PER_SEC << std::endl;

    ///////////////////////////////////////
    ///     GPU CUDA CUBLAS transpose    //
    ///////////////////////////////////////

    begin_time = clock();
    std::cout << std::endl << begin_time << " , GPU CUDA CUBLAS transpose" << std::endl;

    double* d_matrix2_T;

    // Allocate device memory
    if (cudaMalloc(&d_matrix1   , sizeof(double) * n * m) != cudaSuccess) std::cout << "cudaMalloc failed!" << std::endl;
    if (cudaMalloc(&d_matrix2_T , sizeof(double) * n * m) != cudaSuccess) std::cout << "cudaMalloc failed!" << std::endl;
    if (cudaMalloc(&d_matrix2   , sizeof(double) * m * r) != cudaSuccess) std::cout << "cudaMalloc failed!" << std::endl;
    if (cudaMalloc(&d_result    , sizeof(double) * n * r) != cudaSuccess) std::cout << "cudaMalloc failed!" << std::endl;

    if (cudaMemcpy(d_matrix1    , matrix1   , n * m * sizeof(double), cudaMemcpyHostToDevice)) std::cout << "cudaMemcpy failed!" << std::endl;
    if (cudaMemcpy(d_matrix2_T  , matrix2_T , n * m * sizeof(double), cudaMemcpyHostToDevice)) std::cout << "cudaMemcpy failed!" << std::endl;

    // Tranpose d_matrix2
    cublasDgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N, n, m, &alpha, d_matrix2_T, m, &beta, d_matrix2_T, n, d_matrix2, n);

    // Calculate: c = (alpha*a) * b + (beta*c)
    // nxr = nxm * mxr
    // Signature: handle, operation, operation, n, r, m, alpha, A, lda, B, ldb,
    // beta, C, ldc
    cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, r, m, &alpha, d_matrix1, n, d_matrix2, m, &beta, d_result, r);

    // Copy back the three matrices
    //cudaMemcpy(matrix1, d_matrix1   , sizeof(double) * n * m, cudaMemcpyDeviceToHost);
    //cudaMemcpy(matrix2, d_matrix2   , sizeof(double) * n * m, cudaMemcpyDeviceToHost);
    cudaMemcpy(result, d_result     , sizeof(double) * n * r, cudaMemcpyDeviceToHost);

    // Free our memory
    cudaFree(d_matrix1);
    cudaFree(d_matrix2);
    cudaFree(d_matrix2_T);
    cudaFree(d_result);
    
    cublasDestroy(handle);

    //std::cout << std::endl << "matrix 1" << std::endl;
    //printMatrix(n, m, matrix1);
    //std::cout << std::endl << "matrix 2" << std::endl;
    //printMatrix(m, r, matrix2);
    std::cout << std::endl << "result" << std::endl;
    printMatrix(n, r, result);
    
    std::cout << clock() << " , CUDA CUBLAS ranspose , " << double(clock() - begin_time) / CLOCKS_PER_SEC << std::endl;

    cublasDestroy(handle);

    return 0;
}