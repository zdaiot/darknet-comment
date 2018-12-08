#include "gemm.h"
#include "utils.h"
#include "cuda.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

void gemm_bin(int M, int N, int K, float ALPHA, 
        char  *A, int lda, 
        float *B, int ldb,
        float *C, int ldc)
{
    int i,j,k;
    for(i = 0; i < M; ++i){
        for(k = 0; k < K; ++k){
            char A_PART = A[i*lda+k];
            if(A_PART){
                for(j = 0; j < N; ++j){
                    C[i*ldc+j] += B[k*ldb+j];
                }
            } else {
                for(j = 0; j < N; ++j){
                    C[i*ldc+j] -= B[k*ldb+j];
                }
            }
        }
    }
}

float *random_matrix(int rows, int cols)
{
    int i;
    float *m = calloc(rows*cols, sizeof(float));
    for(i = 0; i < rows*cols; ++i){
        m[i] = (float)rand()/RAND_MAX;
    }
    return m;
}

void time_random_matrix(int TA, int TB, int m, int k, int n)
{
    float *a;
    if(!TA) a = random_matrix(m,k);
    else a = random_matrix(k,m);
    int lda = (!TA)?k:m;
    float *b;
    if(!TB) b = random_matrix(k,n);
    else b = random_matrix(n,k);
    int ldb = (!TB)?n:k;

    float *c = random_matrix(m,n);
    int i;
    clock_t start = clock(), end;
    for(i = 0; i<10; ++i){
        gemm_cpu(TA,TB,m,n,k,1,a,lda,b,ldb,1,c,n);
    }
    end = clock();
    printf("Matrix Multiplication %dx%d * %dx%d, TA=%d, TB=%d: %lf ms\n",m,k,k,n, TA, TB, (float)(end-start)/CLOCKS_PER_SEC);
    free(a);
    free(b);
    free(c);
}

/*
输入：TA 判断矩阵A是否转置
     TB 判断矩阵B是否转置
     M,N,k 见下面功能，与矩阵A,B,C的维度对应
     ALPHA 广义矩阵乘积操作(gemm)参数
     *A 广义矩阵乘积操作(gemm)中的矩阵A
     lda 矩阵*A一行有多少个元素
     *B 广义矩阵乘积操作(gemm)中的矩阵B
     ldb 矩阵*B一行有多少个元素
     BETA 广义矩阵乘积操作(gemm)参数
     *C 广义矩阵乘积操作(gemm)中的矩阵C
     ldc 矩阵*C一行有多少个元素
功能：当TA=TB=0的时候，A大小为M*K，B大小为K*N，C的大小为M*N，完成广义矩阵乘积操作(gemm) C = ALPHA*A*B + BETA*C
     当TA=1,TB=0的时候，A大小为K*M，B大小为K*N，C的大小为M*N，完成广义矩阵乘积操作(gemm) C = ALPHA*(A^T)*B + BETA*C
     当TA=0,TB=1的时候，A大小为M*K，B大小为N*K，C的大小为M*N，完成广义矩阵乘积操作(gemm) C = ALPHA*A*(B^T) + BETA*C
     当TA=TB=1的时候，A大小为K*M，B大小为N*K，C的大小为M*N，完成广义矩阵乘积操作(gemm) C = ALPHA*(A^T)*(B^T) + BETA*C
输出：广义矩阵乘积结果 C
返回：无
*/
void gemm(int TA, int TB, int M, int N, int K, float ALPHA, 
        float *A, int lda, 
        float *B, int ldb,
        float BETA,
        float *C, int ldc)
{
    gemm_cpu( TA,  TB,  M, N, K, ALPHA,A,lda, B, ldb,BETA,C,ldc);
}

/*
输入：M 矩阵A的行数与矩阵C的行数
     N 矩阵B的列数与矩阵C的列数
     K 矩阵A的列数与矩阵B的行数
     ALPHA 广义矩阵乘积操作(gemm)参数
     *A 广义矩阵乘积操作(gemm)中的矩阵A，大小为 M*K
     lda 矩阵*A一行有多少个元素
     *B 广义矩阵乘积操作(gemm)中的矩阵B，大小为 K*N
     ldb 矩阵*B一行有多少个元素
     *C 广义矩阵乘积操作(gemm)中的矩阵C，大小为 M*N
     ldc 矩阵*C一行有多少个元素
功能：矩阵乘积操作(gemm) C = ALPHA*A*B
输出：带权的矩阵A与矩阵B相乘结果 *C
返回：无
*/
void gemm_nn(int M, int N, int K, float ALPHA, 
        float *A, int lda, 
        float *B, int ldb,
        float *C, int ldc)
{
    int i,j,k;
    // OpenMP 并不是一个简单的函数库，而是一个诸多编译器支持的框架，或者说是协议吧，
    // 总之，不需要任何配置，你就可以在 Visual Studio 或者 gcc 中使用它了。
    #pragma omp parallel for // OpenMP的并行计算
    for(i = 0; i < M; ++i){  // 对于A中的每一行或C中每一行
        for(k = 0; k < K; ++k){  // 对于B中的每一行或A中每一列
            register float A_PART = ALPHA*A[i*lda+k]; // 寄存器变量
            for(j = 0; j < N; ++j){ //对于C中的每一列或B中每一列
                // 当i,k固定，j滑动，可以理解为C中第i行是由B中各行按照A中第i行的值线性组合得到的
                C[i*ldc+j] += A_PART*B[k*ldb+j]; 
            }
        }
    }

    /*当然代码还可以这么写，只是这样写的话就A与B均无法放到第三层循环外面，增加了计算量，不过容易理解
    int i,j,k;
    #pragma omp parallel for // OpenMP的并行计算
    for(i = 0; i < M; ++i){  // 对于A中的每一行
        for(j = 0; j < N; ++j){
            for(k = 0; k < K; ++k){  // 对于B中的每一行
                // 当i,j固定，k滑动，可以理解为C中第i行第j列是由A中第i行与B中第j列做内积得到
                C[i*ldc+j] +=  ALPHA*A[i*lda+k]*B[k*ldb+j]; 
            }
        }
    }
    */
}

/*
输入：M 矩阵A的行数与矩阵C的行数
     N 矩阵B的行数与矩阵C的列数
     K 矩阵A的列数与矩阵B的列数
     ALPHA 广义矩阵乘积操作(gemm)参数
     *A 广义矩阵乘积操作(gemm)中的矩阵A，大小为 M*K
     lda 矩阵*A一行有多少个元素
     *B 广义矩阵乘积操作(gemm)中的矩阵B，大小为 N*K
     ldb 矩阵*B一行有多少个元素
     *C 广义矩阵乘积操作(gemm)中的矩阵C，大小为 M*N
     ldc 矩阵*C一行有多少个元素
功能：矩阵乘积操作(gemm) C = ALPHA*A*(B^T)
输出：带权的矩阵A与矩阵B的转置相乘结果 *C
返回：无
*/
void gemm_nt(int M, int N, int K, float ALPHA, 
        float *A, int lda, 
        float *B, int ldb,
        float *C, int ldc)
{
    int i,j,k;
    #pragma omp parallel for 
    for(i = 0; i < M; ++i){  // A的每一行或C中每一行
        for(j = 0; j < N; ++j){ // B的每一行或C中每一列
            register float sum = 0;
            for(k = 0; k < K; ++k){ // B的每一列或A中每一列
                // 当i,j固定，k滑动，可以可以理解为C中第i行第j列是由A中第i行与B中第j行做内积得到
                sum += ALPHA*A[i*lda+k]*B[j*ldb + k];
            }
            C[i*ldc+j] += sum;  // 用sum临时变量猜测是因为可以使用到寄存器临时变量，另外这里的 += 有什么用还不清楚
        }
    }
}

/*
输入：M 矩阵A的列数与矩阵C的行数
     N 矩阵B的列数与矩阵C的列数
     K 矩阵A的行数与矩阵B的行数
     ALPHA 广义矩阵乘积操作(gemm)参数
     *A 广义矩阵乘积操作(gemm)中的矩阵A，大小为 K*M
     lda 矩阵*A一行有多少个元素
     *B 广义矩阵乘积操作(gemm)中的矩阵B，大小为 K*N
     ldb 矩阵*B一行有多少个元素
     *C 广义矩阵乘积操作(gemm)中的矩阵C，大小为 M*N
     ldc 矩阵*C一行有多少个元素
功能：矩阵乘积操作(gemm) C = ALPHA*(A^T)*B
输出：带权的矩阵A的转置与矩阵B相乘结果 *C
返回：无
*/
void gemm_tn(int M, int N, int K, float ALPHA, 
        float *A, int lda, 
        float *B, int ldb,
        float *C, int ldc)
{
    int i,j,k;
    #pragma omp parallel for
    for(i = 0; i < M; ++i){ // 对于A中每一列或C中每一行
        for(k = 0; k < K; ++k){ // 对于B中每一行或者A中每一行
            register float A_PART = ALPHA*A[k*lda+i];
            for(j = 0; j < N; ++j){ // 对于C中每一行或B中每一列
                // 当i,k固定，j滑动，可以理解为C中第i行是由B中各行按照A中第i列的值线性组合得到的
                C[i*ldc+j] += A_PART*B[k*ldb+j];
            }
        }
    }

    /*当然代码还可以这么写，只是这样写的话就A与B均无法放到第三层循环外面，增加了计算量，不过容易理解
    int i,j,k;
    #pragma omp parallel for
    for(i = 0; i < M; ++i){ // 对于A中每一列
        for(j = 0; j < N; ++j){ // 对于C中每一行
            for(k = 0; k < K; ++k){ // 对于B中每一行
                // 当i,j固定，k滑动，可以理解为C中第i行第j列是由A中第i列与B中第j列做内积得到
                C[i*ldc+j] += ALPHA*A[k*lda+i]*B[k*ldb+j];
            }
        }
    }
    */
}

/*
输入：M 矩阵A的列数与矩阵C的行数
     N 矩阵B的行数与矩阵C的列数
     K 矩阵A的行数与矩阵B的列数
     ALPHA 广义矩阵乘积操作(gemm)参数
     *A 广义矩阵乘积操作(gemm)中的矩阵A，大小为 K*M
     lda 矩阵*A一行有多少个元素
     *B 广义矩阵乘积操作(gemm)中的矩阵B，大小为 N*K
     ldb 矩阵*B一行有多少个元素
     *C 广义矩阵乘积操作(gemm)中的矩阵C，大小为 M*N
     ldc 矩阵*C一行有多少个元素
功能：矩阵乘积操作(gemm) C = ALPHA*(A^T)*(B^T)
输出：带权的矩阵A的转置与矩阵B的转置相乘结果 *C
返回：无
*/
void gemm_tt(int M, int N, int K, float ALPHA, 
        float *A, int lda, 
        float *B, int ldb,
        float *C, int ldc)
{
    int i,j,k;
    #pragma omp parallel for
    for(i = 0; i < M; ++i){ // 对于A的每一列或C中每一行
        for(j = 0; j < N; ++j){ // 对于B中每一行或C中每一列
            register float sum = 0;
            for(k = 0; k < K; ++k){ // 对于B中每一列或者A中每一行
                // 当i,j固定，k滑动，可以理解为C中第i行第j列是由A中第i列与B中第j行做内积得到
                sum += ALPHA*A[i+k*lda]*B[k+j*ldb];
            }
            C[i*ldc+j] += sum;
        }
    }
}


void gemm_cpu(int TA, int TB, int M, int N, int K, float ALPHA, 
        float *A, int lda, 
        float *B, int ldb,
        float BETA,
        float *C, int ldc)
{
    //printf("cpu: %d %d %d %d %d %f %d %d %f %d\n",TA, TB, M, N, K, ALPHA, lda, ldb, BETA, ldc);
    int i, j;
    for(i = 0; i < M; ++i){
        for(j = 0; j < N; ++j){
            C[i*ldc + j] *= BETA; // 参考广义矩阵乘积操作(gemm)，这里的BETA为1
        }
    }
    if(!TA && !TB) // 判断转置
        gemm_nn(M, N, K, ALPHA,A,lda, B, ldb,C,ldc);
    else if(TA && !TB)
        gemm_tn(M, N, K, ALPHA,A,lda, B, ldb,C,ldc);
    else if(!TA && TB)
        gemm_nt(M, N, K, ALPHA,A,lda, B, ldb,C,ldc);
    else
        gemm_tt(M, N, K, ALPHA,A,lda, B, ldb,C,ldc);
}

#ifdef GPU

#include <math.h>

void gemm_gpu(int TA, int TB, int M, int N, int K, float ALPHA, 
        float *A_gpu, int lda, 
        float *B_gpu, int ldb,
        float BETA,
        float *C_gpu, int ldc)
{
    cublasHandle_t handle = blas_handle();
    cudaError_t status = cublasSgemm(handle, (TB ? CUBLAS_OP_T : CUBLAS_OP_N), 
            (TA ? CUBLAS_OP_T : CUBLAS_OP_N), N, M, K, &ALPHA, B_gpu, ldb, A_gpu, lda, &BETA, C_gpu, ldc);
    check_error(status);
}

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

void time_gpu_random_matrix(int TA, int TB, int m, int k, int n)
{
    float *a;
    if(!TA) a = random_matrix(m,k);
    else a = random_matrix(k,m);
    int lda = (!TA)?k:m;
    float *b;
    if(!TB) b = random_matrix(k,n);
    else b = random_matrix(n,k);
    int ldb = (!TB)?n:k;

    float *c = random_matrix(m,n);
    int i;
    clock_t start = clock(), end;
    for(i = 0; i<32; ++i){
        gemm_gpu(TA,TB,m,n,k,1,a,lda,b,ldb,1,c,n);
    }
    end = clock();
    printf("Matrix Multiplication %dx%d * %dx%d, TA=%d, TB=%d: %lf s\n",m,k,k,n, TA, TB, (float)(end-start)/CLOCKS_PER_SEC);
    free(a);
    free(b);
    free(c);
}

void time_gpu(int TA, int TB, int m, int k, int n)
{
    int iter = 10;
    float *a = random_matrix(m,k);
    float *b = random_matrix(k,n);

    int lda = (!TA)?k:m;
    int ldb = (!TB)?n:k;

    float *c = random_matrix(m,n);

    float *a_cl = cuda_make_array(a, m*k);
    float *b_cl = cuda_make_array(b, k*n);
    float *c_cl = cuda_make_array(c, m*n);

    int i;
    clock_t start = clock(), end;
    for(i = 0; i<iter; ++i){
        gemm_gpu(TA,TB,m,n,k,1,a_cl,lda,b_cl,ldb,1,c_cl,n);
        cudaThreadSynchronize();
    }
    double flop = ((double)m)*n*(2.*k + 2.)*iter;
    double gflop = flop/pow(10., 9);
    end = clock();
    double seconds = sec(end-start);
    printf("Matrix Multiplication %dx%d * %dx%d, TA=%d, TB=%d: %lf s, %lf GFLOPS\n",m,k,k,n, TA, TB, seconds, gflop/seconds);
    cuda_free(a_cl);
    cuda_free(b_cl);
    cuda_free(c_cl);
    free(a);
    free(b);
    free(c);
}


void test_gpu_accuracy(int TA, int TB, int m, int k, int n)
{
    srand(0);
    float *a;
    if(!TA) a = random_matrix(m,k);
    else a = random_matrix(k,m);
    int lda = (!TA)?k:m;
    float *b;
    if(!TB) b = random_matrix(k,n);
    else b = random_matrix(n,k);
    int ldb = (!TB)?n:k;

    float *c = random_matrix(m,n);
    float *c_gpu = random_matrix(m,n);
    memset(c, 0, m*n*sizeof(float));
    memset(c_gpu, 0, m*n*sizeof(float));
    int i;
    //pm(m,k,b);
    gemm_gpu(TA,TB,m,n,k,1,a,lda,b,ldb,1,c_gpu,n);
    //printf("GPU\n");
    //pm(m, n, c_gpu);

    gemm_cpu(TA,TB,m,n,k,1,a,lda,b,ldb,1,c,n);
    //printf("\n\nCPU\n");
    //pm(m, n, c);
    double sse = 0;
    for(i = 0; i < m*n; ++i) {
        //printf("%f %f\n", c[i], c_gpu[i]);
        sse += pow(c[i]-c_gpu[i], 2);
    }
    printf("Matrix Multiplication %dx%d * %dx%d, TA=%d, TB=%d: %g SSE\n",m,k,k,n, TA, TB, sse/(m*n));
    free(a);
    free(b);
    free(c);
    free(c_gpu);
}

int test_gpu_blas()
{
    /*
       test_gpu_accuracy(0,0,10,576,75); 

       test_gpu_accuracy(0,0,17,10,10); 
       test_gpu_accuracy(1,0,17,10,10); 
       test_gpu_accuracy(0,1,17,10,10); 
       test_gpu_accuracy(1,1,17,10,10); 

       test_gpu_accuracy(0,0,1000,10,100); 
       test_gpu_accuracy(1,0,1000,10,100); 
       test_gpu_accuracy(0,1,1000,10,100); 
       test_gpu_accuracy(1,1,1000,10,100); 

       test_gpu_accuracy(0,0,10,10,10); 

       time_gpu(0,0,64,2916,363); 
       time_gpu(0,0,64,2916,363); 
       time_gpu(0,0,64,2916,363); 
       time_gpu(0,0,192,729,1600); 
       time_gpu(0,0,384,196,1728); 
       time_gpu(0,0,256,196,3456); 
       time_gpu(0,0,256,196,2304); 
       time_gpu(0,0,128,4096,12544); 
       time_gpu(0,0,128,4096,4096); 
     */
    time_gpu(0,0,64,75,12544); 
    time_gpu(0,0,64,75,12544); 
    time_gpu(0,0,64,75,12544); 
    time_gpu(0,0,64,576,12544); 
    time_gpu(0,0,256,2304,784); 
    time_gpu(1,1,2304,256,784); 
    time_gpu(0,0,512,4608,196); 
    time_gpu(1,1,4608,512,196); 

    return 0;
}
#endif

