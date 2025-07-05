#include "f8.h"
#include "gemm.h"

#include "cutlass/numeric_conversion.h"

using namespace cute;

/////////////////////////////////////////////////////////////////////////////////////////////////
/// GEMM kernel configurations
/////////////////////////////////////////////////////////////////////////////////////////////////
using ElementA = cutlass::float_e4m3_t;
using ElementB = cutlass::float_e4m3_t;
using ElementC = float;
using ElementD = float;

int main() {
    const int M = 2048;
    const int N = 4096;
    const int K = 4096;
    const int block_size = 32;
    
    ElementA *A;
    ElementB *B;
    ElementC *C;
    ElementD *D;
    A = new ElementA[M * K];
    B = new ElementB[K * N];
    C = new ElementC[M * N];
    D = new ElementD[M * N];
    
    

    std::srand(static_cast<unsigned int>(std::time(0)));
    cutlass::NumericConverter<ElementA, float, cutlass::FloatRoundStyle::round_to_nearest> converterA;
    cutlass::NumericConverter<ElementB, float, cutlass::FloatRoundStyle::round_to_nearest> converterB;
    
    for (int i = 0; i < M * K; ++i) {
        // 模拟浮点值
        float f = static_cast<float>(std::rand()) / RAND_MAX * 480.0f - 240.0f;
        
        // 这里可以使用 CUTLASS 的量化转换器（如果你使用完整的库）
        // 否则使用构造函数转换
        A[i] = converterA(f);
    }

    for (int i = 0; i < M * N; ++i) {
        // 模拟浮点值
        ElementC f = static_cast<ElementC>(12.0 * std::rand() / RAND_MAX - 6.0);
        
        // 这里可以使用 CUTLASS 的量化转换器（如果你使用完整的库）
        // 否则使用构造函数转换
        C[i] = f;
    }
    for (int i = 0; i < N * K; ++i) {
        // 模拟浮点值
        float f = static_cast<float>(std::rand()) / RAND_MAX * 480.0f - 240.0f;
        
        // 这里可以使用 CUTLASS 的量化转换器（如果你使用完整的库）
        // 否则使用构造函数转换
        B[i] = converterB(f);
    }


    ElementA *A_d;
    ElementB *B_d;
    ElementC *C_d;
    ElementD *D_d;    

    cudaMalloc((void**)&A_d, M * K * sizeof(ElementA));
    cudaMalloc((void**)&B_d, K * N * sizeof(ElementB));
    cudaMalloc((void**)&C_d, M * N * sizeof(ElementC));
    cudaMalloc((void**)&D_d, M * N * sizeof(ElementD));
    cudaMemcpy(A_d, A, M * K * sizeof(ElementA), cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B, K * N * sizeof(ElementB), cudaMemcpyHostToDevice);
    cudaMemcpy(C_d, C, M * N * sizeof(ElementC), cudaMemcpyHostToDevice);
    
    
    // Timing using CUDA events
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    
    for (int it = 0; it < 200; it ++) {
        matmul_hostf8(A_d, B_d, M, N, K, C_d, D_d);
    }
    CHECK_CUDA(cudaEventRecord(start));
    for (int it = 0; it < 400; it ++) {
        matmul_hostf8(A_d, B_d, M, N, K, C_d, D_d);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    float milliseconds = 0;
    CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));
    cudaMemcpy(D, D_d, M * N * sizeof(ElementD), cudaMemcpyDeviceToHost);

    std::printf("GEMM completed in %.3f ms\n", milliseconds / 400);
    std::cout << "fp8 gemm finished." << std::endl;
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
    cudaFree(D_d);
    return 0;
}