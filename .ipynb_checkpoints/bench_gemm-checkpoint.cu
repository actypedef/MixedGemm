#include "gemm.h"

/////////////////////////////////////////////////////////////////////////////////////////////////
/// GEMM kernel configurations
/////////////////////////////////////////////////////////////////////////////////////////////////

// A matrix configuration
using         ElementANormal    = cutlass::mx_float4_t<cutlass::float_e2m1_t>;    // Element type for A matrix operand
using         ElementASensitive = cutlass::mx_float6_t<cutlass::float_e3m2_t>;    // Element type for A matrix operand
using         ElementAOutlier   = cutlass::mx_float8_t<cutlass::float_e4m3_t>;    // Element type for A matrix operand

// B matrix configuration
using         ElementB    = cutlass::mx_float4_t<cutlass::float_e2m1_t>;    // Element type for B matrix operand

// C/D matrix configuration
using         ElementD    = cutlass::bfloat16_t;                            // Element type for D matrix operand
using         ElementC    = cutlass::bfloat16_t;                            // Element type for C matrix operand

int main() {
    /*
    const int M = 1024;
    const int N = 4096;
    const int KN = 2560;
    const int KS = 1408
    const int KO = 128
    const int block_size = 32; 
    
    ElementA::DataType *A;
    ElementB::DataType *B;
    ElementC *C;
    ElementD *D;
    A = new ElementA::DataType[M * K];
    B = new ElementB::DataType[N * K];
    C = new ElementC[M * N];
    D = new ElementD[M * N];
    
    // 创建 scale 数组（每 block_size 个元素对应一个缩放因子）
    int szA = ((M * K + block_size - 1) / block_size);
    ElementA::ScaleFactorType *scaleA = new ElementA::ScaleFactorType[((M * K + block_size - 1) / block_size)];
    int szB = ((N * K + block_size - 1) / block_size);
    ElementB::ScaleFactorType *scaleB = new ElementB::ScaleFactorType[((N * K + block_size - 1) / block_size)];
    
    std::srand(static_cast<unsigned int>(std::time(0)));
    cutlass::NumericConverter<ElementA::DataType, float, cutlass::FloatRoundStyle::round_to_nearest> converterA;
    cutlass::NumericConverter<ElementB::DataType, float, cutlass::FloatRoundStyle::round_to_nearest> converterB;
    cutlass::NumericConverter<ElementA::ScaleFactorType, float, cutlass::FloatRoundStyle::round_to_nearest> converterSFA;
    cutlass::NumericConverter<ElementB::ScaleFactorType, float, cutlass::FloatRoundStyle::round_to_nearest> converterSFB;
    
    for (int i = 0; i < M * K; ++i) {
        // 模拟浮点值
        float f = static_cast<float>(std::rand()) / RAND_MAX * 2.0f - 1.0f;  // [-1, 1]
        
        // 这里可以使用 CUTLASS 的量化转换器（如果你使用完整的库）
        // 否则使用构造函数转换
        A[i] = converterA(f);
    }

    for (int i = 0; i < M * N; ++i) {
        // 模拟浮点值
        ElementC f = static_cast<ElementC>(2.0 * std::rand() / RAND_MAX - 1.0);  // [-1, 1]
        
        // 这里可以使用 CUTLASS 的量化转换器（如果你使用完整的库）
        // 否则使用构造函数转换
        C[i] = f;
    }
    for (int i = 0; i < N * K; ++i) {
        // 模拟浮点值
        float f = static_cast<float>(std::rand()) / RAND_MAX * 2.0f - 1.0f;  // [-1, 1]
        
        // 这里可以使用 CUTLASS 的量化转换器（如果你使用完整的库）
        // 否则使用构造函数转换
        B[i] = converterB(f);
    }


    // 随机初始化 scale（每 block 一个）
    for (size_t i = 0; i < szA; ++i) {
        scaleA[i] = converterSFA(0.1f + static_cast<float>(std::rand()) / RAND_MAX * 0.9f);  // [0.1, 1.0]
    }
    for (size_t i = 0; i < szB; ++i) {
        scaleB[i] = converterSFB(0.1f + static_cast<float>(std::rand()) / RAND_MAX * 0.9f);  // [0.1, 1.0]
    }
    
    // Timing using CUDA events
    // cudaEvent_t start, stop;
    // CHECK_CUDA(cudaEventCreate(&start));
    // CHECK_CUDA(cudaEventCreate(&stop));
    // CHECK_CUDA(cudaEventRecord(start));
    float ms = 0;
    for (int it = 0; it < 200; it ++) {
        float t = matmul_host4(A, B, M, N, K, C, D, scaleA, scaleB);
    }
    for (int it = 0; it < 400; it ++) {
        float t = matmul_host4(A, B, M, N, K, C, D, scaleA, scaleB);
        ms += t;
    }
    std::printf("GEMM completed in %.3f ms\n", ms / 400);
    // CHECK_CUDA(cudaEventRecord(stop));
    // CHECK_CUDA(cudaEventSynchronize(stop));
    // float milliseconds = 0;
    // CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));

    // std::printf("GEMM completed in %.3f ms\n", milliseconds);
    std::cout << "mxfp4 gemm finished." << std::endl;*/
    return 0; 
}