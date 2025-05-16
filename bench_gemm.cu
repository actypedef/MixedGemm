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
    
    const int M = 1024;
    const int N = 4096;
    const int KN = 2560;
    const int KS = 1408;
    const int KO = 128;
    const int block_size = 32; 
    
    ElementANormal::DataType *AN;
    ElementASensitive::DataType *AS;
    ElementAOutlier::DataType *AO;
    ElementB::DataType *BN;
    ElementB::DataType *BS;
    ElementB::DataType *BO;
    ElementC *C;
    ElementD *D;
    AN = new ElementANormal::DataType[M * KN];
    AS = new ElementASensitive::DataType[M * KS];
    AO = new ElementAOutlier::DataType[M * KO];
    BN = new ElementB::DataType[N * KN];
    BS = new ElementB::DataType[N * KS];
    BO = new ElementB::DataType[N * KO];
    C = new ElementC[M * N];
    D = new ElementD[M * N];
    
    // 创建 scale 数组（每 block_size 个元素对应一个缩放因子）
    int szAN = ((M * KN + block_size - 1) / block_size);
    ElementANormal::ScaleFactorType *scaleAN = new ElementANormal::ScaleFactorType[((M * KN + block_size - 1) / block_size)];
    int szBN = ((N * KN + block_size - 1) / block_size);
    ElementB::ScaleFactorType *scaleBN = new ElementB::ScaleFactorType[((N * KN + block_size - 1) / block_size)];

    int szAS = ((M * KS + block_size - 1) / block_size);
    ElementASensitive::ScaleFactorType *scaleAS = new ElementASensitive::ScaleFactorType[((M * KS + block_size - 1) / block_size)];
    int szBS = ((N * KS + block_size - 1) / block_size);
    ElementB::ScaleFactorType *scaleBS = new ElementB::ScaleFactorType[((N * KS + block_size - 1) / block_size)];

    int szAO = ((M * KO + block_size - 1) / block_size);
    ElementAOutlier::ScaleFactorType *scaleAO = new ElementAOutlier::ScaleFactorType[((M * KO + block_size - 1) / block_size)];
    int szBO = ((N * KO + block_size - 1) / block_size);
    ElementB::ScaleFactorType *scaleBO = new ElementB::ScaleFactorType[((N * KO + block_size - 1) / block_size)];
    
    std::srand(static_cast<unsigned int>(std::time(0)));
    cutlass::NumericConverter<ElementANormal::DataType, float, cutlass::FloatRoundStyle::round_to_nearest> converterAN;
    cutlass::NumericConverter<ElementASensitive::DataType, float, cutlass::FloatRoundStyle::round_to_nearest> converterAS;
    cutlass::NumericConverter<ElementAOutlier::DataType, float, cutlass::FloatRoundStyle::round_to_nearest> converterAO;
    cutlass::NumericConverter<ElementB::DataType, float, cutlass::FloatRoundStyle::round_to_nearest> converterB;
    cutlass::NumericConverter<ElementANormal::ScaleFactorType, float, cutlass::FloatRoundStyle::round_to_nearest> converterSFA;
    cutlass::NumericConverter<ElementB::ScaleFactorType, float, cutlass::FloatRoundStyle::round_to_nearest> converterSFB;
    
    for (int i = 0; i < M * KN; ++i) {
        // 模拟浮点值
        float f = static_cast<float>(std::rand()) / RAND_MAX * 2.0f - 1.0f;  // [-1, 1]
        
        // 这里可以使用 CUTLASS 的量化转换器（如果你使用完整的库）
        // 否则使用构造函数转换
        AN[i] = converterAN(f);
    }

    for (int i = 0; i < M * KS; ++i) {
        // 模拟浮点值
        float f = static_cast<float>(std::rand()) / RAND_MAX * 2.0f - 1.0f;  // [-1, 1]
        
        // 这里可以使用 CUTLASS 的量化转换器（如果你使用完整的库）
        // 否则使用构造函数转换
        AS[i] = converterAS(f);
    }

    for (int i = 0; i < M * KO; ++i) {
        // 模拟浮点值
        float f = static_cast<float>(std::rand()) / RAND_MAX * 2.0f - 1.0f;  // [-1, 1]
        
        // 这里可以使用 CUTLASS 的量化转换器（如果你使用完整的库）
        // 否则使用构造函数转换
        AO[i] = converterAO(f);
    }

    for (int i = 0; i < M * N; ++i) {
        // 模拟浮点值
        ElementC f = static_cast<ElementC>(2.0 * std::rand() / RAND_MAX - 1.0);  // [-1, 1]
        
        // 这里可以使用 CUTLASS 的量化转换器（如果你使用完整的库）
        // 否则使用构造函数转换
        C[i] = f;
    }
    for (int i = 0; i < N * KN; ++i) {
        // 模拟浮点值
        float f = static_cast<float>(std::rand()) / RAND_MAX * 2.0f - 1.0f;  // [-1, 1]
        
        // 这里可以使用 CUTLASS 的量化转换器（如果你使用完整的库）
        // 否则使用构造函数转换
        BN[i] = converterB(f);
    }
    for (int i = 0; i < N * KS; ++i) {
        // 模拟浮点值
        float f = static_cast<float>(std::rand()) / RAND_MAX * 2.0f - 1.0f;  // [-1, 1]
        
        // 这里可以使用 CUTLASS 的量化转换器（如果你使用完整的库）
        // 否则使用构造函数转换
        BS[i] = converterB(f);
    }

    for (int i = 0; i < N * KO; ++i) {
        // 模拟浮点值
        float f = static_cast<float>(std::rand()) / RAND_MAX * 2.0f - 1.0f;  // [-1, 1]
        
        // 这里可以使用 CUTLASS 的量化转换器（如果你使用完整的库）
        // 否则使用构造函数转换
        BO[i] = converterB(f);
    }


    // 随机初始化 scale（每 block 一个）
    for (size_t i = 0; i < szAN; ++i) {
        scaleAN[i] = converterSFA(0.1f + static_cast<float>(std::rand()) / RAND_MAX * 0.9f);  // [0.1, 1.0]
    }
    for (size_t i = 0; i < szBN; ++i) {
        scaleBN[i] = converterSFB(0.1f + static_cast<float>(std::rand()) / RAND_MAX * 0.9f);  // [0.1, 1.0]
    }
    for (size_t i = 0; i < szAS; ++i) {
        scaleAS[i] = converterSFA(0.1f + static_cast<float>(std::rand()) / RAND_MAX * 0.9f);  // [0.1, 1.0]
    }
    for (size_t i = 0; i < szBS; ++i) {
        scaleBS[i] = converterSFB(0.1f + static_cast<float>(std::rand()) / RAND_MAX * 0.9f);  // [0.1, 1.0]
    }
    for (size_t i = 0; i < szAO; ++i) {
        scaleAO[i] = converterSFA(0.1f + static_cast<float>(std::rand()) / RAND_MAX * 0.9f);  // [0.1, 1.0]
    }
    for (size_t i = 0; i < szBO; ++i) {
        scaleBO[i] = converterSFB(0.1f + static_cast<float>(std::rand()) / RAND_MAX * 0.9f);  // [0.1, 1.0]
    }
    
    // Timing using CUDA events
    // cudaEvent_t start, stop;
    // CHECK_CUDA(cudaEventCreate(&start));
    // CHECK_CUDA(cudaEventCreate(&stop));
    // CHECK_CUDA(cudaEventRecord(start));
    float ms = 0;
    for (int it = 0; it < 200; it ++) {
        float t = matmul_host(AN, BN, AS, BS, AO, BO, M, N, KN, KS, KO, C, D, scaleAN, scaleBN, scaleAS, scaleBS, scaleAO, scaleBO);
    }
    for (int it = 0; it < 400; it ++) {
        float t = matmul_host(AN, BN, AS, BS, AO, BO, M, N, KN, KS, KO, C, D, scaleAN, scaleBN, scaleAS, scaleBS, scaleAO, scaleBO);
        ms += t;
    }
    std::printf("GEMM completed in %.3f ms\n", ms / 400);
    // CHECK_CUDA(cudaEventRecord(stop));
    // CHECK_CUDA(cudaEventSynchronize(stop));
    // float milliseconds = 0;
    // CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));

    // std::printf("GEMM completed in %.3f ms\n", milliseconds);
    std::cout << "mixed gemm finished." << std::endl;
    return 0; 
}