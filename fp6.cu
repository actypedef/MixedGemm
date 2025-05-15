#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h>
#include <chrono>

#include "cutlass/cutlass.h"

#include "cute/tensor.hpp"
#include "cutlass/tensor_ref.h"
#include "cutlass/numeric_conversion.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/kernel/tile_scheduler_params.h"

#include "cutlass/util/command_line.h"
#include "cutlass/util/distribution.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/tensor_view_io.h"
#include "cutlass/util/reference/device/gemm.h"
#include "cutlass/util/reference/device/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/reference/host/gett.hpp"
#include "cutlass/util/reference/host/tensor_norm.h"
#include "cutlass/util/reference/host/tensor_compare.h"


#include <iostream>

#include "helper.h"

#define CHECK_CUDA(call)                                                       \
    do {                                                                       \
        cudaError_t err = call;                                                \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA Error at %s:%d - %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err));                                  \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)


using namespace cute;






/////////////////////////////////////////////////////////////////////////////////////////////////
/// GEMM kernel configurations
/////////////////////////////////////////////////////////////////////////////////////////////////

// A matrix configuration
using         ElementA    = cutlass::mx_float6_t<cutlass::float_e3m2_t>;    // Element type for A matrix operand
using         LayoutATag  = cutlass::layout::RowMajor;                      // Layout type for A matrix operand
constexpr int AlignmentA  = 128;                                             // Memory access granularity/alignment of A matrix in units of elements (up to 16 bytes)

// B matrix configuration
using         ElementB    = cutlass::mx_float4_t<cutlass::float_e2m1_t>;    // Element type for B matrix operand
using         LayoutBTag  = cutlass::layout::ColumnMajor;                   // Layout type for B matrix operand
constexpr int AlignmentB  = 32;                                             // Memory access granularity/alignment of B matrix in units of elements (up to 16 bytes)

// C/D matrix configuration
using         ElementD    = cutlass::bfloat16_t;                            // Element type for D matrix operand
using         ElementC    = cutlass::bfloat16_t;                            // Element type for C matrix operand
using         LayoutCTag  = cutlass::layout::RowMajor;                      // Layout type for C matrix operand
using         LayoutDTag  = cutlass::layout::RowMajor;                      // Layout type for D matrix operand
constexpr int AlignmentD  = 128 / cutlass::sizeof_bits<ElementD>::value;    // Memory access granularity/alignment of C matrix in units of elements (up to 16 bytes)
constexpr int AlignmentC  = 128 / cutlass::sizeof_bits<ElementC>::value;    // Memory access granularity/alignment of C matrix in units of elements (up to 16 bytes)
// Kernel functional config
using ElementAccumulator  = float;                                          // Element type for internal accumulation
using ArchTag             = cutlass::arch::Sm120;                           // Tag indicating the minimum SM that supports the intended feature
using OperatorClass       = cutlass::arch::OpClassBlockScaledTensorOp;      // Operator class tag

// Kernel Perf config
using ThreadBlockShape    = Shape<_128,_128,_128>;                          // Threadblock's tile size
using ClusterShape        = Shape<_1,_1,_1>;                                // Shape of the threadblocks in a cluster

float matmul_host(
        const ElementA::DataType *A,
        const ElementB::DataType *B,
        int M,
        int N,
        int K,
        ElementC *C,
        ElementD *D,
        const ElementA::ScaleFactorType *SFA,
        const ElementB::ScaleFactorType *SFB
)
{

    using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
        ArchTag, OperatorClass,                      
        ThreadBlockShape, ClusterShape,
        cutlass::epilogue::collective::EpilogueTileAuto,
        ElementAccumulator, ElementAccumulator,
        ElementC, LayoutCTag, AlignmentC,
        ElementD, LayoutDTag, AlignmentD,
        cutlass::epilogue::collective::EpilogueScheduleAuto                      // Epilogue schedule policy
        >::CollectiveOp;

    using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
        ArchTag, OperatorClass,
        ElementA, LayoutATag, AlignmentA,
        ElementB, LayoutBTag, AlignmentB,
        ElementAccumulator,
        ThreadBlockShape, ClusterShape,
        cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
        cutlass::gemm::collective::KernelScheduleAuto                             // Kernel schedule policy. Auto defaults to cooperative kernel schedule
        >::CollectiveOp;

    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
        Shape<int,int,int,int>,                                                   // Indicates ProblemShape
        CollectiveMainloop,
        CollectiveEpilogue,
        void>;

    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

    // Reference device GEMM implementation type
    using StrideA   = typename Gemm::GemmKernel::StrideA;
    using LayoutA   = decltype(cute::make_layout(make_shape(0,0,0), StrideA{}));
    using LayoutSFA = typename Gemm::GemmKernel::CollectiveMainloop::LayoutSFA;      // Scale Factor tensors have an interleaved layout. Bring Layout instead of stride.
    using StrideB   = typename Gemm::GemmKernel::StrideB;
    using LayoutB   = decltype(cute::make_layout(make_shape(0,0,0), StrideB{}));
    using LayoutSFB = typename Gemm::GemmKernel::CollectiveMainloop::LayoutSFB;      // Scale Factor tensors have an interleaved layout. Bring Layout instead of stride.
    using StrideC   = typename Gemm::GemmKernel::StrideC;
    using LayoutC   = decltype(cute::make_layout(make_shape(0,0,0), StrideC{}));
    using StrideD   = typename Gemm::GemmKernel::StrideD;
    using LayoutD   = decltype(cute::make_layout(make_shape(0,0,0), StrideD{}));

    //
    // Data members
    //

    /// Initialization
    StrideA stride_A;
    LayoutA layout_A;
    LayoutSFA layout_SFA;
    StrideB stride_B;
    LayoutB layout_B;
    LayoutSFB layout_SFB;
    StrideC stride_C;
    LayoutC layout_C;
    StrideD stride_D;
    LayoutD layout_D;

    cutlass::HostTensor<ElementA::DataType, cutlass::layout::PackedVectorLayout> block_A;
    cutlass::HostTensor<ElementA::ScaleFactorType, cutlass::layout::PackedVectorLayout> block_SFA;
    cutlass::HostTensor<ElementB::DataType, cutlass::layout::PackedVectorLayout> block_B;
    cutlass::HostTensor<ElementB::ScaleFactorType, cutlass::layout::PackedVectorLayout> block_SFB;
    cutlass::HostTensor<ElementC, cutlass::layout::PackedVectorLayout> block_C;
    // Output Tensor
    cutlass::HostTensor<ElementD, cutlass::layout::PackedVectorLayout> block_D;

    // For SFA and SFB tensors layouts
    using Sm1xxBlkScaledConfig =  typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;

    stride_A = cutlass::make_cute_packed_stride(StrideA{}, {M, K, 1});
    stride_B = cutlass::make_cute_packed_stride(StrideB{}, {N, K, 1});
    stride_C = cutlass::make_cute_packed_stride(StrideC{}, {M, N, 1});
    stride_D = cutlass::make_cute_packed_stride(StrideD{}, {M, N, 1});

    layout_A = make_layout(make_shape(M, K, 1), stride_A);
    layout_B = make_layout(make_shape(N, K, 1), stride_B);
    layout_C = make_layout(make_shape(M, N, 1), stride_C);
    layout_D = make_layout(make_shape(M, N, 1), stride_D);
    layout_SFA = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(cute::make_shape(M, N, K, 1));
    layout_SFB = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(cute::make_shape(M, N, K, 1));


    block_A.reset(cutlass::make_Coord(size(layout_A)));
    block_B.reset(cutlass::make_Coord(size(layout_B)));
    block_C.reset(cutlass::make_Coord(size(layout_C)));
    block_D.reset(cutlass::make_Coord(size(layout_D)));
    block_SFA.reset(cutlass::make_Coord(size(filter_zeros(layout_SFA))));
    block_SFB.reset(cutlass::make_Coord(size(filter_zeros(layout_SFB))));

    block_A.copy_in_host_to_device(A);
    block_B.copy_in_host_to_device(B);
    block_C.copy_in_host_to_device(C);
    block_SFA.copy_in_host_to_device(SFA);
    block_SFB.copy_in_host_to_device(SFB);

    block_A.sync_device();
    block_B.sync_device();
    block_C.sync_device();
    block_SFA.sync_device();
    block_SFB.sync_device();
    
    // Timing using CUDA events
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    CHECK_CUDA(cudaEventRecord(start));
    Gemm gemmOp;

    typename Gemm::Arguments arguments {
        cutlass::gemm::GemmUniversalMode::kGemm,
        {M, N, K, 1},
        { // Mainloop arguments
            block_A.device_data(), stride_A,
            block_B.device_data(), stride_B,
            block_SFA.device_data(), layout_SFA,
            block_SFB.device_data(), layout_SFB
        },
        { // Epilogue arguments
            {1.0, 0},
            block_C.device_data(), stride_C,
            block_D.device_data(), stride_D
        }
    };

    auto status = gemmOp(arguments);

    assert(status == cutlass::Status::kSuccess);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    float milliseconds = 0;
    CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));
    return milliseconds;
    // std::printf("GEMM completed in %.3f ms\n", milliseconds);

}

int main() {
    // 假设我们创建一个 M x K 的矩阵
    const int M = 1024;
    const int N = 4096;
    const int K = 1408;
    const int block_size = 32;  // NVFP4 的缩放通常是 per-block 的，16 是常见值
    
    // 创建并初始化 nvfp4 矩阵（按行主序）
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
    

    // 随机初始化（这里我们模拟 float -> nvfp4 的量化过程）
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
    for (int it = 0; it < 100; it ++) {
        float t = matmul_host(A, B, M, N, K, C, D, scaleA, scaleB);
    }
    for (int it = 0; it < 200; it ++) {
        float t = matmul_host(A, B, M, N, K, C, D, scaleA, scaleB);
        ms += t;
    }
    std::printf("GEMM completed in %.3f ms\n", ms / 200);
    // CHECK_CUDA(cudaEventRecord(stop));
    // CHECK_CUDA(cudaEventSynchronize(stop));
    // float milliseconds = 0;
    // CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));

    // std::printf("GEMM completed in %.3f ms\n", milliseconds);
    std::cout << "nvfp6 gemm finished." << std::endl;
    return 0;
}