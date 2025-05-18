cmake_minimum_required(VERSION 3.18)
project(MixedGemm LANGUAGES CXX CUDA)

# 查找 CUDA Toolkit
find_package(CUDAToolkit REQUIRED)

# 设置 CUTLASS 根目录
set(CUTLASS_ROOT /root/autodl-tmp/cutlass)

# 包含 CUTLASS 头文件
include_directories(./include)
include_directories(${CUTLASS_ROOT}/include)
include_directories(${CUTLASS_ROOT}/tools/util/include)
include_directories(${CUTLASS_ROOT}/examples/common)

# 设置 CMake CUDA 架构目标
set(CMAKE_CUDA_ARCHITECTURES 120a)

# 创建目标可执行文件
add_executable(bench_fp4 bench_fp4.cu src/fp4.cu)
add_executable(bench_fp6 bench_fp6.cu src/fp6.cu)
add_executable(bench_fp8 bench_fp8.cu src/fp8.cu)
add_executable(bench_gemm bench_gemm.cu src/gemm.cu src/fp4.cu src/fp6.cu src/fp8.cu)

# 设置 CUDA 编译选项（允许宽松 constexpr）
target_compile_options(bench_fp4 PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
target_compile_options(bench_fp6 PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
target_compile_options(bench_fp8 PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
target_compile_options(bench_gemm PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

# 添加 CUTLASS 的架构宏定义（确保针对 SM120 架构）
target_compile_definitions(bench_fp4 PRIVATE CUTLASS_ARCH_SM120_SUPPORTED)
target_compile_definitions(bench_fp6 PRIVATE CUTLASS_ARCH_SM120_SUPPORTED)
target_compile_definitions(bench_fp8 PRIVATE CUTLASS_ARCH_SM120_SUPPORTED)
target_compile_definitions(bench_gemm PRIVATE CUTLASS_ARCH_SM120_SUPPORTED)

