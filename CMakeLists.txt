cmake_minimum_required(VERSION 3.18)
project(fp4 LANGUAGES CXX CUDA)

# 查找 CUDA Toolkit
find_package(CUDAToolkit REQUIRED)

# 设置 CUTLASS 根目录
set(CUTLASS_ROOT /home/actypedef/cutlass-main)

# 包含 CUTLASS 头文件
include_directories(${CUTLASS_ROOT}/include)
include_directories(${CUTLASS_ROOT}/tools/util/include)
include_directories(${CUTLASS_ROOT}/examples/common)

# 设置 CMake CUDA 架构目标
set(CMAKE_CUDA_ARCHITECTURES 120a)

# 创建目标可执行文件
add_executable(fp4 fp4.cu)
add_executable(fp6 fp6.cu)
add_executable(fp8 fp8.cu)

# 设置 CUDA 编译选项（允许宽松 constexpr）
target_compile_options(fp4 PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
target_compile_options(fp6 PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
target_compile_options(fp8 PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

# 添加 CUTLASS 的架构宏定义（确保针对 SM120 架构）
target_compile_definitions(fp4 PRIVATE CUTLASS_ARCH_SM120_SUPPORTED)
target_compile_definitions(fp6 PRIVATE CUTLASS_ARCH_SM120_SUPPORTED)
target_compile_definitions(fp8 PRIVATE CUTLASS_ARCH_SM120_SUPPORTED)

