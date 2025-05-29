# MixedGemm


A simple mixed-precision GEMM kernel on RTX5090.


We use [CUTLASS](https://github.com/NVIDIA/cutlass) to perform the mxfp4, mxfp6, mxfp8 GEMM.

benchmarks should be performed with several lines commented out.