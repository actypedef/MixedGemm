# MixedGemm


A simple mixed-precision GEMM kernel on RTX5090.


We use [CUTLASS](https://github.com/NVIDIA/cutlass) to perform the mxfp4, mxfp6, mxfp8 GEMM.

In this example, we quantized Weight to 100% mxfp4 ,Activation to 62.5% mxfp4 ＋ 34.375% mxfp6 ＋ 3.125% mxfp8 to achieve best performance with tolerant accuracy loss.

[CUDA TOOLKIT 12.8.1](https://developer.nvidia.com/cuda-12-8-1-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04&target_type=runfile_local) is required.

