# MixedGemm-benchmark


**MixedGemm-samedtype** is a mixed-precision GEMM with quantize and reorder kernel performed on Blackwell GPUs(RTX5090).

We use [CUTLASS](https://github.com/NVIDIA/cutlass) to perform the mxfp4, mxfp6, mxfp8 GEMM.
> RTX 5070 Ti Laptop 

> M = 2048, N = 4096, K = 4096 (50%FP4, 0%FP6, 50%FP8)

| Stage | FP8-TRT | FP8 | MXFP8 | MXFP6 | MXFP4 | MicroMix | FP16 |
|---|---|---|---|---|---|---|---|
| Quantize | 0ms | 0ms | 0.078ms | 0.105ms | 0.096ms | 0.106ms | 0ms |
| GEMM | 0.795 ms | 0.566 ms | 0.501ms | 0.671ms | 0.229ms | 0.386ms | 1.253ms |
| Dequantize | 0ms | 0ms | 0ms | 0ms | 0ms | 0ms | 0ms |

In this branch, we perform benchmarks of various Quantize, Dequantize and GEMM kernels.

[CUDA TOOLKIT 12.8.1](https://developer.nvidia.com/cuda-12-8-1-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04&target_type=runfile_local) is required.

## Installation

1. Clone this repo and CUTLASS (Make sure you install Git, and Conda)
```
git clone https://github.com/actypedef/MixedGemm.git
git clone https://github.com/NVIDIA/cutlass.git
cd MixedGemm
git switch samedtype
```
2. Prepare environment
```
sudo apt-get update
sudo apt-get install python3-dev

curl -s https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | tee /etc/apt/trusted.gpg.d/kitware.gpg >/dev/null
sudo apt update
sudo apt install cmake

conda create -n mixedgemm python=3.12
conda activate mixedgemm
conda install pybind11
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```
3. Replace following paths in CMakeLists.txt with your actual paths
```
CMAKE_PREFIX_PATH
torch_python PATHS
PYTHON_ROOT
CUTLASS_ROOT
```
4. Make and run
```
bash remake.sh
python main.py
```
