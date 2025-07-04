# MixedGemm-benchmark


**MixedGemm** is a mixed-precision GEMM with quantize and reorder kernel performed on Blackwell GPUs(RTX5090).

We use [CUTLASS](https://github.com/NVIDIA/cutlass) to perform the mxfp4, mxfp6, mxfp8 GEMM.
> RTX 5070 Ti Laptop 

> M = 2048, N = 4096, K = 4096 (47%FP4, 0%FP6, 53%FP8)

| Stage      | W4A16-TRT | FP8-TRT  | MXFP8   | MXFP6   | MXFP4   | QuaRot(INT4)  | Atom(INT4) | MicroMix | FP16    |
|---         |---        |---       |---      |---      |---      |---            |---         |---       |---      |
| Quantize   | 0.075ms   | 0.082ms  | 0.078ms | 0.105ms | 0.096ms | 0.143ms       | 0.174ms    | 0.106ms  | 0ms     |
| GEMM       | 0.775ms   | 0.682 ms | 0.501ms | 0.671ms | 0.229ms | 3.012ms       | 0.523ms    | 0.386ms  | 1.253ms |
| Dequantize | 0ms       | 0ms      | 0ms     | 0ms     | 0ms     | 0.133ms       | 2.784ms    | 0ms      | 0ms     |
| Total      | 0.850ms   | 0.764ms  | 0.579ms | 0.706ms | 0.325ms | 3.284ms       | 3.481ms    | 0.492ms  | 1.253ms |


---

> RTX 5090 

> M = 2048, N = 4096, K = 4096 (50%FP4, 0%FP6, 50%FP8)

| Stage      | W4A16-TRT | FP8-TRT  | MXFP8   | MXFP6   | MXFP4   | QuaRot(INT4) | Atom(INT4) | MicroMix | FP16    |
|---         |---        |---       |---      |---      |---      |---           |---         |---       |---      |
| Quantize   | 0.016ms   | 0.009ms  | 0.027ms | 0.033ms | 0.031ms | 0.031ms      | 0.044ms    | 0.031ms  | 0ms     |
| GEMM       | 0.224ms   | 0.177 ms | 0.139ms | 0.202ms | 0.064ms | 0.947ms      | 0.081ms    | 0.106ms  | 0.389ms |
| Dequantize | 0ms       | 0ms      | 0ms     | 0ms     | 0ms     | 0.032ms      | 0.840ms    | 0ms      | 0ms     |
| Total      | 0.240ms   | 0.186ms  | 0.166ms | 0.235ms | 0.095ms | 1.010ms      | 0.965ms    | 0.137ms  | 0.389ms |


![](/img/llama.png)
![](/img/qwen.png)
![](/img/result.png)

In this branch, we perform benchmarks of various Quantize, Dequantize and GEMM kernels.

[CUDA TOOLKIT 12.8.1](https://developer.nvidia.com/cuda-12-8-1-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04&target_type=runfile_local) is required.

## Installation

0. If you do not have CUDA TOOLKIT 12.8.1, please refer to [this](https://developer.nvidia.com/cuda-12-8-1-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04&target_type=runfile_local), make sure you are on RTX50 Series or other BlackWell GPUs

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

conda create -n mixedgemm python=3.10
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
4. Run benchmark
```
pip install --upgrade tensorrt

bash remake.sh

./build/bench_gemm
./build/bench_reorder
./build/bench_fp8

python fp16-test.py
python trt-fp8-profiler.py
python trt-w4a16-profiler.py
```
