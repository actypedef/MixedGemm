# MixedGemm

[![pV9V2gx.png](https://s21.ax1x.com/2025/05/30/pV9V2gx.png)](https://imgse.com/i/pV9V2gx)

**MixedGemm** is a mixed-precision GEMM with quantize and reorder kernel performed on Blackwell GPUs(RTX5090).

We use [CUTLASS](https://github.com/NVIDIA/cutlass) to perform the mxfp4, mxfp6, mxfp8 GEMM.

In this example, we quantized Weight to 100% mxfp4, Activation to 62.5% mxfp4, 34.375% mxfp6 and 3.125% mxfp8 to achieve best performance with tolerant accuracy loss.

[CUDA TOOLKIT 12.8.1](https://developer.nvidia.com/cuda-12-8-1-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04&target_type=runfile_local) is required.

## Installation

1. Clone this repo and CUTLASS (Make sure you install Git, and Conda)
```
git clone https://github.com/actypedef/MixedGemm.git
git clone https://github.com/NVIDIA/cutlass.git
cd MixedGemm
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
