apt update
apt upgrade -y
apt install git -y
apt install python-is-python3 -y 
apt install python3 -y 
apt install python3-pip -y
alias python=python3
bash scripts/init_submodules.sh
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128 --break-system-packages
pip install -r requirements.txt --break-system-packages
export MAKEFLAGS="-j32"
export CMAKE_BUILD_PARALLEL_LEVEL=32
export NINJA_FLAGS="-j32"   # sometimes used
export MAX_JOBS=32          # common in torch extensions
export USE_NINJA=1
pip install -v --break-system-packages   https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.6.8/flash_attn-2.7.4+cu128torch2.9-cp312-cp312-linux_x86_64.whl
pip install flashinfer-python --break-system-packages
pip install git+https://github.com/huggingface/diffusers --break-system-packages 

cd third_party/SageAttention/ # install SageAttention
export EXT_PARALLEL=4 NVCC_APPEND_FLAGS="--threads 18" MAX_JOBS=128 # parallel compiling (Optional)
python3 setup.py install  
cd ../..

cd third_party/sparse_sageattn_2 # if you want to use Radial Attention with SageAttention v2 backend
pip install ninja  --break-system-packages  # for parallel compilation
pip install -e . --break-system-packages
cd ../..

pip install termcolor --break-system-packages
pip install transformers --break-system-packages
pip install matplotlib --break-system-packages
pip install accelerate --break-system-packages
pip install ftfy --break-system-packages
pip install opencv-python --break-system-packages
pip install imageio imageio-ffmpeg --break-system-packages
pip install peft --break-system-packages
pip install pandas --break-system-packages
pip install seaborn --break-system-packages


## Fixes in the script for dense attention layers
