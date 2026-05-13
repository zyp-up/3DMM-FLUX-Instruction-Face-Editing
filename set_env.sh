conda create -n controlface310 python=3.10 -y
conda activate controlface310

pip install --no-cache-dir torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121

pip install -U setuptools wheel ninja cmake
conda install -y -c fvcore -c iopath -c conda-forge fvcore iopath
conda install -y -c conda-forge mpi4py dlib scikit-learn "scikit-image<0.25" tqdm

pip install -r requirements.txt

# 若需要 PyTorch3D，请使用本地源码安装（不要直接使用 pytorch3d channel 的旧包）
# git clone https://github.com/facebookresearch/pytorch3d.git
# cd pytorch3d
# pip install --no-build-isolation -e .
# 说明：使用 --no-build-isolation 是为了强制复用当前环境里的 torch；
# 若直接 pip install -e .，可能在隔离构建环境中报 No module named 'torch'。
