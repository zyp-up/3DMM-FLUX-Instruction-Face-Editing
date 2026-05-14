conda create -n controlface310 python=3.10 -y
conda activate controlface310

pip install --no-cache-dir torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121

pip install -U setuptools wheel ninja cmake
conda install -y -c fvcore -c iopath -c conda-forge fvcore iopath
conda install -y -c conda-forge mpi4py dlib scikit-learn "scikit-image<0.25" tqdm

pip install -r requirements.txt

# Install PyTorch3D from source when needed; avoid stale channel builds.
# git clone https://github.com/facebookresearch/pytorch3d.git
# cd pytorch3d
# pip install --no-build-isolation -e .
# --no-build-isolation forces the build to reuse the active torch install and
# avoids missing-torch errors inside an isolated build environment.
