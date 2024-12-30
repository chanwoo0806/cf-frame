# Requirements

```bash
conda create -n cf-frame python=3.10
# conda install pytorch==1.12.1 cudatoolkit=11.3 -c pytorch
conda install pytorch pytorch-cuda=11.8 -c pytorch -c nvidia
conda install numpy==1.22.3
conda install pyyaml
conda install tqdm

pip install scipy==1.9.3
pip install tensorboard

# recommended
conda install jupyter
pip install pybind11
pip install torchdiffeq
```