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
pip install torchdiffeq
```

# Run
```bash
python main.py \
    turbocf \
    --default default \
    --dataset gowalla \
    --loss nonparam \
    --trainer nonparam \
    --summary turbocf-gowalla \
    --rand_seed 2024 \
    --alpha 0.6 \
    --power 0.7 \
    --filter 1
```