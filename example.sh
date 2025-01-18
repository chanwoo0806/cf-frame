#!/bin/bash

#SBATCH --job-name=NeuMF
#SBATCH --partition=a6000
#SBATCH --gres=gpu:1
#SBATCH --mem=30G
#SBATCH --time=3-00:00:00
#SBATCH --cpus-per-task=4
#SBATCH --output=./log_slurm/%j-%x.out

ml purge
ml load cuda/12.1
eval "$(conda shell.bash hook)"
conda activate cf-frame

lr_vals=(1.0e-3 5.0e-4)
layer_num_vals=(3 4)
dropout_vals=(0.0 0.1)

tst_batch=256
dataset=last-fm

for lr in "${lr_vals[@]}"; do
    for layer_num in "${layer_num_vals[@]}"; do
        for dropout in "${dropout_vals[@]}"; do
            srun python main.py \
                neumf \
                --default default \
                --loss bce \
                --dataset $dataset \
                --summary neumf \
                --comment "NeuMF, $dataset, $lr, $layer_num, $dropout" \
                --epoch 500 \
                --trn_batch 8192 \
                --tst_batch $tst_batch \
                --tst_step 1 \
                --device cuda \
                --embed_dim 32 \
                --negative_num 4 \
                --lr $lr \
                --dropout $dropout \
                --layer_num $layer_num
        done
    done
done
