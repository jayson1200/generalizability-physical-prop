#!/bin/bash
#SBATCH --account=iliad
#SBATCH --partition=iliad
#SBATCH --job-name=vlm-scripted-bottle-w-cond
#SBATCH --output=logs/vlm_scripted_bottle_w_cond_%j.out
#SBATCH --error=logs/vlm_scripted_bottle_w_cond_%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1

# Load conda
source /iliad/u/jmeribe/miniconda3/bin/activate
conda activate vlm-dex

# Go to your project directory
cd /iliad2/u/jmeribe/projects/vlm-scaffolding

# Run training
python -m train.train \
    --run-name EnvBottle-v0-phys-rand-with-cond \
    --env-name EnvBottle-v0-phys-rand-with-cond \
    --num-iterations 2000 \
    --method scripted \
    --split train1 \
    --distribution "beta(a=2.5, b=1.9)" \
    --num-envs 1024 \

