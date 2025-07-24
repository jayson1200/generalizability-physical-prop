#!/bin/bash
#SBATCH --account=iliad
#SBATCH --partition=iliad
#SBATCH --job-name=vlm-scripted-apple
#SBATCH --output=logs/vlm_scripted_apple_%j.out
#SBATCH --error=logs/vlm_scripted_apple_%j.err
#SBATCH --time=15:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --gres=gpu:1

# Load conda
source /iliad/u/jmeribe/miniconda3/bin/activate
conda activate vlm-dex

# Go to your project directory
cd /iliad2/u/jmeribe/projects/vlm-scaffolding

# Run training
python -m train.train \
    --run-name EnvApple-v0 \
    --env-name EnvApple-v0 \
    --num-iterations 2000 \
    --method scripted \
    --split train1

