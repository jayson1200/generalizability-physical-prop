#!/bin/bash
#SBATCH --account=iliad
#SBATCH --partition=iliad
#SBATCH --job-name=bottle-cond-vs-uncond
#SBATCH --output=logs/bottle_cond_vs_uncond_%j.out
#SBATCH --error=logs/bottle_cond_vs_uncond_%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1

# Load conda
source /iliad/u/jmeribe/miniconda3/bin/activate
conda activate vlm-dex

# Go to your project directory
cd /iliad2/u/jmeribe/projects/vlm-scaffolding

MODELS=(1999 1750 1250 750 250)
SPLITS=(test train1)
#DISTS=(
#  "beta(a=6.1, b=50.0)"
#  "beta(a=50, b=6.1)"
#  "beta(a=50, b=50)"
#)

DISTS=(
  "beta(a=2.5, b=1.9)"
  "beta(a=2.5, b=6.8)"
)

for MODEL in "${MODELS[@]}"; do
  for SPLIT in "${SPLITS[@]}"; do
    for DIST in "${DISTS[@]}"; do

      python -m train.eval_specific_two \
        --method scripted \
        --split "${SPLIT}" \
        --model-path /iliad/u/jmeribe/projects/vlm-scaffolding/logs/scripted/EnvBottle-v0-phys-rand-with-cond/train1/2025-07-23_14-44-16/model_${MODEL}.pt \
        --env-name EnvBottle-v0-phys-rand-with-cond \
        --distribution "${DIST}"

      python -m train.eval_specific_two \
        --method scripted \
        --split "${SPLIT}" \
        --model-path /iliad/u/jmeribe/projects/vlm-scaffolding/logs/scripted/EnvBottle-v0-phys-rand/train1/2025-07-23_15-23-08/model_${MODEL}.pt \
        --env-name EnvBottle-v0-phys-rand \
        --distribution "${DIST}"

    done
  done
done