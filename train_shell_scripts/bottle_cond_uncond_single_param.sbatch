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

# MODELS=(1999 1750 1250 750 250)
MODELS=(1999)
SPLITS=(test train1)
DISTS=(
 "uniform(loc=0.001, scale=0.25)"
 "uniform(loc=0.25, scale=0.50)"
 "uniform(loc=0.50, scale=0.75)"
 "uniform(loc=0.75, scale=0.999)"
)

VARYINGPARAMS=(
  "[True, False, False, False, False, False]"
  "[False, True, False, False, False, False]"
  "[False, False, True, False, False, False]"
  "[False, False, False, False, False, True]"
)

OUTFILE="bottle-single-param-vary.csv"


for MODEL in "${MODELS[@]}"; do
  for SPLIT in "${SPLITS[@]}"; do
    for DIST in "${DISTS[@]}"; do
      for PARAM in "${VARYINGPARAMS[@]}"; do
        python -m train.eval_specific \
          --method scripted \
          --split "${SPLIT}" \
          --model-path /iliad/u/jmeribe/projects/vlm-scaffolding/logs/scripted/EnvBottle-v0-phys-rand-with-cond/train1/2025-07-23_14-44-16/model_${MODEL}.pt \
          --env-name EnvBottle-v0-phys-rand-with-cond \
          --distribution "${DIST}" \
          --varying-params "${PARAM}" \
          --output-file-name "${OUTFILE}"

        python -m train.eval_specific \
          --method scripted \
          --split "${SPLIT}" \
          --model-path /iliad/u/jmeribe/projects/vlm-scaffolding/logs/scripted/EnvBottle-v0-phys-rand/train1/2025-07-23_15-23-08/model_${MODEL}.pt \
          --env-name EnvBottle-v0-phys-rand \
          --distribution "${DIST}" \
          --varying-params "${PARAM}" \
          --output-file-name "${OUTFILE}"

        python -m train.eval_specific \
          --method scripted \
          --split "${SPLIT}" \
          --model-path /iliad/u/jmeribe/projects/vlm-scaffolding/logs/scripted/EnvBottle-v0/train1/2025-07-06_19-08-10/model_${MODEL}.pt \
          --env-name EnvBottle-v0-phys-rand-for-base \
          --distribution "${DIST}" \
          --varying-params "${PARAM}" \
          --output-file-name "${OUTFILE}"
      done
    done
  done
done