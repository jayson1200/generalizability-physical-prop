#!/bin/bash
#SBATCH --account=iliad
#SBATCH --partition=iliad
#SBATCH --job-name=vlm-vary-bottle-params
#SBATCH --output=logs/vlm_scripted_vary_params_bottle_%j.out
#SBATCH --error=logs/vlm_scripted_vary_params_bottle_%j.err
#SBATCH --time=15:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --gres=gpu:1

# Load conda
source /iliad/u/jmeribe/miniconda3/bin/activate
conda activate vlm-dex

# Go to your project directory
cd /iliad2/u/jmeribe/projects/vlm-scaffolding


python -m train.vary_task_params.vary_multiple_bottle_params

