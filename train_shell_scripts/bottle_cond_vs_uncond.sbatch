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

python -m train.eval_specific --method scripted --split test --model-path /iliad/u/jmeribe/projects/vlm-scaffolding/logs/scripted/EnvBottle-v0-phys-rand-with-cond/train1/2025-07-23_14-44-16/model_1999.pt  --env-name EnvBottle-v0-phys-rand-with-cond --distribution "beta(a=6.1, b=50.0)"
python -m train.eval_specific --method scripted --split test --model-path /iliad/u/jmeribe/projects/vlm-scaffolding/logs/scripted/EnvBottle-v0-phys-rand/train1/2025-07-23_15-23-08/model_1999.pt  --env-name EnvBottle-v0-phys-rand --distribution "beta(a=6.1, b=50.0)"

python -m train.eval_specific --method scripted --split test --model-path /iliad/u/jmeribe/projects/vlm-scaffolding/logs/scripted/EnvBottle-v0-phys-rand-with-cond/train1/2025-07-23_14-44-16/model_1999.pt  --env-name EnvBottle-v0-phys-rand-with-cond --distribution "beta(a=6.1, b=50.0)"
python -m train.eval_specific --method scripted --split test --model-path /iliad/u/jmeribe/projects/vlm-scaffolding/logs/scripted/EnvBottle-v0-phys-rand/train1/2025-07-23_15-23-08/model_1999.pt  --env-name EnvBottle-v0-phys-rand --distribution "beta(a=6.1, b=50.0)"

python -m train.eval_specific --method scripted --split test --model-path /iliad/u/jmeribe/projects/vlm-scaffolding/logs/scripted/EnvBottle-v0-phys-rand-with-cond/train1/2025-07-23_14-44-16/model_1999.pt  --env-name EnvBottle-v0-phys-rand-with-cond --distribution "beta(a=50, b=6.1)"
python -m train.eval_specific --method scripted --split test --model-path /iliad/u/jmeribe/projects/vlm-scaffolding/logs/scripted/EnvBottle-v0-phys-rand/train1/2025-07-23_15-23-08/model_1999.pt  --env-name EnvBottle-v0-phys-rand --distribution "beta(a=50, b=6.1)"

python -m train.eval_specific --method scripted --split test --model-path /iliad/u/jmeribe/projects/vlm-scaffolding/logs/scripted/EnvBottle-v0-phys-rand-with-cond/train1/2025-07-23_14-44-16/model_1999.pt  --env-name EnvBottle-v0-phys-rand-with-cond --distribution "beta(a=50, b=6.1)"
python -m train.eval_specific --method scripted --split test --model-path /iliad/u/jmeribe/projects/vlm-scaffolding/logs/scripted/EnvBottle-v0-phys-rand/train1/2025-07-23_15-23-08/model_1999.pt  --env-name EnvBottle-v0-phys-rand --distribution "beta(a=50, b=6.1)"