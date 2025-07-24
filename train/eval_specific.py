import glob
import re

import gymnasium as gym
import numpy as np
import torch
import os
import pandas as pd
import argparse
import tqdm
import yaml
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

from rsl_code.on_policy_runner import OnPolicyRunner
from rsl_code.wrapper import RslRlVecEnvWrapper
from tasks import *

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate a policty on a ManiSkill environment"
    )

    parser.add_argument("--method", type=str, default="scripted", help="Method to evaluate on")
    parser.add_argument("--split", type=str, help="Dataset to test on")
    parser.add_argument("--model-path", type=str, help="Path to model")
    parser.add_argument("--env-name", type=str, default="TrackTomato-v0", help="Environment name to train on")
    parser.add_argument("--distribution", type=str, default="uniform(loc=0.0, scale=1.0)", help="Split to train on")
    
    return parser.parse_args()

args = parse_args()
method = args.method
split = args.split
env_name = args.env_name
model_path = args.model_path
log_dir = os.path.dirname(model_path)
run_name = "irrelevant"
model_name = model_path.split("/")[-1]
distribution = args.distribution
num_envs = 1024


results_file = "tracking_results.csv"
if os.path.exists(results_file):
    results_df = pd.read_csv(results_file, index_col=0)
else:
    results_df = pd.DataFrame(
        columns=[
            "method",
            "environment",
            "success",
            "num_envs",
            "run_name",
            "model_name",
            "split",
            "distribution",
        ]
    )


# Load config and update it
with open("rsl_code/config.yaml", "r") as f:
    train_cfg = yaml.load(f, Loader=yaml.FullLoader)
train_cfg["runner"]["run_name"] = run_name
for key in train_cfg["runner"]:
    train_cfg[key] = train_cfg["runner"][key]
train_cfg["env_name"] = env_name
train_cfg["gemini_traj"] = True


# Create environment
eval_envs = gym.make(
    env_name,
    num_envs=num_envs,
    render_mode="rgb_array",
    method=method,
    use_wrist=True,
    start_limit=10,
    control_mode="pd_joint_pos",
    split=split,
    distribution=distribution,
)
max_len = eval_envs.unwrapped.max_episode_steps
eval_envs = ManiSkillVectorEnv(
    eval_envs, num_envs, ignore_terminations=False, record_metrics=True
)
eval_vec_envs = RslRlVecEnvWrapper(eval_envs)

# Run evaluation
runner = OnPolicyRunner(
    eval_vec_envs, train_cfg, log_dir=log_dir, device="cuda"
)
runner.load(model_path)
torch.manual_seed(0)
np.random.seed(0)
policy = runner.get_noisy_inference_policy()

obs, info = eval_vec_envs.reset(seed=0)
success = np.zeros(eval_envs.num_envs, dtype=bool)

with torch.inference_mode():
    for i in tqdm.trange(max_len):
        act = policy(obs)
        obs, rew, done, info = eval_vec_envs.step(act)
        success = (
            success | eval_vec_envs.unwrapped.is_success().cpu().numpy()
        )

print(info)
success_rate = success.mean()
print(success_rate)

# Update results DataFrame
row_data = {
    "method": method,
    "environment": env_name,
    "success": success_rate,
    "num_envs": num_envs,
    "run_name": run_name,
    "model_name": model_name,
    "split": split,
    "distribution": distribution,
}


results_df = pd.concat(
    [results_df, pd.DataFrame([row_data])], ignore_index=True
)

# Save results
results_df.to_csv(results_file)

# Clean up
del runner, eval_envs, eval_vec_envs
