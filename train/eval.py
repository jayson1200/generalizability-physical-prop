import glob
import re

import gymnasium as gym
import numpy as np
import torch
import tqdm
import yaml
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

from rsl_code.on_policy_runner import OnPolicyRunner
from rsl_code.wrapper import RslRlVecEnvWrapper
from tasks import *


def get_latest_model_path(log_dir):
    model_files = glob.glob(f"{log_dir}/model_*.pt")
    if not model_files:
        raise FileNotFoundError(f"No model checkpoints found in {log_dir}")

    model_numbers = []
    for file in model_files:
        match = re.search(r"model_(\d+)\.pt", file)
        if match:
            model_numbers.append(int(match.group(1)))

    if not model_numbers:
        raise ValueError(f"Could not parse model numbers from files in {log_dir}")

    latest_model_num = max(model_numbers)
    return f"{log_dir}/model_{latest_model_num}.pt"


import os

import pandas as pd

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
        ]
    )

for split in [
    "train1",
    "train2",
    "train3",
]:
    for method in [
        "replay",
        "scripted",
        "gemini",
        "gemini_few_shot_tracking",
        "gemini_few_shot_oracle",
        "gemini_3",
        "gemini_5",
        "gemini_10",
        "gemini_40",
        "gemini_keypoint_oracle",
        "gemini_trajectory_oracle",
        "gemini_iteration_2_oracle",
        "gemini_iteration_3_oracle",
    ]:
        for env_name in [
            "EnvApple-v0",
            "EnvBottle-v0",
            "EnvDrawer-v0",
            "EnvFridge-v0",
            "EnvHammer-v0",
            "EnvPliers-v0",
            "EnvScissors-v0",
            "EnvSponge-v0",
        ]:
            num_envs = 1000

            subfolder = method if method != "replay" else "gemini"
            run_folders = glob.glob(f"logs/{subfolder}/{env_name}/{split}/*")
            if not run_folders:
                print(
                    f"WARNING: No run folders found for {subfolder} {split} {env_name}. Skipping evaluation."
                )
                continue

            run_folders.sort(key=os.path.getctime)
            run_name = os.path.basename(run_folders[-1])
            log_dir = f"logs/{subfolder}/{env_name}/{split}/{run_name}"

            # Load config and update it
            with open("rsl_code/config.yaml", "r") as f:
                train_cfg = yaml.load(f, Loader=yaml.FullLoader)
            train_cfg["runner"]["run_name"] = run_name
            for key in train_cfg["runner"]:
                train_cfg[key] = train_cfg["runner"][key]
            train_cfg["env_name"] = env_name
            train_cfg["gemini_traj"] = True

            # Load model
            latest_model_path = get_latest_model_path(log_dir)
            model_name = latest_model_path.split("/")[-1]

            # Check if this exact run is already in the results
            mask = (
                (results_df["method"] == method)
                & (results_df["environment"] == env_name)
                & (results_df["num_envs"] >= num_envs)
                & (results_df["run_name"] == run_name)
                & (results_df["model_name"] == model_name)
                & (results_df["split"] == split)
            )

            if mask.any():
                print(
                    f"Skipping evaluation for {method} on {env_name} with model {model_name} - already in results."
                )
                continue

            # Evaluate on test set
            train_mode = "test"
            test_split = "test"
            if method in [
                "gemini_few_shot_tracking",
                "gemini_few_shot_oracle",
                "gemini_iteration_2_oracle",
                "gemini_iteration_3_oracle",
            ]:
                test_split = split.replace("train", "test")

            # Create environment
            eval_envs = gym.make(
                env_name,
                num_envs=num_envs,
                render_mode="rgb_array",
                method=method,
                use_wrist=True,
                start_limit=10,
                control_mode="pd_joint_pos",
                split=test_split,
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
            runner.load(latest_model_path)
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
            }

            # Update or add row
            mask = (
                (results_df["method"] == method)
                & (results_df["environment"] == env_name)
                & (results_df["split"] == split)
            )
            if mask.any():
                for col, value in row_data.items():
                    results_df.loc[mask, col] = value
            else:
                results_df = pd.concat(
                    [results_df, pd.DataFrame([row_data])], ignore_index=True
                )

            # Save results
            results_df.to_csv(results_file)

            # Clean up
            del runner, eval_envs, eval_vec_envs
