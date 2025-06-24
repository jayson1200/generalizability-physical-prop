import argparse
import glob
import os
import re

import gymnasium as gym
import numpy as np
import torch
import tqdm
import yaml
from mani_skill.utils.wrappers.record import RecordEpisode
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

from rsl_code.on_policy_runner import OnPolicyRunner
from rsl_code.wrapper import RslRlVecEnvWrapper
from tasks import *


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate rollout videos for a trained policy"
    )
    parser.add_argument(
        "--env-name", type=str, required=True, help="Environment name to evaluate on"
    )
    parser.add_argument(
        "--split", type=str, default="train1", help="Split to evaluate on"
    )
    parser.add_argument("--method", type=str, required=True, help="Method to evaluate")
    parser.add_argument(
        "--num-envs", type=int, default=4, help="Number of parallel environments"
    )
    parser.add_argument("--rt", action="store_true", help="Use RT rendering")
    return parser.parse_args()


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


def main():
    args = parse_args()

    # Determine subfolder for logs
    subfolder = args.method if args.method != "replay" else "gemini"

    # Find the latest run folder
    run_folders = glob.glob(f"logs/{subfolder}/{args.env_name}/{args.split}/*")
    if not run_folders:
        raise FileNotFoundError(
            f"No run folders found for {subfolder} {args.split} {args.env_name}"
        )

    run_folders.sort(key=os.path.getctime)
    run_name = os.path.basename(run_folders[-1])
    log_dir = f"logs/{subfolder}/{args.env_name}/{args.split}/{run_name}"

    # Load config and update it
    with open("rsl_code/config.yaml", "r") as f:
        train_cfg = yaml.load(f, Loader=yaml.FullLoader)
    train_cfg["runner"]["run_name"] = run_name
    for key in train_cfg["runner"]:
        train_cfg[key] = train_cfg["runner"][key]
    train_cfg["env_name"] = args.env_name
    train_cfg["gemini_traj"] = True

    # Load model
    latest_model_path = get_latest_model_path(log_dir)

    # Determine test split
    test_split = "test"
    # test_split = 'train1'
    if args.method in ["gemini_few_shot_tracking", "gemini_few_shot_oracle"]:
        test_split = args.split.replace("train", "test")

    if args.rt:
        rt_args = {
            "human_render_camera_configs": dict(
                shader_pack="rt", width=1000, height=1000
            ),
            "parallel_in_single_scene": True,
            "all_objects": True,
            "build_background": True,
        }
    else:
        rt_args = {}

    # Create environment
    eval_envs = gym.make(
        args.env_name,
        num_envs=args.num_envs,
        render_mode="rgb_array",
        method=args.method,
        use_wrist=True,
        start_limit=10,
        control_mode="pd_joint_pos",
        visualize_keypoints=False,
        split=test_split,
        **rt_args,
    )

    length = eval_envs.unwrapped.max_episode_steps
    eval_envs = RecordEpisode(
        eval_envs,
        output_dir=f'videos/{args.method}/{eval_envs.unwrapped.category}/{args.split}{"_rt" if args.rt else ""}',
        save_trajectory=False,
        save_video=True,
        max_steps_per_video=length,
        video_fps=60,
    )
    eval_envs = ManiSkillVectorEnv(
        eval_envs, args.num_envs, ignore_terminations=False, record_metrics=True
    )
    eval_vec_envs = RslRlVecEnvWrapper(eval_envs)

    # Run evaluation
    runner = OnPolicyRunner(eval_vec_envs, train_cfg, log_dir=log_dir, device="cuda")
    runner.load(latest_model_path)
    torch.manual_seed(0)
    np.random.seed(0)
    policy = runner.get_noisy_inference_policy()

    obs, _ = eval_vec_envs.reset(seed=0)

    with torch.inference_mode():
        for j in range(3):
            eval_vec_envs.reset()
            for i in tqdm.trange(length):
                act = policy(obs)
                obs, rew, done, info = eval_vec_envs.step(act)


if __name__ == "__main__":
    main()
