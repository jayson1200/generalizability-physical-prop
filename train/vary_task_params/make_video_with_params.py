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
from .utils import MJCFHandler


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a policy on a ManiSkill environment"
    )
    parser.add_argument("--method", type=str, default="scripted",
                        help="Method to evaluate on")
    parser.add_argument("--split", type=str, default="test", help="Dataset to test on")
    parser.add_argument("--model-path", type=str, help="Path to model")
    parser.add_argument("--friction", type=str,
                        help="(linear friction, z-torque, xy-torque)")
    parser.add_argument("--density", type=str, help="density")
    parser.add_argument("--center_of_mass", type=str, help="pos x y z for new COM")
    parser.add_argument("--scale", type=str, help="uniform scale factor")
    parser.add_argument("--env_name", type=str, default="TrackTomato-v0", help="Environment name to train on")
    parser.add_argument("--obj_path", type=str, help="Environment name to train on")
    parser.add_argument("--distribution", type=str, default="uniform(loc=0.0, scale=1.0)", help="Split to train on")
    return parser.parse_args()


def main():
    args = parse_args()

    # Load config and update it
    with open("rsl_code/config.yaml", "r") as f:
        train_cfg = yaml.load(f, Loader=yaml.FullLoader)
    train_cfg["runner"]["run_name"] = "irrelevant"
    for key in train_cfg["runner"]:
        train_cfg[key] = train_cfg["runner"][key]
    train_cfg["env_name"] = args.env_name
    train_cfg["gemini_traj"] = True

    log_dir = os.path.dirname(args.model_path)
    test_split = "test"
    num_envs = 6
    rt_args = {
        # "human_render_camera_configs": dict(
        #     shader_pack="rt", width=1000, height=1000
        # ),
        # "parallel_in_single_scene": True,
        # "all_objects": True,
        # "build_background": True,
    }



    with MJCFHandler(args) as param_name:
        # Create environment
        eval_envs = gym.make(
            args.env_name,
            num_envs=num_envs,
            render_mode="rgb_array",
            method=args.method,
            use_wrist=True,
            start_limit=10,
            control_mode="pd_joint_pos",
            visualize_keypoints=False,
            split=test_split,
            model_scale=float(args.scale) if args.scale else 1.5,
            distribution=args.distribution,
            **rt_args,
        )

        length = eval_envs.unwrapped.max_episode_steps
        eval_envs = RecordEpisode(
            eval_envs,
            output_dir=f'videos/{args.method}/{eval_envs.unwrapped.category}/{args.split}/{param_name}',
            save_trajectory=False,
            save_video=True,
            max_steps_per_video=length,
            video_fps=60,
        )
        eval_envs = ManiSkillVectorEnv(
            eval_envs, num_envs, ignore_terminations=False, record_metrics=True
        )
        eval_vec_envs = RslRlVecEnvWrapper(eval_envs)

        # Run evaluation
        runner = OnPolicyRunner(eval_vec_envs, train_cfg, log_dir=log_dir, device="cuda")
        runner.load(args.model_path)
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
