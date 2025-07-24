import argparse
import os
import shutil
from datetime import datetime

import gymnasium as gym
import yaml
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
from mani_skill.utils.structs.types import SimConfig

from rsl_code.on_policy_runner import OnPolicyRunner
from rsl_code.wrapper import RslRlVecEnvWrapper
from tasks import *


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train a policy on a ManiSkill environment"
    )
    parser.add_argument(
        "--run-name", type=str, default=None, help="Name for this training run"
    )
    parser.add_argument(
        "--num-envs", type=int, default=2048, help="Number of parallel environments"
    )
    parser.add_argument(
        "--num-iterations", type=int, default=5, help="Number of training iterations"
    )
    parser.add_argument(
        "--method", type=str, default="gemini", help="Method to train on"
    )
    parser.add_argument(
        "--delta-action", action="store_true", help="Use delta action space"
    )
    parser.add_argument("--split", type=str, default="train1", help="Split to train on")
    parser.add_argument(
        "--env-name",
        type=str,
        default="TrackTomato-v0",
        help="Environment name to train on",
    )
    parser.add_argument("--distribution", type=str, default="uniform(loc=0.0, scale=1.0)", help="Split to train on")
    
    return parser.parse_args()


args = parse_args()


def create_env(env_name, num_envs):
    """Create and wrap a ManiSkill environment."""
    delta_action = args.delta_action
    use_wrist = True
    if args.method == "rl":
        use_wrist = False
        delta_action = True

    env = gym.make(
        env_name,
        num_envs=num_envs,
        total_steps=args.num_iterations * 24,
        method=args.method,
        render_mode="rgb_array",
        control_mode="pd_joint_pos",
        use_wrist=use_wrist,
        delta_action=delta_action,
        split=args.split,
        distribution=args.distribution,
    )
    env = ManiSkillVectorEnv(
        env, num_envs, ignore_terminations=False, record_metrics=True
    )
    return RslRlVecEnvWrapper(env)


def load_config(config_path, run_name, env_name):
    """Load and prepare training configuration."""
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    config["runner"]["run_name"] = run_name

    for key, value in config["runner"].items():
        config[key] = value

    config["env_name"] = env_name
    config["method"] = args.method
    config["split"] = args.split
    return config


def main():
    vec_env = create_env(args.env_name, args.num_envs)
    vec_env.reset()

    time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = (
        args.run_name
        if args.run_name is not None
        else args.method + "_" + args.env_name + "_" + args.split + "_" + time_str
    )
    train_cfg = load_config("rsl_code/config.yaml", run_name, args.env_name)

    log_dir = f"logs/{args.method}/{args.env_name}/{args.split}/{time_str}"
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    runner = OnPolicyRunner(vec_env, train_cfg, log_dir=log_dir, device="cuda")
    runner.learn(args.num_iterations, True)
    runner.writer.stop()


if __name__ == "__main__":
    main()
