import glob
import os
import re

import gymnasium as gym
import yaml
from mani_skill.utils.wrappers.record import RecordEpisode
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

from rsl_code.on_policy_runner import OnPolicyRunner
from rsl_code.wrapper import RslRlVecEnvWrapper


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


def create_eval_env(
    env_name,
    num_envs,
    log_dir,
    make_video=False,
    create_viewer=False,
    load_keypoints=True,
    load_trajectories=True,
):
    """
    Create evaluation environments with appropriate wrappers.

    Args:
        env_name: Name of the environment to create
        num_envs: Number of environments to create
        make_video: Whether to record videos

    Returns:
        Wrapped environment ready for evaluation
    """
    # Create base environment
    eval_envs = gym.make(
        env_name,
        num_envs=num_envs,
        render_mode="rgb_array",
        # is_train=False,
        # use_gemini_traj=True,
        # start_limits=(100, 100),
        control_mode="pd_joint_pos",
        load_keypoints=load_keypoints,
        load_trajectories=load_trajectories,
    )
    if create_viewer:
        viewer = eval_envs.render_human()
    else:
        viewer = None

    # Add video recording if requested
    if make_video:
        video_folder = f"{log_dir}/videos"
        os.makedirs(video_folder, exist_ok=True)
        eval_envs = RecordEpisode(
            eval_envs,
            output_dir=video_folder,
            save_trajectory=False,
            save_video=True,
            trajectory_name="trajectory",
            max_steps_per_video=600,
            video_fps=60,
        )

    # Add ManiSkill vector wrapper
    eval_envs = ManiSkillVectorEnv(
        eval_envs, num_envs, ignore_terminations=False, record_metrics=True
    )

    # Add RSL wrapper
    return RslRlVecEnvWrapper(eval_envs), viewer


def load_config_and_runner(env_name, run_name, vec_env, log_dir):
    """
    Load configuration and create a runner for evaluation.

    Args:
        env_name: Name of the environment
        run_name: Name of the training run
        vec_env: Vectorized environment

    Returns:
        Configured runner and path to the latest model
    """
    # Load configuration
    with open("rsl_code/configs/config2.yaml", "r") as f:
        train_cfg = yaml.load(f, Loader=yaml.FullLoader)

    # Update configuration
    train_cfg["runner"]["run_name"] = run_name
    for key in train_cfg["runner"]:
        train_cfg[key] = train_cfg["runner"][key]
    train_cfg["env_name"] = env_name
    train_cfg["gemini_traj"] = True

    # Create runner
    runner = OnPolicyRunner(vec_env, vec_env, train_cfg, log_dir=log_dir, device="cuda")

    # Load latest model
    latest_model_path = get_latest_model_path(log_dir)
    print(f"Loading model from {latest_model_path}")
    runner.load(latest_model_path)

    return runner
