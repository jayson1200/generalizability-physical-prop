import argparse
import glob
import json
import logging
import os
import pickle
import re

import gymnasium as gym
import numpy as np
import torch
import yaml
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
from tqdm import tqdm

from build_dataset.task_specification import TASK_SPECIFICATIONS
from rsl_code.on_policy_runner import OnPolicyRunner
from rsl_code.wrapper import RslRlVecEnvWrapper
from tasks import *

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments for the few-shot dataset builder."""
    parser = argparse.ArgumentParser(
        description="Build few-shot dataset for a specific task"
    )
    parser.add_argument(
        "--task",
        choices=list(TASK_SPECIFICATIONS.keys()),
        default=None,
        help="Environment name to generate dataset for (e.g., TrackFridge-v0)",
    )
    parser.add_argument(
        "--num-envs", type=int, default=2000, help="Number of environments to run"
    )
    parser.add_argument(
        "--method",
        type=str,
        default="gemini",
        choices=["gemini", "gemini_few_shot_oracle", "gemini_iteration_2_oracle"],
        help="Method to use for trajectory generation",
    )
    parser.add_argument(
        "--criteria",
        type=str,
        default="oracle",
        choices=["tracking", "oracle"],
        help="Criteria to use for trajectory generation",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train1",
        choices=["train1", "train2", "train3"],
        help="Split to use for trajectory generation",
    )

    return parser.parse_args()


def get_latest_model_path(log_dir):
    logger.info(f"Searching for latest model in {log_dir}")
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
    model_path = f"{log_dir}/model_{latest_model_num}.pt"
    logger.info(f"Found latest model: {model_path}")
    return model_path


def main():
    args = parse_args()
    logger.info(f"Starting dataset generation with arguments: {args}")

    if args.task is None:
        raise ValueError("Please specify a task using --task argument")

    task_spec = TASK_SPECIFICATIONS[args.task].create()
    env_name = task_spec.env_name
    num_envs = args.num_envs
    method = args.method
    split = args.split

    logger.info(f"Looking for run folders for {env_name}")
    run_folders = glob.glob(f"logs/gemini/{env_name}/{split}/*")
    if not run_folders:
        raise FileNotFoundError(f"No run folders found for {env_name}")

    run_folders.sort(key=os.path.getctime)
    latest_run = os.path.basename(run_folders[-1])
    run_name = latest_run
    log_dir = f"logs/gemini/{env_name}/{split}/{run_name}"

    logger.info(f"Using latest run: {run_name}")
    with open("rsl_code/config.yaml", "r") as f:
        train_cfg = yaml.load(f, Loader=yaml.FullLoader)

    # Update configuration
    train_cfg["runner"]["run_name"] = run_name
    for key in train_cfg["runner"]:
        train_cfg[key] = train_cfg["runner"][key]
    train_cfg["env_name"] = env_name
    train_cfg["gemini_traj"] = True

    latest_model_path = get_latest_model_path(log_dir)
    logger.info(f"Loading model from {latest_model_path}")

    is_train = True
    logger.info(f"Creating environment with {num_envs} parallel instances")
    eval_envs = gym.make(
        env_name,
        num_envs=num_envs,
        render_mode="rgb_array",
        is_train=is_train,
        method=method,
        use_wrist=True,
        control_mode="pd_joint_pos",
        split=split,
    )
    if args.criteria == "oracle":
        eval_envs.unwrapped.start_limit = 10

    max_len = eval_envs.unwrapped.max_episode_steps
    eval_envs = ManiSkillVectorEnv(
        eval_envs, num_envs, ignore_terminations=False, record_metrics=True
    )
    eval_vec_envs = RslRlVecEnvWrapper(eval_envs)

    # Create runner
    logger.info("Initializing runner")
    runner = OnPolicyRunner(eval_vec_envs, train_cfg, log_dir=log_dir, device="cuda")

    # Load model
    logger.info("Loading model weights")
    runner.load(latest_model_path)

    # Run with noisy inference policy
    logger.info("Starting trajectory generation")
    policy = runner.get_noisy_inference_policy()
    obs, info = eval_vec_envs.reset()
    trajectories = eval_vec_envs.unwrapped.trajectories
    num_traj = len(trajectories["hand"])
    success_per_traj = np.zeros(num_traj, dtype=int)
    traj_count = np.zeros(num_traj, dtype=int)
    success = np.zeros(num_envs, dtype=bool)
    traj_idx = eval_vec_envs.unwrapped.traj_idx.cpu().numpy()
    with torch.no_grad():
        for i in tqdm(range(max_len), desc="Generating trajectories"):
            act = policy(obs)
            obs, rew, done, info = eval_vec_envs.step(act)

            if args.criteria == "tracking":
                success |= (
                    (
                        eval_vec_envs.unwrapped.elapsed_steps
                        >= eval_vec_envs.unwrapped.max_episode_steps - 1
                    )
                    .cpu()
                    .numpy()
                )
            elif args.criteria == "oracle":
                success |= eval_vec_envs.unwrapped.is_success().cpu().numpy()
            if i == 0:
                for j in traj_idx:
                    traj_count[j] += 1
    for j in np.where(success)[0]:
        success_per_traj[traj_idx[j]] += 1

    logger.info("Processing successful trajectories")
    success_rate = success_per_traj / (traj_count + 1e-6)
    num_best = 10
    best_idxs = np.argsort(-success_rate)[:num_best]
    logger.info(
        f"Success rates for top {num_best} trajectories: {success_rate[best_idxs]}"
    )

    best_imgs = []
    keypoints = eval_vec_envs.unwrapped.keypoints
    names = list(keypoints.keys())
    first_keypoint = keypoints[names[0]]
    for i in best_idxs:
        best_imgs.append(first_keypoint.imgs[i])

    best_image_coordinates = []
    for i in best_idxs:
        image_coordinates = []
        for name, keypoint in keypoints.items():
            y, x = keypoint.image_point[i].cpu().numpy()
            image_coordinates.append({"name": name, "point": [int(y), int(x)]})
        best_image_coordinates.append(image_coordinates)

    best_trajs = {}
    for name in tqdm(trajectories.keys(), desc="Processing trajectory data"):
        best_trajs[name] = trajectories[name][best_idxs, :300].clone()
        sample_indices = np.linspace(0, best_trajs[name].shape[1] - 1, 20, dtype=int)
        best_trajs[name] = best_trajs[name][:, sample_indices]

    best_trajs["hand"] = best_trajs["hand"][..., :3]
    names = task_spec.item_names + ["hand"]
    base_pos = best_trajs[names[0]][:, 0]
    for name in best_trajs.keys():
        best_trajs[name] = best_trajs[name] - base_pos[:, None]

    logger.info("Converting trajectories to JSON format")
    json_trajs = []
    for traj_idx in tqdm(
        range(best_trajs[names[0]].shape[0]), desc="Converting trajectories"
    ):
        json_traj = []
        for point_idx in range(best_trajs[names[0]].shape[1]):
            waypoint = {}
            for name in names:
                point = best_trajs[name][traj_idx, point_idx].cpu().numpy()
                waypoint[name] = {
                    "x": float(point[0]),
                    "y": float(point[1]),
                    "z": float(point[2]),
                }
            waypoint["waypoint_num"] = point_idx
            json_traj.append(waypoint)
        json_trajs.append(json_traj)

    trajectory_examples = []
    for i, traj in enumerate(tqdm(json_trajs, desc="Formatting trajectories")):
        formatted_traj = []
        for waypoint in traj:
            formatted_waypoint = {"waypoint_num": waypoint["waypoint_num"]}
            for name in names:
                formatted_waypoint[name] = {
                    "x": round(waypoint[name]["x"], 2),
                    "y": round(waypoint[name]["y"], 2),
                    "z": round(waypoint[name]["z"], 2),
                }
            formatted_traj.append(formatted_waypoint)
        example = f"{json.dumps(formatted_traj, indent=2)}"
        trajectory_examples.append(example)

        print(example)

    logger.info("Saving examples to file")
    if method == "gemini":
        subfolder = f"gemini_few_shot_{args.criteria}"
    elif method == "gemini_few_shot_oracle":
        subfolder = f"gemini_iteration_2_{args.criteria}"
    elif method == "gemini_iteration_2_oracle":
        subfolder = f"gemini_iteration_3_{args.criteria}"
    for new_split in [args.split, args.split.replace("train", "test")]:
        folder = f"data/{subfolder}/{task_spec.folder}/{new_split}"
        if not os.path.exists(folder):
            os.makedirs(folder)
        output_path = f"{folder}/few_shot_examples.pkl"
        examples = {
            "trajectories": trajectory_examples,
            "image_coordinates": best_image_coordinates,
            "images": best_imgs,
        }
        with open(output_path, "wb") as f:
            pickle.dump(examples, f)
        logger.info(f"Successfully saved examples to {output_path}")


if __name__ == "__main__":
    main()
