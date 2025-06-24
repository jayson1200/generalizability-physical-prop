import logging

logging.basicConfig(
    level=logging.WARNING, format="%(asctime)s - %(levelname)s - %(message)s"
)
import argparse
import asyncio
import json
import os
import pickle
import shutil
from collections import Counter, defaultdict
from typing import Any, Dict, List, Tuple

import cv2
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import sapien
import torch
import tqdm
from google import genai
from google.auth import default
from google.genai import types
from mani_skill.utils import structs
from mani_skill.utils.geometry.geometry import transform_points
from mani_skill.utils.geometry.trimesh_utils import (
    get_render_body_meshes,
    get_render_shape_meshes,
)
from PIL import Image
from scipy.interpolate import interp1d

from build_dataset.task_specification import TASK_SPECIFICATIONS, TaskSpecification
from tasks import *
from utils.keypoint import Keypoint

# Configuration
LOCATION = "us-central1"
PROJECT = ""  # replace
MODEL = "gemini-2.5-flash-preview-04-17"
EPISODE_LENGTH = 300


def get_gemini_client():
    """Initialize and return a Gemini client using environment-based authentication."""
    try:
        credentials, _ = default()
        return genai.Client(
            vertexai=True, project=PROJECT, location=LOCATION, credentials=credentials
        )
    except Exception as e:
        logging.error(f"Failed to initialize Gemini client: {e}")
        raise


def add_pos_noise(pos, noise: bool = True):
    if noise:
        return pos + torch.rand_like(pos) * 0.2 - 0.1
    else:
        return pos


def add_ori_noise(ori, noise: bool = True):
    if noise:
        return ori + torch.rand_like(ori) * np.pi / 4 - np.pi / 8
    else:
        return ori


client = get_gemini_client()


def parse_trajectory_response(ans: Any) -> Dict:
    """Parse the trajectory response from the model."""
    json_ans = json.loads(ans)
    traj = {}
    for i in range(len(json_ans)):
        for name in json_ans[i].keys():
            if name == "waypoint_num":
                continue
            pred = json_ans[i][name]
            pred = [pred["x"], pred["y"], pred["z"]] if isinstance(pred, dict) else pred
            traj[name] = traj.get(name, []) + [pred]
    return {k: np.array(v) for k, v in traj.items()}


def interpolate_trajectory(
    points: np.ndarray, num_steps: int = EPISODE_LENGTH
) -> np.ndarray:
    """Interpolate trajectory points to create a smooth path."""
    t_original = np.linspace(0, 1, len(points))
    t_interp = np.linspace(0, 1, num_steps)
    return np.column_stack(
        [interp1d(t_original, points[:, i])(t_interp) for i in range(3)]
    )


def save_trajectory_plots(all_trajectories: List[Dict], folder: str) -> None:
    """Save plots of all trajectories for each dimension."""
    keypoint_names = list(all_trajectories[0].keys())
    colors = plt.cm.rainbow(np.linspace(0, 1, len(keypoint_names)))
    name_to_color = {name: color for name, color in zip(keypoint_names, colors)}

    for i, dim in enumerate(["x", "y", "z"]):
        plt.figure(figsize=(15, 8))
        for traj_idx, future_pos in enumerate(all_trajectories[:5]):
            for name in future_pos.keys():
                plt.plot(future_pos[name][:, i], alpha=0.7, color=name_to_color[name])

        for name in keypoint_names:
            plt.plot([], [], label=name, color=name_to_color[name])

        plt.xlabel("Time Step")
        plt.ylabel(f"{dim.upper()} Position (m)")
        plt.title(f"Trajectories in {dim.upper()} Dimension")
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                folder, "visualizations", "trajectories", f"all_trajectories_{dim}.png"
            ),
            bbox_inches="tight",
        )
        plt.close()


async def query_keypoints_async(
    img: Image.Image, query: str, i: int, folder: str, pbar: tqdm.tqdm, examples
) -> str:
    """Query Gemini for keypoint detection asynchronously."""
    loop = asyncio.get_event_loop()
    contents = [img, query]

    try:
        resp = await asyncio.wait_for(
            loop.run_in_executor(
                None,
                lambda: client.models.generate_content(
                    model=MODEL,
                    contents=contents,
                    config=types.GenerateContentConfig(temperature=0.5),
                ),
            ),
            timeout=120,
        )
        with open(
            os.path.join(folder, "responses", f"keypoint_detection_response_{i}.txt"),
            "w",
        ) as f:
            f.write(resp.text)
        pbar.update(1)
        return resp.text
    except asyncio.TimeoutError:
        logging.warning(f"Keypoint query for sample {i} timed out after 6 minutes")
        pbar.update(1)
        return None
    except Exception as e:
        logging.warning(f"Failed to get keypoint response for sample {i}: {e}")
        pbar.update(1)
        return None


async def query_trajectory_async(
    img_with_points: Image.Image,
    query: str,
    initial_positions: Dict,
    i: int,
    task_spec: TaskSpecification,
    folder: str,
    pbar: tqdm.tqdm,
) -> str:
    """Query Gemini for trajectory generation asynchronously."""
    loop = asyncio.get_event_loop()
    try:
        resp = await asyncio.wait_for(
            loop.run_in_executor(
                None,
                lambda: client.models.generate_content(
                    model=MODEL,
                    contents=[img_with_points, query],
                    config=types.GenerateContentConfig(
                        response_modalities=["TEXT"],
                        tools=[types.Tool(code_execution=types.ToolCodeExecution)],
                        thinking_config=types.ThinkingConfig(
                            thinking_budget=1000 if task_spec.test_mode else None
                        ),
                    ),
                ),
            ),
            timeout=600,
        )
        if resp.candidates is None:
            return None
        last_out = next(
            (
                part.code_execution_result.output
                for part in resp.candidates[0].content.parts[::-1]
                if part.code_execution_result is not None
            ),
            None,
        )
        with open(
            os.path.join(
                folder, "responses", f"trajectory_generation_response_{i}.txt"
            ),
            "w",
        ) as f:
            try:
                formatted_json = json.dumps(json.loads(last_out), indent=2)
                f.write(formatted_json)
            except json.JSONDecodeError:
                f.write(last_out)
        pbar.update(1)
        return last_out
    except asyncio.TimeoutError:
        logging.warning(f"Trajectory query for sample {i} timed out after 10 minutes")
        pbar.update(1)
        return None
    except Exception as e:
        logging.warning(f"Failed to get trajectory response for sample {i}: {e}")
        pbar.update(1)
        return None


def get_object_name(obj):
    """Extract the object name from the object."""
    return obj.name.split("_", 1)[1].split("-", 1)[0] if "-" in obj.name else obj.name


def process_all_keypoint_responses(
    responses: List[str],
    imgs: List[Image.Image],
    positions: List[torch.Tensor],
    env: gym.Env,
    task_spec: TaskSpecification,
    folder: str,
    object_poses: Dict[str, torch.Tensor],
) -> Tuple[
    List[Image.Image],
    Dict[str, torch.Tensor],
    Dict[str, Keypoint],
    Dict[str, torch.Tensor],
]:
    """Process all keypoint detection responses.

    Returns:
        List[Image.Image]: List of images with keypoints drawn
        Dict[str, torch.Tensor]: Dictionary of grasp positions for each item
        Dict[str, Keypoint]: Dictionary of keypoints for each item
    """
    valid_images = []
    valid_grasp_pos = defaultdict(list)
    all_points = []
    all_obj_names = []
    all_link_names = []

    # Pre-compute meshes for all objects
    object_meshes = {}
    for obj in list(env.unwrapped.all_objects.values()) + list(
        env.unwrapped.all_articulations.values()
    ):
        if get_object_name(obj) not in task_spec.affected_objects:
            continue

        if isinstance(obj, structs.articulation.Articulation):
            for link in obj.links:
                if not hasattr(link, "render_shapes"):
                    continue

                points = []
                for render_shape in link.render_shapes:
                    for shape in render_shape:
                        meshes = get_render_shape_meshes(shape)
                        points.extend(
                            torch.tensor(mesh.sample(10000), dtype=torch.float32)
                            for mesh in meshes
                        )
                if not points:
                    continue

                points = torch.cat(points, dim=0).cuda().clone()
                object_meshes[(obj.name, link.name)] = points
        else:
            for obj2 in obj._objs:
                comp = obj2.find_component_by_type(sapien.render.RenderBodyComponent)
                if comp is None:
                    continue

                points = []
                for mesh in get_render_body_meshes(comp):
                    points.append(torch.tensor(mesh.sample(10000), dtype=torch.float32))
                if not points:
                    continue

                points = torch.cat(points, dim=0).cuda().clone()
                object_meshes[obj.name] = points

    # Process each response
    new_object_poses = {}
    yx = {}
    for query_idx, resp_text in enumerate(
        tqdm.tqdm(responses, desc="Processing keypoint responses")
    ):
        try:
            data = json.loads(resp_text.strip("'").strip("`json\n").strip("`"))
            with open(
                os.path.join(
                    folder, "responses", f"keypoint_detection_response_{query_idx}.txt"
                ),
                "w",
            ) as f:
                json.dump(data, f, indent=2)

            # Check if all required items are present
            required_names = set(task_spec.item_names)
            present_names = {item["name"] for item in data}
            if not required_names.issubset(present_names):
                missing_names = required_names - present_names
                logging.warning(
                    f"Missing required items in keypoint response {query_idx}: {missing_names}"
                )
                continue

            # Draw keypoints on image
            img_cv = np.array(imgs[query_idx])
            img_with_points = img_cv.copy()
            for item in data:
                name = item["name"]
                y, x = [int(p * 800 / 1000) for p in item["point"]]
                cv2.circle(img_with_points, (x, y), 5, (0, 0, 255), -1)
                cv2.putText(
                    img_with_points,
                    name,
                    (x + 10, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 0, 0),
                    1,
                )

            img_with_points_pil = Image.fromarray(img_with_points)
            img_with_points_pil.save(
                os.path.join(
                    folder,
                    "visualizations",
                    "keypoints",
                    f"initial_with_points_query_{query_idx}.png",
                )
            )

            # Process each keypoint
            grasp_pos = {}
            valid_points = True
            for item in tqdm.tqdm(
                data, desc=f"Finding closest points for query {query_idx}", leave=False
            ):
                try:
                    name = item["name"]
                    image_pos = item["point"]
                    y, x = [int(p * 800 / 1000) for p in image_pos]
                    grasp_pos[name] = positions[query_idx][y, x].cuda()

                    closest_point = None
                    closest_dist = float("inf")
                    closest_obj_name = None
                    closest_link_name = None

                    for obj in list(env.unwrapped.all_objects.values()) + list(
                        env.unwrapped.all_articulations.values()
                    ):
                        if get_object_name(obj) not in task_spec.affected_objects:
                            continue

                        if isinstance(obj, structs.articulation.Articulation):
                            for link in obj.links:
                                key = (obj.name, link.name)
                                if key not in object_meshes:
                                    continue
                                points = object_meshes[key].clone()
                                pose = object_poses[key][query_idx].clone()
                                pose = structs.pose.Pose.create_from_pq(
                                    p=pose[None, :3], q=pose[None, 3:]
                                )
                                trans = pose.to_transformation_matrix().clone()[0]
                                world_points = (trans[:3, :3] @ points.T).T + trans[
                                    :3, 3
                                ]
                                dist = torch.norm(
                                    world_points - grasp_pos[name][None], dim=-1
                                )

                                if dist.min() < closest_dist:
                                    closest_dist = dist.min()
                                    closest_point = points[dist.argmin()]
                                    closest_obj_name = get_object_name(obj)
                                    closest_link_name = link.name
                        else:
                            if obj.name not in object_meshes:
                                continue
                            points = object_meshes[obj.name].clone()
                            pose = object_poses[obj.name][query_idx].clone()
                            pose = structs.pose.Pose.create_from_pq(
                                p=pose[None, :3], q=pose[None, 3:]
                            )
                            trans = pose.to_transformation_matrix().clone()[0]
                            world_points = (trans[:3, :3] @ points.T).T + trans[:3, 3]
                            dist = torch.norm(
                                world_points - grasp_pos[name][None], dim=-1
                            )

                            if dist.min() < closest_dist:
                                closest_dist = dist.min()
                                closest_point = points[dist.argmin()]
                                closest_obj_name = get_object_name(obj)
                                closest_link_name = None

                    if closest_point is None:
                        valid_points = False
                        logging.warning(
                            f"Could not find closest point for {name} in query {query_idx}"
                        )
                        break

                    all_points.append(closest_point)
                    all_obj_names.append(closest_obj_name)
                    all_link_names.append(closest_link_name)
                except Exception as e:
                    logging.warning(
                        f"Failed to process keypoint {item.get('name', 'unknown')} in query {query_idx}: {e}"
                    )
                    valid_points = False
                    break

            # Only add if all points are valid and all required names are present
            if valid_points and set(grasp_pos.keys()) == required_names:
                valid_images.append(imgs[query_idx])
                for name in object_poses.keys():
                    if name not in new_object_poses:
                        new_object_poses[name] = []
                    new_object_poses[name].append(object_poses[name][query_idx])
                for name, pos in grasp_pos.items():
                    valid_grasp_pos[name].append(pos)
                for item in data:
                    name = item["name"]
                    if name not in yx:
                        yx[name] = []
                    yx[name].append(item["point"])
        except Exception as e:
            logging.warning(f"Failed to process keypoint response {query_idx}: {e}")
            continue

    # Group keypoints
    keypoints = {}
    if all_points:
        grouped_points = defaultdict(list)
        grouped_obj_names = defaultdict(list)
        grouped_link_names = defaultdict(list)

        current_idx = 0
        for query_idx, resp_text in enumerate(responses):
            try:
                data = json.loads(resp_text.strip("'").strip("`json\n").strip("`"))
                required_names = set(task_spec.item_names)
                present_names = {item["name"] for item in data}
                if not required_names.issubset(present_names):
                    continue

                for item in data:
                    name = item["name"]
                    grouped_points[name].append(all_points[current_idx])
                    grouped_obj_names[name].append(all_obj_names[current_idx])
                    grouped_link_names[name].append(all_link_names[current_idx])
                    current_idx += 1
            except Exception as e:
                logging.warning(f"Failed to group keypoints for query {query_idx}: {e}")
                continue

        # Create keypoints only if all required items are present
        if all(name in grouped_points for name in task_spec.item_names):
            for name in grouped_points:
                try:
                    stacked_points = torch.stack(grouped_points[name])
                    obj_name = Counter(grouped_obj_names[name]).most_common(1)[0][0]
                    link_name = (
                        Counter(grouped_link_names[name]).most_common(1)[0][0]
                        if any(grouped_link_names[name])
                        else None
                    )
                    image_points = torch.tensor(np.array(yx[name]))
                    keypoints[name] = Keypoint(
                        obj_name, link_name, stacked_points, image_points, valid_images
                    )
                except Exception as e:
                    logging.warning(f"Failed to create keypoint for {name}: {e}")
                    # Log more detailed information about the exception
                    import traceback

                    logging.error(
                        f"Detailed error for {name} keypoint creation: {str(e)}"
                    )
                    logging.error(f"Exception type: {type(e).__name__}")
                    logging.error(f"Stack trace: {traceback.format_exc()}")
                    continue

    # Stack grasp positions
    stacked_grasp_pos = {
        name: torch.stack(pos) for name, pos in valid_grasp_pos.items()
    }
    stacked_object_poses = {
        name: torch.stack(pos) for name, pos in new_object_poses.items()
    }

    return valid_images, stacked_grasp_pos, keypoints, stacked_object_poses


def process_trajectory_response(
    resp_text: str,
    grasp_pos: Dict,
    initial_positions: Dict,
    task_spec: TaskSpecification,
    folder: str,
    noise: bool = True,
    num_steps: int = EPISODE_LENGTH,
) -> Dict:
    """Process a single trajectory response."""
    if resp_text is None:
        return None
    try:
        with open(
            os.path.join(
                folder,
                "responses",
                f"trajectory_generation_response_{task_spec.folder.split('/')[-1]}.txt",
            ),
            "w",
        ) as f:
            try:
                data = json.loads(resp_text)
                json.dump(data, f, indent=2)
            except json.JSONDecodeError:
                f.write(resp_text)
                logging.warning("Invalid JSON in trajectory response, skipping")
                return None

        future_pos = parse_trajectory_response(resp_text)

        # Check if all required names are present
        required_names = set(task_spec.item_names + ["hand"])
        if not all(name in future_pos for name in required_names):
            missing_names = required_names - set(future_pos.keys())
            logging.warning(
                f"Missing required names in trajectory response: {missing_names}"
            )
            return None

        # Process the trajectory
        base_pos = grasp_pos[task_spec.item_names[0]].cpu().numpy()
        for name in future_pos.keys():
            future_pos[name] = future_pos[name] + base_pos
            future_pos[name][0] = initial_positions[name].cpu().numpy()
            future_pos[name] = interpolate_trajectory(
                np.array(future_pos[name]), num_steps
            )
            future_pos[name] = torch.tensor(future_pos[name])
            if name == "hand":
                future_pos[name] = torch.cat(
                    (future_pos[name], torch.zeros(future_pos[name].shape[0], 3)), dim=1
                )
            future_pos[name] = future_pos[name].squeeze(0)

        initial_ori = torch.tensor(task_spec.initial_hand_orientation)
        initial_ori = add_ori_noise(initial_ori, noise)
        future_pos["hand"][..., 3:6] = initial_ori
        return future_pos
    except Exception as e:
        logging.warning(f"Failed to process trajectory response: {e}")
        return None


def batch_trajectories(all_trajectories: List[Dict]) -> Dict:
    """Batch all trajectories by stacking vectors in the dictionaries."""
    return {
        name: torch.stack([traj[name] for traj in all_trajectories])
        for name in all_trajectories[0].keys()
    }


async def process_trajectory_response_async(
    resp_text: str,
    grasp_pos: Dict,
    initial_positions: Dict,
    task_spec: TaskSpecification,
    folder: str,
    noise: bool = True,
    num_steps: int = EPISODE_LENGTH,
) -> Dict:
    """Process a single trajectory response asynchronously."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        lambda: process_trajectory_response(
            resp_text, grasp_pos, initial_positions, task_spec, folder, noise, num_steps
        ),
    )


async def main_async():
    parser = argparse.ArgumentParser(description="Generate dataset for a specific task")
    parser.add_argument(
        "--task",
        choices=list(TASK_SPECIFICATIONS.keys()),
        required=True,
        help="Task to generate dataset for",
    )
    parser.add_argument(
        "--num-samples", type=int, default=5, help="Number of samples to generate"
    )
    parser.add_argument(
        "--test", action="store_true", help="Run in test mode with limited tokens"
    )
    parser.add_argument(
        "--method",
        choices=[
            "gemini",
            "gemini_3",
            "gemini_5",
            "gemini_10",
            "gemini_40",
            "gemini_few_shot_tracking",
            "gemini_few_shot_oracle",
            "gemini_iteration_2_oracle",
            "gemini_iteration_3_oracle",
            "gemini_keypoint_oracle",
            "gemini_trajectory_oracle",
            "scripted",
        ],
        default="gemini",
        help="Method to use for trajectory generation",
    )
    parser.add_argument(
        "--split",
        choices=["train1", "train2", "train3", "test", "test1", "test2", "test3"],
        default="train1",
        help="Split to generate dataset for",
    )
    parser.add_argument(
        "--no-noise", action="store_false", help="Add noise to the trajectories"
    )
    args = parser.parse_args()

    task_spec = TASK_SPECIFICATIONS[args.task].create()
    task_spec.test_mode = args.test

    folder = f"data/{args.method}/{task_spec.folder}/{args.split}"
    if os.path.exists(folder):
        if args.method not in (
            "gemini_few_shot_tracking",
            "gemini_few_shot_oracle",
            "gemini_iteration_2_oracle",
            "gemini_iteration_3_oracle",
        ):
            shutil.rmtree(folder)
        else:
            for item in os.listdir(folder):
                if item != "few_shot_examples.pkl":
                    item_path = os.path.join(folder, item)
                    if os.path.isdir(item_path):
                        shutil.rmtree(item_path)
                    else:
                        os.remove(item_path)

    os.makedirs(folder, exist_ok=True)
    for subdir in ["queries", "responses", "visualizations"]:
        os.makedirs(os.path.join(folder, subdir), exist_ok=True)
    os.makedirs(os.path.join(folder, "visualizations", "keypoints"), exist_ok=True)
    os.makedirs(os.path.join(folder, "visualizations", "trajectories"), exist_ok=True)

    env = gym.make(
        task_spec.env_name,
        num_envs=4,
        all_objects=True,
        render_mode="rgb_array",
        method=args.method,
        use_wrist=False,
        load_keypoints=False,
        load_trajectories=False,
        human_render_camera_configs=dict(shader_pack="rt", width=800, height=800),
        parallel_in_single_scene=True,
        build_background=True,
    )

    imgs = []
    positions = []
    object_poses = {}
    os.makedirs(os.path.join(folder, "visualizations", "initial"), exist_ok=True)
    split_hash = args.split
    if split_hash in ["test1", "test2", "test3"]:
        split_hash = "test"
    seed = hash(split_hash) % (2**31)
    for i in tqdm.trange(args.num_samples, desc="Generating initial images"):
        env.reset(seed=seed + i)

        for name in list(env.unwrapped.all_objects.keys()) + list(
            env.unwrapped.all_articulations.keys()
        ):
            if name in env.unwrapped.all_objects:
                obj = env.unwrapped.all_objects[name]
            else:
                obj = env.unwrapped.all_articulations[name]
                for link in obj.links:
                    object_poses[(obj.name, link.name)] = object_poses.get(
                        (obj.name, link.name), []
                    ) + [link.pose.raw_pose[-1].clone()]

            object_poses[obj.name] = object_poses.get(obj.name, []) + [
                obj.pose.raw_pose[-1].clone()
            ]
        img = Image.fromarray(env.render()[0, :, :800].cpu().numpy())
        img.save(os.path.join(folder, "visualizations", f"initial/{i}.png"))

        camera = env.unwrapped.scene.human_render_cameras["render_camera"]
        position = camera.get_obs()["position"][0, :, :800].cpu().clone()
        position = position.float() / 1000.0
        model_matrix = camera.camera.get_model_matrix().cpu()[0]
        position = (
            position.reshape(-1, 3) @ model_matrix[:3, :3].T + model_matrix[:3, 3]
        )
        position = position.reshape(800, 800, 3)

        depth_normalized = (
            (position[..., 2] / position[..., 2].max() * 255).numpy().astype(np.uint8)
        )
        depth_img = Image.fromarray(depth_normalized)
        os.makedirs(os.path.join(folder, "visualizations", "depth"), exist_ok=True)
        depth_img.save(os.path.join(folder, "visualizations", "depth", f"{i}.png"))

        positions.append(position.clone())
        imgs.append(img)

    if args.method in (
        "gemini_few_shot_tracking",
        "gemini_few_shot_oracle",
        "gemini_iteration_2_oracle",
        "gemini_iteration_3_oracle",
    ):
        examples = pickle.load(
            open(
                f"data/{args.method}/{task_spec.folder}/{args.split}/few_shot_examples.pkl",
                "rb",
            )
        )
    else:
        examples = None

    if args.method in ["gemini_keypoint_oracle", "scripted"]:
        gemini_folder = f"data/gemini/{task_spec.folder}/test"
        if os.path.exists(gemini_folder):
            keypoints_file = os.path.join(gemini_folder, "keypoints.pt")
            if os.path.exists(keypoints_file):
                keypoints = torch.load(keypoints_file, weights_only=False)
                logging.info(f"Loaded keypoints: {keypoints}")

        median_keypoints = {}
        for name, keypoint in keypoints.items():
            point = torch.median(keypoint.point, dim=0).values
            point = point.repeat(args.num_samples, 1)
            image_point = torch.zeros(args.num_samples, 2)
            imgs = [keypoint.imgs[0]] * args.num_samples
            median_keypoints[name] = Keypoint(
                keypoint.obj_name, keypoint.link_name, point, image_point, imgs
            )

        valid_images = imgs
        valid_grasp_pos = {}
        for name, keypoint in median_keypoints.items():
            obj_name = keypoint.obj_name
            link_name = keypoint.link_name
            if link_name is not None:
                obj = env.unwrapped.all_articulations[obj_name]
                for link in obj.links:
                    if link.name == link_name:
                        pose = object_poses[(obj.name, link.name)]
                        break
            else:
                obj = env.unwrapped.all_objects[obj_name]
                pose = object_poses[obj.name]
            pose = torch.stack(pose)
            pose = structs.pose.Pose.create_from_pq(p=pose[..., :3], q=pose[..., 3:])
            trans = pose.to_transformation_matrix().clone()
            world_point = transform_points(trans, keypoint.point.clone())
            valid_grasp_pos[name] = world_point
        object_poses = {
            name: torch.stack(value) for name, value in object_poses.items()
        }
        keypoints = median_keypoints
    else:
        query = f"""{task_spec.keypoint_task}
The answer should follow the json format: [{', '.join([f'{{"name": "{name}", "point": [...]}}' for name in task_spec.item_names])}]
The points are in [y, x] format normalized to 0-1000."""

        with open(os.path.join(folder, "queries", "keypoint_query.txt"), "w") as f:
            f.write(query)

        logging.info("Querying Gemini for keypoint detection...")
        with tqdm.tqdm(
            total=args.num_samples, desc="Getting keypoint responses"
        ) as pbar:
            keypoint_responses = await asyncio.gather(
                *[
                    query_keypoints_async(imgs[i], query, i, folder, pbar, examples)
                    for i in range(args.num_samples)
                ]
            )

        valid_images, valid_grasp_pos, keypoints, object_poses = (
            process_all_keypoint_responses(
                keypoint_responses,
                imgs,
                positions,
                env,
                task_spec,
                folder,
                object_poses,
            )
        )

    if args.method in ["gemini_trajectory_oracle", "scripted"]:
        valid_trajectories = []
        valid_indices = []
        for i in range(len(valid_images)):
            valid_indices.append(i)
            init_hand_pos = add_pos_noise(
                torch.tensor(np.array(task_spec.initial_hand_pos)), not args.no_noise
            ).cuda()
            if args.task == "fridge":
                waypoints = {"hand": [], "handle": []}
                pos = valid_grasp_pos["handle"][i]
                for j in range(5):
                    t = j / 4
                    interpolated_pos = init_hand_pos * (1 - t) + pos * t
                    waypoints["hand"].append(interpolated_pos.cpu().numpy())
                    waypoints["handle"].append(pos.cpu().numpy())
                hinge_pos = pos + torch.tensor(
                    [0, -0.65, 0], device=init_hand_pos.device
                )
                for j in range(5, 20):
                    t = (j - 5) / 14
                    angle = torch.tensor(t * np.pi / 2, device=init_hand_pos.device)
                    radius = torch.norm(hinge_pos - pos)
                    x = pos[0] + radius * torch.sin(angle)
                    y = pos[1] - radius * (1 - torch.cos(angle))
                    z = pos[2]
                    new_pos = torch.tensor([x, y, z], device=init_hand_pos.device)
                    waypoints["hand"].append(new_pos.cpu().numpy())
                    waypoints["handle"].append(new_pos.cpu().numpy())
            elif args.task == "drawer":
                pos = valid_grasp_pos["handle"][i]
                waypoints = {"hand": [], "handle": []}
                for j in range(5):
                    t = j / 4
                    interpolated_pos = init_hand_pos * (1 - t) + pos * t
                    waypoints["hand"].append(interpolated_pos.cpu().numpy())
                    waypoints["handle"].append(pos.cpu().numpy())
                for j in range(5, 20):
                    t = (j - 5) / 14
                    x = pos[0] + 0.3 * t
                    new_pos = torch.tensor(
                        [x, pos[1], pos[2]], device=init_hand_pos.device
                    )
                    waypoints["hand"].append(new_pos.cpu().numpy())
                    waypoints["handle"].append(new_pos.cpu().numpy())
            elif args.task == "scissors":
                left_pos = valid_grasp_pos["loop 1"][i]
                right_pos = valid_grasp_pos["loop 2"][i]

                waypoints = {"hand": [], "loop 1": [], "loop 2": []}

                target = (left_pos + right_pos) / 2
                for j in range(5):
                    t = j / 4
                    interpolated_pos = init_hand_pos * (1 - t) + target * t
                    waypoints["hand"].append(interpolated_pos.cpu().numpy())
                    waypoints["loop 1"].append(left_pos.cpu().numpy())
                    waypoints["loop 2"].append(right_pos.cpu().numpy())

                for j in range(5, 20):
                    t = (j - 5) / 14
                    left_new = left_pos * (1 - t) + target * t
                    right_new = right_pos * (1 - t) + target * t
                    waypoints["hand"].append(target.cpu().numpy())
                    waypoints["loop 1"].append(left_new.cpu().numpy())
                    waypoints["loop 2"].append(right_new.cpu().numpy())
            elif args.task == "pliers":
                left_pos = valid_grasp_pos["handle left"][i]
                right_pos = valid_grasp_pos["handle right"][i]

                waypoints = {"hand": [], "handle left": [], "handle right": []}

                target = (left_pos + right_pos) / 2
                for j in range(5):
                    t = j / 4
                    interpolated_pos = init_hand_pos * (1 - t) + target * t
                    waypoints["hand"].append(interpolated_pos.cpu().numpy())
                    waypoints["handle left"].append(left_pos.cpu().numpy())
                    waypoints["handle right"].append(right_pos.cpu().numpy())

                for j in range(5, 20):
                    t = (j - 5) / 14
                    left_new = left_pos * (1 - t) + target * t
                    right_new = right_pos * (1 - t) + target * t
                    waypoints["hand"].append(target.cpu().numpy())
                    waypoints["handle left"].append(left_new.cpu().numpy())
                    waypoints["handle right"].append(right_new.cpu().numpy())
            elif args.task == "bottle":
                bottle_pos = valid_grasp_pos["bottle"][i]
                point_pos = valid_grasp_pos["point"][i]

                waypoints = {"hand": [], "bottle": [], "point": []}
                for j in range(5):
                    t = j / 4
                    interpolated_pos = init_hand_pos * (1 - t) + bottle_pos * t
                    waypoints["hand"].append(interpolated_pos.cpu().numpy())
                    waypoints["bottle"].append(bottle_pos.cpu().numpy())
                    waypoints["point"].append(point_pos.cpu().numpy())

                target1 = bottle_pos + torch.tensor(
                    [0, 0, 0.1], device=init_hand_pos.device
                )
                for j in range(5, 9):
                    t = (j - 5) / 3
                    new_pos = bottle_pos * (1 - t) + target1 * t
                    waypoints["hand"].append(new_pos.cpu().numpy())
                    waypoints["bottle"].append(new_pos.cpu().numpy())
                    waypoints["point"].append(point_pos.cpu().numpy())

                target2 = point_pos + torch.tensor(
                    [0, 0, 0.15], device=init_hand_pos.device
                )
                for j in range(9, 18):
                    t = (j - 9) / 8
                    new_pos = target1 * (1 - t) + target2 * t
                    waypoints["hand"].append(new_pos.cpu().numpy())
                    waypoints["bottle"].append(new_pos.cpu().numpy())
                    waypoints["point"].append(point_pos.cpu().numpy())

                target3 = point_pos + torch.tensor(
                    [0, 0, 0.05], device=init_hand_pos.device
                )
                for j in range(18, 20):
                    t = (j - 18) / 1
                    new_pos = target2 * (1 - t) + target3 * t
                    waypoints["hand"].append(new_pos.cpu().numpy())
                    waypoints["bottle"].append(new_pos.cpu().numpy())
                    waypoints["point"].append(point_pos.cpu().numpy())

            elif args.task == "apple":
                apple_pos = valid_grasp_pos["apple"][i]
                board_pos = valid_grasp_pos["cutting board"][i]

                waypoints = {"hand": [], "apple": [], "cutting board": []}
                for j in range(5):
                    t = j / 4
                    interpolated_pos = init_hand_pos * (1 - t) + apple_pos * t
                    waypoints["hand"].append(interpolated_pos.cpu().numpy())
                    waypoints["apple"].append(apple_pos.cpu().numpy())
                    waypoints["cutting board"].append(board_pos.cpu().numpy())

                target1 = apple_pos + torch.tensor(
                    [0, 0, 0.1], device=init_hand_pos.device
                )
                for j in range(5, 9):
                    t = (j - 5) / 3
                    new_pos = apple_pos * (1 - t) + target1 * t
                    waypoints["hand"].append(new_pos.cpu().numpy())
                    waypoints["apple"].append(new_pos.cpu().numpy())
                    waypoints["cutting board"].append(board_pos.cpu().numpy())

                target2 = board_pos + torch.tensor(
                    [0, 0, 0.15], device=init_hand_pos.device
                )
                for j in range(9, 18):
                    t = (j - 9) / 8
                    new_pos = target1 * (1 - t) + target2 * t
                    waypoints["hand"].append(new_pos.cpu().numpy())
                    waypoints["apple"].append(new_pos.cpu().numpy())
                    waypoints["cutting board"].append(board_pos.cpu().numpy())

                target3 = board_pos + torch.tensor(
                    [0, 0, 0.05], device=init_hand_pos.device
                )
                for j in range(18, 20):
                    t = (j - 18) / 1
                    new_pos = target2 * (1 - t) + target3 * t
                    waypoints["hand"].append(new_pos.cpu().numpy())
                    waypoints["apple"].append(new_pos.cpu().numpy())
                    waypoints["cutting board"].append(board_pos.cpu().numpy())
            elif args.task == "sponge":
                sponge_pos = valid_grasp_pos["sponge"][i]

                waypoints = {"hand": [], "sponge": []}
                for j in range(5):
                    t = j / 4
                    interpolated_pos = init_hand_pos * (1 - t) + sponge_pos * t
                    waypoints["hand"].append(interpolated_pos.cpu().numpy())
                    waypoints["sponge"].append(sponge_pos.cpu().numpy())

                target1 = sponge_pos + torch.tensor(
                    [-0.2, 0, 0], device=init_hand_pos.device
                )
                for j in range(5, 8):
                    t = (j - 5) / 2
                    new_pos = sponge_pos * (1 - t) + target1 * t
                    waypoints["hand"].append(new_pos.cpu().numpy())
                    waypoints["sponge"].append(new_pos.cpu().numpy())

                target2 = target1 + torch.tensor(
                    [0, -0.2, 0], device=init_hand_pos.device
                )
                for j in range(8, 12):
                    t = (j - 7) / 4
                    new_pos = target1 * (1 - t) + target2 * t
                    waypoints["hand"].append(new_pos.cpu().numpy())
                    waypoints["sponge"].append(new_pos.cpu().numpy())

                target3 = target2 + torch.tensor(
                    [0.2, 0, 0], device=init_hand_pos.device
                )
                for j in range(12, 16):
                    t = (j - 11) / 4
                    new_pos = target2 * (1 - t) + target3 * t
                    waypoints["hand"].append(new_pos.cpu().numpy())
                    waypoints["sponge"].append(new_pos.cpu().numpy())

                target4 = target3 + torch.tensor(
                    [0, 0.2, 0], device=init_hand_pos.device
                )
                for j in range(16, 20):
                    t = (j - 15) / 4
                    new_pos = target3 * (1 - t) + target4 * t
                    waypoints["hand"].append(new_pos.cpu().numpy())
                    waypoints["sponge"].append(new_pos.cpu().numpy())

            elif args.task == "hammer":
                handle_pos = valid_grasp_pos["handle"][i]
                head_pos = valid_grasp_pos["head"][i]

                waypoints = {"hand": [], "handle": [], "head": []}
                for j in range(5):
                    t = j / 4
                    interpolated_pos = init_hand_pos * (1 - t) + handle_pos * t
                    waypoints["hand"].append(interpolated_pos.cpu().numpy())
                    waypoints["handle"].append(handle_pos.cpu().numpy())
                    waypoints["head"].append(head_pos.cpu().numpy())

                for i in range(3):
                    handle_target = handle_pos + torch.tensor(
                        [0, 0, 0.2], device=init_hand_pos.device
                    )
                    head_target = head_pos + torch.tensor(
                        [0, 0, 0.2], device=init_hand_pos.device
                    )
                    for j in range(5 + i * 5, 8 + i * 5):
                        t = (j - (4 + i * 5)) / 3
                        new_handle_pos = handle_pos * (1 - t) + handle_target * t
                        new_head_pos = head_pos * (1 - t) + head_target * t

                        waypoints["hand"].append(new_handle_pos.cpu().numpy())
                        waypoints["handle"].append(new_handle_pos.cpu().numpy())
                        waypoints["head"].append(new_head_pos.cpu().numpy())

                    handle_target2 = handle_pos + torch.tensor(
                        [0, 0, 0.05], device=init_hand_pos.device
                    )
                    head_target2 = head_pos + torch.tensor(
                        [0, 0, 0.05], device=init_hand_pos.device
                    )
                    for j in range(8 + i * 5, 10 + i * 5):
                        t = (j - (7 + i * 5)) / 2
                        new_handle_pos = handle_target * (1 - t) + handle_target2 * t
                        new_head_pos = head_target * (1 - t) + head_target2 * t

                        waypoints["hand"].append(new_handle_pos.cpu().numpy())
                        waypoints["handle"].append(new_handle_pos.cpu().numpy())
                        waypoints["head"].append(new_head_pos.cpu().numpy())

            for name in waypoints.keys():
                waypoints[name] = interpolate_trajectory(np.array(waypoints[name]))
                waypoints[name] = torch.tensor(waypoints[name])
                if name == "hand":
                    waypoints[name] = torch.cat(
                        (waypoints[name], torch.zeros(waypoints[name].shape[0], 3)),
                        dim=1,
                    )
                waypoints[name] = waypoints[name].squeeze(0)

            initial_ori = torch.tensor(task_spec.initial_hand_orientation)
            initial_ori = add_ori_noise(initial_ori, not args.no_noise)
            waypoints["hand"][..., 3:6] = initial_ori
            valid_trajectories.append(waypoints)
    else:
        trajectory_queries = []
        for i in tqdm.trange(len(valid_images), desc="Preparing trajectory queries"):
            initial_positions = {}
            for name in task_spec.item_names:
                initial_positions[name] = valid_grasp_pos[name][i].clone()

            initial_positions["hand"] = torch.tensor(
                task_spec.initial_hand_pos,
                device=valid_grasp_pos[task_spec.item_names[0]].device,
            )
            initial_positions["hand"] = add_pos_noise(
                initial_positions["hand"], not args.no_noise
            )

            normalized_initial_positions = {}
            for name in initial_positions.keys():
                normalized_initial_positions[name] = (
                    initial_positions[name]
                    - valid_grasp_pos[task_spec.item_names[0]][i]
                )
                normalized_initial_positions[name] = (
                    normalized_initial_positions[name].cpu().numpy()
                )

            position_info = "\n".join(
                f"The initial position of the {name} is [{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}]."
                for name, pos in normalized_initial_positions.items()
            )

            json_format = (
                '[{\n"waypoint_num": 0,\n'
                + ",\n".join(
                    f'"{name}": {{"x": float, "y": float, "z": float}}'
                    for name in task_spec.item_names + ["hand"]
                )
                + "\n} ...]"
            )

            if args.method.endswith("_3"):
                length = 3
            elif args.method.endswith("_4"):
                length = 4
            elif args.method.endswith("_5"):
                length = 5
            elif args.method.endswith("_10"):
                length = 10
            elif args.method.endswith("_40"):
                length = 40
            else:
                length = 20
            query = f"""Your are controlling a robot hand to {task_spec.trajectory_task}.
The coordinates are measured in meters.
The x axis is forward, the y axis is left and the z axis is up.
{task_spec.detailed_task}
Describe a very realistic trajectory of exactly {length} waypoints.
Use code to generate the output.
"""
            if examples is not None:
                import random

                selected_examples = random.sample(
                    examples["trajectories"], min(3, len(examples["trajectories"]))
                )
                example_texts = []
                for j, example in enumerate(selected_examples, 1):
                    example_texts.append(f"Example {j}:\n{example}\n")
                few_shot_examples = "\n".join(example_texts)
                query += f"""Here are some examples of trajectories. Note that the initial hand and object positions might be different.
{few_shot_examples}
Use these examples to generate a new trajectory.
"""
            query += f"""{position_info}
Use the following json format for the trajectory:
{json_format}
**Only** print the json output. Do **not** print anything else with the code."""

            with open(
                os.path.join(folder, "queries", f"trajectory_query_{i}.txt"), "w"
            ) as f:
                f.write(query)

            trajectory_queries.append((valid_images[i], query, initial_positions, i))

        logging.info("Querying Gemini for trajectory generation...")
        with tqdm.tqdm(
            total=len(trajectory_queries), desc="Getting trajectory responses"
        ) as pbar:
            trajectory_responses = await asyncio.gather(
                *[
                    query_trajectory_async(
                        *args, task_spec=task_spec, pbar=pbar, folder=folder
                    )
                    for args in trajectory_queries
                ]
            )

        logging.info("Processing trajectory responses...")
        valid_trajectories = []
        valid_indices = []
        length = EPISODE_LENGTH
        # if 'arm' in args.task:
        # length = EPISODE_LENGTH * 2
        for idx, resp_text in enumerate(
            tqdm.tqdm(trajectory_responses, desc="Processing trajectories")
        ):
            grasp_pos = {
                name: valid_grasp_pos[name][idx] for name in task_spec.item_names
            }
            initial_positions = {
                name: trajectory_queries[idx][2][name]
                for name in task_spec.item_names + ["hand"]
            }
            trajectory = await process_trajectory_response_async(
                resp_text,
                grasp_pos,
                initial_positions,
                task_spec,
                folder,
                not args.no_noise,
                length,
            )
            if trajectory is not None:
                valid_trajectories.append(trajectory)
                valid_indices.append(idx)

    if not valid_trajectories:
        logging.error("No valid trajectories generated")
        return

    save_trajectory_plots(valid_trajectories, folder)

    batched_trajectories = batch_trajectories(valid_trajectories)
    batched_trajectories = {
        k: batched_trajectories[k] for k in task_spec.item_names + ["hand"]
    }

    torch.save(batched_trajectories, os.path.join(folder, "trajectories.pt"))

    # Save only the keypoints that correspond to valid trajectories
    valid_keypoints = {}
    for name in task_spec.item_names:
        keypoint = keypoints[name]
        valid_points = keypoint.point[valid_indices]
        valid_image_points = keypoint.image_point[valid_indices]
        valid_imgs = []
        for j in valid_indices:
            valid_imgs.append(keypoint.imgs[j])
        valid_keypoints[name] = Keypoint(
            keypoint.obj_name,
            keypoint.link_name,
            valid_points,
            valid_image_points,
            valid_imgs,
        )

    torch.save(valid_keypoints, os.path.join(folder, "keypoints.pt"))

    for name in object_poses.keys():
        object_poses[name] = object_poses[name][valid_indices]
    torch.save(object_poses, os.path.join(folder, "object_poses.pt"))

    env_with_trajectories = gym.make(
        task_spec.env_name,
        num_envs=4,
        method=args.method,
        render_mode="rgb_array",
        use_wrist=True,
        load_keypoints=True,
        load_trajectories=True,
        split=args.split,
    )

    frames = []
    obs, _ = env_with_trajectories.reset()
    for _ in tqdm.tqdm(range(length), desc="Visualizing trajectories"):
        action = env_with_trajectories.action_space.sample()
        obs, reward, terminated, truncated, info = env_with_trajectories.step(action)

        # Render all 4 environments and combine them into a single image
        rendered_frames = env_with_trajectories.render().cpu().numpy()
        h, w, c = rendered_frames[0].shape
        combined_frame = np.zeros((h * 2, w * 2, c), dtype=np.uint8)

        # Arrange the 4 frames in a 2x2 grid
        combined_frame[:h, :w] = rendered_frames[0]
        combined_frame[:h, w:] = rendered_frames[1]
        combined_frame[h:, :w] = rendered_frames[2]
        combined_frame[h:, w:] = rendered_frames[3]

        frames.append(combined_frame)

    import imageio

    imageio.mimsave(
        os.path.join(folder, "visualizations", "loaded_trajectories.mp4"),
        frames,
        fps=60,
    )

    logging.info("Dataset generation completed successfully")


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
