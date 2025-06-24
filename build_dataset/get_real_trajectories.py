import argparse
import asyncio
import json
import logging
import os
from typing import Any, Dict, List

import numpy as np
import sapien
import torch
import tqdm
from google import genai
from google.auth import default
from google.genai import types
from mani_skill.utils import structs
from mani_skill.utils.geometry.trimesh_utils import (
    get_render_body_meshes,
    get_render_shape_meshes,
)
from PIL import Image
from scipy.interpolate import interp1d

from build_dataset.task_specification import TASK_SPECIFICATIONS, TaskSpecification
from tasks import *
from utils.keypoint import Keypoint

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


client = get_gemini_client()


async def query_trajectory_async(
    query: str, i: int, task_spec: TaskSpecification, folder: str, pbar: tqdm.tqdm
) -> str:
    """Query Gemini for trajectory generation asynchronously."""
    loop = asyncio.get_event_loop()
    try:
        resp = await asyncio.wait_for(
            loop.run_in_executor(
                None,
                lambda: client.models.generate_content(
                    model=MODEL,
                    contents=[query],
                    config=types.GenerateContentConfig(
                        response_modalities=["TEXT"],
                        tools=[types.Tool(code_execution=types.ToolCodeExecution)],
                        thinking_config=types.ThinkingConfig(thinking_budget=1000),
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


def batch_trajectories(all_trajectories: List[Dict]) -> Dict:
    return {
        name: torch.stack([traj[name] for traj in all_trajectories])
        for name in all_trajectories[0].keys()
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate dataset for a specific task")
    parser.add_argument(
        "--task",
        choices=list(TASK_SPECIFICATIONS.keys()),
        required=True,
        help="Task to generate dataset for",
    )
    parser.add_argument(
        "--method",
        choices=["gemini"],
        default="gemini",
        help="Method to use for trajectory generation",
    )
    parser.add_argument(
        "--split", default="train1", help="Split to generate dataset for"
    )
    args = parser.parse_args()

    task_spec = TASK_SPECIFICATIONS[args.task].create()

    folder = f"data/real/{args.task}/{args.split}"
    os.makedirs(folder, exist_ok=True)

    with open(os.path.join(folder, "responses", "keypoint_3d.json"), "r") as f:
        keypoints_3d = json.load(f)

    keypoints_3d = {
        name: [
            keypoints_3d[name]["x"],
            keypoints_3d[name]["y"],
            keypoints_3d[name]["z"],
        ]
        for name in task_spec.item_names
    }
    keypoints_3d = {
        name: torch.tensor(np.array(keypoints_3d[name]), device="cpu")
        for name in task_spec.item_names
    }

    trajectory_queries = []
    initial_positions = {}
    for name in task_spec.item_names:
        initial_positions[name] = torch.tensor(
            np.array(keypoints_3d[name]), device="cpu"
        )

    initial_positions["hand"] = torch.tensor(task_spec.initial_hand_pos, device="cpu")
    initial_positions["hand"] = initial_positions["hand"]

    normalized_initial_positions = {}
    for name in initial_positions.keys():
        normalized_initial_positions[name] = (
            initial_positions[name] - keypoints_3d[task_spec.item_names[0]]
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

    length = 20
    query = f"""Your are controlling a robot hand to {task_spec.trajectory_task}.
The coordinates are measured in meters.
The x axis is forward, the y axis is left and the z axis is up.
{task_spec.detailed_task}
Describe a very realistic trajectory of exactly {length} waypoints.
Use code to generate the output.
{position_info}
Use the following json format for the trajectory:
{json_format}
**Only** print the json output. Do **not** print anything else with the code."""

    print(query)

    logging.info("Querying Gemini for trajectory generation...")
    with tqdm.tqdm(total=1, desc="Getting trajectory responses") as pbar:

        async def main():
            tasks = [
                query_trajectory_async(query, i, task_spec, folder, pbar)
                for i in range(1)
            ]
            trajectory_responses = await asyncio.gather(*tasks)
            return trajectory_responses

        trajectory_responses = asyncio.run(main())

    future_pos = parse_trajectory_response(trajectory_responses[0])

    base_pos = keypoints_3d[task_spec.item_names[0]].cpu().numpy()
    for name in future_pos.keys():
        future_pos[name] = future_pos[name] + base_pos
        future_pos[name][0] = initial_positions[name].cpu().numpy()
        future_pos[name] = interpolate_trajectory(np.array(future_pos[name]))
        future_pos[name] = torch.tensor(future_pos[name])
        if name == "hand":
            future_pos[name] = torch.cat(
                (future_pos[name], torch.zeros(future_pos[name].shape[0], 3)), dim=1
            )
        future_pos[name] = future_pos[name].squeeze(0)

    initial_ori = torch.tensor(task_spec.initial_hand_orientation)
    initial_ori = initial_ori
    future_pos["hand"][..., 3:6] = initial_ori
    print(future_pos)
    future_pos = batch_trajectories([future_pos])
    print(future_pos)
    torch.save(future_pos, os.path.join(folder, "trajectories.pt"))
