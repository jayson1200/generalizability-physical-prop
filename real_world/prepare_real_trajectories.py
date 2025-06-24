import json
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import tyro


@dataclass
class Args:
    task_name: str
    experiment_dir: str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_experiment"
    split_name: str = "latest"


TASK_NAME_TO_OBJECTS = {
    "box_arm": ["box", "bottle"],
    "bottle_arm3": ["bottle", "plate"],
}


def run_command(command: str, title: Optional[str] = None):
    print("\n")
    print("#" * 100)
    if title is not None:
        print(title)
    print(f"Running command: {command}")
    print("#" * 100)
    print("\n")
    subprocess.run(command, shell=True, check=True)


def main():
    args = tyro.cli(Args)
    print(args)
    print()

    Path(args.experiment_dir).mkdir(parents=True, exist_ok=True)

    # 1. Capture rgbd image
    rgbd_dir = f"{args.experiment_dir}/rgbd"
    run_command(
        command=f"python -m real_world.capture_rgbd_image --output-dir {rgbd_dir}",
        title="Step 1: Capture RGBD image",
    )

    # Check if the rgbd image is captured
    rgb_path = f"{rgbd_dir}/rgb/00000.png"
    depth_path = f"{rgbd_dir}/depth/00000.png"
    cam_K_path = f"{rgbd_dir}/cam_K.txt"
    assert Path(rgb_path).exists(), f"RGB image not found: {rgb_path}"
    assert Path(depth_path).exists(), f"Depth image not found: {depth_path}"
    assert Path(cam_K_path).exists(), f"Camera intrinsic matrix not found: {cam_K_path}"

    # 2. Run Gemini for 2d keypoint detection
    TASK_NAME = args.task_name
    SPLIT_NAME = args.split_name
    run_command(
        command=f"python -m build_dataset.get_real_keypoints \
                --task {TASK_NAME} --method gemini --split {SPLIT_NAME} \
                --img {rgb_path}",
        title="Step 2: Run Gemini for 2D keypoint detection",
    )

    # Check if output json is generated
    keypoint_2d_json_path = (
        f"data/real/{TASK_NAME}/{SPLIT_NAME}/responses/all_keypoint_responses.json"
    )
    assert Path(
        keypoint_2d_json_path
    ).exists(), f"Keypoint 2D json not found: {keypoint_2d_json_path}"

    # Extract keypoint 2D
    with open(keypoint_2d_json_path, "r") as f:
        keypoint_2d_data = json.load(f)

    OBJECT_1, OBJECT_2 = TASK_NAME_TO_OBJECTS[TASK_NAME]
    names = [x["name"] for x in keypoint_2d_data]
    points = [x["point"] for x in keypoint_2d_data]
    assert (
        OBJECT_1 in names
    ), f"Object 1 not found in keypoint 2D data: {OBJECT_1}. Keys: {names}"
    assert (
        OBJECT_2 in names
    ), f"Object 2 not found in keypoint 2D data: {OBJECT_2}. Keys: {names}"

    keypoint_2d_y_object_1, keypoint_2d_x_object_1 = points[names.index(OBJECT_1)]
    keypoint_2d_y_object_2, keypoint_2d_x_object_2 = points[names.index(OBJECT_2)]

    # 3. Run keypoint backprojection
    keypoint_3d_json_path_object_1 = f"{args.experiment_dir}/{OBJECT_1}_3d.json"
    keypoint_3d_json_path_object_2 = f"{args.experiment_dir}/{OBJECT_2}_3d.json"
    run_command(
        command=f"python -m real_world.select_keypoint \
                --rgb-path {rgb_path} \
                --depth-path {depth_path} \
                --cam-K-path {cam_K_path} \
                --keypoint-x-idx {keypoint_2d_x_object_1} --keypoint-y-idx {keypoint_2d_y_object_1} \
                --output-keypoint-3d-path {keypoint_3d_json_path_object_1}",
        title="Step 3: Run keypoint backprojection for object 1",
    )
    run_command(
        command=f"python -m real_world.select_keypoint \
                --rgb-path {rgb_path} \
                --depth-path {depth_path} \
                --cam-K-path {cam_K_path} \
                --keypoint-x-idx {keypoint_2d_x_object_2} --keypoint-y-idx {keypoint_2d_y_object_2} \
                --output-keypoint-3d-path {keypoint_3d_json_path_object_2}",
        title="Step 3: Run keypoint backprojection for object 2",
    )

    # Check if output json is generated
    assert Path(
        keypoint_3d_json_path_object_1
    ).exists(), f"Keypoint 3D json not found: {keypoint_3d_json_path_object_1}"
    assert Path(
        keypoint_3d_json_path_object_2
    ).exists(), f"Keypoint 3D json not found: {keypoint_3d_json_path_object_2}"

    # Extract keypoint 3D
    with open(keypoint_3d_json_path_object_1, "r") as f:
        keypoint_3d_object_1 = json.load(f)
    keypoint_3d_x_object_1 = keypoint_3d_object_1["x"]
    keypoint_3d_y_object_1 = keypoint_3d_object_1["y"]
    keypoint_3d_z_object_1 = keypoint_3d_object_1["z"]

    with open(keypoint_3d_json_path_object_2, "r") as f:
        keypoint_3d_object_2 = json.load(f)
    keypoint_3d_x_object_2 = keypoint_3d_object_2["x"]
    keypoint_3d_y_object_2 = keypoint_3d_object_2["y"]
    keypoint_3d_z_object_2 = keypoint_3d_object_2["z"]

    combined_keypoint_3d_path = (
        f"data/real/{TASK_NAME}/{SPLIT_NAME}/responses/keypoint_3d.json"
    )
    combined_keypoint_3d = {
        OBJECT_1: {
            "x": keypoint_3d_x_object_1,
            "y": keypoint_3d_y_object_1,
            "z": keypoint_3d_z_object_1,
        },
        OBJECT_2: {
            "x": keypoint_3d_x_object_2,
            "y": keypoint_3d_y_object_2,
            "z": keypoint_3d_z_object_2,
        },
    }
    with open(combined_keypoint_3d_path, "w") as f:
        json.dump(combined_keypoint_3d, f)

    # 4. Run Gemini for trajectory prediction
    run_command(
        command=f"python -m build_dataset.get_real_trajectories --task {TASK_NAME} --method gemini --split {SPLIT_NAME}",
        title="Step 4: Run Gemini for trajectory prediction",
    )

    # Check that trajectory is generatedj
    trajectories_path = f"data/real/{TASK_NAME}/{SPLIT_NAME}/trajectories.pt"
    assert Path(
        trajectories_path
    ).exists(), f"Trajectories not found: {trajectories_path}"

    # 5. Copy split data to experiment dir
    split_data_dir = f"data/real/{TASK_NAME}/{SPLIT_NAME}"
    run_command(
        command=f"cp -r {split_data_dir} {args.experiment_dir}",
        title="Step 5: Copy split data to experiment dir",
    )

    # Summary
    print("~" * 100)
    print(f"Experiment directory: {args.experiment_dir}")
    print(f"OBJECT_1: {OBJECT_1}")
    print(
        f"Keypoint 2D: {keypoint_2d_x_object_1}, {keypoint_2d_y_object_1} (x, y) [{keypoint_2d_json_path}]"
    )
    print(
        f"Keypoint 3D: {keypoint_3d_x_object_1}, {keypoint_3d_y_object_1}, {keypoint_3d_z_object_1} (x, y, z) [{keypoint_3d_json_path_object_1}]"
    )
    print(f"OBJECT_2: {OBJECT_2}")
    print(
        f"Keypoint 2D: {keypoint_2d_x_object_2}, {keypoint_2d_y_object_2} (x, y) [{keypoint_2d_json_path}]"
    )
    print(
        f"Keypoint 3D: {keypoint_3d_x_object_2}, {keypoint_3d_y_object_2}, {keypoint_3d_z_object_2} (x, y, z) [{keypoint_3d_json_path_object_2}]"
    )
    print(f"Combined keypoint 3D: {combined_keypoint_3d_path}")
    print(f"Trajectories: {trajectories_path}")

    print("TO RUN:")
    print(
        f"python -m real_world.keypoint_ros_node --keypoint_x_idx {keypoint_2d_x_object_1} --keypoint_y_idx {keypoint_2d_y_object_1}"
    )
    print(
        f"python -m real_world.keypoint_ros_node_2 --keypoint_x_idx {keypoint_2d_x_object_2} --keypoint_y_idx {keypoint_2d_y_object_2}"
    )
    print("~" * 100)


if __name__ == "__main__":
    main()
