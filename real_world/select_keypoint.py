import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import tyro
from PIL import Image

from real_world.camera_extrinsics import ZED_CAMERA_T_R_C
from real_world.misc_utils import transform_point
from real_world.table_constants import TABLE_LENGTH_Z, TABLE_Z


@dataclass
class Args:
    rgb_path: Path
    depth_path: Path
    cam_K_path: Path
    keypoint_x_idx: Optional[int] = None
    keypoint_y_idx: Optional[int] = None
    keypoint_2d_path: Optional[Path] = None
    output_keypoint_3d_path: Optional[Path] = None
    visualize: bool = False


def select_2d_keypoint(rgb: np.ndarray) -> Tuple[int, int]:
    # Get prompt as click
    plt.figure(figsize=(9, 6))
    plt.title("Click on the image to select a 2D keypoint")
    plt.imshow(rgb)
    plt.axis("off")
    points = plt.ginput(1)  # get one click
    plt.close()

    x, y = int(points[0][0]), int(points[0][1])
    return x, y


def estimate_3d_keypoint(
    keypoint_x_idx: int,
    keypoint_y_idx: int,
    points: np.ndarray,
    method: str = "area_mean",
    size: int = 1,
) -> np.ndarray:
    assert points.ndim == 3, f"Points must be 3D, got {points.ndim}"
    height, width, _ = points.shape

    if method == "simple":
        return points[keypoint_y_idx, keypoint_x_idx]
    elif method in ["area_mean", "area_median"]:
        min_y_idx = np.clip(keypoint_y_idx - size, a_min=0, a_max=None)
        min_x_idx = np.clip(keypoint_x_idx - size, a_min=0, a_max=None)
        max_y_idx = np.clip(keypoint_y_idx + size, a_min=None, a_max=height - 1)
        max_x_idx = np.clip(keypoint_x_idx + size, a_min=None, a_max=width - 1)

        area_points = points[min_y_idx:max_y_idx, min_x_idx:max_x_idx].reshape(-1, 3)
        filtered_points = area_points[area_points[:, 2] > 0]
        if filtered_points.size == 0:
            print(
                f"WARNING: No valid points found in the area with size {size} around the keypoint ({keypoint_x_idx}, {keypoint_y_idx})"
            )
            return estimate_3d_keypoint(
                keypoint_x_idx, keypoint_y_idx, points, method, size=size + 1
            )

        if method == "area_mean":
            return filtered_points.mean(axis=0)
        elif method == "area_median":
            return np.median(filtered_points, axis=0)
        else:
            raise ValueError(f"Invalid method: {method}")
    else:
        raise ValueError(f"Invalid method: {method}")


def get_points(depth: np.ndarray, cam_K: np.ndarray) -> np.ndarray:
    height, width = depth.shape
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    cx = cam_K[0, 2]
    cy = cam_K[1, 2]
    fx = cam_K[0, 0]
    fy = cam_K[1, 1]
    x = (x - cx) / fx
    y = (y - cy) / fy
    z = np.array(depth)
    return np.stack((np.multiply(x, z), np.multiply(y, z), z), axis=-1)


def visualize_with_origin_camera(
    geometries, width=1280, height=720, rescale_factor=2.0, y_up=True, cam_K=None
):
    w, h = int(width * rescale_factor), int(height * rescale_factor)

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="DepthAnything Point Cloud Viewer", width=w, height=h)
    for geometry in geometries:
        vis.add_geometry(geometry)

    view_control = vis.get_view_control()
    cam_params = view_control.convert_to_pinhole_camera_parameters()

    # === Set extrinsics ===
    extrinsic = np.eye(4)
    extrinsic[:3, 3] = np.array([0, 0, 0])  # Camera at origin
    extrinsic[:3, 0] = np.array([1, 0, 0])  # X right
    extrinsic[:3, 1] = np.array([0, 1 if y_up else -1, 0])  # Y up or down
    extrinsic[:3, 2] = np.array([0, 0, 1])  # Z forward

    cam_params.extrinsic = extrinsic

    # === Set intrinsics ===
    if cam_K is not None:
        # cam_K is assumed to correspond to the (width, height) resolution
        # supplied to this function *before* rescaling.
        sx = sy = rescale_factor  # same scale in x and y
        fx = cam_K[0, 0] * sx
        fy = cam_K[1, 1] * sy
        cx = cam_K[0, 2] * sx
        cy = cam_K[1, 2] * sy

        intrinsic = o3d.camera.PinholeCameraIntrinsic()
        intrinsic.set_intrinsics(w, h, fx, fy, cx, cy)

        cam_params.intrinsic = intrinsic

    # Apply extrinsic only — keep default intrinsics
    view_control.convert_from_pinhole_camera_parameters(
        cam_params, allow_arbitrary=True
    )

    vis.run()
    vis.destroy_window()


def select_keypoint(args: Args) -> np.ndarray:
    assert args.rgb_path.exists(), f"RGB path {args.rgb_path} does not exist"
    assert args.depth_path.exists(), f"Depth path {args.depth_path} does not exist"
    assert (
        args.cam_K_path.exists()
    ), f"Camera info path {args.cam_K_path} does not exist"

    # Get data
    rgb = np.array(Image.open(args.rgb_path))
    depth = np.array(Image.open(args.depth_path))
    cam_K = np.loadtxt(args.cam_K_path)

    assert rgb.ndim == 3, "RGB image must be 3D"
    assert depth.ndim == 2, "Depth image must be 2D"
    assert cam_K.shape == (3, 3), "Camera info must be 3x3"

    height, width, _ = rgb.shape
    assert rgb.shape == (
        height,
        width,
        3,
    ), f"RGB image shape must be (height, width, 3), got {rgb.shape}"
    assert depth.shape == (
        height,
        width,
    ), f"Depth image shape must be (height, width), got {depth.shape}"
    assert rgb.dtype == np.uint8, f"RGB image must be uint8, got {rgb.dtype}"
    assert np.issubdtype(
        depth.dtype, np.integer
    ), f"Depth image must be integer, got {depth.dtype}"

    # Depth is in mm
    depth = depth / 1000.0

    points = get_points(depth=depth, cam_K=cam_K)
    colors = np.array(rgb) / 255.0

    # Select keypoint
    if args.keypoint_x_idx is not None and args.keypoint_y_idx is not None:
        print("Using provided keypoint...")
        keypoint_x_idx, keypoint_y_idx = args.keypoint_x_idx, args.keypoint_y_idx
    elif args.keypoint_2d_path is not None:
        print("Using provided keypoint...")
        assert (
            args.keypoint_2d_path.exists()
        ), f"Keypoint 2D path {args.keypoint_2d_path} does not exist"
        assert (
            args.keypoint_2d_path.suffix == ".json"
        ), f"Keypoint 2D path must be a JSON file, got {args.keypoint_2d_path.suffix}"
        json_data = json.load(args.keypoint_2d_path.open())
        keypoint_x_idx, keypoint_y_idx = json_data["x"], json_data["y"]
    else:
        print("Selecting keypoint...")
        keypoint_x_idx, keypoint_y_idx = select_2d_keypoint(rgb)

    keypoint_point = estimate_3d_keypoint(
        keypoint_x_idx=keypoint_x_idx,
        keypoint_y_idx=keypoint_y_idx,
        points=points,
        method="area_median",
        size=1,
    )
    print(f"Selected 2d keypoint: (x={keypoint_x_idx}, y={keypoint_y_idx})")
    print(f"Selected 3d keypoint: {keypoint_point}")

    print("Converting from camera frame to robot frame...")
    keypoint_point_robot = transform_point(
        T=ZED_CAMERA_T_R_C,
        point=keypoint_point,
    )
    print(f"Selected 3d keypoint in robot frame: {keypoint_point_robot}")

    keypoint_point_robot[2] = np.clip(
        keypoint_point_robot[2], a_min=TABLE_Z + TABLE_LENGTH_Z / 2, a_max=None
    )

    if args.visualize:
        # Create the point cloud and view it
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points.reshape(-1, 3))
        pcd.colors = o3d.utility.Vector3dVector(colors.reshape(-1, 3))

        # ▼ 3‑cm pink sphere centred on the key‑point ------------------------------
        sphere_radius = 0.03  # metres; tweak if it looks too big
        sphere = o3d.geometry.TriangleMesh.create_sphere(
            radius=sphere_radius,
            resolution=12,
        )
        sphere.translate(keypoint_point)  # move sphere to the clicked point
        sphere.paint_uniform_color([1.0, 0.1, 0.8])  # hot‑pink
        sphere.compute_vertex_normals()  # so lighting looks nice
        # --------------------------------------------------------------------------

        visualize_with_origin_camera([pcd, sphere], cam_K=cam_K)

    if args.output_keypoint_3d_path is not None:
        print(f"Saving keypoint 3D to {args.output_keypoint_3d_path}")
        args.output_keypoint_3d_path.parent.mkdir(parents=True, exist_ok=True)
        assert (
            args.output_keypoint_3d_path.suffix == ".json"
        ), f"Output keypoint 3D path must be a JSON file, got {args.output_keypoint_3d_path.suffix}"
        with args.output_keypoint_3d_path.open("w") as f:
            json.dump(
                {
                    "x": keypoint_point_robot[0],
                    "y": keypoint_point_robot[1],
                    "z": keypoint_point_robot[2],
                },
                f,
            )

    return keypoint_point


def main():
    args = tyro.cli(Args)
    print(args)

    select_keypoint(args)


if __name__ == "__main__":
    main()
