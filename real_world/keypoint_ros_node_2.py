#!/usr/bin/env python

import time
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
import open3d as o3d
import rospy
import tyro
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Point, Pose
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import CameraInfo
from sensor_msgs.msg import Image as ROSImage

from real_world.camera_extrinsics import ZED_CAMERA_T_R_C
from real_world.select_keypoint import (
    estimate_3d_keypoint,
    get_points,
    select_2d_keypoint,
    visualize_with_origin_camera,
)
from real_world.table_constants import TABLE_LENGTH_Z, TABLE_Z


@dataclass
class Args:
    keypoint_x_idx: Optional[int] = None
    keypoint_y_idx: Optional[int] = None
    visualize: bool = False


def accurate_sleep(seconds: float):
    start_time = time.perf_counter()
    while time.perf_counter() - start_time < seconds:
        time.sleep(0.0001)


def transform_point(T: np.ndarray, p: np.ndarray) -> np.ndarray:
    assert T.shape == (4, 4)
    assert p.shape == (3,)
    return (T @ np.array([*p, 1]))[:3]


def draw_keypoint(
    img: np.ndarray,
    keypoint_3d: np.ndarray,
    cam_K: np.ndarray,
    sphere_radius_m: float = 0.03,  # physical radius used in Open3D scene
    radius_px: int | None = None,  # fixed pixel radius; overrides auto‑scaling
    color: tuple[int, int, int] = (0, 0, 255),  # BGR (red)
    thickness: int = -1,  # ‑1 → filled circle
) -> np.ndarray:
    """
    Project a 3‑D keypoint in camera coordinates onto the RGB image and
    draw a 2‑D circle at the projected pixel.

    Parameters
    ----------
    img : np.ndarray (H×W×3, BGR)
        Image to draw on (modified in‐place and also returned).
    keypoint_3d : np.ndarray, shape (3,)
        (X, Y, Z) of the point in the **camera** coordinate frame [metres].
    cam_K : np.ndarray, shape (3, 3)
        Camera intrinsic matrix [[fx, 0, cx], [0, fy, cy], [0, 0, 1]].
    sphere_radius_m : float, optional
        Physical radius that should appear as the drawn circle; affects
        automatic pixel‑radius scaling.
    radius_px : int | None, optional
        Fixed pixel radius to use instead of automatic scaling.
    color : tuple[int, int, int], optional
        Circle colour in BGR.
    thickness : int, optional
        OpenCV thickness parameter (‑1 → filled).

    Returns
    -------
    img_out : np.ndarray
        The image with the keypoint drawn.
    """
    assert img.ndim == 3 and img.shape[2] == 3, "img must be H×W×3 BGR"
    assert keypoint_3d.shape == (3,), "keypoint_3d must be length‑3"
    assert cam_K.shape == (3, 3), "cam_K must be 3×3"

    X, Y, Z = keypoint_3d.astype(float)
    if Z <= 0:  # behind camera → nothing to draw
        return img

    fx, fy = cam_K[0, 0], cam_K[1, 1]
    cx, cy = cam_K[0, 2], cam_K[1, 2]

    u = fx * X / Z + cx
    v = fy * Y / Z + cy
    u_int, v_int = int(round(u)), int(round(v))

    h, w = img.shape[:2]
    if 0 <= u_int < w and 0 <= v_int < h:
        if radius_px is None:
            # pixel radius that a sphere of `sphere_radius_m` subtends at depth Z
            radius_px = max(2, int(round(fx * sphere_radius_m / Z)))
        cv2.circle(img, (u_int, v_int), radius_px, color, thickness)

    return img


class KeypointROSNode:
    def __init__(self, args: Args):
        self.args = args

        # Variables for storing the latest images
        self.latest_rgb = None
        self.latest_depth = None
        self.latest_cam_K = None
        self.latest_pose = None

        rospy.init_node("keypoint_node_2")
        self.bridge = CvBridge()

        # Check camera parameter
        camera = rospy.get_param("/camera", None)
        if camera is None:
            DEFAULT_CAMERA = "zed"
            rospy.logwarn(
                f"No /camera parameter found, using default camera {DEFAULT_CAMERA}"
            )
            camera = DEFAULT_CAMERA
        print(f"Using camera: {camera}")
        if camera == "zed":
            self.rgb_sub_topic = "/zed/zed_node/rgb/image_rect_color"
            self.depth_sub_topic = "/zed/zed_node/depth/depth_registered"
            self.camera_info_sub_topic = "/zed/zed_node/rgb/camera_info"
        elif camera == "realsense":
            self.rgb_sub_topic = "/camera/color/image_raw"
            self.depth_sub_topic = "/camera/aligned_depth_to_color/image_raw"
            self.camera_info_sub_topic = "/camera/color/camera_info"
        else:
            raise ValueError(f"Unknown camera: {camera}")

        # Subscribers for RGB and depth images
        self.rgb_sub = rospy.Subscriber(
            self.rgb_sub_topic,
            ROSImage,
            self.rgb_callback,
            queue_size=1,
        )
        self.depth_sub = rospy.Subscriber(
            self.depth_sub_topic,
            ROSImage,
            self.depth_callback,
            queue_size=1,
        )
        self.cam_K_sub = rospy.Subscriber(
            self.camera_info_sub_topic,
            CameraInfo,
            self.cam_K_callback,
            queue_size=1,
        )
        self.pose_sub = rospy.Subscriber(
            "/object_pose_2", Pose, self.pose_callback, queue_size=1
        )

        self.keypoint_3d_pub = rospy.Publisher("/keypoint_3d_2", Point, queue_size=1)

    def rgb_callback(self, data):
        try:
            self.latest_rgb = self.bridge.imgmsg_to_cv2(data, "rgb8")
            print(f"Received RGB image: {self.latest_rgb.shape}")
        except CvBridgeError as e:
            rospy.logerr(f"Could not convert RGB image: {e}")

    def depth_callback(self, data):
        try:
            self.latest_depth = self.bridge.imgmsg_to_cv2(data, "64FC1")
            print(f"Received depth image: {self.latest_depth.shape}")
        except CvBridgeError as e:
            rospy.logerr(f"Could not convert depth image: {e}")

    def cam_K_callback(self, data: CameraInfo):
        self.latest_cam_K = np.array(data.K).reshape(3, 3)

    def pose_callback(self, data: Pose):
        xyz = np.array([data.position.x, data.position.y, data.position.z])
        quat_xyzw = np.array(
            [
                data.orientation.x,
                data.orientation.y,
                data.orientation.z,
                data.orientation.w,
            ]
        )
        T = np.eye(4)
        T[:3, :3] = R.from_quat(quat_xyzw).as_matrix()
        T[:3, 3] = xyz
        self.latest_pose = T

    def run(self):
        ##############################
        # Wait for the first images
        ##############################
        while not rospy.is_shutdown() and (
            self.latest_rgb is None
            or self.latest_depth is None
            or self.latest_cam_K is None
            or self.latest_pose is None
        ):
            print(
                f"Missing one of the required images (RGB, depth, cam_K, pose). Waiting... (rgb missing: {self.latest_rgb is None}, depth missing: {self.latest_depth is None}, cam_K missing: {self.latest_cam_K is None}, pose missing: {self.latest_pose is None})"
            )
            rospy.sleep(0.1)

        assert self.latest_rgb is not None
        assert self.latest_depth is not None
        assert self.latest_cam_K is not None
        assert self.latest_pose is not None

        # Select keypoint
        first_rgb = self.process_rgb(self.latest_rgb)
        first_depth = self.process_depth(self.latest_depth)
        first_cam_K = self.latest_cam_K.copy()
        first_pose = self.latest_pose.copy()
        points = get_points(depth=first_depth, cam_K=first_cam_K)
        colors = np.array(first_rgb) / 255.0

        # Select keypoint
        if self.args.keypoint_x_idx is None or self.args.keypoint_y_idx is None:
            print("Selecting keypoint...")
            keypoint_x_idx, keypoint_y_idx = select_2d_keypoint(first_rgb)
        else:
            print("Using provided keypoint...")
            keypoint_x_idx, keypoint_y_idx = (
                self.args.keypoint_x_idx,
                self.args.keypoint_y_idx,
            )

        keypoint_point = estimate_3d_keypoint(
            keypoint_x_idx=keypoint_x_idx,
            keypoint_y_idx=keypoint_y_idx,
            points=points,
            method="area_median",
            size=1,
        )
        print(f"Selected 2d keypoint: (x={keypoint_x_idx}, y={keypoint_y_idx})")
        print(f"Selected 3d keypoint: {keypoint_point}")

        keypoint_init_C = keypoint_point
        T_C_Oinit = first_pose
        T_Oinit_C = np.linalg.inv(T_C_Oinit)
        keypoint_init_Oinit = transform_point(T_Oinit_C, keypoint_init_C)

        if self.args.visualize:
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

            visualize_with_origin_camera([pcd, sphere], cam_K=first_cam_K)

        first_pose = self.latest_pose.copy()

        # Track keypoint
        while not rospy.is_shutdown():
            start_time = time.perf_counter()

            rgb = self.process_rgb(self.latest_rgb)
            cam_K = self.latest_cam_K.copy()
            pose = self.latest_pose.copy()
            assert pose.shape == (4, 4), f"pose.shape = {pose.shape}"
            T_C_O = pose

            MODE = "default"
            if MODE == "default":
                keypoint_C = transform_point(T=T_C_O, p=keypoint_init_Oinit)
                self.publish_point(keypoint_C)
            elif MODE == "override_with_first_pose":
                # HACK: Overwrite with first pose (not true tracking, just using the first)
                first_T_C_O = first_pose
                T_C_O = first_T_C_O
                keypoint_C = transform_point(T=T_C_O, p=keypoint_init_Oinit)
                self.publish_point(keypoint_C)
            elif MODE == "override_with_first_keypoint":
                # HACK: Override with just the first keypoint computed, no object tracking
                # Clip to be above table
                keypoint_C = keypoint_point
                keypoint_R = transform_point(T=ZED_CAMERA_T_R_C, p=keypoint_C)
                keypoint_R[2] = np.clip(
                    keypoint_R[2], a_min=TABLE_Z + TABLE_LENGTH_Z / 2, a_max=None
                )
                keypoint_C = transform_point(
                    T=np.linalg.inv(ZED_CAMERA_T_R_C), p=keypoint_R
                )
                self.publish_point(keypoint_C)
            else:
                raise ValueError(f"Unknown mode: {MODE}")

            # Must be BGR for cv2
            vis_img = cv2.cvtColor(rgb.copy(), cv2.COLOR_RGB2BGR)
            vis_img = draw_keypoint(img=vis_img, keypoint_3d=keypoint_C, cam_K=cam_K)
            cv2.imshow("Pose Visualization", vis_img)
            cv2.waitKey(1)

            done_time = time.perf_counter()
            self.rate_hz = 100
            extra_time = 1 / self.rate_hz - (done_time - start_time)
            print(f"Extra time: {extra_time}")
            if extra_time > 0:
                # rospy.sleep(extra_time)
                # time.sleep(extra_time)
                accurate_sleep(extra_time)
            else:
                print(f"Extra time: {extra_time}")
            after_sleep_time = time.perf_counter()
            print(
                f"Max rate: {np.round(1.0 / (done_time - start_time))} Hz ({np.round((done_time - start_time) * 1000)} ms), Actual rate with sleep: {np.round(1.0 / (after_sleep_time - start_time))} Hz"
            )

    def process_rgb(self, rgb):
        # rospy.logdebug(f"rgb: {rgb.shape}, {rgb.dtype}, {np.max(rgb)}, {np.min(rgb)}")
        return rgb

    def process_depth(self, depth):
        # Turn nan values into 0
        depth[np.isnan(depth)] = 0
        depth[np.isinf(depth)] = 0

        # depth is either in meters or millimeters
        # Need to convert to meters
        # If the max value is greater than 100, then it's likely in mm
        in_mm = depth.max() > 100
        if in_mm:
            print(f"Converting depth from mm to m since max = {depth.max()}")
            depth = depth / 1000
        else:
            print(f"Depth is in meters since max = {depth.max()}")

        # Clamp
        depth[depth < 0.1] = 0
        depth[depth > 4] = 0

        return depth

    def publish_point(self, point: np.ndarray):
        assert point.shape == (3,), f"point.shape = {point.shape}"
        msg = Point()
        msg.x, msg.y, msg.z = point
        self.keypoint_3d_pub.publish(msg)
        # rospy.logdebug("Point published to /keypoint_3d")


if __name__ == "__main__":
    args = tyro.cli(Args)
    node = KeypointROSNode(args)
    node.run()
