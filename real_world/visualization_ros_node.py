#!/usr/bin/env python

import functools
import math
import struct
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Tuple

import numpy as np
import rospy
import sensor_msgs.point_cloud2 as pc2
import trimesh
import tyro
from geometry_msgs.msg import Point, Pose
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import JointState, PointCloud2
from std_msgs.msg import Float64MultiArray
from termcolor import colored

from real_world.camera_extrinsics import ZED_CAMERA_T_R_C, ZED_CAMERA_T_R_Cptcloud
from real_world.print_utils import get_ros_loop_rate_str
from real_world.pybullet_utils import add_sphere, move_sphere
from real_world.table_constants import (
    TABLE_QW,
    TABLE_QX,
    TABLE_QY,
    TABLE_QZ,
    TABLE_X,
    TABLE_Y,
    TABLE_Z,
)

NUM_ARM_JOINTS = 7
NUM_HAND_JOINTS = 16
BLUE_TRANSLUCENT_RGBA = [0, 0, 1, 0.5]
RED_TRANSLUCENT_RGBA = [1, 0, 0, 0.2]
GREEN_TRANSLUCENT_RGBA = [0, 1, 0, 0.5]
BLACK_TRANSLUCENT_RGBA = [0, 0, 0, 0.5]

BLUE_RGB = [0, 0, 1]
RED_RGB = [1, 0, 0]
GREEN_RGB = [0, 1, 0]
YELLOW_RGB = [1, 1, 0]
CYAN_RGB = [0, 1, 1]
MAGENTA_RGB = [1, 0, 1]
WHITE_RGB = [1, 1, 1]
BLACK_RGB = [0, 0, 0]

BLUE_RGBA = [*BLUE_RGB, 1]
RED_RGBA = [*RED_RGB, 1]
GREEN_RGBA = [*GREEN_RGB, 1]
YELLOW_RGBA = [*YELLOW_RGB, 1]
CYAN_RGBA = [*CYAN_RGB, 1]
MAGENTA_RGBA = [*MAGENTA_RGB, 1]
WHITE_RGBA = [*WHITE_RGB, 1]
BLACK_RGBA = [*BLACK_RGB, 1]


def get_asset_root() -> Path:
    asset_root = Path(__file__).parent.parent / "assets"
    assert asset_root.exists(), f"asset_root not found: {asset_root}"
    return asset_root


@dataclass
class Args:
    load_point_cloud: bool = False
    rate_hz: float = 10
    load_scene_mesh: bool = False


def add_cuboid(halfExtents, position, orientation, rgbaColor=RED_TRANSLUCENT_RGBA):
    # HACK: Hide pybullet import in functions that use it to avoid messy tyro autocomplete
    import pybullet as p

    # Create a visual shape for the cuboid
    visualShapeId = p.createVisualShape(
        shapeType=p.GEOM_BOX, halfExtents=halfExtents, rgbaColor=rgbaColor
    )  # Red color

    # Create a collision shape for the cuboid
    collisionShapeId = p.createCollisionShape(
        shapeType=p.GEOM_BOX, halfExtents=halfExtents
    )

    # Create the cuboid as a rigid body
    cuboidId = p.createMultiBody(
        baseMass=1,  # Mass of the cuboid
        baseCollisionShapeIndex=collisionShapeId,
        baseVisualShapeIndex=visualShapeId,
        basePosition=position,
        baseOrientation=orientation,
    )
    return cuboidId


def create_transform(
    pos: np.ndarray,
    rot: np.ndarray,
) -> np.ndarray:
    assert pos.shape == (3,)
    assert rot.shape == (3, 3)
    T = np.eye(4)
    T[:3, :3] = rot
    T[:3, 3] = pos
    return T


def add_line(start, end, rgbColor=WHITE_RGB, lineWidth=3):
    # HACK: Hide pybullet import in functions that use it to avoid messy tyro autocomplete
    import pybullet as p

    return p.addUserDebugLine(start, end, lineColorRGB=rgbColor, lineWidth=lineWidth)


def move_line(lineId, start, end, rgbColor=WHITE_RGB, lineWidth=3):
    # HACK: Hide pybullet import in functions that use it to avoid messy tyro autocomplete
    import pybullet as p

    p.addUserDebugLine(
        start,
        end,
        replaceItemUniqueId=lineId,
        lineColorRGB=rgbColor,
        lineWidth=lineWidth,
    )


def visualize_transform(
    xyz: np.ndarray,
    rotation_matrix: np.ndarray,
    length: float = 0.2,
    lines: Optional[list] = None,
) -> list:
    T = create_transform(pos=xyz, rot=rotation_matrix)
    assert T.shape == (4, 4), T.shape

    origin = np.array([0, 0, 0])
    x_pos = np.array([length, 0, 0])
    y_pos = np.array([0, length, 0])
    z_pos = np.array([0, 0, length])

    tranformed_origin = T[:3, :3] @ origin + T[:3, 3]
    tranformed_x_pos = T[:3, :3] @ x_pos + T[:3, 3]
    tranformed_y_pos = T[:3, :3] @ y_pos + T[:3, 3]
    tranformed_z_pos = T[:3, :3] @ z_pos + T[:3, 3]

    if lines is None:
        lines = []

        lines.append(add_line(tranformed_origin, tranformed_x_pos, rgbColor=RED_RGB))
        lines.append(add_line(tranformed_origin, tranformed_y_pos, rgbColor=GREEN_RGB))
        lines.append(add_line(tranformed_origin, tranformed_z_pos, rgbColor=BLUE_RGB))
        return lines
    else:
        move_line(
            lines[0],
            tranformed_origin,
            tranformed_x_pos,
            rgbColor=RED_RGB,
        )
        move_line(
            lines[1],
            tranformed_origin,
            tranformed_y_pos,
            rgbColor=GREEN_RGB,
        )
        move_line(
            lines[2],
            tranformed_origin,
            tranformed_z_pos,
            rgbColor=BLUE_RGB,
        )
        return lines


def rgb_to_float(color: float) -> Tuple[float, float, float]:
    """Convert packed RGB float to separate R, G, B components (https://wiki.ros.org/pcl/Overview)."""
    s = struct.pack(">f", color)
    i = struct.unpack(">I", s)[0]
    r = (i >> 16) & 0x0000FF
    g = (i >> 8) & 0x0000FF
    b = (i) & 0x0000FF
    return r / 255.0, g / 255.0, b / 255.0


def transform_points(T: np.ndarray, points: np.ndarray) -> np.ndarray:
    assert T.shape == (4, 4), T.shape
    n_pts = points.shape[0]
    assert points.shape == (n_pts, 3), points.shape

    return (T[:3, :3] @ points.T + T[:3, 3][:, None]).T


def draw_colored_point_cloud(
    point_cloud_and_colors: np.ndarray,
    T_R_Cptcloud: np.ndarray,
    point_size: int = 5,
):
    # HACK: Hide pybullet import in functions that use it to avoid messy tyro autocomplete
    import pybullet as p

    n_pts = point_cloud_and_colors.shape[0]
    assert point_cloud_and_colors.shape == (
        n_pts,
        4,
    ), f"point_cloud_and_colors.shape: {point_cloud_and_colors.shape}"
    point_cloud = point_cloud_and_colors[:, :3]

    # point_cloud is in Cptcloud frame
    point_cloud_R = transform_points(T=T_R_Cptcloud, points=point_cloud)

    # Extract colors from the point cloud
    point_cloud_raw_colors = point_cloud_and_colors[:, 3]
    point_cloud_colors = np.array(
        [rgb_to_float(color) for color in point_cloud_raw_colors]
    )

    FILTER_POINT_CLOUD = False
    if FILTER_POINT_CLOUD:
        # idxs = (point_cloud_R[:, 0] > 0) & (point_cloud_R[:, 1] < 0)
        idxs = (point_cloud_R[:, 0] > 0) & (point_cloud_R[:, 1] < -0.2)
        point_cloud_R = point_cloud_R[idxs]
        point_cloud_colors = point_cloud_colors[idxs]

    num_points = len(point_cloud_colors)

    # Downsample if too many points
    MAX_POINTS = 100_000  # TODO: Tune
    if num_points > MAX_POINTS:
        downsample_factor = math.ceil(num_points / MAX_POINTS)
        print(
            colored(
                f"num_points: {num_points} is greater than MAX_POINTS: {MAX_POINTS}, downsample_factor: {downsample_factor}",
                "yellow",
            )
        )
        point_cloud_R = point_cloud_R[::downsample_factor]
        point_cloud_colors = point_cloud_colors[::downsample_factor]

    # Use debug points instead of spheres for faster rendering
    if not hasattr(draw_colored_point_cloud, "points"):
        print(f"Creating new point cloud with {len(point_cloud_R)} points")
        draw_colored_point_cloud.points = p.addUserDebugPoints(
            point_cloud_R,
            point_cloud_colors,
            point_size,
        )
    else:
        p.addUserDebugPoints(
            point_cloud_R,
            point_cloud_colors,
            point_size,
            replaceItemUniqueId=draw_colored_point_cloud.points,
        )


def create_urdf(obj_path: Path) -> Path:
    assert obj_path.suffix == ".obj"
    filename = obj_path.name
    parent_folder = obj_path.parent
    urdf_path = parent_folder / f"{obj_path.stem}.urdf"
    urdf_text = f"""<?xml version="1.0" ?>
<robot name="model.urdf">
  <link name="baseLink">
    <contact>
      <lateral_friction value="0.8"/>
      <rolling_friction value="0.001"/>
      <contact_cfm value="0.0"/>
      <contact_erp value="1.0"/>
    </contact>
    <inertial>
    <origin rpy="0 0 0" xyz="0.01 0.0 0.01"/>
       <mass value=".066"/>
       <inertia ixx="1e-3" ixy="0" ixz="0" iyy="1e-3" iyz="0" izz="1e-3"/>
    </inertial>
    <visual>
      <geometry>
        <mesh filename="{filename}" scale="1 1 1"/>
      </geometry>
      <material name="white">
        <color rgba="1. 1. 1. 1."/>
      </material>
    </visual>
    <collision>
      <geometry>
        <mesh filename="{filename}" scale="1 1 1"/>
      </geometry>
    </collision>
  </link>
</robot>"""
    with urdf_path.open("w") as f:
        f.write(urdf_text)
    return urdf_path


class VisualizationNode:
    def __init__(self, args: Args):
        # ROS setup
        rospy.init_node("visualization_ros_node")
        self.args = args

        # ROS msgs
        self.iiwa_joint_cmd = None
        self.allegro_joint_cmd = None
        self.iiwa_joint_state = None
        self.allegro_joint_state = None
        self.palm_target = None
        self.object_pose = None
        self.object_pose_2 = None
        self.goal_object_pose = None
        self.goal_object_pose_2 = None
        self.keypoint_3d = None
        self.keypoint_3d_2 = None
        self.target_keypoint_3d = None
        self.target_keypoint_3d_2 = None
        self.point_cloud_and_colors = None

        # Subscribers
        self.iiwa_sub = rospy.Subscriber(
            "/iiwa/joint_states", JointState, self.iiwa_joint_state_callback
        )
        self.allegro_sub = rospy.Subscriber(
            "/allegroHand_0/joint_states", JointState, self.allegro_joint_state_callback
        )
        self.iiwa_cmd_sub = rospy.Subscriber(
            "/iiwa/joint_cmd", JointState, self.iiwa_joint_cmd_callback
        )
        self.allegro_cmd_sub = rospy.Subscriber(
            "/allegroHand_0/joint_cmd", JointState, self.allegro_joint_cmd_callback
        )
        self.palm_target_sub = rospy.Subscriber(
            "/palm_target", Float64MultiArray, self.palm_target_callback
        )
        self.object_pose_sub = rospy.Subscriber(
            "/object_pose", Pose, self.object_pose_callback
        )
        self.object_pose_2_sub = rospy.Subscriber(
            "/object_pose_2", Pose, self.object_pose_2_callback
        )
        self.goal_object_pose_sub = rospy.Subscriber(
            "/goal_object_pose", Pose, self.goal_object_pose_callback
        )
        self.goal_object_pose_2_sub = rospy.Subscriber(
            "/goal_object_pose_2", Pose, self.goal_object_pose_2_callback
        )
        self.keypoint_3d_sub = rospy.Subscriber(
            "/keypoint_3d", Point, self.keypoint_3d_callback, queue_size=1
        )
        self.keypoint_3d_2_sub = rospy.Subscriber(
            "/keypoint_3d_2", Point, self.keypoint_3d_2_callback, queue_size=1
        )
        self.target_keypoint_3d_sub = rospy.Subscriber(
            "/target_keypoint_3d", Point, self.target_keypoint_3d_callback, queue_size=1
        )
        self.target_keypoint_3d_2_sub = rospy.Subscriber(
            "/target_keypoint_3d_2",
            Point,
            self.target_keypoint_3d_2_callback,
            queue_size=1,
        )

        if self.camera == "zed":
            point_cloud_topic = "/zed/zed_node/point_cloud/cloud_registered"
        elif self.camera == "realsense":
            point_cloud_topic = "/camera/depth/color/points"
        else:
            raise ValueError(f"Invalid camera: {self.camera}")
        self.point_cloud_sub = rospy.Subscriber(
            point_cloud_topic,
            PointCloud2,
            self.point_cloud_callback,
        )

        # Initialize PyBullet
        print("~" * 80)
        print("Initializing PyBullet")
        self.initialize_pybullet()
        print("PyBullet initialized!")
        print("~" * 80)

        # Set control rate to 60Hz
        self.rate_hz = args.rate_hz
        self.rate = rospy.Rate(self.rate_hz)

    def initialize_pybullet(self):
        # HACK: Hide pybullet import in functions that use it to avoid messy tyro autocomplete
        import pybullet as p

        """Initialize PyBullet, set up camera, and load the robot URDF."""
        p.connect(p.GUI)

        # Create a real robot (simulating real robot) and a command robot (visualizing commands)
        # Load robot URDF with a fixed base
        robot_urdf_path = get_asset_root() / "kuka_allegro/kuka_allegro.urdf"
        assert robot_urdf_path.exists(), f"robot_urdf_path not found: {robot_urdf_path}"

        # WARNING: After extensive testing, we find that the Allegro hand robot in the real world
        #          is about 1.2cm lower than the simulated Allegro hand for most joint angles.
        #          This difference is severe enough to cause low-profile manipulation tasks to fail
        #          Thus, we manually offset the robot base by 1.2cm in the z-direction.
        # MANUAL_OFFSET_ROBOT_Z = -0.007
        MANUAL_OFFSET_ROBOT_Z = -0.012
        self.robot_id = p.loadURDF(
            str(robot_urdf_path),
            useFixedBase=True,
            basePosition=[0, 0, MANUAL_OFFSET_ROBOT_Z],
            baseOrientation=[0, 0, 0, 1],
        )
        self.robot_cmd_id = p.loadURDF(
            str(robot_urdf_path),
            useFixedBase=True,
            basePosition=[0, 0, MANUAL_OFFSET_ROBOT_Z],
            baseOrientation=[0, 0, 0, 1],
        )

        # Load the scene mesh
        LOAD_SCENE_MESH = self.args.load_scene_mesh
        if LOAD_SCENE_MESH:
            scene_urdf_path = (
                get_asset_root() / "scene_mesh_cropped/scene_mesh_cropped.urdf"
            )
            assert (
                scene_urdf_path.exists()
            ), f"scene_urdf_path not found: {scene_urdf_path}"
            T = np.linalg.inv(
                np.array(
                    [
                        [
                            -9.87544368e-01,
                            -1.57333070e-01,
                            -1.55753395e-03,
                            7.91730212e-02,
                        ],
                        [
                            -9.08047145e-04,
                            -4.19989728e-03,
                            9.99990768e-01,
                            -3.65614006e-01,
                        ],
                        [
                            -1.57338159e-01,
                            9.87536666e-01,
                            4.00471907e-03,
                            5.94016453e-01,
                        ],
                        [0.00000000e00, 0.00000000e00, 0.00000000e00, 1.00000000e00],
                    ]
                )
            )
            x, y, z = T[:3, 3]
            qx, qy, qz, qw = R.from_matrix(T[:3, :3]).as_quat()

            _scene_id = p.loadURDF(
                str(scene_urdf_path),
                useFixedBase=True,
                basePosition=[x, y, z],
                baseOrientation=[qx, qy, qz, qw],
            )

        LOAD_TABLE = True
        if LOAD_TABLE:
            table_urdf_path = get_asset_root() / "table/table.urdf"
            assert (
                table_urdf_path.exists()
            ), f"table_urdf_path not found: {table_urdf_path}"
            _table_id = p.loadURDF(
                str(table_urdf_path),
                useFixedBase=True,
                # basePosition=[TABLE_X, TABLE_Y, TABLE_Z],
                basePosition=[TABLE_X, TABLE_Y, TABLE_Z + 0],
                baseOrientation=[TABLE_QX, TABLE_QY, TABLE_QZ, TABLE_QW],
            )

            TRANSPARENT_TABLE = False
            if TRANSPARENT_TABLE:
                # Make the table black transparent
                # Change the color of each link (including the base)
                for link_index in range(
                    -1, p.getNumJoints(_table_id)
                ):  # -1 is for the base
                    p.changeVisualShape(
                        _table_id, link_index, rgbaColor=BLACK_TRANSLUCENT_RGBA
                    )

        # Load the object mesh
        FAR_AWAY_OBJECT_POSITION = np.ones(3)
        object_mesh_path = rospy.get_param("/mesh_file", None)
        if object_mesh_path is None:
            DEFAULT_MESH_PATH = get_asset_root() / "kiri/snackbox/snackbox.obj"
            object_mesh_path = DEFAULT_MESH_PATH
            print(colored(f"Using default object mesh: {object_mesh_path}", "yellow"))
        assert isinstance(
            object_mesh_path, str
        ), f"object_mesh_path: {object_mesh_path}"
        print("~" * 80)
        print(f"object_mesh_path: {object_mesh_path}")
        print("~" * 80 + "\n")

        goal_object_mesh_path = object_mesh_path

        # Object often has too many faces to have 2 objects and may have issues loading
        MAX_N_FACES = 10_000
        mesh = trimesh.load(object_mesh_path)
        n_faces = mesh.faces.shape[0]
        if n_faces > MAX_N_FACES:
            print(
                colored(
                    f"object_mesh_path: {object_mesh_path} has more than {MAX_N_FACES} faces, has {n_faces} faces",
                    "yellow",
                )
            )

            USE_OPEN3D = False
            if USE_OPEN3D:
                import open3d as o3d

                orig_mesh = o3d.io.read_triangle_mesh(object_mesh_path)
                simplified_mesh = orig_mesh.simplify_quadric_decimation(
                    target_number_of_triangles=MAX_N_FACES,
                )
                object_mesh_Path = Path(object_mesh_path)
                simplified_mesh_path = str(
                    object_mesh_Path.parent
                    / f"{object_mesh_Path.stem}_simplified{object_mesh_Path.suffix}"
                )
                o3d.io.write_triangle_mesh(simplified_mesh_path, simplified_mesh)
                print(f"Saved simplified mesh to: {simplified_mesh_path}")

                object_mesh_path = simplified_mesh_path
                goal_object_mesh_path = simplified_mesh_path
            else:
                import fast_simplification

                points_out, faces_out = fast_simplification.simplify(
                    mesh.vertices, mesh.faces, target_reduction=0.9
                )
                new_mesh = trimesh.Trimesh(vertices=points_out, faces=faces_out)
                object_mesh_Path = Path(object_mesh_path)
                simplified_mesh_path = str(
                    object_mesh_Path.parent
                    / f"{object_mesh_Path.stem}_simplified{object_mesh_Path.suffix}"
                )
                new_mesh.export(simplified_mesh_path)
                print(f"Saved simplified mesh to: {simplified_mesh_path}")

                # object_mesh_path = simplified_mesh_path
                goal_object_mesh_path = simplified_mesh_path

        object_urdf_path = create_urdf(Path(object_mesh_path))
        if goal_object_mesh_path != object_mesh_path:
            goal_object_urdf_path = create_urdf(Path(goal_object_mesh_path))
        else:
            goal_object_urdf_path = object_urdf_path

        self.object_id = p.loadURDF(str(object_urdf_path), useFixedBase=True)
        p.resetBasePositionAndOrientation(
            self.object_id, FAR_AWAY_OBJECT_POSITION, [0, 0, 0, 1]
        )

        self.goal_object_id = p.loadURDF(str(goal_object_urdf_path), useFixedBase=True)
        p.resetBasePositionAndOrientation(
            self.goal_object_id,
            FAR_AWAY_OBJECT_POSITION + np.array([0.2, 0.2, 0.2]),
            [0, 0, 0, 1],
        )
        p.changeVisualShape(self.goal_object_id, -1, rgbaColor=GREEN_TRANSLUCENT_RGBA)

        # Load the object mesh 2
        FAR_AWAY_OBJECT_POSITION = np.ones(3)
        object_mesh_2_path = rospy.get_param("/mesh_file_2", None)
        if object_mesh_2_path is None:
            DEFAULT_MESH_PATH = get_asset_root() / "kiri/snackbox/snackbox.obj"
            object_mesh_2_path = DEFAULT_MESH_PATH
            print(
                colored(f"Using default object mesh 2: {object_mesh_2_path}", "yellow")
            )
        assert isinstance(
            object_mesh_2_path, str
        ), f"object_mesh_2_path: {object_mesh_2_path}"
        print("~" * 80)
        print(f"object_mesh_2_path: {object_mesh_2_path}")
        print("~" * 80 + "\n")

        goal_object_mesh_2_path = object_mesh_2_path

        # Object often has too many faces to have 2 objects and may have issues loading
        MAX_N_FACES = 10_000
        mesh_2 = trimesh.load(object_mesh_2_path)
        n_faces_2 = mesh_2.faces.shape[0]
        if n_faces_2 > MAX_N_FACES:
            print(
                colored(
                    f"object_mesh_2_path: {object_mesh_2_path} has more than {MAX_N_FACES} faces, has {n_faces} faces",
                    "yellow",
                )
            )

            USE_OPEN3D = False
            if USE_OPEN3D:
                import open3d as o3d

                orig_mesh_2 = o3d.io.read_triangle_mesh(object_mesh_2_path)
                simplified_mesh_2 = orig_mesh_2.simplify_quadric_decimation(
                    target_number_of_triangles=MAX_N_FACES,
                )
                object_mesh_2_Path = Path(object_mesh_2_path)
                simplified_mesh_2_path = str(
                    object_mesh_2_Path.parent
                    / f"{object_mesh_2_Path.stem}_simplified{object_mesh_2_Path.suffix}"
                )
                o3d.io.write_triangle_mesh(simplified_mesh_2_path, simplified_mesh_2)
                print(f"Saved simplified mesh to: {simplified_mesh_2_path}")

                object_mesh_2_path = simplified_mesh_2_path
                goal_object_mesh_2_path = simplified_mesh_2_path
            else:
                import fast_simplification

                points_out_2, faces_out_2 = fast_simplification.simplify(
                    mesh_2.vertices, mesh_2.faces, target_reduction=0.9
                )
                new_mesh_2 = trimesh.Trimesh(vertices=points_out_2, faces=faces_out_2)
                object_mesh_2_Path = Path(object_mesh_2_path)
                simplified_mesh_2_path = str(
                    object_mesh_2_Path.parent
                    / f"{object_mesh_2_Path.stem}_simplified{object_mesh_2_Path.suffix}"
                )
                new_mesh_2.export(simplified_mesh_2_path)
                print(f"Saved simplified mesh to: {simplified_mesh_2_path}")

                # object_mesh_path = simplified_mesh_path
                goal_object_mesh_2_path = simplified_mesh_2_path

        object_urdf_path_2 = create_urdf(Path(object_mesh_2_path))
        if goal_object_mesh_2_path != object_mesh_2_path:
            goal_object_urdf_path_2 = create_urdf(Path(goal_object_mesh_2_path))
        else:
            goal_object_urdf_path_2 = object_urdf_path_2

        self.object_id_2 = p.loadURDF(str(object_urdf_path_2), useFixedBase=True)
        p.resetBasePositionAndOrientation(
            self.object_id_2, FAR_AWAY_OBJECT_POSITION, [0, 0, 0, 1]
        )

        self.goal_object_id_2 = p.loadURDF(
            str(goal_object_urdf_path_2), useFixedBase=True
        )
        p.resetBasePositionAndOrientation(
            self.goal_object_id_2, FAR_AWAY_OBJECT_POSITION, [0, 0, 0, 1]
        )
        p.changeVisualShape(self.goal_object_id_2, -1, rgbaColor=GREEN_TRANSLUCENT_RGBA)

        TRANSLUCENT_ROBOT = True
        if TRANSLUCENT_ROBOT:
            # Make the robot translucent
            # Change the color of each link (including the base)
            robot_visual_data = p.getVisualShapeData(self.robot_id)
            for visual_shape in robot_visual_data:
                link_idx = visual_shape[1]  # Link index
                rgba = visual_shape[7]  # RGBA color
                TRANSPARENCY = 0.5
                new_rgba = (rgba[0], rgba[1], rgba[2], TRANSPARENCY)
                p.changeVisualShape(self.robot_id, link_idx, rgbaColor=new_rgba)

        # Make the robot blue
        # Change the color of each link (including the base)
        for link_index in range(
            -1, p.getNumJoints(self.robot_cmd_id)
        ):  # -1 is for the base
            p.changeVisualShape(
                self.robot_cmd_id, link_index, rgbaColor=BLUE_TRANSLUCENT_RGBA
            )

        # Set the robot to a default pose
        DEFAULT_ARM_Q = np.zeros(NUM_ARM_JOINTS)
        DEFAULT_HAND_Q = np.zeros(NUM_HAND_JOINTS)
        assert DEFAULT_ARM_Q.shape == (NUM_ARM_JOINTS,)
        assert DEFAULT_HAND_Q.shape == (NUM_HAND_JOINTS,)
        DEFAULT_Q = np.concatenate([DEFAULT_ARM_Q, DEFAULT_HAND_Q])
        self.set_robot_state(self.robot_id, DEFAULT_Q)
        self.set_robot_state(self.robot_cmd_id, DEFAULT_Q)

        # Keep track of the link names and IDs
        self.robot_link_name_to_id = {}
        for i in range(p.getNumJoints(self.robot_id)):
            self.robot_link_name_to_id[
                p.getJointInfo(self.robot_id, i)[12].decode("utf-8")
            ] = i
        self.robot_cmd_link_name_to_id = {}
        for i in range(p.getNumJoints(self.robot_cmd_id)):
            self.robot_cmd_link_name_to_id[
                p.getJointInfo(self.robot_cmd_id, i)[12].decode("utf-8")
            ] = i

        # Create the hand target
        FAR_AWAY_PALM_TARGET = np.concatenate([np.ones(3), np.zeros(3)])
        self.palm_target_lines = visualize_transform(
            xyz=FAR_AWAY_PALM_TARGET[:3],
            rotation_matrix=R.from_euler("ZYX", FAR_AWAY_PALM_TARGET[3:]).as_matrix(),
        )
        self.hand_lines = visualize_transform(
            xyz=FAR_AWAY_PALM_TARGET[:3],
            rotation_matrix=R.from_euler("ZYX", FAR_AWAY_PALM_TARGET[3:]).as_matrix(),
        )
        self.hand_cmd_lines = visualize_transform(
            xyz=FAR_AWAY_PALM_TARGET[:3],
            rotation_matrix=R.from_euler("ZYX", FAR_AWAY_PALM_TARGET[3:]).as_matrix(),
        )

        self.object_lines = visualize_transform(
            xyz=FAR_AWAY_PALM_TARGET[:3],
            rotation_matrix=R.from_euler("ZYX", FAR_AWAY_PALM_TARGET[3:]).as_matrix(),
        )
        self.goal_object_lines = visualize_transform(
            xyz=FAR_AWAY_PALM_TARGET[:3],
            rotation_matrix=R.from_euler("ZYX", FAR_AWAY_PALM_TARGET[3:]).as_matrix(),
        )
        self.object_lines_2 = visualize_transform(
            xyz=FAR_AWAY_PALM_TARGET[:3],
            rotation_matrix=R.from_euler("ZYX", FAR_AWAY_PALM_TARGET[3:]).as_matrix(),
        )
        self.goal_object_lines_2 = visualize_transform(
            xyz=FAR_AWAY_PALM_TARGET[:3],
            rotation_matrix=R.from_euler("ZYX", FAR_AWAY_PALM_TARGET[3:]).as_matrix(),
        )

        self.keypoint_3d_point = add_sphere(
            radius=0.03, position=FAR_AWAY_PALM_TARGET[:3], rgbaColor=[1, 0, 0, 0.5]
        )
        self.keypoint_3d_2_point = add_sphere(
            radius=0.03, position=FAR_AWAY_PALM_TARGET[:3], rgbaColor=[0, 0, 1, 0.5]
        )
        self.target_keypoint_3d_point = add_sphere(
            radius=0.03, position=FAR_AWAY_PALM_TARGET[:3], rgbaColor=[1, 0, 0, 0.2]
        )
        self.target_keypoint_3d_2_point = add_sphere(
            radius=0.03, position=FAR_AWAY_PALM_TARGET[:3], rgbaColor=[0, 0, 1, 0.2]
        )

        # Create the camera lines
        CAMERA_LINES_TO_DRAW: Literal["C", "Cptcloud"] = "C"
        if CAMERA_LINES_TO_DRAW == "C":
            self.camera_lines = visualize_transform(
                xyz=self.T_R_C[:3, 3],
                rotation_matrix=self.T_R_C[:3, :3],
            )
        elif CAMERA_LINES_TO_DRAW == "Cptcloud":
            self.camera_lines = visualize_transform(
                xyz=self.T_R_Cptcloud[:3, 3],
                rotation_matrix=self.T_R_Cptcloud[:3, :3],
            )
        else:
            raise ValueError(f"Invalid CAMERA_LINES_TO_DRAW: {CAMERA_LINES_TO_DRAW}")

        # Set the camera parameters
        self.set_pybullet_camera()

        # Set gravity for simulation
        p.setGravity(0, 0, -9.81)

    def set_pybullet_camera(
        self,
        cameraDistance=2,
        cameraYaw=90,
        cameraPitch=-15,
        cameraTargetPosition=[0, 0, 0],
    ):
        """Configure the PyBullet camera view."""
        # HACK: Hide pybullet import in functions that use it to avoid messy tyro autocomplete
        import pybullet as p

        p.resetDebugVisualizerCamera(
            cameraDistance=cameraDistance,
            cameraYaw=cameraYaw,
            cameraPitch=cameraPitch,
            cameraTargetPosition=cameraTargetPosition,
        )

    def set_robot_state(self, robot, q: np.ndarray) -> None:
        # HACK: Hide pybullet import in functions that use it to avoid messy tyro autocomplete
        import pybullet as p

        assert q.shape == (23,)

        num_total_joints = p.getNumJoints(robot)
        actuatable_joint_idxs = [
            i
            for i in range(num_total_joints)
            if p.getJointInfo(robot, i)[2] != p.JOINT_FIXED
        ]
        num_actuatable_joints = len(actuatable_joint_idxs)
        assert (
            num_actuatable_joints == 23
        ), f"num_actuatable_joints: {num_actuatable_joints}"

        for i, joint_idx in enumerate(actuatable_joint_idxs):
            p.resetJointState(robot, joint_idx, q[i])

    def get_robot_state(self, robot) -> np.ndarray:
        # HACK: Hide pybullet import in functions that use it to avoid messy tyro autocomplete
        import pybullet as p

        num_total_joints = p.getNumJoints(robot)
        actuatable_joint_idxs = [
            i
            for i in range(num_total_joints)
            if p.getJointInfo(robot, i)[2] != p.JOINT_FIXED
        ]
        num_actuatable_joints = len(actuatable_joint_idxs)
        assert (
            num_actuatable_joints == 23
        ), f"num_actuatable_joints: {num_actuatable_joints}"

        q = np.zeros(num_actuatable_joints)
        for i, joint_idx in enumerate(actuatable_joint_idxs):
            q[i] = p.getJointState(robot, joint_idx)[0]  # Joint position
        return q

    def iiwa_joint_cmd_callback(self, msg: JointState):
        """Callback to update the commanded joint positions."""
        self.iiwa_joint_cmd = np.array(msg.position)

    def allegro_joint_cmd_callback(self, msg: JointState):
        """Callback to update the commanded joint positions."""
        self.allegro_joint_cmd = np.array(msg.position)

    def iiwa_joint_state_callback(self, msg: JointState):
        """Callback to update the current joint positions."""
        self.iiwa_joint_state = np.array(msg.position)

    def allegro_joint_state_callback(self, msg: JointState):
        """Callback to update the current joint positions."""
        self.allegro_joint_state = np.array(msg.position)

    def palm_target_callback(self, msg: Float64MultiArray):
        """Callback to update the current hand target."""
        self.palm_target = np.array(msg.data)

    def object_pose_callback(self, msg: Pose):
        """ "Callback to update the current object pose."""
        xyz = np.array([msg.position.x, msg.position.y, msg.position.z])
        quat_xyzw = np.array(
            [
                msg.orientation.x,
                msg.orientation.y,
                msg.orientation.z,
                msg.orientation.w,
            ]
        )
        latest_pose = np.eye(4)
        latest_pose[:3, 3] = xyz
        latest_pose[:3, :3] = R.from_quat(quat_xyzw).as_matrix()
        self.object_pose = latest_pose

    def goal_object_pose_callback(self, msg: Pose):
        """ "Callback to update the goal object pose."""
        xyz = np.array([msg.position.x, msg.position.y, msg.position.z])
        quat_xyzw = np.array(
            [
                msg.orientation.x,
                msg.orientation.y,
                msg.orientation.z,
                msg.orientation.w,
            ]
        )
        latest_pose = np.eye(4)
        latest_pose[:3, 3] = xyz
        latest_pose[:3, :3] = R.from_quat(quat_xyzw).as_matrix()
        self.goal_object_pose = latest_pose

    def object_pose_2_callback(self, msg: Pose):
        """ "Callback to update the current object pose."""
        xyz = np.array([msg.position.x, msg.position.y, msg.position.z])
        quat_xyzw = np.array(
            [
                msg.orientation.x,
                msg.orientation.y,
                msg.orientation.z,
                msg.orientation.w,
            ]
        )
        latest_pose = np.eye(4)
        latest_pose[:3, 3] = xyz
        latest_pose[:3, :3] = R.from_quat(quat_xyzw).as_matrix()
        self.object_pose_2 = latest_pose

    def goal_object_pose_2_callback(self, msg: Pose):
        """ "Callback to update the goal object pose."""
        xyz = np.array([msg.position.x, msg.position.y, msg.position.z])
        quat_xyzw = np.array(
            [
                msg.orientation.x,
                msg.orientation.y,
                msg.orientation.z,
                msg.orientation.w,
            ]
        )
        latest_pose = np.eye(4)
        latest_pose[:3, 3] = xyz
        latest_pose[:3, :3] = R.from_quat(quat_xyzw).as_matrix()
        self.goal_object_pose_2 = latest_pose

    def keypoint_3d_callback(self, msg: Point):
        self.keypoint_3d = np.array([msg.x, msg.y, msg.z])

    def keypoint_3d_2_callback(self, msg: Point):
        self.keypoint_3d_2 = np.array([msg.x, msg.y, msg.z])

    def target_keypoint_3d_callback(self, msg: Point):
        self.target_keypoint_3d = np.array([msg.x, msg.y, msg.z])

    def target_keypoint_3d_2_callback(self, msg: Point):
        self.target_keypoint_3d_2 = np.array([msg.x, msg.y, msg.z])

    def point_cloud_callback(self, msg: PointCloud2):
        self.point_cloud_and_colors = np.array(
            list(
                pc2.read_points(msg, field_names=("x", "y", "z", "rgb"), skip_nans=True)
            )
        )

        n_pts = self.point_cloud_and_colors.shape[0]
        assert self.point_cloud_and_colors.shape == (
            n_pts,
            4,
        ), f"self.point_cloud_and_colors.shape: {self.point_cloud_and_colors.shape}"

    def update_pybullet(self):
        """Update the PyBullet simulation with the commanded joint positions."""
        # HACK: Hide pybullet import in functions that use it to avoid messy tyro autocomplete
        import pybullet as p

        if self.iiwa_joint_cmd is None:
            print(colored("iiwa_joint_cmd is None", "yellow"))
            iiwa_joint_cmd = np.zeros(NUM_ARM_JOINTS)
        else:
            iiwa_joint_cmd = self.iiwa_joint_cmd

        if self.allegro_joint_cmd is None:
            print(colored("allegro_joint_cmd is None", "yellow"))
            allegro_joint_cmd = np.zeros(NUM_HAND_JOINTS)
        else:
            allegro_joint_cmd = self.allegro_joint_cmd

        if self.iiwa_joint_state is None:
            print(colored("iiwa_joint_state is None", "yellow"))
            iiwa_joint_state = np.zeros(NUM_ARM_JOINTS)
        else:
            iiwa_joint_state = self.iiwa_joint_state

        if self.allegro_joint_state is None:
            print(colored("allegro_joint_state is None", "yellow"))
            allegro_joint_state = np.zeros(NUM_HAND_JOINTS)
        else:
            allegro_joint_state = self.allegro_joint_state

        if self.palm_target is None:
            print(colored("palm_target is None", "yellow"))
            palm_target = np.zeros(6) + 100  # Far away
        else:
            palm_target = self.palm_target

        if self.object_pose is None:
            print(colored("object_pose is None", "yellow"))
            object_pose = np.eye(4)
            object_pose[:3, 3] = np.zeros(3) + 100  # Far away
        else:
            object_pose = self.object_pose

        if self.object_pose_2 is None:
            print(colored("object_pose_2 is None", "yellow"))
            object_pose_2 = np.eye(4)
            object_pose_2[:3, 3] = np.zeros(3) + 100  # Far away
        else:
            object_pose_2 = self.object_pose_2

        if self.keypoint_3d is None:
            print(colored("keypoint_3d is None", "yellow"))
            keypoint_3d = np.zeros(3) + 100  # Far away
        else:
            keypoint_3d = self.keypoint_3d

        if self.keypoint_3d_2 is None:
            print(colored("keypoint_3d_2 is None", "yellow"))
            keypoint_3d_2 = np.zeros(3) + 100  # Far away
        else:
            keypoint_3d_2 = self.keypoint_3d_2

        if self.target_keypoint_3d is None:
            print(colored("target_keypoint_3d is None", "yellow"))
            target_keypoint_3d = np.zeros(3) + 100  # Far away
        else:
            target_keypoint_3d = self.target_keypoint_3d

        if self.target_keypoint_3d_2 is None:
            print(colored("target_keypoint_3d_2 is None", "yellow"))
            target_keypoint_3d_2 = np.zeros(3) + 100  # Far away
        else:
            target_keypoint_3d_2 = self.target_keypoint_3d_2

        if self.goal_object_pose is None:
            print(colored("goal_object_pose is None", "yellow"))
            goal_object_pose = np.eye(4)
            goal_object_pose[:3, 3] = np.zeros(3) + 100  # Far away
        else:
            goal_object_pose = self.goal_object_pose

        if self.goal_object_pose_2 is None:
            print(colored("goal_object_pose_2 is None", "yellow"))
            goal_object_pose_2 = np.eye(4)
            goal_object_pose_2[:3, 3] = np.zeros(3) + 100  # Far away
        else:
            goal_object_pose_2 = self.goal_object_pose_2

        # print(colored(f"object_rot: {object_pose[:3, :3]}, goal_object_rot: {goal_object_pose[:3, :3]}", "red"))

        # Command Robot: Set the commanded joint positions
        q_cmd = np.concatenate([iiwa_joint_cmd, allegro_joint_cmd])
        q_state = np.concatenate([iiwa_joint_state, allegro_joint_state])
        self.set_robot_state(self.robot_cmd_id, q_cmd)
        self.set_robot_state(self.robot_id, q_state)

        # Update the hand target
        visualize_transform(
            xyz=palm_target[:3],
            rotation_matrix=R.from_euler("ZYX", palm_target[3:]).as_matrix(),
            lines=self.palm_target_lines,
        )

        # Visualize the palm of the robot
        robot_palm_com, robot_palm_quat, *_ = p.getLinkState(
            self.robot_id,
            self.robot_link_name_to_id["base_link"],
            computeForwardKinematics=1,
        )
        robot_cmd_palm_com, robot_cmd_palm_quat, *_ = p.getLinkState(
            self.robot_cmd_id,
            self.robot_cmd_link_name_to_id["base_link"],
            computeForwardKinematics=1,
        )
        visualize_transform(
            xyz=np.array(robot_palm_com),
            rotation_matrix=R.from_quat(robot_palm_quat).as_matrix(),
            lines=self.hand_lines,
        )
        visualize_transform(
            xyz=np.array(robot_cmd_palm_com),
            rotation_matrix=R.from_quat(robot_cmd_palm_quat).as_matrix(),
            lines=self.hand_cmd_lines,
        )

        robot_palm_euler_ZYX = R.from_quat(robot_palm_quat).as_euler(
            "ZYX", degrees=False
        )

        # Log to debug palm position and orientation in robot frame
        DEBUG_PALM = False
        if DEBUG_PALM:
            print(f"robot_palm_com = {robot_palm_com}")
            print(f"robot_palm_quat = {robot_palm_quat}")
            print(f"robot_palm_euler_ZYX = {robot_palm_euler_ZYX}")

        # Update the object pose
        # Object pose is in camera frame = C frame
        # We want it in world frame = robot frame = R frame
        T_C_O = object_pose
        T_R_O = self.T_R_C @ T_C_O
        print(colored(f"T_R_O = {T_R_O}", "red"))
        object_pos = T_R_O[:3, 3]
        object_quat_xyzw = R.from_matrix(T_R_O[:3, :3]).as_quat()
        p.resetBasePositionAndOrientation(self.object_id, object_pos, object_quat_xyzw)

        # Update the goal object pose
        # Goal object pose is in camera frame = C frame
        # We want it in world frame = robot frame = R frame
        T_C_G = goal_object_pose
        T_R_G = self.goal_T_R_C @ T_C_G
        goal_object_pos = T_R_G[:3, 3]
        goal_object_quat_xyzw = R.from_matrix(T_R_G[:3, :3]).as_quat()
        p.resetBasePositionAndOrientation(
            self.goal_object_id, goal_object_pos, goal_object_quat_xyzw
        )

        # Visualize object transforms
        visualize_transform(
            xyz=np.array(object_pos),
            rotation_matrix=R.from_quat(object_quat_xyzw).as_matrix(),
            lines=self.object_lines,
        )
        visualize_transform(
            xyz=np.array(goal_object_pos),
            rotation_matrix=R.from_quat(goal_object_quat_xyzw).as_matrix(),
            lines=self.goal_object_lines,
        )

        # Update the object pose
        # Object pose is in camera frame = C frame
        # We want it in world frame = robot frame = R frame
        T_C_O_2 = object_pose_2
        T_R_O_2 = self.T_R_C @ T_C_O_2
        print(colored(f"T_R_O_2 = {T_R_O_2}", "red"))
        object_pos_2 = T_R_O_2[:3, 3]
        object_quat_xyzw_2 = R.from_matrix(T_R_O_2[:3, :3]).as_quat()
        p.resetBasePositionAndOrientation(
            self.object_id_2, object_pos_2, object_quat_xyzw_2
        )

        # Update the goal object pose
        # Goal object pose is in camera frame = C frame
        # We want it in world frame = robot frame = R frame
        T_C_G_2 = goal_object_pose_2
        T_R_G_2 = self.goal_T_R_C @ T_C_G_2
        goal_object_pos_2 = T_R_G_2[:3, 3]
        goal_object_quat_xyzw_2 = R.from_matrix(T_R_G_2[:3, :3]).as_quat()
        p.resetBasePositionAndOrientation(
            self.goal_object_id_2, goal_object_pos_2, goal_object_quat_xyzw_2
        )

        # Visualize object transforms
        visualize_transform(
            xyz=np.array(object_pos_2),
            rotation_matrix=R.from_quat(object_quat_xyzw_2).as_matrix(),
            lines=self.object_lines_2,
        )
        visualize_transform(
            xyz=np.array(goal_object_pos_2),
            rotation_matrix=R.from_quat(goal_object_quat_xyzw_2).as_matrix(),
            lines=self.goal_object_lines_2,
        )

        # Update the keypoint
        # Keypoint is in camera frame = C frame
        # We want it in world frame = robot frame = R frame
        keypoint_3d_R = transform_points(self.T_R_C, keypoint_3d[None]).squeeze(axis=0)
        move_sphere(self.keypoint_3d_point, keypoint_3d_R)
        print(colored(f"keypoint_3d_R = {keypoint_3d_R}", "red"))

        keypoint_3d_2_R = transform_points(self.T_R_C, keypoint_3d_2[None]).squeeze(
            axis=0
        )
        move_sphere(self.keypoint_3d_2_point, keypoint_3d_2_R)

        # target_keypoint_3d_R = transform_points(self.T_R_C, target_keypoint_3d[None]).squeeze(axis=0)
        # NOTE: Target keypoint is already in robot frame
        target_keypoint_3d_R = target_keypoint_3d
        move_sphere(self.target_keypoint_3d_point, target_keypoint_3d_R)

        target_keypoint_3d_2_R = transform_points(
            self.T_R_C, target_keypoint_3d_2[None]
        ).squeeze(axis=0)
        move_sphere(self.target_keypoint_3d_2_point, target_keypoint_3d_2_R)

        # Update the point cloud
        LOAD_POINT_CLOUD = self.args.load_point_cloud
        if LOAD_POINT_CLOUD:
            if self.point_cloud_and_colors is not None:
                draw_colored_point_cloud(
                    point_cloud_and_colors=self.point_cloud_and_colors,
                    T_R_Cptcloud=self.T_R_Cptcloud,
                )
            else:
                print(colored("point_cloud_and_colors is None", "yellow"))

    def run(self):
        """Main loop to run the node, update simulation, and publish joint states."""
        # HACK: Hide pybullet import in functions that use it to avoid messy tyro autocomplete
        import pybullet as p

        while not rospy.is_shutdown():
            start_time = rospy.Time.now()

            # Update the PyBullet simulation with the current joint commands
            self.update_pybullet()

            # Sleep to maintain the loop rate
            before_sleep_time = rospy.Time.now()

            SLEEP_MODE: Literal[
                "ROS_SLEEP", "WHILE_SLEEP", "WHILE_NO_SLEEP", "NO_SLEEP"
            ] = "WHILE_SLEEP"
            if SLEEP_MODE == "ROS_SLEEP":
                # Seems to cause segfault?
                self.rate.sleep()
            elif SLEEP_MODE == "WHILE_SLEEP":
                loops_waiting = 0
                while (
                    not rospy.is_shutdown()
                    and (rospy.Time.now() - start_time).to_sec() < 1 / self.rate_hz
                ):
                    # print(
                    #     f"loops_waiting: {loops_waiting}, (rospy.Time.now() - start_time).to_sec() = {(rospy.Time.now() - start_time).to_sec()}"
                    # )
                    loops_waiting += 1
                    time.sleep(0.0001)
            elif SLEEP_MODE == "WHILE_NO_SLEEP":
                loops_waiting = 0
                while (
                    not rospy.is_shutdown()
                    and (rospy.Time.now() - start_time).to_sec() < 1 / self.rate_hz
                ):
                    print(
                        f"loops_waiting: {loops_waiting}, (rospy.Time.now() - start_time).to_sec() = {(rospy.Time.now() - start_time).to_sec()}"
                    )
                    loops_waiting += 1
            elif SLEEP_MODE == "NO_SLEEP":
                pass
            else:
                raise ValueError(f"Invalid SLEEP_MODE: {SLEEP_MODE}")

            after_sleep_time = rospy.Time.now()
            print(
                get_ros_loop_rate_str(
                    start_time=start_time,
                    before_sleep_time=before_sleep_time,
                    after_sleep_time=after_sleep_time,
                    node_name=rospy.get_name(),
                )
            )

        # Disconnect from PyBullet when shutting down
        p.disconnect()

    @property
    @functools.lru_cache()
    def camera(self) -> Literal["zed", "realsense"]:
        # Check camera parameter
        camera = rospy.get_param("/camera", None)
        if camera is None:
            DEFAULT_CAMERA = "zed"
            print(
                colored(
                    f"No /camera parameter found, using default camera {DEFAULT_CAMERA}",
                    "yellow",
                )
            )
            camera = DEFAULT_CAMERA
        print(f"Using camera: {camera}")
        assert camera in ["zed", "realsense"], f"camera: {camera}"
        return camera

    @property
    @functools.lru_cache()
    def goal_camera(self) -> Literal["zed", "realsense"]:
        # Check goal_camera parameter
        goal_camera = rospy.get_param("/goal_camera", None)
        if goal_camera is None:
            DEFAULT_CAMERA = "zed"
            print(
                colored(
                    f"No /goal_camera parameter found, using default camera {DEFAULT_CAMERA}",
                    "yellow",
                )
            )
            goal_camera = DEFAULT_CAMERA
        print(f"Using goal_camera: {goal_camera}")
        assert goal_camera in ["zed", "realsense"], f"goal_camera: {goal_camera}"
        return goal_camera

    @property
    @functools.lru_cache()
    def T_R_C(self) -> np.ndarray:
        if self.camera == "zed":
            return ZED_CAMERA_T_R_C
        else:
            raise ValueError(f"Unknown camera: {self.camera}")

    @property
    @functools.lru_cache()
    def goal_T_R_C(self) -> np.ndarray:
        # Check goal_camera parameter
        if self.goal_camera == "zed":
            return ZED_CAMERA_T_R_C
        else:
            raise ValueError(f"Unknown goal_camera: {self.goal_camera}")

    @property
    @functools.lru_cache()
    def T_R_Cptcloud(self) -> np.ndarray:
        if self.camera == "zed":
            return ZED_CAMERA_T_R_Cptcloud
        else:
            raise ValueError(f"Unknown camera: {self.camera}")


def main():
    args = tyro.cli(Args)
    try:
        # Create and run the FakeRobotNode
        node = VisualizationNode(args)
        node.run()
    except rospy.ROSInterruptException:
        pass


if __name__ == "__main__":
    main()
