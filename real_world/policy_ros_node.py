#!/usr/bin/env python

import os
import time

import numpy as np
import pybullet as pb
import pytorch_kinematics as pk
import rospy
import torch
from geometry_msgs.msg import Point
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from termcolor import colored

from real_world.camera_extrinsics import ZED_CAMERA_T_R_C
from real_world.load_policy_utils import create_eval_env, load_config_and_runner
from real_world.misc_utils import transform_point
from real_world.pybullet_utils import get_link_name_to_idx
from tasks.env_arm import (
    control_ik,
    matrix_to_axis_angle,
    normalize,
    quaternion_to_matrix,
    unnormalize,
)
from utils.kuka_allegro import KukaAllegro

ENV_NAME = "EnvBottleArm-v0"

TRAJECTORY_FILE = "data/real/bottle_arm3/latest/trajectories.pt"

POLICY_FOLDER = ""  # replace

DEVICE = "cuda"
TRAJ_IDX = 0

NUM_XYZ = 3
NUM_RPY = 3
NUM_XYZRPY = NUM_XYZ + NUM_RPY
TRAJ_IDX = 0

DISABLE_ACTIONS = False
DISABLE_ACTIONS_FOR_N_STEPS_FLAG = False
DISABLE_ACTIONS_FOR_N_STEPS = 30
DISABLE_ROTATION_ACTION = False
IGNORE_KEYPOINT_3D = False
INCLUDE_KEYPOINT_3D_2 = True
IGNORE_KEYPOINT_3D_2 = False
ZERO_OBS = False
DISABLE_AVERAGING = False

NUM_QUAT = 4
NUM_XYZQUAT = NUM_XYZ + NUM_QUAT
NUM_FINGERS = 4

BATCH_SIZE = 1

OBS_LIST = []
ACTION_LIST = []
SCALED_ACTION_WITHOUT_AVG_LIST = []
SCALED_ACTION_LIST = []

NUM_OBS = 229
NUM_ACTIONS = 23
NUM_ARM_JOINTS = 7
NUM_HAND_JOINTS = 16

if INCLUDE_KEYPOINT_3D_2:
    NUM_OBS += NUM_XYZ * 16


def accurate_sleep(seconds: float):
    start_time = time.perf_counter()
    while time.perf_counter() - start_time < seconds:
        # pass
        time.sleep(0.0001)


class PolicyROSNode:
    def __init__(
        self,
        traj_file: str,
    ):
        # Init policy
        log_dir = f"logs/{POLICY_FOLDER}"
        video_envs, viewer = create_eval_env(
            env_name=ENV_NAME,
            num_envs=16,
            log_dir=log_dir,
            make_video=False,
            create_viewer=False,
        )
        runner = load_config_and_runner(
            env_name=ENV_NAME,
            run_name=POLICY_FOLDER,
            vec_env=video_envs,
            log_dir=log_dir,
        )
        # self.policy = runner.get_inference_policy()
        self.policy = runner.get_noisy_inference_policy()

        self.joint_limits = video_envs.env._env.joint_limits
        assert self.joint_limits.shape == (
            NUM_ARM_JOINTS + NUM_HAND_JOINTS,
            2,
        ), f"self.joint_limits.shape: {self.joint_limits.shape}"

        self.conservative_joint_limits = self.joint_limits.detach().clone()
        self.conservative_joint_limits[:NUM_ARM_JOINTS, 0] += np.deg2rad(10)
        self.conservative_joint_limits[:NUM_ARM_JOINTS, 1] -= np.deg2rad(10)

        # Create PK chain
        with open(KukaAllegro.urdf_path, "rb") as f:
            urdf_str = f.read()
        self.arm_pk_chain = pk.build_serial_chain_from_urdf(
            urdf_str,
            end_link_name="base_link",
        ).to(device=DEVICE)

        self.init_targets(
            traj_file=traj_file,
        )

        # Variables for storing the latest images
        rospy.init_node("policy_node")
        self.rate_hz = 60
        self.rate = rospy.Rate(self.rate_hz)

        # State variables
        self.latest_iiwa_q = None
        self.latest_iiwa_qd = None
        self.latest_allegro_q = None
        self.latest_allegro_qd = None
        self.latest_keypoint_3d = None
        self.latest_keypoint_3d_2 = None

        self.REAL_ELAPSED_STEPS = 0
        self.elapsed_steps = 0
        self.action_avg = torch.zeros(BATCH_SIZE, NUM_ACTIONS, device=DEVICE)
        # init_arm_config = torch.tensor(
        #     np.deg2rad([-65.64, 31.7, -9.58, -93.92, 47.16, 103.39, -52.01]),
        #     device=DEVICE,
        # )
        # init_hand_config = self.joint_limits[NUM_ARM_JOINTS:, 0]
        # self.start_qpos = torch.cat([init_arm_config, init_hand_config], dim=-1)[None]
        # self.normalized_start_qpos = normalize(
        #     self.start_qpos, low=self.joint_limits[:, 0], high=self.joint_limits[:, 1]
        # )
        # assert self.start_qpos.shape == (1, NUM_ARM_JOINTS + NUM_HAND_JOINTS), (
        #     f"self.start_qpos.shape: {self.start_qpos.shape}"
        # )
        # assert self.normalized_start_qpos.shape == (1, NUM_ARM_JOINTS + NUM_HAND_JOINTS), (
        #     f"self.normalized_start_qpos.shape: {self.normalized_start_qpos.shape}"
        # )

        self.iiwa_joint_state_sub = rospy.Subscriber(
            "/iiwa/joint_states", JointState, self.iiwa_joint_state_callback
        )
        self.allegro_joint_state_sub = rospy.Subscriber(
            "/allegroHand_0/joint_states", JointState, self.allegro_joint_state_callback
        )
        self.keypoint_3d_sub = rospy.Subscriber(
            "/keypoint_3d", Point, self.keypoint_3d_callback, queue_size=1
        )
        self.keypoint_3d_2_sub = rospy.Subscriber(
            "/keypoint_3d_2", Point, self.keypoint_3d_2_callback, queue_size=1
        )

        self.iiwa_joint_cmd_pub = rospy.Publisher(
            "/iiwa/joint_cmd", JointState, queue_size=1
        )
        self.allegro_joint_cmd_pub = rospy.Publisher(
            "/allegroHand_0/joint_cmd", JointState, queue_size=1
        )
        self.palm_target_pub = rospy.Publisher(
            "/palm_target", Float64MultiArray, queue_size=1
        )
        self.target_keypoint_3d_pub = rospy.Publisher(
            "/target_keypoint_3d", Point, queue_size=1
        )
        self.target_keypoint_3d_2_pub = rospy.Publisher(
            "/target_keypoint_3d_2", Point, queue_size=1
        )

        pb.connect(pb.DIRECT)
        self.pb_robot = pb.loadURDF(KukaAllegro.urdf_path)
        self.pb_link_name_to_idx = get_link_name_to_idx(self.pb_robot)

    def init_targets(self, traj_file: str):
        assert os.path.exists(traj_file), f"File {traj_file} does not exist"
        assert traj_file.endswith(".pt"), f"File {traj_file} must end with .pt"
        trajectories_dict = torch.load(traj_file, weights_only=False)
        self.target_object_keypoint = trajectories_dict["bottle"].to(DEVICE)
        self.target_wrist_pose = trajectories_dict["hand"].to(DEVICE)
        if INCLUDE_KEYPOINT_3D_2:
            self.target_object_keypoint_2 = trajectories_dict["plate"].to(DEVICE)

        NUM_PADDING = 400  # Extend the end of the trajectory with the last pose
        self.target_object_keypoint = torch.cat(
            [self.target_object_keypoint]
            + NUM_PADDING * [self.target_object_keypoint[:, -1:]],
            dim=1,
        )
        self.target_wrist_pose = torch.cat(
            [self.target_wrist_pose] + NUM_PADDING * [self.target_wrist_pose[:, -1:]],
            dim=1,
        )
        if INCLUDE_KEYPOINT_3D_2:
            self.target_object_keypoint_2 = torch.cat(
                [self.target_object_keypoint_2]
                + NUM_PADDING * [self.target_object_keypoint_2[:, -1:]],
                dim=1,
            )

        N_TRAJS, N_TIMESTEPS, _ = self.target_object_keypoint.shape
        assert self.target_object_keypoint.shape == (
            N_TRAJS,
            N_TIMESTEPS,
            3,
        ), f"target_object_keypoint.shape: {self.target_object_keypoint.shape}"
        assert self.target_wrist_pose.shape == (
            N_TRAJS,
            N_TIMESTEPS,
            6,
        ), f"target_wrist_pose.shape: {self.target_wrist_pose.shape}"
        if INCLUDE_KEYPOINT_3D_2:
            assert self.target_object_keypoint_2.shape == (
                N_TRAJS,
                N_TIMESTEPS,
                3,
            ), f"target_object_keypoint_2.shape: {self.target_object_keypoint_2.shape}"

        target_wrist_euler_xyz = self.target_wrist_pose[:, :, 3:]
        wrist_euler_xyz_np = target_wrist_euler_xyz.cpu().numpy()
        wrist_quat_xyzw = (
            R.from_euler("xyz", wrist_euler_xyz_np.reshape(N_TRAJS * N_TIMESTEPS, 3))
            .as_quat()
            .reshape(N_TRAJS, N_TIMESTEPS, 4)
        )
        wrist_quat_wxyz = wrist_quat_xyzw[:, :, [3, 0, 1, 2]]
        self.target_wrist_pos = self.target_wrist_pose[:, :, :3]
        self.target_wrist_quat = torch.from_numpy(wrist_quat_wxyz).float().to(DEVICE)

    def iiwa_joint_state_callback(self, data: JointState):
        self.latest_iiwa_q = np.array(data.position)
        self.latest_iiwa_qd = np.array(data.velocity)

    def allegro_joint_state_callback(self, data: JointState):
        self.latest_allegro_q = np.array(data.position)
        self.latest_allegro_qd = np.array(data.velocity)

    def keypoint_3d_callback(self, data: Point):
        self.latest_keypoint_3d = np.array([data.x, data.y, data.z])

    def keypoint_3d_2_callback(self, data: Point):
        self.latest_keypoint_3d_2 = np.array([data.x, data.y, data.z])

    def compute_tip_poses(self) -> np.ndarray:
        tip_link_names = [
            "link_3.0_tip",
            "link_7.0_tip",
            "link_11.0_tip",
            "link_15.0_tip",
        ]

        tip_poses = []
        for tip_link_name in tip_link_names:
            tip_xyz, tip_quat_xyzw, *_ = pb.getLinkState(
                self.pb_robot,
                self.pb_link_name_to_idx[tip_link_name],
                computeForwardKinematics=1,
            )
            tip_xyz = np.array(tip_xyz)
            tip_quat_xyzw = np.array(tip_quat_xyzw)
            assert tip_xyz.shape == (3,)
            assert tip_quat_xyzw.shape == (4,)
            tip_quat_wxyz = tip_quat_xyzw[[3, 0, 1, 2]]
            tip_pose = np.concatenate([tip_xyz, tip_quat_wxyz], axis=-1)
            tip_poses.append(tip_pose)
        tip_poses = np.stack(tip_poses, axis=0)
        assert tip_poses.shape == (len(tip_link_names), 7)
        return tip_poses

    def run(self):
        if IGNORE_KEYPOINT_3D:
            print(colored("IGNORE_KEYPOINT_3D", "yellow"))
            # reasonable_keypoint_3d_R = np.array(
            #     [0.57080256, -0.25907539, 0.28410709]
            # )  # Reasonable start
            reasonable_keypoint_3d_R = np.array(
                # [0.77491437, -0.00268608, 0.28758481]
                # [0.79447297, 0.00459389, 0.28796183]
                [0.83, 0.06, 0.28]
            )  # Reasonable start
            # [ 0.77383087 -0.00115028  0.28779086]
            # self.latest_keypoint_3d = np.zeros(3) + 100  # Far away
            T_C_R = np.linalg.inv(ZED_CAMERA_T_R_C)
            keypoint_3d_C = transform_point(T=T_C_R, point=reasonable_keypoint_3d_R)
            self.latest_keypoint_3d = keypoint_3d_C
        if IGNORE_KEYPOINT_3D_2:
            print(colored("IGNORE_KEYPOINT_3D_2", "yellow"))
            self.latest_keypoint_3d_2 = (
                np.zeros(3) + 100
            )  # Far away, TODO: Make this better

        ##############################
        # Wait for the first observations
        ##############################
        while not rospy.is_shutdown() and (
            self.latest_iiwa_q is None
            or self.latest_iiwa_qd is None
            or self.latest_allegro_q is None
            or self.latest_allegro_qd is None
            or self.latest_keypoint_3d is None
            or self.latest_keypoint_3d_2 is None
        ):
            print(
                f"Missing one of the required images (iiwa_q (missing? {self.latest_iiwa_q is None}), iiwa_qd (missing? {self.latest_iiwa_qd is None}), allegro_q (missing? {self.latest_allegro_q is None}), allegro_qd (missing? {self.latest_allegro_qd is None}), keypoint_3d (missing? {self.latest_keypoint_3d is None}), keypoint_3d_2 (missing? {self.latest_keypoint_3d_2 is None})). Waiting..."
            )
            rospy.sleep(0.1)

        assert self.latest_iiwa_q is not None
        assert self.latest_iiwa_qd is not None
        assert self.latest_allegro_q is not None
        assert self.latest_allegro_qd is not None
        assert self.latest_keypoint_3d is not None
        assert self.latest_keypoint_3d_2 is not None

        # Loop sending same PD target as current
        INIT_IIWA_Q = self.latest_iiwa_q.copy()
        INIT_ALLEGR_Q = self.latest_allegro_q.copy()
        self.start_qpos = (
            torch.from_numpy(
                np.concatenate([INIT_IIWA_Q, INIT_ALLEGR_Q], axis=-1)[None]
            )
            .float()
            .to(DEVICE)
        )
        self.normalized_start_qpos = normalize(
            self.start_qpos, low=self.joint_limits[:, 0], high=self.joint_limits[:, 1]
        )
        assert self.start_qpos.shape == (
            1,
            NUM_ARM_JOINTS + NUM_HAND_JOINTS,
        ), f"self.start_qpos.shape: {self.start_qpos.shape}"
        assert self.normalized_start_qpos.shape == (
            1,
            NUM_ARM_JOINTS + NUM_HAND_JOINTS,
        ), f"self.normalized_start_qpos.shape: {self.normalized_start_qpos.shape}"

        start_time_of_sending_same_pd_target = rospy.Time.now()
        while True:
            NUM_SECONDS_TO_SEND_SAME_PD_TARGET = 1
            if rospy.Time.now() - start_time_of_sending_same_pd_target > rospy.Duration(
                NUM_SECONDS_TO_SEND_SAME_PD_TARGET
            ):
                print("~" * 100)
                print("Done sending same PD target")
                print("~" * 100)
                break

            # Publish
            # Arm command
            iiwa_msg = JointState()
            iiwa_msg.header.stamp = rospy.Time.now()
            iiwa_msg.name = [f"iiwa_joint_{i}" for i in range(NUM_ARM_JOINTS)]
            iiwa_msg.position = INIT_IIWA_Q.tolist()
            iiwa_msg.velocity = []
            iiwa_msg.effort = []
            self.iiwa_joint_cmd_pub.publish(iiwa_msg)

            # Hand command
            allegro_msg = JointState()
            allegro_msg.header.stamp = rospy.Time.now()
            allegro_msg.name = [f"allegro_joint_{i}" for i in range(NUM_HAND_JOINTS)]
            allegro_msg.position = INIT_ALLEGR_Q.tolist()
            allegro_msg.velocity = []
            allegro_msg.effort = []
            self.allegro_joint_cmd_pub.publish(allegro_msg)

            print(
                f"Published same PD target, time: {rospy.Time.now() - start_time_of_sending_same_pd_target}"
            )
            self.rate.sleep()

        # Loop
        while not rospy.is_shutdown():
            start_time = time.perf_counter()

            MAX_ELAPSED_STEPS = self.target_wrist_pos.shape[1] - 400
            # MAX_ELAPSED_STEPS = 120  # 2 seconds
            # MAX_ELAPSED_STEPS = 100  # <2 seconds
            if self.elapsed_steps >= MAX_ELAPSED_STEPS:
                SAVE_THINGS = True
                if SAVE_THINGS:
                    obs_tensor = torch.stack(OBS_LIST, dim=0)
                    action_tensor = torch.stack(ACTION_LIST, dim=0)
                    scaled_action_tensor = torch.stack(SCALED_ACTION_LIST, dim=0)
                    scaled_action_without_avg_tensor = torch.stack(
                        SCALED_ACTION_WITHOUT_AVG_LIST, dim=0
                    )
                    N = obs_tensor.shape[0]
                    assert obs_tensor.shape == (N, NUM_OBS)
                    assert action_tensor.shape == (N, NUM_ACTIONS)
                    assert scaled_action_tensor.shape == (
                        N,
                        NUM_ARM_JOINTS + NUM_HAND_JOINTS,
                    )

                    datetime_str = time.strftime("%Y-%m-%d_%H-%M-%S")
                    # filename = f"2025-05-11_actual_real_world_recorded_data.pt"
                    filename = f"{datetime_str}_actual_real_world_recorded_data.pt"
                    print(f"Saving to {filename}")
                    torch.save(
                        {
                            "obs": obs_tensor,
                            "action": action_tensor,
                            "scaled_action": scaled_action_tensor,
                            "scaled_action_without_avg": scaled_action_without_avg_tensor,
                        },
                        filename,
                    )
                print(
                    f"DONE: self.elapsed_steps: {self.elapsed_steps}, self.target_wrist_pos.shape[1]: {self.target_wrist_pos.shape[1]}"
                )
                return

            # Get observation dictionary
            obs_dict = {}
            obs_dict["iiwa_q"] = torch.from_numpy(self.latest_iiwa_q).float().to(DEVICE)
            obs_dict["allegro_q"] = (
                torch.from_numpy(self.latest_allegro_q).float().to(DEVICE)
            )
            # HACK: Overwrite allegro_q's -4'th joint to 0.2630
            # obs_dict["allegro_q"][-4] = 0.2630
            obs_dict["allegro_q"][-4] = 0.7

            obs_dict["iiwa_qd"] = (
                torch.from_numpy(self.latest_iiwa_qd).float().to(DEVICE)
            )
            obs_dict["allegro_qd"] = (
                torch.from_numpy(self.latest_allegro_qd).float().to(DEVICE)
            )

            tip_poses = self.compute_tip_poses()
            obs_dict["tip_poses"] = (
                torch.from_numpy(tip_poses).float().to(DEVICE).reshape(-1)
            )
            # HACK: tip_poses to 0
            obs_dict["tip_poses"] *= 0
            obs_dict["action_avg"] = self.action_avg.squeeze(dim=0)

            keypoint_3d_C = self.latest_keypoint_3d
            keypoint_3d_R = transform_point(T=ZED_CAMERA_T_R_C, point=keypoint_3d_C)
            print(colored(f"keypoint_3d_R: {keypoint_3d_R}", "red"))
            obs_dict["obj_keypoint_3d"] = (
                torch.from_numpy(keypoint_3d_R).float().to(DEVICE)
            )
            if INCLUDE_KEYPOINT_3D_2:
                keypoint_3d_2_C = self.latest_keypoint_3d_2
                keypoint_3d_2_R = transform_point(
                    T=ZED_CAMERA_T_R_C, point=keypoint_3d_2_C
                )
                obs_dict["obj_keypoint_3d_2"] = (
                    torch.from_numpy(keypoint_3d_2_R).float().to(DEVICE)
                )

            # From
            # >>> x.keys()
            # dict_keys(['bottle', 'hand'])
            for i in list(range(10)) + [20, 30, 40, 50]:
                idx = TRAJ_IDX, self.elapsed_steps + i
                target_wrist_pose = self.target_wrist_pose[idx]
                target_object_keypoint = self.target_object_keypoint[idx]
                if INCLUDE_KEYPOINT_3D_2:
                    target_object_keypoint_2 = self.target_object_keypoint_2[idx]

                obs_dict[f"target_obj_keypoint_{i}"] = target_object_keypoint
                if INCLUDE_KEYPOINT_3D_2:
                    obs_dict[f"target_obj_keypoint_2_{i}"] = target_object_keypoint_2
                obs_dict[f"target_wrist_pose_{i}"] = target_wrist_pose

            obs_dict["initial_obj_keypoint"] = self.target_object_keypoint[TRAJ_IDX, 0]
            if INCLUDE_KEYPOINT_3D_2:
                obs_dict["initial_obj_keypoint_2"] = self.target_object_keypoint_2[
                    TRAJ_IDX, 0
                ]

            for k, v in obs_dict.items():
                assert v.ndim == 1, f"{k} has shape {v.shape}"

            obs = (
                torch.cat([obs_dict[k] for k in obs_dict.keys()], dim=-1)
                .float()
                .to(DEVICE)
            )

            if ZERO_OBS:
                # HACK
                print(colored("ZERO_OBS", "yellow"))
                obs = torch.zeros(NUM_OBS, device=DEVICE)
            assert obs.shape == (NUM_OBS,), f"obs.shape: {obs.shape}"

            OBS_LIST.append(obs.detach().cpu())

            # Get action
            action = self.policy(obs.unsqueeze(dim=0))
            ACTION_LIST.append(action.squeeze(dim=0).detach().cpu())

            assert action.shape == (
                BATCH_SIZE,
                NUM_ACTIONS,
            ), f"action.shape: {action.shape}"
            action = torch.clamp(action, min=-1, max=1)

            """
            Controller: Joint PD Position Targets (NUM_ARM_JOINTS + NUM_HAND_JOINTS)
            Action: Delta Wrist Pose (NUM_XYZRPY) + Hand Joint PD Position Targets (NUM_HAND_JOINTS)
            """
            assert action.shape == (
                BATCH_SIZE,
                NUM_ARM_JOINTS + NUM_HAND_JOINTS,
            ), f"action.shape: {action.shape}"

            normalized_arm_action = action[:, :NUM_XYZRPY].detach().clone()
            normalized_hand_action = (
                action[:, NUM_XYZRPY : NUM_XYZRPY + NUM_HAND_JOINTS].detach().clone()
            )

            ARM_LOWER_XYZ_LIMITS = torch.tensor([-0.1, -0.1, -0.1], device=DEVICE)
            ARM_UPPER_XYZ_LIMITS = torch.tensor([0.1, 0.1, 0.1], device=DEVICE)
            ARM_LOWER_RPY_LIMITS = torch.deg2rad(
                torch.tensor([-20, -20, -20], device=DEVICE)
            )
            ARM_UPPER_RPY_LIMITS = torch.deg2rad(
                torch.tensor([20, 20, 20], device=DEVICE)
            )
            ARM_LOWER_LIMITS = torch.cat(
                [ARM_LOWER_XYZ_LIMITS, ARM_LOWER_RPY_LIMITS], dim=-1
            )
            ARM_UPPER_LIMITS = torch.cat(
                [ARM_UPPER_XYZ_LIMITS, ARM_UPPER_RPY_LIMITS], dim=-1
            )
            arm_action = unnormalize(
                normalized_arm_action, low=ARM_LOWER_LIMITS, high=ARM_UPPER_LIMITS
            )

            # Only use one action for fingers
            normalized_hand_action[:, :] = normalized_hand_action[:, 0:1]
            # hand_action = unnormalize(
            #     normalized_hand_action,
            #     low=self.joint_limits[NUM_ARM_JOINTS:, 0],
            #     high=self.joint_limits[NUM_ARM_JOINTS:, 1],
            # )
            hand_action = normalized_hand_action * 2

            # Compute current wrist pose
            current_qpos = (
                torch.cat([obs_dict["iiwa_q"], obs_dict["allegro_q"]], dim=-1)
                .float()
                .to(DEVICE)[None]
            )
            current_arm_qpos = current_qpos[:, :NUM_ARM_JOINTS]
            _current_hand_qpos = current_qpos[:, NUM_ARM_JOINTS:]
            wrist_pose = self.arm_pk_chain.forward_kinematics(
                current_arm_qpos
            ).get_matrix()
            assert wrist_pose.shape == (
                BATCH_SIZE,
                4,
                4,
            ), f"wrist_pose.shape: {wrist_pose.shape}"

            # Target wrist pose
            target_wrist_pos = self.target_wrist_pos[TRAJ_IDX, self.elapsed_steps]
            target_wrist_quat = self.target_wrist_quat[TRAJ_IDX, self.elapsed_steps]
            target_wrist_pose = (
                torch.eye(4)
                .unsqueeze(dim=0)
                .repeat_interleave(BATCH_SIZE, dim=0)
                .to(DEVICE)
            )
            target_wrist_pose[:, :3, 3] = target_wrist_pos
            target_wrist_pose[:, :3, :3] = quaternion_to_matrix(target_wrist_quat)
            assert target_wrist_pose.shape == (
                BATCH_SIZE,
                4,
                4,
            ), f"target_wrist_pose.shape: {target_wrist_pose.shape}"

            # Compute wrist error
            wrist_pos_error = target_wrist_pose[:, :3, 3] - wrist_pose[:, :3, 3]
            wrist_rot_error = torch.bmm(
                target_wrist_pose[:, :3, :3], wrist_pose[:, :3, :3].transpose(1, 2)
            )
            wrist_rot_error = matrix_to_axis_angle(wrist_rot_error)
            wrist_error = torch.cat([wrist_pos_error, wrist_rot_error], dim=-1)
            assert wrist_error.shape == (
                BATCH_SIZE,
                NUM_XYZRPY,
            ), f"wrist_error.shape: {wrist_error.shape}"

            # Arm action: wrist pose targets (delta from target wrist pose)
            if not (
                DISABLE_ACTIONS
                or (
                    DISABLE_ACTIONS_FOR_N_STEPS_FLAG
                    and self.elapsed_steps < DISABLE_ACTIONS_FOR_N_STEPS
                )
            ):
                if DISABLE_ROTATION_ACTION:
                    wrist_error[:, 0:3] += arm_action[:, 0:3]
                else:
                    wrist_error += arm_action

                # wrist_error[:, 2] = torch.clamp(wrist_error[:, 2], min=0, max=0.1)

                # Clamp wrist z error such that the resulting new wrist z is at least MIN_Z
                # MIN_Z = 0.25
                MIN_Z = 0.3
                wrist_z = wrist_pose[:, 2, 3]
                wrist_z_error = wrist_error[:, 2]
                new_wrist_z = wrist_z + wrist_z_error
                new_wrist_z = torch.clamp(new_wrist_z, min=MIN_Z, max=None)
                wrist_error[:, 2] = new_wrist_z - wrist_z
            else:
                print(colored("DISABLE_ACTIONS", "yellow"))

            # Compute jacobian
            jacobian = self.arm_pk_chain.jacobian(current_arm_qpos)
            assert jacobian.shape == (
                BATCH_SIZE,
                NUM_XYZRPY,
                NUM_ARM_JOINTS,
            ), f"jacobian.shape: {jacobian.shape}"
            pos_only = False
            if pos_only:
                jacobian = jacobian[:, 0:3]
                assert jacobian.shape == (
                    BATCH_SIZE,
                    NUM_XYZ,
                    NUM_ARM_JOINTS,
                ), f"jacobian.shape: {jacobian.shape}"

            # Compute delta arm joint position
            delta_arm_joint_pos = control_ik(
                j_eef=jacobian,
                dpose=wrist_error,
            )

            new_arm_qpos = current_arm_qpos + delta_arm_joint_pos
            assert new_arm_qpos.shape == (
                BATCH_SIZE,
                NUM_ARM_JOINTS,
            ), f"new_arm_qpos.shape: {new_arm_qpos.shape}"

            # Hand action: Hand joint position targets (delta from default)
            if not (
                DISABLE_ACTIONS
                or (
                    DISABLE_ACTIONS_FOR_N_STEPS_FLAG
                    and self.elapsed_steps < DISABLE_ACTIONS_FOR_N_STEPS
                )
            ):
                normalized_new_hand_qpos = (
                    self.normalized_start_qpos[:, NUM_ARM_JOINTS:] + hand_action
                )
            else:
                normalized_new_hand_qpos = self.normalized_start_qpos[
                    :, NUM_ARM_JOINTS:
                ]
                print(colored("DISABLE_ACTIONS", "yellow"))

            normalized_new_arm_qpos = normalize(
                new_arm_qpos,
                low=self.joint_limits[:, 0][:NUM_ARM_JOINTS],
                high=self.joint_limits[:, 1][:NUM_ARM_JOINTS],
            )

            normalized_new_qpos = torch.cat(
                [normalized_new_arm_qpos, normalized_new_hand_qpos], dim=-1
            )
            assert normalized_new_qpos.shape == (
                BATCH_SIZE,
                NUM_ARM_JOINTS + NUM_HAND_JOINTS,
            ), f"normalized_new_qpos.shape: {normalized_new_qpos.shape}"

            new_qpos = unnormalize(
                normalized_new_qpos,
                low=self.joint_limits[:, 0],
                high=self.joint_limits[:, 1],
            )
            SCALED_ACTION_WITHOUT_AVG_LIST.append(new_qpos.detach().cpu())
            if not DISABLE_AVERAGING:
                # gamma = 0.9
                # gamma = 0.95
                gamma = 0.98
                self.action_avg = self.action_avg * gamma + normalized_new_qpos * (
                    1 - gamma
                )
                # weight = gamma ** (self.elapsed_steps + 1)
                weight = gamma ** (self.REAL_ELAPSED_STEPS + 1)
                total_action = self.action_avg + self.normalized_start_qpos * weight
                total_action = torch.clamp(total_action, min=-1.0, max=1.0)

                # Go from [-1, 1] to [self.limits[:, 0], self.limits[:, 1]]
                scaled_action = unnormalize(
                    total_action,
                    low=self.joint_limits[:, 0],
                    high=self.joint_limits[:, 1],
                ).squeeze(dim=0)
            else:
                print(colored("DISABLE_AVERAGING", "yellow"))
                scaled_action = unnormalize(
                    normalized_new_qpos,
                    low=self.joint_limits[:, 0],
                    high=self.joint_limits[:, 1],
                ).squeeze(dim=0)

            SCALED_ACTION_LIST.append(scaled_action.detach().cpu())

            # rospy.logerr(f"At {self.elapsed_steps}: obs: {obs}, action: {action}, scaled_action: {scaled_action}")

            # Publish action
            # Arm command
            iiwa_msg = JointState()
            iiwa_msg.header.stamp = rospy.Time.now()
            iiwa_msg.name = [f"iiwa_joint_{i}" for i in range(NUM_ARM_JOINTS)]
            iiwa_msg.position = torch.clamp(
                scaled_action[:NUM_ARM_JOINTS],
                min=self.conservative_joint_limits[:NUM_ARM_JOINTS, 0],
                max=self.conservative_joint_limits[:NUM_ARM_JOINTS, 1],
            ).tolist()
            iiwa_msg.velocity = []
            iiwa_msg.effort = []
            self.iiwa_joint_cmd_pub.publish(iiwa_msg)

            # Hand command
            allegro_msg = JointState()
            allegro_msg.header.stamp = rospy.Time.now()
            allegro_msg.name = [f"allegro_joint_{i}" for i in range(NUM_HAND_JOINTS)]
            allegro_msg.position = scaled_action[NUM_ARM_JOINTS:].tolist()
            # HACK: Overwrite allegro_q's -4'th joint to 0.2630
            # allegro_msg.position[-4] = 0.2630
            allegro_msg.position[-4] = 0.7
            allegro_msg.velocity = []
            allegro_msg.effort = []
            self.allegro_joint_cmd_pub.publish(allegro_msg)

            # Palm target
            # Needed in xyz,ZYX euler angles
            target_wrist_pos_np = (
                self.target_wrist_pos[TRAJ_IDX, self.elapsed_steps]
                .detach()
                .cpu()
                .numpy()
            )
            target_wrist_quat_wxyz_np = (
                self.target_wrist_quat[TRAJ_IDX, self.elapsed_steps]
                .detach()
                .cpu()
                .numpy()
            )
            assert target_wrist_pos_np.shape == (
                3,
            ), f"target_wrist_pos_np.shape: {target_wrist_pos_np.shape}"
            assert target_wrist_quat_wxyz_np.shape == (
                4,
            ), f"target_wrist_quat_wxyz_np.shape: {target_wrist_quat_wxyz_np.shape}"
            target_wrist_quat_xyzw_np = target_wrist_quat_wxyz_np[[1, 2, 3, 0]]
            palm_target_np = np.concatenate(
                [
                    target_wrist_pos_np,
                    R.from_quat(target_wrist_quat_xyzw_np).as_euler(
                        "ZYX", degrees=False
                    ),
                ],
                axis=-1,
            )
            assert palm_target_np.shape == (
                6,
            ), f"palm_target_np.shape: {palm_target_np.shape}"
            palm_target_msg = Float64MultiArray()
            palm_target_msg.data = palm_target_np.tolist()
            self.palm_target_pub.publish(palm_target_msg)

            # Target keypoint
            target_keypoint_3d_np = (
                self.target_object_keypoint[TRAJ_IDX, self.elapsed_steps]
                .detach()
                .cpu()
                .numpy()
            )
            target_keypoint_3d_msg = Point()
            target_keypoint_3d_msg.x = target_keypoint_3d_np[0]
            target_keypoint_3d_msg.y = target_keypoint_3d_np[1]
            target_keypoint_3d_msg.z = target_keypoint_3d_np[2]
            self.target_keypoint_3d_pub.publish(target_keypoint_3d_msg)

            if INCLUDE_KEYPOINT_3D_2:
                target_keypoint_3d_2_np = (
                    self.target_object_keypoint_2[TRAJ_IDX, self.elapsed_steps]
                    .detach()
                    .cpu()
                    .numpy()
                )
                target_keypoint_3d_2_msg = Point()
                target_keypoint_3d_2_msg.x = target_keypoint_3d_2_np[0]
                target_keypoint_3d_2_msg.y = target_keypoint_3d_2_np[1]
                target_keypoint_3d_2_msg.z = target_keypoint_3d_2_np[2]
                self.target_keypoint_3d_2_pub.publish(target_keypoint_3d_2_msg)

            # HACK: Slow down the rate of going through the trajectory
            self.REAL_ELAPSED_STEPS += 1
            # if self.REAL_ELAPSED_STEPS % 1 == 0:
            # if self.REAL_ELAPSED_STEPS % 2 == 0:
            # if self.REAL_ELAPSED_STEPS % 3 == 0:
            if self.REAL_ELAPSED_STEPS % 4 == 0:
                self.elapsed_steps += 1

            done_time = time.perf_counter()
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


if __name__ == "__main__":
    node = PolicyROSNode(
        traj_file=TRAJECTORY_FILE,
    )
    node.run()
