from typing import Optional, Union

import numpy as np
import pytorch_kinematics as pk
import sapien
import torch
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils, structs
from mani_skill.utils.geometry.geometry import transform_points
from mani_skill.utils.geometry.rotation_conversions import (
    matrix_to_axis_angle,
    matrix_to_quaternion,
    quaternion_to_matrix,
)
from mani_skill.utils.structs.types import (
    DefaultMaterialsConfig,
    GPUMemoryConfig,
    SceneConfig,
    SimConfig,
)
from scipy.spatial.transform import Rotation as R

from real_world.table_constants import X_R_TC
from utils.kuka_allegro import KukaAllegro


def normalize(x: torch.Tensor, low: torch.Tensor, high: torch.Tensor) -> torch.Tensor:
    # (-1, 1) => (low, high)
    assert x.ndim == 2, f"x.ndim: {x.ndim}"
    N, D = x.shape
    assert (
        low.shape == high.shape == (D,)
    ), f"low.shape: {low.shape}, high.shape: {high.shape}, D: {D}"
    return (x - low) / (high - low) * 2 - 1


def unnormalize(x: torch.Tensor, low: torch.Tensor, high: torch.Tensor) -> torch.Tensor:
    # (low, high) -> (-1, 1)
    assert x.ndim == 2, f"x.ndim: {x.ndim}"
    N, D = x.shape
    assert (
        low.shape == high.shape == (D,)
    ), f"low.shape: {low.shape}, high.shape: {high.shape}, D: {D}"
    return (x + 1) / 2 * (high - low) + low


def TC_to_R(
    pos: torch.Tensor, quat_wxyz: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_dims = pos.shape[:-1]
    assert pos.shape == (*batch_dims, 3) and quat_wxyz.shape == (
        *batch_dims,
        4,
    ), f"pos.shape: {pos.shape}, quat_wxyz.shape: {quat_wxyz.shape}"

    # Build a single 4×4 identity, then broadcast it across all batch dimensions
    T_TC_W = (
        torch.eye(4, device=pos.device, dtype=torch.float32)  # (4,4)
        .view(*(1,) * len(batch_dims), 4, 4)  # (1, …, 1, 4, 4)
        .expand(*batch_dims, 4, 4)  # (*batch_dims, 4, 4)
        .clone()  # make it writable
    )

    T_TC_W[..., :3, :3] = quaternion_to_matrix(quat_wxyz)
    T_TC_W[..., :3, 3] = pos

    T_R_TC = torch.tensor(X_R_TC).to(pos.device).float()  # (4,4)

    T_R_W = torch.bmm(
        T_R_TC[None].repeat_interleave(np.prod(batch_dims), dim=0),
        T_TC_W.reshape(-1, 4, 4),
    ).reshape(*batch_dims, 4, 4)

    new_pos = T_R_W[..., :3, 3]
    new_quat_wxyz = matrix_to_quaternion(T_R_W[..., :3, :3])
    assert new_pos.shape == (*batch_dims, 3) and new_quat_wxyz.shape == (
        *batch_dims,
        4,
    ), f"new_pos.shape: {new_pos.shape}, new_quat_wxyz.shape: {new_quat_wxyz.shape}"

    return new_pos, new_quat_wxyz


def control_ik(
    j_eef: torch.Tensor, dpose: torch.Tensor, damping: float = 0.5
) -> torch.Tensor:
    # solve damped least squares

    # Set controller parameters
    NUM_ENVS, NUM_EE_DIMS, NUM_JOINTS = j_eef.shape
    assert dpose.shape == (NUM_ENVS, NUM_EE_DIMS), f"dpose.shape: {dpose.shape}"

    j_eef_T = torch.transpose(j_eef, 1, 2)
    assert j_eef_T.shape == (
        NUM_ENVS,
        NUM_JOINTS,
        NUM_EE_DIMS,
    ), f"j_eef_T.shape: {j_eef_T.shape}"

    lmbda = torch.eye(NUM_EE_DIMS, device=j_eef.device) * (damping**2)
    assert lmbda.shape == (NUM_EE_DIMS, NUM_EE_DIMS), f"lmbda.shape: {lmbda.shape}"

    # u = J.T @ (J @ J.T + lambda)^-1 @ dpose
    u = torch.bmm(
        j_eef_T,
        torch.bmm(
            torch.inverse(torch.bmm(j_eef, j_eef_T) + lmbda[None]),
            dpose.unsqueeze(dim=-1),
        ),
    ).squeeze(dim=-1)
    assert u.shape == (NUM_ENVS, NUM_JOINTS), f"u.shape: {u.shape}"

    return u


NUM_XYZ = 3
NUM_XYZRPY = 6
NUM_ARM_JOINTS = 7
NUM_HAND_JOINTS = 16

CREATE_DEBUG_PYBULLET = False


class EnvArm(BaseEnv):
    SUPPORTED_ROBOTS = ["kuka_allegro"]
    agent: Union[KukaAllegro]

    def __init__(
        self,
        *args,
        robot_uids="kuka_allegro",
        category="",
        method="gemini",
        split="train1",
        all_objects=False,
        use_wrist=True,
        num_train_traj=1000000,
        total_steps=1000000,
        start_limit=0.15,
        load_keypoints=True,
        load_trajectories=True,
        visualize_keypoints=True,
        individual_fingers=False,
        build_background=False,
        gamma=0.9,
        reset_to=None,
        delta_action=False,
        num_envs=1,
        touch_keypoints=None,
        max_episode_steps=300,
        random_objs=None,
        **kwargs,
    ):
        self.load_keypoints = load_keypoints
        self.load_trajectories = load_trajectories

        self.random_objs = random_objs
        self.build_background = build_background

        self.total_steps = total_steps
        self.cur_step = 0
        self.category = category
        self.use_wrist = use_wrist
        self.num_train_traj = num_train_traj
        self.start_limit = start_limit
        self.end_limit = self.start_limit / 2
        self.gamma = gamma
        self.reset_to = reset_to
        self.delta_action = delta_action
        self.method = method
        self.visualize_keypoints = visualize_keypoints
        self.all_objects = all_objects
        self.individual_fingers = individual_fingers

        self.folder = f"data/{method}/{category}/{split}"

        self.keypoint_file = f"{self.folder}/keypoints.pt"
        if load_keypoints:
            self.keypoints = torch.load(self.keypoint_file, weights_only=False)
        else:
            self.keypoints = {}

        if touch_keypoints is None:
            self.touch_keypoints = list(self.keypoints.keys())
        else:
            self.touch_keypoints = touch_keypoints

        self.trajectories_file = f"{self.folder}/trajectories.pt"
        self.object_poses_file = f"{self.folder}/object_poses.pt"

        self.action_avg = None
        self.max_episode_steps = max_episode_steps

        # Initialize keypoint mappings
        self.keypoint_mappings = {}

        super().__init__(*args, robot_uids=robot_uids, num_envs=num_envs, **kwargs)

    def reset(self, seed=None, options=None):
        if options is None:
            options = dict(reconfigure=False)
        self._set_episode_rng(seed, options.get("env_idx", torch.arange(self.num_envs)))
        obs, info = super().reset(seed=seed, options=options)
        if seed is not None:
            if isinstance(seed, list):
                seed = seed[0]
            torch.manual_seed(seed)
            np.random.seed(seed)
            torch.cuda.manual_seed(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        info = self.compute_infos(info)
        return obs, info

    def compute_dense_reward(self, obs, action, info):
        target_obj_pos = self.get_target_positions()
        keypoint_pos = self.get_keypoint_positions()

        self.trans_dist = []
        for name in self.keypoints:
            self.trans_dist.append(
                torch.norm(keypoint_pos[name] - target_obj_pos[name], dim=-1)
            )
        if self.load_trajectories:
            self.trans_dist = torch.stack(self.trans_dist, dim=1)
        else:
            self.trans_dist = torch.zeros((self.num_envs, 1), device=self.device)

        self.sum_contact_force = self.compute_contact_forces()

        self.contact_force_reward = torch.exp(-1 / (self.sum_contact_force + 1e-6))
        self.object_tracking_reward = torch.exp(-(self.trans_dist.mean(dim=1) * 20))
        self.action_penalty = 0  # -0.001 * (action**2).sum(dim=-1)

        reward = (
            self.object_tracking_reward
            + self.contact_force_reward
            + self.action_penalty
        )
        return reward

    def _compute_keypoint_mappings(self):
        """Pre-compute mappings between keypoints and their corresponding links/objects."""
        for keypoint in self.keypoints:
            obj_name = self.keypoints[keypoint].obj_name
            if obj_name in self.all_objects:
                self.keypoint_mappings[keypoint] = self.all_objects[obj_name]
            elif obj_name in self.all_articulations:
                link_name = self.keypoints[keypoint].link_name
                for link in self.all_articulations[obj_name].links:
                    if link.name == link_name:
                        self.keypoint_mappings[keypoint] = link
                        break

    def get_keypoint_positions(self, env_idx: Optional[torch.Tensor] = None):
        """Get the positions of all keypoints."""
        res = {}
        for name in self.keypoints:
            if name not in self.keypoint_mappings:
                continue
            obj = self.keypoint_mappings[name]
            pos = self.keypoints[name].point[self.traj_idx]
            if env_idx is None:
                res[name] = transform_points(
                    obj.pose.to_transformation_matrix().clone(),
                    common.to_tensor(pos, device=self.device),
                )
            else:
                res[name] = transform_points(
                    obj.pose.to_transformation_matrix().clone(),
                    common.to_tensor(pos[env_idx], device=self.device),
                )
        return res

    def compute_infos(self, info):
        """Compute additional information for logging."""
        if "log" not in info:
            info["log"] = {}

        logs = info["log"]
        keypoint_pos = self.get_keypoint_positions()
        logs.update(
            {
                "sum_contact_force": self.sum_contact_force,
                "contact_force_reward": self.contact_force_reward,
                "object_tracking_reward": self.object_tracking_reward,
                "trans_dist": self.trans_dist,
                "trans_limit": self.trans_limit,
                "action_penalty": self.action_penalty,
                "success": self.elapsed_steps == (self.max_episode_steps - 1),
            }
        )
        for name in self.keypoints:
            logs[f"{name}_x"] = keypoint_pos[name][..., 0]
            logs[f"{name}_y"] = keypoint_pos[name][..., 1]
            logs[f"{name}_z"] = keypoint_pos[name][..., 2]
        info["log"] = logs
        return info

    def compute_normalized_dense_reward(self, obs, action, info):
        """Compute normalized dense reward."""
        return self.compute_dense_reward(obs=obs, action=action, info=info)

    @property
    def min_z(self):
        return 0.3

    def step(self, action, *args, **kwargs):
        """
        Execute one step in the environment.

        This method:
        1. Processes the action
        2. Applies the action to the environment
        3. Computes rewards and termination conditions
        """
        self.cur_step += 1

        # Convert action to tensor if needed
        if isinstance(action, np.ndarray):
            action = torch.from_numpy(action)
        action = action.to(self.device)
        action = torch.clamp(action, min=-1, max=1)

        gamma = 0.9
        self.action_avg = self.action_avg * gamma + action * (1 - gamma)
        action = self.action_avg

        """
        Controller: Joint PD Position Targets (NUM_ARM_JOINTS + NUM_HAND_JOINTS)
        Action: Delta Wrist Pose (NUM_XYZRPY) + Hand Joint PD Position Targets (NUM_HAND_JOINTS)
        """
        assert action.shape == (
            self.num_envs,
            NUM_ARM_JOINTS + NUM_HAND_JOINTS,
        ), f"action.shape: {action.shape}"

        normalized_arm_action = action[:, :NUM_XYZRPY].detach().clone()
        normalized_hand_action = (
            action[:, NUM_XYZRPY : NUM_XYZRPY + NUM_HAND_JOINTS].detach().clone()
        )

        ARM_LOWER_XYZ_LIMITS = torch.tensor([-0.1, -0.1, -0.1], device=self.device)
        ARM_UPPER_XYZ_LIMITS = torch.tensor([0.1, 0.1, 0.1], device=self.device)
        ARM_LOWER_RPY_LIMITS = torch.deg2rad(
            torch.tensor([-20, -20, -20], device=self.device)
        )
        ARM_UPPER_RPY_LIMITS = torch.deg2rad(
            torch.tensor([20, 20, 20], device=self.device)
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
        hand_action = normalized_hand_action * 2

        # Compute current wrist pose
        current_qpos = self.agent.robot.get_qpos()
        current_arm_qpos = current_qpos[:, :NUM_ARM_JOINTS]
        current_hand_qpos = current_qpos[:, NUM_ARM_JOINTS:]
        wrist_pose = self.arm_pk_chain.forward_kinematics(current_arm_qpos).get_matrix()
        assert wrist_pose.shape == (
            self.num_envs,
            4,
            4,
        ), f"wrist_pose.shape: {wrist_pose.shape}"

        # Target wrist pose
        target_wrist_pos = torch.tensor([0.6197, -0.3303, 0.4018], device=self.device)
        # target_wrist_quat = torch.tensor(
        #     [0.7189, 0.5713, 0.2716, 0.2882], device=self.device
        # )
        x, y, z, w = R.from_euler(
            "xyz", [0, -np.pi / 2, np.pi * 3 / 4], degrees=False
        ).as_quat()
        target_wrist_quat = torch.tensor([w, x, y, z], device=self.device)
        if self.load_trajectories:
            target_wrist_pos = self.cur_traj["hand"][
                torch.arange(self.num_envs), self.elapsed_steps, :3
            ]
            target_wrist_euler = self.cur_traj["hand"][
                torch.arange(self.num_envs), self.elapsed_steps, 3:
            ]
            target_wrist_quat = R.from_euler(
                "xyz", target_wrist_euler.cpu().numpy()
            ).as_quat()
            target_wrist_quat = np.concatenate(
                [target_wrist_quat[..., 3:], target_wrist_quat[..., :3]], axis=-1
            )
            target_wrist_quat = (
                torch.from_numpy(target_wrist_quat).to(self.device).to(torch.float32)
            )

        target_wrist_pose = (
            torch.eye(4)
            .unsqueeze(dim=0)
            .repeat_interleave(self.num_envs, dim=0)
            .to(self.device)
        )
        target_wrist_pose[:, :3, 3] = target_wrist_pos
        target_wrist_pose[:, :3, :3] = quaternion_to_matrix(target_wrist_quat)
        assert target_wrist_pose.shape == (
            self.num_envs,
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
            self.num_envs,
            NUM_XYZRPY,
        ), f"wrist_error.shape: {wrist_error.shape}"

        # Arm action: wrist pose targets (delta from target wrist pose)
        wrist_error += arm_action
        new_wrist_pos = wrist_pose[:, :3, 3] + wrist_error[:, :3]
        new_wrist_pos[:, 2] = torch.clamp(new_wrist_pos[:, 2], min=self.min_z)
        wrist_error[:, :3] = new_wrist_pos - wrist_pose[:, :3, 3]

        # Compute jacobian
        jacobian = self.arm_pk_chain.jacobian(current_arm_qpos)
        assert jacobian.shape == (
            self.num_envs,
            NUM_XYZRPY,
            NUM_ARM_JOINTS,
        ), f"jacobian.shape: {jacobian.shape}"
        pos_only = False
        if pos_only:
            jacobian = jacobian[:, 0:3]
            assert jacobian.shape == (
                self.num_envs,
                NUM_XYZ,
                NUM_ARM_JOINTS,
            ), f"jacobian.shape: {jacobian.shape}"

        # Compute delta arm joint position
        delta_arm_joint_pos = control_ik(
            j_eef=jacobian,
            dpose=wrist_error,
        )

        new_arm_qpos = current_arm_qpos + delta_arm_joint_pos
        if not self.load_trajectories:
            new_arm_qpos = current_arm_qpos

        assert new_arm_qpos.shape == (
            self.num_envs,
            NUM_ARM_JOINTS,
        ), f"new_arm_qpos.shape: {new_arm_qpos.shape}"

        # Hand action: Hand joint position targets (delta from default)
        new_hand_qpos = self.normalized_start_qpos[..., NUM_ARM_JOINTS:] + hand_action
        new_arm_qpos = normalize(
            new_arm_qpos,
            low=self.joint_limits[:NUM_ARM_JOINTS, 0],
            high=self.joint_limits[:NUM_ARM_JOINTS, 1],
        )

        new_qpos = torch.cat([new_arm_qpos, new_hand_qpos], dim=-1)
        assert new_qpos.shape == (
            self.num_envs,
            NUM_ARM_JOINTS + NUM_HAND_JOINTS,
        ), f"new_qpos.shape: {new_qpos.shape}"

        normalized_new_qpos = new_qpos

        normalized_new_qpos = torch.clamp(normalized_new_qpos, min=-1.0, max=1.0)

        # Execute step in parent environment
        new_qpos = unnormalize(
            normalized_new_qpos,
            low=self.joint_limits[:, 0],
            high=self.joint_limits[:, 1],
        )
        new_qpos = torch.clamp(
            new_qpos,
            min=self.conservative_joint_limits[:, 0],
            max=self.conservative_joint_limits[:, 1],
        )
        normalized_new_qpos = normalize(
            new_qpos, low=self.joint_limits[:, 0], high=self.joint_limits[:, 1]
        )

        obs, reward, terminated, truncated, info = super().step(
            normalized_new_qpos[..., self.agent.custom2loaded], *args, **kwargs
        )

        truncated |= self.episode_length_buf >= self.max_episode_steps - 1
        self.episode_length_buf += 1

        self.trans_limit = (
            self.start_limit
            + (self.end_limit - self.start_limit) * self.cur_step / self.total_steps
        )
        terminated |= (self.trans_dist > self.trans_limit).any(dim=1)

        # Update info
        info = self.compute_infos(info)

        return obs, reward, terminated, truncated, info

    def get_target_positions(self):
        """Get the target object pose for the current trajectory and step."""
        res = {}
        for name in self.keypoints:
            res[name] = self.cur_traj[name][
                torch.arange(self.num_envs), self.elapsed_steps
            ]
        return res

    def _load_agent(self, options: dict):
        """Load the agent (robot) into the environment."""
        super()._load_agent(options, sapien.Pose(p=[0, 0, 0], q=[1, 0, 0, 0]))
        self.joint_limits = []
        for control in self.agent.controller.controllers.values():
            self.joint_limits.append(control._get_joint_limits())
        self.joint_limits = np.concatenate(self.joint_limits, axis=0)
        self.joint_limits = torch.tensor(self.joint_limits).to(self.device)

        self.conservative_joint_limits = self.joint_limits.clone()
        self.conservative_joint_limits[:, 0] += np.deg2rad(10)
        self.conservative_joint_limits[:, 1] -= np.deg2rad(10)

    def add_table(self):
        """Add a table to the scene."""
        table_path = "assets/table/table.obj"
        material = sapien.pysapien.physx.PhysxMaterial(
            static_friction=0.5, dynamic_friction=0.5, restitution=0
        )
        table_builder = self.scene.create_actor_builder()
        table_builder.add_visual_from_file(table_path)
        table_builder.add_multiple_convex_collisions_from_file(
            table_path, material=material
        )

        pos = X_R_TC[:3, 3]
        quat_xyzw = R.from_matrix(X_R_TC[:3, :3]).as_quat()
        quat_wxyz = quat_xyzw[[3, 0, 1, 2]]

        table_builder.initial_pose = sapien.Pose(p=pos, q=quat_wxyz)
        self.table = table_builder.build_kinematic(name="table")

    def _load_scene(self, options: dict):
        from mani_skill import ASSET_DIR

        if self.build_background:
            bg_path = str(
                ASSET_DIR
                / f"scene_datasets/replica_cad_dataset/stages/frl_apartment_stage.glb"
            )
            builder = self.scene.create_actor_builder()
            bg_pose = sapien.Pose(p=[-3, 2, 0], q=[1, 0, 0, 0])
            builder.add_visual_from_file(bg_path)
            builder.add_nonconvex_collision_from_file(bg_path)
            builder.initial_pose = bg_pose

            self.bg = builder.build_static(name=f"scene_background")

        self.add_lights()
        with open(KukaAllegro.urdf_path, "rb") as f:
            urdf_str = f.read()
        self.arm_pk_chain = pk.build_serial_chain_from_urdf(
            urdf_str,
            end_link_name="base_link",
        ).to(device=self.device)
        self.all_objects = {}
        self.all_articulations = {}

        for joint in self.agent.robot.active_joints:
            for obj in joint._objs:
                damping = (0.3 + np.random.rand() * 2.7) * obj.damping
                stiffness = (0.3 + np.random.rand() * 2.7) * obj.stiffness
                obj.set_drive_properties(stiffness=stiffness, damping=damping)

        if self.visualize_keypoints:
            self.goals = {}
            self.tracked_points = {}
            for name in self.keypoints:
                builder = self.scene.create_actor_builder()
                material = sapien.pysapien.render.RenderMaterial(
                    transmission=0.7,
                    base_color=[1, 0, 0, 1],
                )
                builder.add_capsule_visual(
                    radius=0.02, half_length=0.0, material=material
                )
                builder.initial_pose = sapien.Pose()
                self.goals[name] = builder.build(name=f"goal_{name}")

                builder = self.scene.create_actor_builder()
                material = sapien.pysapien.render.RenderMaterial(
                    transmission=0.7,
                    base_color=[0, 1, 0, 1],
                )
                builder.add_capsule_visual(
                    radius=0.02, half_length=0.0, material=material
                )
                builder.initial_pose = sapien.Pose()
                self.tracked_points[name] = builder.build(name=f"keypoint_{name}")

        self._load_targets()
        self._load_objects()
        self.add_table()
        self._compute_keypoint_mappings()

        self.arm_pik = pk.PseudoInverseIK(
            self.arm_pk_chain,
            joint_limits=self.joint_limits[:NUM_ARM_JOINTS],
            early_stopping_any_converged=True,
            max_iterations=200,
            num_retries=1,
        )

    def add_table(self):
        """Add a table to the scene."""
        table_path = "assets/table/table.obj"
        material = sapien.pysapien.physx.PhysxMaterial(
            static_friction=1, dynamic_friction=1, restitution=0
        )
        table_builder = self.scene.create_actor_builder()
        table_builder.add_visual_from_file(table_path)
        table_builder.add_multiple_convex_collisions_from_file(
            table_path, material=material
        )

        pos = X_R_TC[:3, 3]
        quat_xyzw = R.from_matrix(X_R_TC[:3, :3]).as_quat()
        quat_wxyz = quat_xyzw[[3, 0, 1, 2]]

        table_builder.initial_pose = sapien.Pose(p=pos, q=quat_wxyz)
        self.table = table_builder.build_kinematic(name="table")

    def _load_objects(self):
        pass

    def _load_targets(self):
        """Load target poses for objects and wrists."""
        if self.load_trajectories:
            self.trajectories = torch.load(self.trajectories_file, weights_only=False)
            self.object_poses = torch.load(self.object_poses_file, weights_only=False)
            self.cur_traj = {}

            for name in self.trajectories.keys():
                self.trajectories[name] = self.trajectories[name].to(self.device)
                self.trajectories[name] = torch.cat(
                    [self.trajectories[name]] + [self.trajectories[name][:, -1:]] * 400,
                    dim=1,
                )
                self.trajectories[name] = self.trajectories[name].float()

                self.cur_traj[name] = torch.zeros(
                    (self.num_envs, *self.trajectories[name].shape[1:]),
                    device=self.device,
                    dtype=torch.float32,
                )
        else:
            self.trajectories = {}
            self.cur_traj = {}

    def evaluate(self):
        """Evaluate the current episode."""
        return {
            "success": torch.zeros(self.num_envs, device=self.device, dtype=bool),
            "fail": torch.zeros(self.num_envs, device=self.device, dtype=bool),
        }

    def add_lights(self):
        """Add lights to the scene."""
        self.scene.set_ambient_light([0.5, 0.5, 0.5])
        self.scene.add_directional_light([0, 0, -1], [0.9, 0.9, 0.9])

    def _get_obs_agent(self):
        obs = self.agent.get_proprioception()
        obs["qpos"] = obs["qpos"].clone()
        obs["qvel"] = obs["qvel"].clone()
        obs["qpos"] += torch.randn_like(obs["qpos"]) * 0.05
        obs["qvel"] += torch.randn_like(obs["qvel"]) * 0.05
        return obs

    @property
    def _default_arm_config(self):
        return torch.tensor(
            np.deg2rad([-31.95, 40.95, -11.87, -97.48, 94.71, 91.45, -39.06 - 20]),
            device=self.device,
        )

    def _set_start_joints(self, env_idx):
        """Set the initial joint positions for the environment."""
        env_traj_idx = self.traj_idx[env_idx]

        start_qpos = torch.zeros(
            (env_traj_idx.shape[0], NUM_ARM_JOINTS + NUM_HAND_JOINTS),
            dtype=torch.float32,
        )
        if self.load_trajectories:
            pos = self.cur_traj["hand"][env_idx, 0, :3]
            rot = self.cur_traj["hand"][env_idx, 0, 3:]
            rot = R.from_euler("xyz", rot.cpu().numpy()).as_quat()
            rot = np.concatenate([rot[..., 3:], rot[..., :3]], axis=-1)
            rot = torch.from_numpy(rot).to(self.device).to(torch.float32)
        else:
            pos = [
                0.6197,
                -0.3303,
                0.4018,
            ]
            rot = [
                1.36934039,
                0.06124581,
                0.8125544,
            ]

        tf = pk.Transform3d(
            pos=pos,
            rot=rot,
            device=self.device,
        )
        DEFAULT_ARM_CONFIG = np.deg2rad([0, 0, 0, -90, 0, 90, 0]).tolist()
        self.arm_pik.initial_config = torch.tensor(DEFAULT_ARM_CONFIG)[None]
        result = self.arm_pik.solve(tf)
        init_arm_config = result.solutions[:, 0]

        init_arm_config = self._default_arm_config
        init_hand_config = self.joint_limits[NUM_ARM_JOINTS:, 0].clone()
        # init_hand_config = self.joint_limits[NUM_ARM_JOINTS:, 0].clone() + (self.joint_limits[NUM_ARM_JOINTS:, 1].clone() - self.joint_limits[NUM_ARM_JOINTS:, 0].clone()) * .2
        # init_hand_config[-4] = 0.7
        # init_hand_config[-4] = 1.3960
        start_qpos[:, :NUM_ARM_JOINTS] = init_arm_config
        start_qpos[:, NUM_ARM_JOINTS:] = init_hand_config

        start_qpos = torch.clamp(
            start_qpos, min=self.joint_limits[:, 0], max=self.joint_limits[:, 1]
        )

        self.agent.reset(start_qpos[..., self.agent.custom2loaded])
        self.start_qpos[env_idx] = start_qpos
        self.normalized_start_qpos[env_idx] = normalize(
            start_qpos, low=self.joint_limits[:, 0], high=self.joint_limits[:, 1]
        )

    def _after_control_step(self):
        """Perform actions after the control step."""
        if self.gpu_sim_enabled:
            self.scene._gpu_fetch_all()

        if self.visualize_keypoints:
            target_positions = self.get_target_positions()
            keypoint_pos = self.get_keypoint_positions()
            for name in self.keypoints:
                pose = structs.pose.Pose.create_from_pq(
                    p=target_positions[name], q=[1, 0, 0, 0]
                )
                self.goals[name].set_pose(pose)

                pose = structs.pose.Pose.create_from_pq(
                    p=keypoint_pos[name], q=[1, 0, 0, 0]
                )
                self.tracked_points[name].set_pose(pose)

        if self.gpu_sim_enabled:
            self.scene._gpu_apply_all()

    def compute_contact_forces(self):
        """Compute contact forces between finger tips and the object."""
        finger_links = [
            "link_3.0_tip",
            "link_7.0_tip",
            "link_11.0_tip",
            "link_15.0_tip",
        ]
        sum_contact_force = torch.zeros(self.num_envs, device=self.device)
        self.contact_forces = {}

        for link_name in finger_links:
            link = self.agent.robot.find_link_by_name(link_name)
            contact_forces = torch.zeros((self.num_envs, 3), device=self.device)
            for keypoint in self.touch_keypoints:
                if keypoint not in self.keypoint_mappings:
                    continue
                obj = self.keypoint_mappings[keypoint]
                contact_forces += self.scene.get_pairwise_contact_forces(link, obj)

            contact_forces = torch.norm(contact_forces, dim=-1)
            self.contact_forces[link_name] = contact_forces
            sum_contact_force += torch.clamp(contact_forces, min=0.0, max=1.0)

        return sum_contact_force

    def _get_obs_extra(self, info):
        """Get extra observations."""
        tip_poses = self.agent.tip_poses.clone().reshape(self.num_envs, -1)
        tip_poses = torch.zeros_like(tip_poses)
        action_avg = self.action_avg.clone()
        action_avg += torch.randn_like(action_avg) * 0.05
        obs = {
            "tip_poses": tip_poses,
            "action_avg": action_avg,
        }

        keypoint_pos = self.get_keypoint_positions()
        for name in self.keypoints:
            obs[f"actual_{name}"] = keypoint_pos[name].clone()

        for i in list(range(10)) + [20, 30, 40, 50]:
            for name in self.cur_traj:
                traj = (
                    self.cur_traj[name][
                        torch.arange(self.num_envs), self.elapsed_steps + i
                    ]
                    .view(self.num_envs, -1)
                    .clone()
                )
                obs[f"target_{name}_{i}"] = traj

        for name in self.keypoints:
            traj = self.cur_traj[name][:, 0].view(self.num_envs, -1).clone()
            obs[f"initial_{name}"] = traj
        return obs

    @property
    def _default_sim_config(self):
        """Get default simulation configuration."""
        return SimConfig(
            sim_freq=120,
            control_freq=60,
            spacing=50,
            gpu_memory_config=GPUMemoryConfig(
                max_rigid_contact_count=self.num_envs * max(2048, self.num_envs) * 8,
                max_rigid_patch_count=self.num_envs * max(2048, self.num_envs) * 2,
                found_lost_pairs_capacity=2**27,
            ),
            scene_config=SceneConfig(
                gravity=np.array([0.0, 0.0, -9.81]),
                bounce_threshold=2.0,
                solver_position_iterations=8,
                solver_velocity_iterations=0,
            ),
            default_materials_config=DefaultMaterialsConfig(
                dynamic_friction=1,
                static_friction=1,
                restitution=0,
            ),
        )

    @property
    def _default_human_render_camera_configs(self):
        room_camera_pose = sapien_utils.look_at([-2.7, -1.0, 1.3], [-2.0, -1.0, 1])
        return CameraConfig(
            "render_camera",
            room_camera_pose,
            512,
            512,
            1,
            0.01,
            100,
        )

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        """Initialize an episode for the given environment indices."""
        # Using torch.device context manager to auto create tensors
        # on CPU/CUDA depending on self.device, the device the env runs on
        with torch.device(self.device):
            if self.action_avg is None:
                # First time env is called
                self.action_avg = torch.zeros(
                    (self.num_envs, self.agent.action_space.shape[-1]),
                    dtype=torch.float32,
                )
                self.cum_action = torch.zeros(
                    (self.num_envs, self.agent.action_space.shape[-1]),
                    dtype=torch.float32,
                )
                self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.int32)
                self.normalized_start_qpos = torch.zeros(
                    (self.num_envs, self.agent.action_space.shape[-1]),
                    dtype=torch.float32,
                )
                self.traj_idx = torch.zeros(self.num_envs, dtype=torch.int32)
                self.start_qpos = torch.zeros(
                    (self.num_envs, NUM_ARM_JOINTS + NUM_HAND_JOINTS),
                    dtype=torch.float32,
                )
                self.contact_force_reward = torch.zeros(
                    self.num_envs, dtype=torch.float32
                )
                self.sum_contact_force = torch.zeros(self.num_envs, dtype=torch.float32)
                self.action_penalty = torch.zeros(self.num_envs, dtype=torch.float32)

                # Initialize reward components
                self.object_tracking_reward = torch.zeros(
                    self.num_envs, dtype=torch.float32
                )
                self.trans_dist = torch.zeros(self.num_envs, dtype=torch.float32)
                self.trans_limit = self.start_limit
            else:
                # Reset tensors for the specified environments
                self.action_avg[env_idx] = torch.zeros_like(
                    self.agent.robot.qpos[env_idx], dtype=torch.float32
                )
                self.cum_action[env_idx] = torch.zeros_like(
                    self.agent.robot.qpos[env_idx], dtype=torch.float32
                )
                self.episode_length_buf[env_idx] = 0

            if self.load_trajectories:
                traj_range = 0, self.trajectories["hand"].shape[0]
            else:
                traj_range = 0, 1

            # Randomly select trajectories
            idxs = torch.randint(*traj_range, (len(env_idx),), dtype=torch.int32)
            self.traj_idx[env_idx] = idxs
            if self.reset_to is not None:
                self.traj_idx[env_idx] = self.reset_to

            for name in self.trajectories.keys():
                self.cur_traj[name][env_idx] = self.trajectories[name][
                    self.traj_idx[env_idx]
                ]
                if self.method == "replay":
                    random_idxs = torch.randint(
                        0,
                        self.trajectories["hand"].shape[0] // 2,
                        (len(env_idx),),
                        dtype=torch.int32,
                    )
                    if name == "hand":
                        self.cur_traj[name][env_idx, 1:] = self.trajectories[name][
                            random_idxs, 1:
                        ]
                    else:
                        self.cur_traj[name][env_idx] = self.trajectories[name][
                            random_idxs
                        ]

            self._set_start_joints(env_idx)
            for obj in self.all_objects.values():
                obj.set_pose(obj.initial_pose)

            for art in self.all_articulations.values():
                art.set_pose(art.initial_pose)
                art.set_qpos(
                    torch.zeros((len(env_idx), art.dof[0].item()), device=self.device)
                )
                art.set_qvel(
                    torch.zeros((len(env_idx), art.dof[0].item()), device=self.device)
                )
                art.set_qf(
                    torch.zeros((len(env_idx), art.dof[0].item()), device=self.device)
                )

            if self.load_trajectories:
                for name in self.random_objs:
                    if name in self.all_objects:
                        obj = self.all_objects[name]
                    else:
                        obj = self.all_articulations[name]
                    obj_pose = self.object_poses[name][self.traj_idx[env_idx]]
                    obj_pose = structs.pose.Pose.create_from_pq(
                        p=obj_pose[..., :3], q=obj_pose[..., 3:]
                    )
                    obj.set_pose(obj_pose)
            else:
                for name in self.random_objs:
                    if name in self.all_objects:
                        obj = self.all_objects[name]
                    else:
                        obj = self.all_articulations[name]

                    p = obj.initial_pose.p.clone()
                    p = p.repeat(len(env_idx), 1)
                    p = p + (torch.rand_like(p) * 2 - 1) * torch.tensor(
                        [0.05, 0.05, 0], device=p.device
                    )
                    w, x, y, z = obj.initial_pose.q.cpu().numpy()[0]
                    rot_batch = R.from_quat(np.tile([x, y, z, w], (len(env_idx), 1)))

                    random_rotation = R.from_euler(
                        "z",
                        np.random.uniform(-0.2, 0.2, size=len(env_idx)),
                    )

                    combined_rot = random_rotation * rot_batch

                    q = combined_rot.as_quat()
                    q = torch.tensor(q, device=p.device)
                    q = torch.cat([q[:, 3:], q[:, :3]], dim=1)

                    pose = structs.pose.Pose.create_from_pq(p=p, q=q)
                    obj.set_pose(pose)
