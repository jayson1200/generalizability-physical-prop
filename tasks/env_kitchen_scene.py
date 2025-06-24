from typing import Optional, Union

import numpy as np
import sapien
import torch
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils, structs
from mani_skill.utils.geometry.geometry import transform_points
from mani_skill.utils.scene_builder import SceneBuilder
from mani_skill.utils.scene_builder.registration import REGISTERED_SCENE_BUILDERS
from mani_skill.utils.structs.types import (
    DefaultMaterialsConfig,
    GPUMemoryConfig,
    SceneConfig,
    SimConfig,
)
from scipy.spatial.transform import Rotation as R

from tasks.utils import replicacad_scene_builder
from utils.allegro import AllegroHand


class EnvKitchenSceneEnv(BaseEnv):
    SUPPORTED_ROBOTS = ["allegro_hand"]
    agent: Union[AllegroHand]

    def __init__(
        self,
        *args,
        robot_uids="allegro_hand",
        category="",
        method="gemini",
        split="train1",
        is_train=True,
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
        scene_builder_cls: Union[str, SceneBuilder] = "CustomReplicaCAD",
        num_envs=1,
        selected_objs=[],
        selected_arts=[],
        exclude_objs=[],
        random_objs=[],
        touch_keypoints=None,
        max_episode_steps=300,
        **kwargs,
    ):
        self.selected_objs = selected_objs
        self.selected_arts = selected_arts
        self.exclude_objs = exclude_objs
        self.random_objs = random_objs

        self.load_keypoints = load_keypoints
        self.load_trajectories = load_trajectories
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
        if self.method == "replay":
            self.keypoint_file = f"data/gemini/{category}/{split}/keypoints.pt"
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

        if self.method == "replay":
            self.trajectories_file = f"data/gemini/{category}/{split}/trajectories.pt"
            self.object_poses_file = f"data/gemini/{category}/test/object_poses.pt"

        self.action_avg = None
        self.max_episode_steps = max_episode_steps

        # Initialize keypoint mappings
        self.keypoint_mappings = {}

        if isinstance(scene_builder_cls, str):
            scene_builder_cls = REGISTERED_SCENE_BUILDERS[
                scene_builder_cls
            ].scene_builder_cls
        self.scene_builder: SceneBuilder = scene_builder_cls(self)
        self.build_config_idxs = [0] * num_envs
        self.init_config_idxs = [0] * num_envs

        super().__init__(
            *args,
            robot_uids=robot_uids,
            reconfiguration_freq=0,
            num_envs=num_envs,
            **kwargs,
        )

    def reset(self, seed=None, options=None):
        if options is None:
            options = dict(reconfigure=False)
        self._set_episode_rng(seed, options.get("env_idx", torch.arange(self.num_envs)))
        if "reconfigure" in options and options["reconfigure"]:
            self.build_config_idxs = options.get(
                "build_config_idxs", self.build_config_idxs
            )
            self.init_config_idxs = options.get("init_config_idxs", None)
        else:
            assert (
                "build_config_idxs" not in options
            ), "options dict cannot contain build_config_idxs without reconfigure=True"
            self.init_config_idxs = options.get(
                "init_config_idxs", self.init_config_idxs
            )
        if isinstance(self.build_config_idxs, int):
            self.build_config_idxs = [self.build_config_idxs]
        if isinstance(self.init_config_idxs, int):
            self.init_config_idxs = [self.init_config_idxs]
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
        self.action_penalty = -0.001 * (action**2).sum(dim=-1)

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

    def step(self, action, *args, **kwargs):
        self.cur_step += 1

        if isinstance(action, np.ndarray):
            action = torch.from_numpy(action)
        action = action.clone()
        action = action.to(self.device)
        action = torch.clamp(action, min=-1, max=1)

        residual_limit = torch.tensor(
            [0.1, 0.1, 0.1, 1.0, 1.0, 1.0], device=self.device
        )
        action[:, :6] = (
            action[:, :6]
            * 2
            * residual_limit
            / (self.limits[:6, 1] - self.limits[:6, 0])
        )

        if self.individual_fingers:
            action[:, 6:] *= 2
            action[:, [6, 7, 8, 9]] = action[:, [6]]
            action[:, [10, 11, 12, 13]] = action[:, [7]]
            action[:, [14, 15, 16, 17]] = action[:, [8]]
            action[:, [18, 19, 20, 21]] = action[:, [9]]
        else:
            action[:, 6:] = action[:, 6][..., None] * 2

        total_action = action
        total_action[:, 6:] += self.normalized_start_qpos[..., 6:]

        if self.load_trajectories:
            if self.use_wrist:
                normalized_wrist_pos = self.cur_traj["hand"][
                    torch.arange(self.num_envs), self.elapsed_steps
                ].clone()
            else:
                normalized_wrist_pos = self.cur_traj["hand"][
                    torch.arange(self.num_envs), 0
                ].clone()
        else:
            normalized_wrist_pos = torch.zeros((self.num_envs, 6), device=self.device)

        normalized_wrist_pos = (
            (normalized_wrist_pos - self.limits[:6, 0])
            / (self.limits[:6, 1] - self.limits[:6, 0])
        ) * 2 - 1
        total_action[:, :6] += normalized_wrist_pos

        self.action_avg = self.action_avg * self.gamma + total_action * (1 - self.gamma)

        weight = self.gamma ** (self.elapsed_steps + 1)
        total_action = self.action_avg + self.normalized_start_qpos * weight[..., None]

        total_action = torch.clamp(total_action, min=-1.0, max=1.0)

        obs, reward, terminated, truncated, info = super().step(
            total_action, *args, **kwargs
        )

        truncated |= self.episode_length_buf >= self.max_episode_steps - 1
        self.episode_length_buf += 1

        self.trans_limit = (
            self.start_limit
            + (self.end_limit - self.start_limit) * self.cur_step / self.total_steps
        )
        terminated |= (self.trans_dist > self.trans_limit).any(dim=1)

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

        self.limits = []
        for control in self.agent.controller.controllers.values():
            self.limits.append(control._get_joint_limits())
        self.limits = np.concatenate(self.limits, axis=0)
        self.limits = torch.tensor(self.limits).to(self.device)

    def _load_scene(self, options: dict):
        if self.scene_builder.build_configs is not None:
            self.scene_builder.build(
                (
                    self.build_config_idxs
                    if self.build_config_idxs is not None
                    else self.scene_builder.sample_build_config_idxs()
                ),
                selected_objs=None if self.all_objects else self.selected_objs,
                selected_arts=None if self.all_objects else self.selected_arts,
                exclude_objs=self.exclude_objs,
                build_background=self.build_background,
            )
        else:
            self.scene_builder.build(
                selected_objs=None if self.all_objects else self.selected_objs,
                selected_arts=None if self.all_objects else self.selected_arts,
                exclude_objs=self.exclude_objs,
                build_background=self.build_background,
            )

        self.all_objects = {}
        for obj in self.scene_builder.movable_objects.values():
            name = obj.name.split("_", 1)[1].split("-", 1)[0]
            self.all_objects[name] = obj

        self.all_articulations = {}
        for art in self.scene_builder.articulations.values():
            name = art.name.split("_", 1)[1].split("-", 1)[0]
            self.all_articulations[name] = art

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
        self._compute_keypoint_mappings()

    def _load_objects(self):
        pass

    def _load_lighting(self, options: dict):
        if self.scene_builder.builds_lighting:
            return
        return super()._load_lighting(options)

    def _load_targets(self):
        """Load target poses for objects and wrists."""
        self.cur_traj = {}
        if self.load_trajectories:
            self.trajectories = torch.load(self.trajectories_file, weights_only=False)
            self.object_poses = torch.load(self.object_poses_file, weights_only=False)

            num_trajs = self.trajectories["hand"].shape[0]
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

    def evaluate(self):
        """Evaluate the current episode."""
        return {
            "success": torch.zeros(self.num_envs, device=self.device, dtype=bool),
            "fail": torch.zeros(self.num_envs, device=self.device, dtype=bool),
        }

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
        obs = {
            "tip_poses": self.agent.tip_poses.clone().reshape(self.num_envs, -1),
            "action_avg": self.action_avg.clone(),
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
        room_camera_pose = sapien_utils.look_at([-1.3, -1.0, 1.3], [-2.0, -1.0, 1])
        return CameraConfig(
            "render_camera",
            room_camera_pose,
            512,
            512,
            1,
            0.01,
            100,
        )

    def _set_start_joints(self, env_idx):
        """Set the initial joint positions for the environment."""
        env_traj_idx = self.traj_idx[env_idx]
        start_qpos = torch.zeros(
            (env_traj_idx.shape[0], self.agent.action_space.shape[-1]),
            dtype=torch.float32,
            device=self.device,
        )
        start_qpos[:, 6:] = self.limits[6:, 0]
        if self.load_trajectories:
            start_qpos[:, :6] = self.cur_traj["hand"][env_idx, 0]
        start_qpos = torch.clamp(
            start_qpos, min=self.limits[:, 0], max=self.limits[:, 1]
        )

        self.agent.reset(start_qpos)
        self.agent.robot.set_pose(sapien.Pose())
        self.normalized_start_qpos[env_idx] = (
            (start_qpos - self.limits[:, 0]) / (self.limits[:, 1] - self.limits[:, 0])
        ) * 2 - 1

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        """Initialize an episode for the given environment indices."""
        # Using torch.device context manager to auto create tensors
        # on CPU/CUDA depending on self.device, the device the env runs on
        with torch.device(self.device):
            if self.scene_builder.init_configs is not None:
                self.scene_builder.initialize(
                    env_idx,
                    (
                        self.init_config_idxs
                        if self.init_config_idxs is not None
                        else self.scene_builder.sample_init_config_idxs()
                    ),
                )
            else:
                self.scene_builder.initialize(env_idx)
            # Initialize tensors if this is the first time
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
                # self.best_tip_dist = torch.ones(self.num_envs, dtype=torch.float32) * 100
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
                    p = p + torch.rand_like(p) * torch.tensor(
                        [0.1, 0.1, 0], device=p.device
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
