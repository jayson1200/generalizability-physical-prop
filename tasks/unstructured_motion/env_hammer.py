import numpy as np
import sapien
import torch
from mani_skill.utils import structs
from scipy.spatial.transform import Rotation as R

from rsl_code.registration import register_env
from tasks.env_kitchen_scene import EnvKitchenSceneEnv

max_episode_steps = 300


@register_env("EnvHammer-v0", max_episode_steps=max_episode_steps)
class EnvHammerEnv(EnvKitchenSceneEnv):
    def __init__(self, *args, start_limit=0.08, **kwargs):
        self.phase = None
        super().__init__(
            *args,
            start_limit=start_limit,
            category="hammer",
            selected_objs=[],
            selected_arts=["kitchen_counter"],
            exclude_objs=[
                "objects/frl_apartment_small_appliance_01",
                "objects/frl_apartment_bowl_06",
                "objects/frl_apartment_cup_01",
                "objects/frl_apartment_cup_02",
                "objects/frl_apartment_cup_03",
                "objects/frl_apartment_sponge_dish",
            ],
            random_objs=["hammer_0"],
            max_episode_steps=max_episode_steps,
            individual_fingers=False,
            **kwargs,
        )

    def _load_objects(self):
        obj_path = f"assets/objects/hammer/hammer.glb"
        material = sapien.pysapien.physx.PhysxMaterial(
            static_friction=1, dynamic_friction=1, restitution=0
        )
        builder = self.scene.create_actor_builder()
        builder.add_visual_from_file(obj_path, scale=np.ones(3) * 0.15)
        builder.add_multiple_convex_collisions_from_file(
            obj_path, material=material, decomposition="cocad", scale=np.ones(3) * 0.15
        )
        x, y, z, w = (
            R.from_euler("z", np.pi / 2) * R.from_euler("x", np.pi / 2)
        ).as_quat()
        builder.initial_pose = sapien.Pose(p=[-2, -0.8, 1.02], q=[w, x, y, z])
        hammer = builder.build(name="hammer_0")
        self.all_objects["hammer_0"] = hammer

    def is_success(self):
        end = self.elapsed_steps >= self.max_episode_steps - 1
        success = self.phase >= 6
        return end & success

    def _after_control_step(self):
        super()._after_control_step()

        obj = self.all_objects["hammer_0"]
        begin = self.elapsed_steps < 20
        self.highest = torch.maximum(self.highest, obj.pose.p[..., 2])
        self.lowest = torch.minimum(self.lowest, obj.pose.p[..., 2])

        above_upper = obj.pose.p[..., 2] > self.lowest + 0.05
        under_lower = obj.pose.p[..., 2] < self.highest - 0.05

        enter_upper = above_upper & (self.phase % 2 == 0) & ~begin
        enter_lower = under_lower & (self.phase % 2 == 1) & ~begin

        self.highest[enter_upper] = obj.pose.p[enter_upper, 2]
        self.lowest[enter_lower] = obj.pose.p[enter_lower, 2]

        self.phase[enter_upper] += 1
        self.phase[enter_lower] += 1

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        super()._initialize_episode(env_idx, options)

        with torch.device(self.device):
            if self.phase is None:
                self.phase = torch.zeros(self.num_envs, dtype=torch.float32)
                self.highest = torch.zeros(self.num_envs, dtype=torch.float32) + 1.02
                self.lowest = torch.zeros(self.num_envs, dtype=torch.float32) + 0.98
            else:
                self.phase[env_idx] = 0
                self.highest[env_idx] = 1.02
                self.lowest[env_idx] = 0.98
