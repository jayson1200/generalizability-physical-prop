import numpy as np
import sapien
import sapien.physx as physx
import torch
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.scene_builder.robocasa.objects.objects import MJCFObject
from scipy.spatial.transform import Rotation as R

from real_world.table_constants import X_R_TC
from rsl_code.registration import register_env
from tasks.env_arm import EnvArm

max_episode_steps = 300


@register_env("EnvAppleArm-v0", max_episode_steps=max_episode_steps)
class EnvAppleArm(EnvArm):
    def __init__(self, *args, start_limit=0.10, **kwargs):

        super().__init__(
            *args,
            start_limit=start_limit,
            category="apple_arm",
            touch_keypoints=["apple"],
            random_objs=["apple_0", "plate_0"],
            max_episode_steps=max_episode_steps,
            individual_fingers=False,
            **kwargs
        )

    @property
    def _default_arm_config(self):
        return torch.tensor(
            np.deg2rad([-31.95 - 20, 40.95, -11.87, -97.48, 94.71, 91.45, -39.06 - 20]),
            device=self.device,
        )

    def _load_objects(self):
        apple_path = "assets/objects/apple/apple.obj"

        builder = self.scene.create_actor_builder()
        builder.add_visual_from_file(apple_path)
        material = physx.PhysxMaterial(1.0, 1.0, 0.0)
        builder.add_multiple_convex_collisions_from_file(apple_path, material=material)
        pos = X_R_TC[:3, 3].copy()
        pos[0] += 0.20
        pos[1] += 0.20
        pos[2] += 0.117

        mat = np.array(
            [
                [0.90477828, -0.42570829, -0.01219521, 0.79165334 - 0.2],
                [0.42582437, 0.90475775, 0.00932881, 0.03923106 - 0.2],
                [0.00706236, -0.01363352, 0.99988212, 0.2536],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )

        rot = [1, 0, 0, 0]
        pos = mat[:3, 3].copy()
        builder.initial_pose = sapien.Pose(pos, rot)
        apple = builder.build(name="apple_0")
        apple.set_mass(0.4)

        self.all_objects["apple_0"] = apple

        plate_path = "assets/objects/plate/plate.obj"

        builder = self.scene.create_actor_builder()
        builder.add_visual_from_file(plate_path)
        material = physx.PhysxMaterial(1.0, 1.0, 0.0)
        builder.add_multiple_convex_collisions_from_file(plate_path, material=material)
        pos = X_R_TC[:3, 3].copy()
        pos[0] += 0.20
        pos[1] += 0.20
        mat = np.array(
            [
                [1, 0, 0, 0.79165334 + 0.1],
                [0, 1, 0, 0.03923106 + 0.1],
                [0, 0, 1, 0.1654],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        rot = [1, 0, 0, 0]
        pos = mat[:3, 3].copy()
        builder.initial_pose = sapien.Pose(pos, rot)
        plate = builder.build(name="plate_0")
        plate.set_mass(0.1)

        self.all_objects["plate_0"] = plate

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([1.0, -1.0, 0.7], [0.6, -0.1, 0.2])
        return CameraConfig(
            "render_camera",
            pose,
            512,
            512,
            1,
            0.01,
            100,
        )

    def is_success(self):
        return self.all_objects["apple_0"].pose.p[:, 1] > 0.2
