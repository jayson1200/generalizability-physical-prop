import numpy as np
import sapien
import sapien.physx as physx
import torch
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from scipy.spatial.transform import Rotation as R

from real_world.table_constants import X_R_TC
from rsl_code.registration import register_env
from tasks.env_arm import EnvArm

max_episode_steps = 300


@register_env("EnvBoxArm-v0", max_episode_steps=max_episode_steps)
class EnvBoxArmEnv(EnvArm):
    def __init__(self, *args, start_limit=0.15, **kwargs):

        super().__init__(
            *args,
            start_limit=start_limit,
            category="box_arm",
            touch_keypoints=["box"],
            random_objs=["box_0", "bottle_0"],
            max_episode_steps=max_episode_steps,
            individual_fingers=False,
            **kwargs
        )

    @property
    def min_z(self):
        return 0.25

    def _load_objects(self):
        import os

        current_dir = os.getcwd()
        bottle_path = os.path.join(
            current_dir, "assets/objects/starbucks_bottle/starbucks_bottle.obj"
        )

        builder = self.scene.create_actor_builder()
        builder.add_visual_from_file(bottle_path)
        material = physx.PhysxMaterial(1.0, 1.0, 0.0)
        builder.add_multiple_convex_collisions_from_file(bottle_path, material=material)

        mat = np.array(
            [
                [0.90477828, -0.42570829, -0.01219521, 0.79165334 + 0.1],
                [0.42582437, 0.90475775, 0.00932881, 0.03923106 + 0.1],
                [0.00706236, -0.01363352, 0.99988212, 0.2536],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )

        rot = [1, 0, 0, 0]
        pos = mat[:3, 3].copy()
        builder.initial_pose = sapien.Pose(pos, rot)
        bottle = builder.build(name="bottle_0")
        bottle.set_mass(0.2)
        self.all_objects["bottle_0"] = bottle

        box_path = os.path.join(current_dir, "assets/objects/snackbox/snackbox.obj")

        builder = self.scene.create_actor_builder()
        builder.add_visual_from_file(box_path)
        material = physx.PhysxMaterial(1.0, 1.0, 0.0)
        builder.add_multiple_convex_collisions_from_file(box_path, material=material)
        mat = np.array(
            [
                [1, 0, 0, 0.79165334 - 0.15],
                [0, 1, 0, 0.03923106 - 0.15],
                [0, 0, 1, 0.1931],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        x, y, z, w = R.from_matrix(mat[:3, :3]).as_quat()
        pos = mat[:3, 3].copy()
        builder.initial_pose = sapien.Pose(pos, [w, x, y, z])
        box = builder.build(name="box_0")
        box.set_mass(0.1)

        self.all_objects["box_0"] = box

    @property
    def _default_arm_config(self):
        return torch.tensor(
            np.deg2rad(
                [-31.95, 40.95 - 20, -11.87, -97.48, 94.71 - 90, 91.45 - 30, -39.06]
            ),
            device=self.device,
        )

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([0.8, -0.8, 0.7], [0.6, -0.1, 0.2])
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
        return self.all_objects["box_0"].pose.p[1] > 0.2
