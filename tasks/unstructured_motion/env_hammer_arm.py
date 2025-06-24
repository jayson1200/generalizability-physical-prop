import numpy as np
import sapien
import sapien.physx as physx
import torch
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils, structs
from scipy.spatial.transform import Rotation as R

from rsl_code.registration import register_env
from tasks.env_arm import EnvArm

max_episode_steps = 300


@register_env("EnvHammerArm-v0", max_episode_steps=max_episode_steps)
class EnvHammerArm3Env(EnvArm):
    def __init__(self, *args, start_limit=0.1, **kwargs):

        super().__init__(
            *args,
            start_limit=start_limit,
            category="hammer_arm",
            random_objs=["hammer_0"],
            max_episode_steps=max_episode_steps,
            individual_fingers=False,
            **kwargs
        )

    @property
    def _default_arm_config(self):
        return torch.tensor(
            np.deg2rad([-35.4, 21.89, -1.45, -102.38, 1.11, 54.53, 0]),
            device=self.device,
        )

    @property
    def min_z(self):
        return 0.27

    def _load_objects(self):
        import os

        current_dir = os.getcwd()
        builder = self.scene.create_actor_builder()
        builder.add_visual_from_file(
            os.path.join(current_dir, "assets/objects/hammer_2/hammer_2.obj")
        )
        material = physx.PhysxMaterial(1.0, 1.0, 0.0)
        builder.add_multiple_convex_collisions_from_file(
            os.path.join(current_dir, "assets/objects/hammer_2/hammer_2.obj"),
            material=material,
        )
        mat = np.array(
            [
                [0.90477828, -0.42570829, -0.01219521, 0.79165334 - 0.3],
                [0.42582437, 0.90475775, 0.00932881, 0.03923106 - 0.4],
                [0.00706236, -0.01363352, 0.99988212, 0.22],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )

        x, y, z, w = R.from_euler(
            "xyz", [-np.pi / 15, 0, -np.pi / 4], degrees=False
        ).as_quat()
        rot = [w, x, y, z]
        pos = mat[:3, 3].copy()
        self.init_pos = pos

        builder.initial_pose = sapien.Pose(pos, rot)
        hammer = builder.build(name="hammer_0")
        hammer.set_mass(0.5)

        self.all_objects["hammer_0"] = hammer

        builder = self.scene.create_actor_builder()
        builder.add_cylinder_visual(radius=0.02, half_length=0.027)
        builder.add_cylinder_collision(radius=0.02, half_length=0.027)
        x, y, z, w = R.from_euler("xyz", [0, np.pi / 2, 0], degrees=False).as_quat()
        rot = [w, x, y, z]
        builder.initial_pose = sapien.Pose(list(pos)[:2] + [0.189], rot)
        self.cylinder = builder.build(name="cylinder")
        self.cylinder.set_mass(1.0)

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([0.5, -0.9, 0.5], [0.4, -0.4, 0.2])
        return CameraConfig(
            "render_camera",
            pose,
            512,
            512,
            1,
            0.01,
            100,
        )

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        super()._initialize_episode(env_idx, options)
        with torch.device(self.device):
            hammer = self.all_objects["hammer_0"]

            hammer_pose = hammer.pose
            rotation_matrix = (
                hammer_pose.to_transformation_matrix()[env_idx, :3, :3].cpu().numpy()
            )
            euler_angles = R.from_matrix(rotation_matrix).as_euler("xyz", degrees=False)
            hammer_angle_xy = euler_angles[:, 2]
            hammer_angle_xy -= np.pi / 2

            offset_distance = 0.18
            offset_x = offset_distance * np.cos(hammer_angle_xy)
            offset_y = offset_distance * np.sin(hammer_angle_xy)

            current_cylinder_pos = hammer_pose.p[env_idx].cpu().numpy()
            new_cylinder_pos = np.stack(
                [
                    current_cylinder_pos[:, 0] + offset_x,
                    current_cylinder_pos[:, 1] + offset_y,
                    current_cylinder_pos[:, 2],
                ],
                axis=1,
            )
            new_cylinder_pos[:, 2] = 0.19

            new_pose = structs.pose.Pose.create_from_pq(
                p=new_cylinder_pos, q=self.cylinder.initial_pose.q
            )
            self.cylinder.set_pose(new_pose)
