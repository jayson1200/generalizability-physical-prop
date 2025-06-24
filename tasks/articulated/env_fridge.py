import numpy as np
import sapien
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils

from rsl_code.registration import register_env
from tasks.env_kitchen_scene import EnvKitchenSceneEnv

max_episode_steps = 300


@register_env("EnvFridge-v0", max_episode_steps=max_episode_steps)
class EnvFridgeEnv(EnvKitchenSceneEnv):
    def __init__(self, *args, start_limit=0.2, **kwargs):

        super().__init__(
            *args,
            start_limit=start_limit,
            category="fridge",
            selected_objs=[],
            selected_arts=["fridge"],
            exclude_objs=[],
            max_episode_steps=max_episode_steps,
            individual_fingers=False,
            # random_objs=['fridge'],
            **kwargs
        )

    def _load_objects(self):
        obj = self.all_articulations["fridge"]

    @property
    def _default_human_render_camera_configs(self):
        room_camera_pose = sapien_utils.look_at([0.5, -2.5, 2], [-2, -3.0, 1])
        return CameraConfig(
            "render_camera",
            room_camera_pose,
            512,
            512,
            1,
            0.01,
            100,
        )

    def is_success(self):
        return self.all_articulations["fridge"].qpos[:, 0] > np.pi / 3
