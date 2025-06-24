import numpy as np
import torch
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.scene_builder.robocasa.objects.objects import MJCFObject
from scipy.spatial.transform import Rotation as R

from rsl_code.registration import register_env
from tasks.env_kitchen_scene import EnvKitchenSceneEnv
from tasks.utils.fix_density import fix_density

max_episode_steps = 300


@register_env("EnvApple-v0", max_episode_steps=max_episode_steps)
class EnvAppleEnv(EnvKitchenSceneEnv):
    def __init__(self, *args, start_limit=0.10, **kwargs):
        super().__init__(
            *args,
            start_limit=start_limit,
            category="apple",
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
            random_objs=["apple_0", "cutting_board_0"],
            max_episode_steps=max_episode_steps,
            individual_fingers=False,
            **kwargs
        )

    def _load_objects(self):
        import os

        current_dir = os.getcwd()
        apple_path = os.path.join(current_dir, "assets/objects/apple/model.xml")
        apple_path = fix_density(apple_path)

        apple = MJCFObject(self.scene, "apple", apple_path, scale=1.5)
        apple.pos = [-2.0, -0.7, 0.924]
        apple = apple.build(list(range(self.num_envs))).actor
        self.all_objects["apple_0"] = apple

        cutting_board_path = os.path.join(
            current_dir, "assets/objects/cutting_board/model.xml"
        )
        cutting_board_path = fix_density(cutting_board_path)
        cutting_board = MJCFObject(
            self.scene, "cutting_board", cutting_board_path, scale=1.5
        )
        cutting_board.pos = [-2.0, -1.1, 0.881]
        cutting_board = cutting_board.build(list(range(self.num_envs))).actor
        self.all_objects["cutting_board_0"] = cutting_board

    @property
    def _default_human_render_camera_configs(self):
        room_camera_pose = sapien_utils.look_at([-1.0, -1.0, 1.4], [-2.0, -1.0, 1])
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
        goal = self.all_objects["cutting_board_0"].pose.p.clone()
        goal[..., 2] += 0.07
        return torch.norm(self.all_objects["apple_0"].pose.p - goal, dim=-1) < 0.1
