import numpy as np
import torch
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.scene_builder.robocasa.objects.objects import MJCFObject

import os
from ..utils.utils import VecMJCFHandler
from ..distributions import *
import sapien

from rsl_code.registration import register_env
from tasks.env_kitchen_scene import EnvKitchenSceneEnv
from tasks.utils.fix_density import fix_density
from tasks.env_kitchen_scene_with_obj_randomization import EnvKitchenSceneEnvWithObjRandomization
from mani_skill.utils.structs.actor import Actor
from scipy.spatial.transform import Rotation

from einops import repeat, einsum

max_episode_steps = 300
num_physical_props = 6

MAX_FRICTION = 2
MAX_DENSITY = 25000
MAX_SCALE = 5
MAX_COM_DIM_VALUE = 1


@register_env("EnvApple-v0-phys-rand-for-base",
              model_scale=1.5, 
              varying_params="[True, True, True, False, False, True]",
              non_varying_params_values="[0.9, 1000, 1.5, 0.0, 0.0, 0.5]",
              distribution="uniform(loc=0.0, scale=1.0)", 
              max_episode_steps=max_episode_steps)
class EnvAppleEnvPhysRandForBase(EnvKitchenSceneEnvWithObjRandomization):
    def __init__(self, 
                 *args, 
                 model_scale = 1.5, 
                 varying_params="[True, True, True, False, False, True]",
                 non_varying_params_values="[0.9, 1000, 1.5, 0.0, 0.0, 0.5]",
                 distribution="uniform(loc=0.0, scale=1.0)", 
                 start_limit=0.10, 
                 **kwargs):
        
        self.m = 0.055 / 1.5
        self.bias = -0.045
        
        self.varying_params = np.array(eval(varying_params)).astype(float).reshape(1, -1)
        self.param_max_values = np.array([MAX_FRICTION,
                                          MAX_DENSITY,
                                          MAX_SCALE,
                                          0,
                                          0,
                                          MAX_COM_DIM_VALUE], dtype=float).reshape(1, -1) * self.varying_params
        self.non_varying_params_values = np.array(eval(non_varying_params_values)).astype(float).reshape(1, -1)
        
        num_envs = kwargs.get("num_envs")

        self.distribution = eval(distribution)
        self.samples = self.distribution.rvs(size=(num_envs, num_physical_props)) 
        self.params = self.samples * self.param_max_values
        
        self.params += (1 - self.varying_params) * self.non_varying_params_values

        self.params[:, -1] -= 0.5

        self.scales = self.params[:, 2]

        self.offsets = np.zeros((num_envs, 3)) 
        self.offsets[:, 2] = self.m * self.scales + self.bias

        
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
            target_obj_name="apple",
            pose_offset_per_environment=self.offsets,
            physical_prop_embed = self.samples,
            **kwargs
        )


    def _load_objects(self):
        current_dir = os.getcwd()
        apple_path = os.path.join(current_dir, "assets/objects/apple/model.xml")
        # apple_path = fix_density(apple_path)

        apples = []
        with VecMJCFHandler(apple_path, self.params) as files:
            for i, file_path in enumerate(files):
                apple = MJCFObject(self.scene, 
                                    f"apple_{i}", 
                                    file_path, 
                                    scale=self.scales[i])
                apple.pos = [-2.0, -0.7, 0.924]
                
                apple = apple.build([i]).actor
                self.remove_from_state_dict_registry(apple)                   
                apples.append(apple)
        
        apple = Actor.merge(apples, name="apple")

        self.add_to_state_dict_registry(apple)
        
        self.all_objects["apple_0"] = apple

        # Cutting Board
        cutting_board_path = os.path.join(
            current_dir, "assets/objects/cutting_board/model.xml"
        )
        # cutting_board_path = fix_density(cutting_board_path)
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
