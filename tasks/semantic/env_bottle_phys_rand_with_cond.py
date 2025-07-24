from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.scene_builder.robocasa.objects.objects import MJCFObject

from rsl_code.registration import register_env
from tasks.env_kitchen_scene import EnvKitchenSceneEnv
from tasks.utils.fix_density import fix_density
from tasks.env_kitchen_scene_with_obj_randomization import EnvKitchenSceneEnvWithObjRandomization
from mani_skill.utils.structs.actor import Actor
from ..utils.utils import VecMJCFHandler
from ..distributions import *

import mujoco
import numpy as np

max_episode_steps = 300
num_physical_props = 6

MAX_FRICTION = 2
MAX_DENSITY = 25000
MAX_SCALE = 7
MAX_COM_DIM_VALUE = 1


@register_env("EnvBottle-v0-phys-rand-with-cond", max_episode_steps=max_episode_steps, model_scale=1.5, distribution="uniform(loc=0.0, scale=1.0)")
class EnvBottleEnvPhysRandWithCond(EnvKitchenSceneEnvWithObjRandomization):
    def __init__(self, *args, distribution="uniform(loc=0.0, scale=1.0)", start_limit=0.10, model_scale = 1.5, **kwargs): 
        self.m = 0.105 / 1.5
        self.bias = -0.1
        self.param_max_values = np.array([MAX_FRICTION,
                                          MAX_DENSITY,
                                          MAX_SCALE,
                                          0,
                                          0,
                                          MAX_COM_DIM_VALUE]).reshape(1, -1)
        
        self.max_friction = MAX_FRICTION
        self.max_density = MAX_DENSITY
        self.max_scale = MAX_SCALE
        self.max_com_dim_value = MAX_COM_DIM_VALUE
        num_envs = kwargs.get("num_envs")

        self.distribution = eval(distribution)
        self.samples = self.distribution.rvs(size=(num_envs, num_physical_props)) 
        self.params = self.samples * self.param_max_values
        self.params[:, -1] -= 0.5

        self.scales = self.params[:, 2]

        self.offsets = np.zeros((num_envs, 3)) 
        self.offsets[:, 2] = self.m * self.scales + self.bias

        
        super().__init__(
            *args,
            start_limit=start_limit,
            category="bottle",
            selected_objs=[],
            selected_arts=["kitchen_counter"],
            touch_keypoints=["bottle"],
            exclude_objs=[
                "objects/frl_apartment_small_appliance_01",
                "objects/frl_apartment_bowl_06",
                "objects/frl_apartment_cup_01",
                "objects/frl_apartment_cup_02",
                "objects/frl_apartment_cup_03",
                "objects/frl_apartment_sponge_dish",
            ],
            random_objs=["bottle_0"],
            max_episode_steps=max_episode_steps,
            individual_fingers=False,
            target_obj_name="bottle",
            pose_offset_per_environment=self.offsets,
            physical_prop_embed = self.samples,
            include_physical_embeds=True,
            **kwargs
        )


    def _load_objects(self):
        import os

        current_dir = os.getcwd()
        bottle_path = os.path.join(
            current_dir, "assets/objects/bottled_water/model.xml"
        )

        # bottle_path = fix_density(bottle_path)

        bottles = []
        with VecMJCFHandler(bottle_path, self.params) as files:
            for i, file_path in enumerate(files):
                bottle = MJCFObject(self.scene, 
                                    f"bottle_{i}", 
                                    file_path, 
                                    scale=self.scales[i])
                bottle.pos = [-2.1, -2.0, 0.974]
                
                # apple = apple.build(list(range(self.num_envs))).actor
                bottle = bottle.build([i]).actor
                self.remove_from_state_dict_registry(bottle)                   
                bottles.append(bottle)
        
        bottle = Actor.merge(bottles, name="bottle")
        self.add_to_state_dict_registry(bottle) 
        self.all_objects["bottle_0"] = bottle

    @property
    def _default_human_render_camera_configs(self):
        room_camera_pose = sapien_utils.look_at([-1.0, -1.5, 1.5], [-2.5, -1.7, 1])
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
        return self.all_objects["bottle_0"].pose.p[:, 1] > -1.4
