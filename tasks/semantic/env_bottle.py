from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.scene_builder.robocasa.objects.objects import MJCFObject

from rsl_code.registration import register_env
from tasks.env_kitchen_scene import EnvKitchenSceneEnv
from tasks.utils.fix_density import fix_density

max_episode_steps = 300


@register_env("EnvBottle-v0", max_episode_steps=max_episode_steps)
class EnvBottleEnv(EnvKitchenSceneEnv):
    def __init__(self, *args, start_limit=0.10, **kwargs):

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
            **kwargs
        )

    def _load_objects(self):
        import os

        current_dir = os.getcwd()
        bottle_path = os.path.join(
            current_dir, "assets/objects/bottled_water/model.xml"
        )

        bottle_path = fix_density(bottle_path)

        bottle_obj = MJCFObject(self.scene, "bottle", bottle_path, scale=1.5)
        bottle_obj.pos = [-2.1, -2.0, 0.974]
        bottle = bottle_obj.build(list(range(self.num_envs))).actor
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
