import sapien
import torch
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from scipy.spatial.transform import Rotation as R

from rsl_code.registration import register_env
from tasks.env_kitchen_scene import EnvKitchenSceneEnv

max_episode_steps = 300


@register_env("EnvScissors-v0", max_episode_steps=max_episode_steps)
class EnvScissorsEnv(EnvKitchenSceneEnv):
    def __init__(self, *args, start_limit=0.03, **kwargs):

        super().__init__(
            *args,
            start_limit=start_limit,
            category="scissors",
            selected_objs=[],
            selected_arts=[
                "kitchen_counter",
            ],
            exclude_objs=[
                "objects/frl_apartment_small_appliance_01",
            ],
            random_objs=["scissors_0"],
            max_episode_steps=max_episode_steps,
            individual_fingers=True,
            **kwargs
        )

    def _load_objects(self):
        loader = self.scene.create_urdf_loader()
        loader.scale = 0.2
        loader.fix_root_link = False
        loader.set_density(1000)
        builder = loader.parse("assets/objects/scissors/mobility.urdf")[
            "articulation_builders"
        ][0]
        builder.disable_self_collision = True
        builder.fix_root_link = False
        x, y, z, w = R.from_euler("z", 0).as_quat()
        builder.initial_pose = sapien.Pose(p=[-2.0, -0.9, 0.875], q=[w, x, y, z])
        self.drawer = builder.build(name="scissors_0")
        self.all_articulations["scissors_0"] = self.drawer

    @property
    def _default_human_render_camera_configs(self):
        room_camera_pose = sapien_utils.look_at([-1.6, -0.9, 1.4], [-2.0, -0.9, 1])
        return CameraConfig(
            "render_camera",
            room_camera_pose,
            512,
            512,
            1,
            0.01,
            100,
        )

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        super()._initialize_episode(env_idx, options)
        with torch.device(self.device):
            scissors = self.all_articulations["scissors_0"]
            scissors.set_qpos(torch.ones(1) * 0.7)
            scissors.set_qvel(torch.zeros(1))
            scissors.set_qf(torch.zeros(1))

    def is_success(self):
        return self.all_articulations["scissors_0"].qpos[:, 0] < 0.1  # radians
