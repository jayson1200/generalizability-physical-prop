import numpy as np
import sapien
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from scipy.spatial.transform import Rotation as R

from rsl_code.registration import register_env
from tasks.env_kitchen_scene import EnvKitchenSceneEnv

max_episode_steps = 300


@register_env("EnvDrawer-v0", max_episode_steps=max_episode_steps)
class EnvDrawerEnv(EnvKitchenSceneEnv):
    def __init__(self, *args, start_limit=0.15, **kwargs):

        super().__init__(
            *args,
            start_limit=start_limit,
            category="drawer",
            selected_objs=[],
            selected_arts=[],
            exclude_objs=[],
            random_objs=["drawer_0"],
            max_episode_steps=max_episode_steps,
            individual_fingers=False,
            **kwargs
        )

    def _load_objects(self):
        loader = self.scene.create_urdf_loader()
        loader.scale = 0.5
        loader.set_density(1000)
        builder = loader.parse("assets/objects/drawer/mobility.urdf")[
            "articulation_builders"
        ][0]
        builder.disable_self_collision = True
        x, y, z, w = R.from_euler("z", np.pi).as_quat()
        builder.initial_pose = sapien.Pose(p=[0.6, -6.5, 0.3], q=[w, x, y, z])
        self.drawer = builder.build(name="drawer_0")
        self.all_articulations["drawer_0"] = self.drawer

    @property
    def _default_human_render_camera_configs(self):
        room_camera_pose = sapien_utils.look_at([2.5, -6, 1.0], [0, -6.5, 0.7])
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
        success = self.all_articulations["drawer_0"].qpos[:, -1] > 0.2
        return success
