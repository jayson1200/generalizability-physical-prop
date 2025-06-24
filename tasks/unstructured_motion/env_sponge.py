import torch
from mani_skill.utils.scene_builder.robocasa.objects.objects import MJCFObject
from scipy.spatial.transform import Rotation as R

from rsl_code.registration import register_env
from tasks.env_kitchen_scene import EnvKitchenSceneEnv
from tasks.utils.fix_density import fix_density

max_episode_steps = 300


@register_env("EnvSponge-v0", max_episode_steps=max_episode_steps)
class EnvSpongeEnv(EnvKitchenSceneEnv):
    def __init__(self, *args, start_limit=0.08, **kwargs):

        self.travelled_distance = None
        super().__init__(
            *args,
            start_limit=start_limit,
            category="sponge",
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
            random_objs=["sponge_0"],
            max_episode_steps=max_episode_steps,
            individual_fingers=False,
            **kwargs
        )

    def _load_objects(self):
        import os

        current_dir = os.getcwd()
        obj_path = os.path.join(current_dir, "assets/objects/sponge/model.xml")
        obj_path = fix_density(obj_path)
        obj = MJCFObject(self.scene, "sponge", obj_path, scale=1.5)
        x, y, z, w = R.from_euler("x", 0).as_quat()
        obj.pos = [-2.1, -1.0, 0.892]
        obj.quat = [w, x, y, z]
        sponge = obj.build(list(range(self.num_envs))).actor
        self.all_objects["sponge_0"] = sponge

    def is_success(self):
        end = self.elapsed_steps >= self.max_episode_steps - 1
        success = self.travelled_distance > 0.3
        return end & success

    def _after_control_step(self):
        super()._after_control_step()

        mask = (self.last_pos[..., 2] < 0.92) & (self.last_pos[..., 2] > 0.88)
        new_pos = self.all_objects["sponge_0"].pose.p.clone()
        dist = (self.last_pos[..., :2] - new_pos[..., :2]).norm(dim=-1)
        self.travelled_distance[mask] += dist[mask]
        self.last_pos = new_pos

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        super()._initialize_episode(env_idx, options)

        with torch.device(self.device):
            if self.travelled_distance is None:
                self.travelled_distance = torch.zeros(
                    self.num_envs, dtype=torch.float32
                )
                self.last_pos = self.all_objects["sponge_0"].pose.p.clone()
            else:
                self.travelled_distance[env_idx] = 0
                self.last_pos[env_idx] = (
                    self.all_objects["sponge_0"].pose.p[env_idx].clone()
                )
