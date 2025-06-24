import copy
import numpy as np
import torch

from mani_skill.agents.base_agent import BaseAgent
from mani_skill.agents.controllers import PDJointPosControllerConfig
from mani_skill.agents.registration import register_agent
from mani_skill.utils import sapien_utils
from mani_skill.utils.structs.pose import vectorize_pose


@register_agent()
class AllegroHand(BaseAgent):
    uid = "allegro_hand"
    urdf_path = "assets/allegro_hand/allegro_hand_right_dummy_joints.urdf"
    fix_root_link = True

    trans_joint_names = [
        "dummy_x_translation_joint",
        "dummy_y_translation_joint",
        "dummy_z_translation_joint",
    ]
    rot_joint_names = [
        "dummy_x_rotation_joint",
        "dummy_y_rotation_joint",
        "dummy_z_rotation_joint",
    ]
    finger_joint_names = [
        "joint_0.0",
        "joint_4.0",
        "joint_8.0",
        "joint_12.0",
        "joint_1.0",
        "joint_5.0",
        "joint_9.0",
        "joint_13.0",
        "joint_2.0",
        "joint_6.0",
        "joint_10.0",
        "joint_14.0",
        "joint_3.0",
        "joint_7.0",
        "joint_11.0",
        "joint_15.0",
    ]

    urdf_config = dict(
        _materials=dict(
            finger_tip=dict(static_friction=2.0, dynamic_friction=1.0, restitution=0.0)
        ),
        link={
            "link_3.0_tip": dict(
                material="finger_tip", patch_radius=0.1, min_patch_radius=0.1
            ),
            "link_7.0_tip": dict(
                material="finger_tip", patch_radius=0.1, min_patch_radius=0.1
            ),
            "link_11.0_tip": dict(
                material="finger_tip", patch_radius=0.1, min_patch_radius=0.1
            ),
            "link_15.0_tip": dict(
                material="finger_tip", patch_radius=0.1, min_patch_radius=0.1
            ),
        },
    )

    @property
    def _controller_configs(self):
        trans_pd = PDJointPosControllerConfig(
            self.trans_joint_names,
            lower=[-10, -10, -10],
            upper=[10, 10, 10],
            stiffness=2000,
            damping=100,
            force_limit=1000,
            normalize_action=True,
        )
        rot_pd = PDJointPosControllerConfig(
            self.rot_joint_names,
            lower=-4 * np.pi,
            upper=4 * np.pi,
            stiffness=2000,
            damping=100,
            force_limit=1000,
            normalize_action=True,
        )
        finger_pd = PDJointPosControllerConfig(
            self.finger_joint_names,
            lower=None,
            upper=None,
            stiffness=16 * [10],
            damping=16 * [0.3],
            force_limit=16 * [10],
            normalize_action=True,
        )

        controller_dict = dict(
            pd_joint_pos=dict(trans=trans_pd, rot=rot_pd, finger=finger_pd),
        )
        return copy.deepcopy(controller_dict)

    def _after_init(self):
        for link in self.robot.links:
            link.disable_gravity = True

        self.tip_links = sapien_utils.get_objs_by_names(
            self.robot.get_links(),
            ["link_3.0_tip", "link_7.0_tip", "link_11.0_tip", "link_15.0_tip"],
        )
        self.wrist_links = sapien_utils.get_objs_by_names(
            self.robot.get_links(), ["wrist"]
        )
        self.base_link = sapien_utils.get_objs_by_names(
            self.robot.get_links(), ["base_link"]
        )

    @property
    def tip_poses(self):
        tip_poses = [
            vectorize_pose(link.pose, device=self.device) for link in self.tip_links
        ]
        return torch.stack(tip_poses, dim=-2)

    @property
    def wrist_poses(self):
        wrist_poses = [
            vectorize_pose(link.pose, device=self.device) for link in self.wrist_links
        ]
        return torch.stack(wrist_poses, dim=-2)

    @property
    def base_pose(self):
        return vectorize_pose(self.base_link[0].pose, device=self.device)
