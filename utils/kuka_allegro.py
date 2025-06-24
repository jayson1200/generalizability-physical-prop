import copy
from typing import List

import numpy as np
import torch
from mani_skill.agents.base_agent import BaseAgent
from mani_skill.agents.controllers import PDJointPosControllerConfig
from mani_skill.agents.controllers.pd_ee_pose import (
    PDEEPosControllerConfig,
    PDEEPoseControllerConfig,
)
from mani_skill.agents.registration import register_agent
from mani_skill.utils import sapien_utils
from mani_skill.utils.structs.pose import vectorize_pose


@register_agent()
class KukaAllegro(BaseAgent):
    uid = "kuka_allegro"
    urdf_path = f"assets/kuka_allegro/kuka_allegro.urdf"
    fix_root_link = True

    arm_joint_names = [
        "iiwa14_joint_1",
        "iiwa14_joint_2",
        "iiwa14_joint_3",
        "iiwa14_joint_4",
        "iiwa14_joint_5",
        "iiwa14_joint_6",
        "iiwa14_joint_7",
    ]
    finger_joint_names = [
        "joint_0.0",
        "joint_1.0",
        "joint_2.0",
        "joint_3.0",
        "joint_4.0",
        "joint_5.0",
        "joint_6.0",
        "joint_7.0",
        "joint_8.0",
        "joint_9.0",
        "joint_10.0",
        "joint_11.0",
        "joint_12.0",
        "joint_13.0",
        "joint_14.0",
        "joint_15.0",
    ]

    arm_stiffness = 200
    arm_damping = 10
    arm_force_limit = 1000

    hand_stiffness = 4
    hand_damping = 0.2
    hand_force_limit = 10

    urdf_config = dict(
        _materials=dict(
            finger_tip=dict(static_friction=1.0, dynamic_friction=1.0, restitution=0.0)
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

    def __init__(self, *args, **kwargs):
        self.joint_names = self.arm_joint_names + self.finger_joint_names
        super().__init__(*args, **kwargs)
        self.loaded2custom = []
        for joint in self.robot.active_joints:
            self.loaded2custom.append(self.joint_names.index(joint.name))
        self.custom2loaded = [0] * len(self.joint_names)
        for i, v in enumerate(self.loaded2custom):
            self.custom2loaded[v] = i
        self.loaded2custom = torch.tensor(self.loaded2custom)
        self.custom2loaded = torch.tensor(self.custom2loaded)

    def get_proprioception(self):
        obs = dict(
            qpos=self.robot.get_qpos()[..., self.loaded2custom],
            qvel=self.robot.get_qvel()[..., self.loaded2custom],
        )
        return obs

    @property
    def _controller_configs(self):
        arm_pd = PDJointPosControllerConfig(
            self.arm_joint_names,
            lower=None,
            upper=None,
            stiffness=[600, 600, 500, 400, 200, 200, 200],
            damping=10,
            force_limit=1000,
            normalize_action=True,
        )
        finger_pd = PDJointPosControllerConfig(
            self.finger_joint_names,
            lower=None,
            upper=None,
            stiffness=16 * [self.hand_stiffness],
            damping=16 * [self.hand_damping],
            force_limit=16 * [self.hand_force_limit],
            normalize_action=True,
        )
        ee_pd = PDEEPoseControllerConfig(
            self.arm_joint_names,
            pos_lower=-1,
            pos_upper=1,
            rot_lower=-np.pi,
            rot_upper=np.pi,
            stiffness=0,
            damping=0,
            force_limit=0,
            ee_link="palm_link",
            urdf_path=self.urdf_path,
            normalize_action=True,
        )

        controller_dict = dict(
            pd_joint_pos=dict(arm=arm_pd, finger=finger_pd),
        )
        return copy.deepcopy(controller_dict)

    def _after_init(self):
        for link in self.robot.links:
            link.disable_gravity = True

        self.tip_links = sapien_utils.get_objs_by_names(
            self.robot.get_links(),
            ["link_3.0_tip", "link_7.0_tip", "link_11.0_tip", "link_15.0_tip"],
        )
        # not used at the moment
        self.wrist_links = sapien_utils.get_objs_by_names(
            self.robot.get_links(), ["palm"]
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
