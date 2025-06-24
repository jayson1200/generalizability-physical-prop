import torch


class Trajectory:
    def __init__(
        self,
        init_obj_pose: torch.Tensor,
        init_hand_pose: torch.Tensor,
        goal_obj_pose: torch.Tensor,
        planned_wrist_poses: torch.Tensor,
        planned_obj_poses: torch.Tensor,
    ):
        self.init_obj_pose = init_obj_pose
        self.init_hand_pose = init_hand_pose
        self.goal_obj_pose = goal_obj_pose

        self.planned_wrist_poses = planned_wrist_poses
        self.planned_obj_poses = planned_obj_poses

    def to(self, device: torch.device):
        self.init_obj_pose = self.init_obj_pose.to(device)
        self.init_hand_pose = self.init_hand_pose.to(device)
        self.goal_obj_pose = self.goal_obj_pose.to(device)
        self.planned_wrist_poses = self.planned_wrist_poses.to(device)
        self.planned_obj_poses = self.planned_obj_poses.to(device)
        return self

    def __len__(self):
        """Return the number of trajectories."""
        if self.init_obj_pose.dim() > 1:
            return self.init_obj_pose.shape[0]
        return 1

    def __getitem__(self, idx):
        """Support slicing to get a subset of trajectories."""
        if isinstance(idx, slice):
            return Trajectory(
                init_obj_pose=self.init_obj_pose[idx],
                init_hand_pose=self.init_hand_pose[idx],
                goal_obj_pose=self.goal_obj_pose[idx],
                planned_wrist_poses=self.planned_wrist_poses[idx],
                planned_obj_poses=self.planned_obj_poses[idx],
            )
        return self.__getitem__(slice(idx, idx + 1))
