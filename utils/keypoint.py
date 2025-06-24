from typing import List

import torch
from PIL import Image


class Keypoint:
    def __init__(
        self,
        obj_name: str,
        link_name: str,
        point: torch.Tensor,
        image_point: torch.Tensor,
        imgs: List[Image.Image],
    ):
        self.obj_name = obj_name
        self.link_name = link_name
        self.point = point
        self.image_point = image_point
        self.imgs = imgs

    def __str__(self):
        return f"{self.obj_name} {self.link_name} {self.point} {self.image_point}"

    def __repr__(self):
        return self.__str__()


class KeypointTrajectory:
    def __init__(self, keypoints, trajectories):
        self.keypoints = keypoints
        self.trajectories = trajectories

    def __str__(self):
        return f"{self.keypoints}"

    def __repr__(self):
        return self.__str__()
