from dataclasses import dataclass
from typing import List, Type

import numpy as np


@dataclass
class TaskSpecification:
    env_name: str
    folder: str
    initial_hand_pos: List[float]
    initial_hand_orientation: List[float]
    item_names: List[str]
    keypoint_task: str
    trajectory_task: str
    detailed_task: str

    @classmethod
    def create(cls) -> "TaskSpecification":
        return cls(
            env_name=cls.env_name,
            folder=cls.folder,
            initial_hand_pos=cls.initial_hand_pos,
            initial_hand_orientation=cls.initial_hand_orientation,
            item_names=cls.item_names,
            keypoint_task=cls.keypoint_task,
            trajectory_task=cls.trajectory_task,
            detailed_task=cls.detailed_task,
        )


class AppleTask(TaskSpecification):
    env_name = "EnvApple-v0"
    folder = "apple"
    initial_hand_pos = [-1.95, -0.71, 1.2]
    initial_hand_orientation = [np.pi, -np.pi / 2, 0]
    item_names = ["apple", "cutting board"]
    affected_objects = ["apple_0", "cutting_board_0"]
    keypoint_task = "Point to the apple and the cutting board in the image."
    trajectory_task = "pick up an apple and put it on a cutting board"
    detailed_task = """First move the robot hand towards the apple.
Then grasp the apple and lift it up.
Finally move the apple on the cutting board and put it down."""


class HammerTask(TaskSpecification):
    env_name = "EnvHammer-v0"
    folder = "hammer"
    initial_hand_pos = [-1.95, -0.9, 1.2]
    initial_hand_orientation = [np.pi, -np.pi / 2, 0]
    item_names = ["handle", "head"]
    affected_objects = ["hammer_0"]
    keypoint_task = (
        "Point to the brown handle and the metal head of the hammer in the image."
    )
    trajectory_task = "make a hammering motion"
    detailed_task = """First move the robot hand towards the handle.
Then grasp the handle.
Finally hit on the kitchen counter 3 times."""


class DrawerTask(TaskSpecification):
    env_name = "EnvDrawer-v0"
    folder = "drawer"
    initial_hand_pos = [1.28, -6.45, 0.60]
    initial_hand_orientation = [0.0, 0.0, np.pi]
    item_names = ["handle"]
    affected_objects = ["drawer_0"]
    keypoint_task = "Point to the handle of the top cabinet drawer in the image."
    trajectory_task = "pull open a cabinet drawer"
    detailed_task = """First move the robot hand towards the handle of the drawer.
Then grasp the handle.
Finally pull the drawer open by at least 30cm."""


class BottleTask(TaskSpecification):
    env_name = "EnvBottle-v0"
    folder = "bottle"
    initial_hand_pos = [-1.77, -1.97, 1.2]
    initial_hand_orientation = [-np.pi / 2, 0, np.pi]
    item_names = ["bottle", "point"]
    affected_objects = ["bottle_0", "kitchen_counter"]
    keypoint_task = "Point to the water bottle on the kitchen counter, and pinpoint a point on the kitchen counter to the right of the kitchen sink in the image."
    trajectory_task = (
        'move a bottle to the target position called "point" on the kitchen counter'
    )
    detailed_task = """First move the robot hand towards the bottle.
Then grasp the bottle and lift it up.
Finally move the bottle to the target position called "point" and put it down."""


class BottleArmTask(TaskSpecification):
    env_name = "EnvBottleArm-v0"
    folder = "bottle_arm"
    initial_hand_pos = [0.4694, -0.5224, 0.4018]
    initial_hand_orientation = [1.36928121, 0.06131303, 0.4635126]
    item_names = ["bottle", "plate"]
    affected_objects = ["bottle_0", "plate_0"]
    keypoint_task = (
        "Point to the middle of the bottle and the plate on the table in the image."
    )
    trajectory_task = "move a bottle onto a plate"
    detailed_task = """First move the robot hand towards the bottle.
Then grasp the bottle and lift it up.
Then place the bottle on to the plate."""


class BoxArmTask(TaskSpecification):
    env_name = "EnvBoxArm-v0"
    folder = "box_arm"
    initial_hand_pos = [0.5163, -0.3471, 0.4074]
    initial_hand_orientation = [-0.83026431, 1.56060727, -0.93857979]
    item_names = ["box", "bottle"]
    affected_objects = ["box_0", "bottle_0"]
    keypoint_task = "Point to the box and the bottle on the table in the image."
    trajectory_task = "slide the box over the table to the bottle"
    detailed_task = """First move the robot hand towards the box.
Then slide the box over the table to the bottle."""


class HammerArmTask(TaskSpecification):
    env_name = "EnvHammerArm-v0"
    folder = "hammer_arm"
    initial_hand_pos = [0.5058, -0.3764, 0.3690]
    initial_hand_orientation = [0.31370497, 1.54948171, -0.33859719]
    item_names = ["handle", "head"]
    affected_objects = ["hammer_0"]
    keypoint_task = "Point to the handle and the head of the hammer in the image."
    trajectory_task = "make a hammering motion"
    detailed_task = """First move the robot hand towards the hammer handle.
Then grasp the hammer handle.
Finally hit on the kitchen counter 3 times."""


class SpongeTask(TaskSpecification):
    env_name = "EnvSponge-v0"
    folder = "sponge"
    initial_hand_pos = [-1.77, -0.9, 1.2]
    initial_hand_orientation = [-np.pi / 2, 0, np.pi / 2]
    item_names = ["sponge"]
    affected_objects = ["sponge_0"]
    keypoint_task = (
        "Point to the green yellow sponge on the kitchen counter in the image."
    )
    trajectory_task = "wipe a kitchen counter with a sponge"
    detailed_task = """First move the robot hand towards the sponge.
Then grasp the sponge.
Finally wipe the kitchen counter with the sponge."""


class PliersTask(TaskSpecification):
    env_name = "EnvPliers-v0"
    folder = "pliers"
    initial_hand_pos = [-1.86, -0.92, 1.2]
    initial_hand_orientation = [np.pi / 2, -np.pi, -np.pi / 2]
    item_names = ["handle left", "handle right"]
    affected_objects = ["pliers_0"]
    keypoint_task = "Point to the left and right handles of the pliers in the image."
    trajectory_task = "close a pair of pliers"
    detailed_task = """First move the robot hand towards the pliers.
Then grasp the left and right handles and entirely close the pliers."""


class ScissorsTask(TaskSpecification):
    env_name = "EnvScissors-v0"
    folder = "scissors"
    initial_hand_pos = [-1.88, -0.96, 1.2]
    initial_hand_orientation = [np.pi, -np.pi / 2, 0]
    item_names = ["loop 1", "loop 2"]
    affected_objects = ["scissors_0"]
    keypoint_task = "Point to the two loops of the scissors in the image."
    trajectory_task = "close a pair of scissors"
    detailed_task = """First move the robot hand towards the scissors.
Then grasp the two loops and entirely close the scissors."""


class FridgeTask(TaskSpecification):
    env_name = "EnvFridge-v0"
    folder = "fridge"
    initial_hand_pos = [-1.25, -2.91, 1.14]
    initial_hand_orientation = [-np.pi / 2, 0, np.pi]
    item_names = ["handle"]
    affected_objects = ["fridge"]
    keypoint_task = "Point to the top handle of the refrigerator door in the image."
    trajectory_task = "open a refrigerator door"
    detailed_task = """The refrigerator faces in x direction.
The y axis points to the right, and the z axis points up.
First figure out how large the door is.
Then describe how the x and y coordinates of the handle change as the door is opened.
Now move the robot hand towards the handle.
Then grasp the handle.
Finally fully open the door."""


TASK_SPECIFICATIONS = {
    "apple": AppleTask,
    "hammer": HammerTask,
    "drawer": DrawerTask,
    "bottle": BottleTask,
    "bottle_arm": BottleArmTask,
    "box_arm": BoxArmTask,
    "hammer_arm": HammerArmTask,
    "sponge": SpongeTask,
    "pliers": PliersTask,
    "scissors": ScissorsTask,
    "fridge": FridgeTask,
}
