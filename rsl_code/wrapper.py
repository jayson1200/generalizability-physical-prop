# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import dataclass, field

import gymnasium as gym
import torch
from mani_skill.utils import gym_utils
from rsl_rl.env import VecEnv


@dataclass
class RslRlVecEnvWrapperConfig:
    pass


class RslRlVecEnvWrapper(VecEnv):
    def __init__(self, env, clip_actions: float | None = None):
        self.env = env
        self.clip_actions = clip_actions
        self.cfg = RslRlVecEnvWrapperConfig()

        # store information required by wrapper
        self.num_envs = self.unwrapped.num_envs
        self.device = self.unwrapped.device

        self.num_actions = self.action_space.shape[-1]
        self.num_obs = self.observation_space.shape[-1]
        self.num_priviledged_obs = self.num_obs

        self.rnd_state = torch.zeros(self.num_envs, 1)

        self.max_episode_length = gym_utils.find_max_episode_steps_value(env)

        # modify the action space to the clip range
        # self._modify_action_space()

        # reset at the start since the RSL-RL runner does not call reset
        self.env.reset()

    def __str__(self):
        """Returns the wrapper name and the :attr:`env` representation string."""
        return f"<{type(self).__name__}{self.env}>"

    def __repr__(self):
        """Returns the string representation of the wrapper."""
        return str(self)

    """
    Properties -- Gym.Wrapper
    """

    @property
    def render_mode(self) -> str | None:
        """Returns the :attr:`Env` :attr:`render_mode`."""
        return self.env.render_mode

    @property
    def observation_space(self) -> gym.Space:
        """Returns the :attr:`Env` :attr:`observation_space`."""
        return self.env.observation_space

    @property
    def action_space(self) -> gym.Space:
        """Returns the :attr:`Env` :attr:`action_space`."""
        return self.env.action_space

    @classmethod
    def class_name(cls) -> str:
        """Returns the class name of the wrapper."""
        return cls.__name__

    @property
    def unwrapped(self):
        """Returns the base environment of the wrapper.

        This will be the bare :class:`gymnasium.Env` environment, underneath all layers of wrappers.
        """
        return self.env.unwrapped

    """
    Properties
    """

    def get_observations(self) -> tuple[torch.Tensor, dict]:
        """Returns the current observations of the environment."""
        obs = self.unwrapped.get_obs()
        return obs, {
            "observations": {
                "critic": obs.clone(),
                "policy": obs.clone(),
                "rnd_state": self.rnd_state,
            }
        }

    @property
    def episode_length_buf(self) -> torch.Tensor:
        """The episode length buffer."""
        return self.unwrapped.episode_length_buf

    @episode_length_buf.setter
    def episode_length_buf(self, value: torch.Tensor):
        """Set the episode length buffer.

        Note:
            This is needed to perform random initialization of episode lengths in RSL-RL.
        """
        value %= self.unwrapped.max_episode_steps
        value = (
            value % (self.unwrapped.max_episode_steps // 2)
        ) + self.unwrapped.max_episode_steps // 2
        self.unwrapped.episode_length_buf = value

    """
    Operations - MDP
    """

    def seed(self, seed: int = -1) -> int:  # noqa: D102
        return self.unwrapped._set_main_rng(seed)

    def reset(self, seed=None) -> tuple[torch.Tensor, dict]:  # noqa: D102
        # reset the environment
        obs, _ = self.env.reset(seed=seed)
        # return observations
        return obs, {
            "observations": {
                "critic": obs.clone(),
                "policy": obs.clone(),
                "rnd_state": self.rnd_state,
            }
        }

    def step(
        self, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        # clip actions
        if self.clip_actions is not None:
            actions = torch.clamp(actions, -self.clip_actions, self.clip_actions)
        # record step information
        obs, rew, terminated, truncated, extras = self.env.step(actions)
        # print(extras)
        # compute dones for compatibility with RSL-RL
        dones = (terminated | truncated).to(dtype=torch.long)

        # move extra observations to the extras dict
        extras["observations"] = {
            "critic": obs.clone(),
            "policy": obs.clone(),
            "rnd_state": self.rnd_state,
        }
        # move time out information to the extras dict
        # this is only needed for infinite horizon tasks
        extras["time_outs"] = truncated

        # return the step information
        return obs, rew, dones, extras

    def close(self):  # noqa: D102
        return self.env.close()

    """
    Helper functions
    """

    # def _modify_action_space(self):
    #     """Modifies the action space to the clip range."""
    #     if self.clip_actions is None:
    #         return

    #     # modify the action space to the clip range
    #     # note: this is only possible for the box action space. we need to change it in the future for other action spaces.
    #     self.env.unwrapped.single_action_space = gym.spaces.Box(
    #         low=-self.clip_actions, high=self.clip_actions, shape=(self.num_actions,)
    #     )
    #     self.env.unwrapped.action_space = gym.vector.utils.batch_space(
    #         self.env.unwrapped.single_action_space, self.num_envs
    #     )
