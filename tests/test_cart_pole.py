"""Test if the CartPoleCost environment still behaves like the original CartPole
environment when the same environment parameters are used.
"""
import math

import gymnasium as gym
import numpy as np
from gymnasium.logger import ERROR

import stable_gym  # noqa: F401

gym.logger.set_level(ERROR)


class TestCartPoleCostEqual:
    # Make original CartPole environment.
    env = gym.make("CartPole")
    # Make CartPoleCost environment.
    env_cost = gym.make("CartPoleCost")

    # Overwrite changed parameters.
    env_cost.unwrapped.force_mag = 10.0
    env_cost.unwrapped.theta_threshold_radians = 12 * 2 * math.pi / 360
    env_cost.unwrapped.x_threshold = 2.4
    env_cost.unwrapped.max_v = np.finfo(np.float32).max
    env_cost.unwrapped.max_x = np.finfo(np.float32).max
    env_cost.unwrapped._init_state_range = {
        "low": [-0.05, -0.05, -0.05, -0.05],
        "high": [0.05, 0.05, 0.05, 0.05],
    }

    def test_equal_reset(self):
        """Test if reset behaves the same."""
        # Perform reset and check if observations are equal.
        observation, _ = self.env.reset(seed=42)
        observation_cost, _ = self.env_cost.reset(seed=42)
        assert np.allclose(
            observation, observation_cost
        ), f"{observation} != {observation_cost}"

    def test_equal_steps(self):
        """Test if steps behave the same."""
        # Perform steps and check if observations are equal.
        observation, _, _, _, _ = self.env.step(0)
        observation_cost, _, _, _, _ = self.env_cost.step(
            np.array([-10], dtype=np.float32)
        )
        assert np.allclose(
            observation, observation_cost
        ), f"{observation} != {observation_cost}"
        observation, _, _, _, _ = self.env.step(1)
        observation_cost, _, _, _, _ = self.env_cost.step(
            np.array([10], dtype=np.float32)
        )
        assert np.allclose(
            observation, observation_cost
        ), f"{observation} != {observation_cost}"
