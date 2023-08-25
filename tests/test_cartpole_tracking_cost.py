"""Test if the CartPoleCost environment still behaves like the original CartPole
environment when the same environment parameters are used.
"""
import math
import random

import gymnasium as gym
import numpy as np
from gymnasium.logger import ERROR

import pytest
import stable_gym  # NOTE: Ensures that the latest version of the environment is used. # noqa: F401, E501

gym.logger.set_level(ERROR)


class TestCartPoleCostEqual:
    @pytest.fixture
    def env_original(self):
        """Create original CartPole environment."""
        env = gym.make("CartPole")
        yield env
        env.close()

    @pytest.fixture
    def env_cost_aligned(self):
        """Create CartPoleTrackingCost environment and align the parameters with the
        original CartPole environment.
        """
        env_cost = gym.make(
            "CartPoleTrackingCost",
        )

        # Overwrite changed parameters.
        env_cost.unwrapped.force_mag = 10.0
        env_cost.unwrapped.theta_threshold_radians = 12 * 2 * math.pi / 360
        env_cost.unwrapped.x_threshold = 2.4
        env_cost.unwrapped.max_v = np.finfo(np.float64).max
        env_cost.unwrapped.max_x = np.finfo(np.float64).max
        env_cost.unwrapped._init_state_range = {
            "low": [-0.05, -0.05, -0.05, -0.05],
            "high": [0.05, 0.05, 0.05, 0.05],
        }
        yield env_cost
        env_cost.close()

    @pytest.fixture
    def env_cost(self):
        """Create CartPoleCost environment."""
        env = gym.make(
            "CartPoleTrackingCost",
            exclude_reference_error_from_observation=False,
        )
        yield env
        env.close()

    def test_equal_reset(self, env_original, env_cost_aligned):
        """Test if reset behaves the same."""
        # Perform reset and check if observations are equal.
        observation, _ = env_original.reset(seed=42)
        observation_cost, _ = env_cost_aligned.reset(seed=42)
        assert np.allclose(
            observation, observation_cost[:-1]
        ), f"{observation} != {observation_cost[:-1]}"

    def test_equal_steps(self, env_original, env_cost_aligned):
        """Test if steps behave the same."""
        # Perform several steps and check if observations are equal.
        env_original.reset(seed=42), env_cost_aligned.reset(seed=42)
        for _ in range(10):
            discrete_action = random.randint(0, 1)
            continuous_action = (
                np.array([-10]) if discrete_action == 0 else np.array([10])
            )
            observation, _, _, _, _ = env_original.step(discrete_action)
            observation_cost, _, _, _, _ = env_cost_aligned.step(continuous_action)
            assert np.allclose(
                observation, observation_cost[:-1]
            ), f"{observation} != {observation_cost[:-1]}"

    def test_snapshot(self, snapshot, env_cost):
        """Test if the 'CartPoleCost' environment is still equal to snapshot."""
        observation, info = env_cost.reset(seed=42)
        assert (observation == snapshot).all()
        assert info == snapshot
        env_cost.action_space.seed(42)
        for _ in range(5):
            action = env_cost.action_space.sample()
            observation, reward, terminated, truncated, info = env_cost.step(action)
            assert (observation == snapshot).all()
            assert reward == snapshot
            assert terminated == snapshot
            assert truncated == snapshot
            assert info == snapshot
