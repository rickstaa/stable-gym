"""Test if the CartPoleCost environment still behaves like the original CartPole
environment when the same environment parameters are used.
"""
import math
import random

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
    env_cost.unwrapped.max_v = np.finfo(np.float64).max
    env_cost.unwrapped.max_x = np.finfo(np.float64).max
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
        # Perform several steps and check if observations are equal.
        self.env.reset(seed=42), self.env_cost.reset(seed=42)
        for _ in range(10):
            discrete_action = random.randint(0, 1)
            continuous_action = (
                np.array([-10]) if discrete_action == 0 else np.array([10])
            )
            observation, _, _, _, _ = self.env.step(discrete_action)
            observation_cost, _, _, _, _ = self.env_cost.step(continuous_action)
            assert np.allclose(
                observation, observation_cost
            ), f"{observation} != {observation_cost}"

    def test_snapshot(self, snapshot):
        """Test if the 'CartPoleCost' environment is still equal to snapshot."""
        observation, info = self.env_cost.reset(seed=42)
        assert (observation == snapshot).all()
        assert info == snapshot
        self.env_cost.action_space.seed(42)
        for _ in range(5):
            action = self.env_cost.action_space.sample()
            observation, reward, terminated, truncated, info = self.env_cost.step(
                action
            )
            assert (observation == snapshot).all()
            assert reward == snapshot
            assert terminated == snapshot
            assert truncated == snapshot
            assert info == snapshot

    def test_reference_tracking_snapshot(self, snapshot):
        """Test if the 'CartPoleCost' environment with a 'reference_tracking' task is
        still equal to snapshot.
        """
        env_cost_reference_tracking = gym.make(
            "CartPoleCost",
            task_type="reference_tracking",
            exclude_reference_error_from_observation=False,
        )
        observation, info = env_cost_reference_tracking.reset(seed=42)
        assert (observation == snapshot).all()
        assert info == snapshot
        env_cost_reference_tracking.action_space.seed(42)
        for _ in range(5):
            action = env_cost_reference_tracking.action_space.sample()
            (
                observation,
                reward,
                terminated,
                truncated,
                info,
            ) = env_cost_reference_tracking.step(action)
            assert (observation == snapshot).all()
            assert reward == snapshot
            assert terminated == snapshot
            assert truncated == snapshot
            assert info == snapshot

    def test_reference_tracking_periodic_snapshot(self, snapshot):
        """Test if the 'CartPoleCost' environment with a 'reference_tracking' task and
        a 'periodic' reference is still equal to snapshot.
        """
        env_cost_reference_tracking_periodic = gym.make(
            "CartPoleCost",
            task_type="reference_tracking",
            reference_type="periodic",
            exclude_reference_error_from_observation=False,
        )
        observation, info = env_cost_reference_tracking_periodic.reset(seed=42)
        assert (observation == snapshot).all()
        assert info == snapshot
        env_cost_reference_tracking_periodic.action_space.seed(42)
        for _ in range(5):
            action = env_cost_reference_tracking_periodic.action_space.sample()
            (
                observation,
                reward,
                terminated,
                truncated,
                info,
            ) = env_cost_reference_tracking_periodic.step(action)
            assert (observation == snapshot).all()
            assert reward == snapshot
            assert terminated == snapshot
            assert truncated == snapshot
            assert info == snapshot
