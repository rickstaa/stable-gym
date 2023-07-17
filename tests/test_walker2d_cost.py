"""Test if the Walker2dCost environment still behaves like the original Walker2d
environment when the same environment parameters are used.
"""
import gymnasium as gym
import numpy as np
from gymnasium.logger import ERROR

from stable_gym.common.utils import change_precision

gym.logger.set_level(ERROR)

PRECISION = 15


class TestWalker2dCostEqual:
    # Make original Walker2d environment.
    env = gym.make("Walker2d")
    # Make Walker2dCost environment.
    env_cost = gym.make(
        "Walker2dCost",
        exclude_reference_from_observation=True,
        exclude_x_velocity_from_observation=True,
    )

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
            self.env.action_space.seed(42)
            action = self.env.action_space.sample()
            observation, _, _, _, _ = self.env.step(action)
            observation_cost, _, _, _, _ = self.env_cost.step(action)
            assert np.allclose(
                observation, observation_cost
            ), f"{observation} != {observation_cost}"

    # NOTE: We decrease the test precision to 16 decimals to ignore numerical
    # differences due to hardware or library differences.
    def test_snapshot(self, snapshot):
        """Test if the 'Walker2dCost' environment is still equal to snapshot."""
        self.env_cost = gym.make(
            "Walker2dCost", exclude_reference_error_from_observation=False
        )  # Check full observation.
        observation, info = self.env_cost.reset(seed=42)
        assert (change_precision(observation, precision=PRECISION) == snapshot).all()
        assert change_precision(info, precision=PRECISION) == snapshot
        self.env_cost.action_space.seed(42)
        for _ in range(5):
            action = self.env_cost.action_space.sample()
            observation, reward, terminated, truncated, info = self.env_cost.step(
                action
            )
            assert (
                change_precision(observation, precision=PRECISION) == snapshot
            ).all()
            assert change_precision(reward, precision=PRECISION) == snapshot
            assert terminated == snapshot
            assert truncated == snapshot
            assert change_precision(info, precision=PRECISION) == snapshot
