"""Test if the SwimmerCost environment still behaves like the original Swimmer
environment when the same environment parameters are used.
"""
# import os

import gymnasium as gym
import numpy as np

# import pytest
from gymnasium.logger import ERROR

import stable_gym  # noqa: F401

gym.logger.set_level(ERROR)


class TestSwimmerCostEqual:
    # Make original Swimmer environment.
    env = gym.make("Swimmer")
    # Make SwimmerCost environment.
    env_cost = gym.make(
        "SwimmerCost",
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

    # # Skip snapshot test during CI.
    # # NOTE: Done because the snapshot can differ between python versions and systems.
    # @pytest.mark.skipif(
    #     os.getenv("CI", False).lower() == "true",
    #     reason="no way to test snapshot in CI",
    # )
    # def test_snapshot(self, snapshot):
    #     """Test if the 'SwimmerCost' environment is still equal to snapshot."""
    #     self.env_cost = gym.make(
    #         "SwimmerCost", exclude_reference_error_from_observation=False
    #     )  # Check full observation.
    #     observation, info = self.env_cost.reset(seed=42)
    #     assert (observation == snapshot).all()
    #     assert info == snapshot
    #     self.env_cost.action_space.seed(42)
    #     for _ in range(5):
    #         action = self.env_cost.action_space.sample()
    #         observation, reward, terminated, truncated, info = self.env_cost.step(
    #             action
    #         )
    #         assert (observation == snapshot).all()
    #         assert reward == snapshot
    #         assert terminated == snapshot
    #         assert truncated == snapshot
    #         assert info == snapshot
