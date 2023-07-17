"""Test if the FetchReachCost environment still behaves like the original FetchReach
environment when the same environment parameters are used.
"""
import gymnasium as gym
import numpy as np
from gymnasium.logger import ERROR

from stable_gym.common.utils import change_precision

gym.logger.set_level(ERROR)

PRECISION = 100


class TestFetchReachCostEqual:
    # Make original FetchReach environment.
    env = gym.make("FetchReach")
    # Make FetchReachCost environment.
    env_cost = gym.make("FetchReachCost")

    def test_equal_reset(self):
        """Test if reset behaves the same."""
        # Perform reset and check if observations are equal.
        observation, _ = self.env.reset(seed=42)
        observation_cost, _ = self.env_cost.reset(seed=42)
        observation = gym.spaces.flatten(self.env.observation_space, observation)
        observation_cost = gym.spaces.flatten(
            self.env_cost.observation_space, observation_cost
        )

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
            observation = gym.spaces.flatten(self.env.observation_space, observation)
            observation_cost = gym.spaces.flatten(
                self.env_cost.observation_space, observation_cost
            )

            assert np.allclose(
                observation, observation_cost
            ), f"{observation} != {observation_cost}"

    # NOTE: We decrease the test precision to 16 decimals to ignore numerical
    # differences due to hardware or library differences.
    def test_snapshot(self, snapshot):
        """Test if the 'FetchReachCost' environment is still equal to snapshot."""
        observation, info = self.env_cost.reset(seed=42)
        observation = gym.spaces.flatten(self.env.observation_space, observation)
        assert (change_precision(observation, precision=PRECISION) == snapshot).all()
        assert change_precision(info, precision=PRECISION) == snapshot
        self.env_cost.action_space.seed(42)
        for _ in range(5):
            action = self.env_cost.action_space.sample()
            observation, reward, terminated, truncated, info = self.env_cost.step(
                action
            )
            observation = gym.spaces.flatten(self.env.observation_space, observation)
            assert (
                change_precision(observation, precision=PRECISION) == snapshot
            ).all()
            assert change_precision(reward, precision=PRECISION) == snapshot
            assert terminated == snapshot
            assert truncated == snapshot
            assert change_precision(info, precision=PRECISION) == snapshot
