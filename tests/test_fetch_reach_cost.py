"""Test if the FetchReachCost environment still behaves like the original FetchReach
environment when the same environment parameters are used.
"""
import gymnasium as gym
import numpy as np
from gymnasium.logger import ERROR

import pytest
import stable_gym  # NOTE: Ensures that the latest version of the environment is used. # noqa: F401, E501
from stable_gym.common.utils import change_precision

gym.logger.set_level(ERROR)

PRECISION = 15


class TestFetchReachCostEqual:
    @pytest.fixture
    def env_original(self):
        """Create original FetchReach environment."""
        env = gym.make("FetchReach")
        yield env
        env.close()

    @pytest.fixture
    def env_cost(self):
        """Create FetchReachCost environment."""
        env = gym.make(
            "FetchReachCost",
        )
        yield env
        env.close()

    def test_equal_reset(self, env_original, env_cost):
        """Test if reset behaves the same."""
        # Perform reset and check if observations are equal.
        observation, _ = env_original.reset(seed=42)
        observation_cost, _ = env_cost.reset(seed=42)
        observation = gym.spaces.flatten(env_original.observation_space, observation)
        observation_cost = gym.spaces.flatten(
            env_cost.observation_space, observation_cost
        )

        assert np.allclose(
            observation, observation_cost
        ), f"{observation} != {observation_cost}"

    def test_equal_steps(self, env_original, env_cost):
        """Test if steps behave the same."""
        # Perform several steps and check if observations are equal.
        env_original.reset(seed=42), env_cost.reset(seed=42)
        for _ in range(10):
            env_original.action_space.seed(42)
            action = env_original.action_space.sample()
            observation, _, _, _, _ = env_original.step(action)
            observation_cost, _, _, _, _ = env_cost.step(action)
            observation = gym.spaces.flatten(
                env_original.observation_space, observation
            )
            observation_cost = gym.spaces.flatten(
                env_cost.observation_space, observation_cost
            )

            assert np.allclose(
                observation, observation_cost
            ), f"{observation} != {observation_cost}"

    # NOTE: We decrease the test precision to 16 decimals to ignore numerical
    # differences due to hardware or library differences.
    def test_snapshot(self, snapshot, env_cost):
        """Test if the 'FetchReachCost' environment is still equal to snapshot."""
        observation, info = env_cost.reset(seed=42)
        observation = gym.spaces.flatten(env_cost.observation_space, observation)
        assert (change_precision(observation, precision=PRECISION) == snapshot).all()
        assert change_precision(info, precision=PRECISION) == snapshot
        env_cost.action_space.seed(42)
        for _ in range(5):
            action = env_cost.action_space.sample()
            observation, reward, terminated, truncated, info = env_cost.step(action)
            observation = gym.spaces.flatten(env_cost.observation_space, observation)
            assert (
                change_precision(observation, precision=PRECISION) == snapshot
            ).all()
            assert change_precision(reward, precision=PRECISION) == snapshot
            assert terminated == snapshot
            assert truncated == snapshot
            assert change_precision(info, precision=PRECISION) == snapshot
