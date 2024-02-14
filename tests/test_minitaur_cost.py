"""Test if the MinitaurBulletCost environment still behaves like the original Minitaur
environment when the same environment parameters are used.
"""

# import os

import gymnasium as gym
import numpy as np
from gym.logger import ERROR as ERROR_ORG
from gymnasium.logger import ERROR

import pytest
import stable_gym  # NOTE: Ensures that the latest version of the environment is used. # noqa: F401, E501
from stable_gym.common.utils import change_precision

gym.logger.set_level(ERROR)

import gym as gym_orig  # noqa: E402

gym_orig.logger.set_level(
    ERROR_ORG
)  # TODO: Can be removed when https://github.com/bulletphysics/bullet3/issues/4369 is resoled. # noqa: E501
import pybullet_envs  # noqa: F401, E402

# Register Minitaur environment. Needed because the original Minitaur environment
# is registered under 'gym' instead of 'gymnasium'.
# TODO: Can be removed when https://github.com/bulletphysics/bullet3/issues/4369 is resoled. # noqa: E501
gym.register(
    id="MinitaurBulletEnv-v0",
    entry_point="pybullet_envs.bullet.minitaur_gym_env:MinitaurBulletEnv",
    max_episode_steps=500,
    reward_threshold=300,
    disable_env_checker=True,
    apply_api_compatibility=True,
)

PRECISION = 15


class TestMinitaurBulletCostEqual:
    @pytest.fixture
    def env_original(self):
        """Create original MinitaurBulletEnv environment."""
        env = gym.make("MinitaurBulletEnv", env_randomizer=None)
        yield env
        env.close()

    @pytest.fixture
    def env_cost(self):
        """Create MinitaurBulletCost environment."""
        env = gym.make(
            "MinitaurBulletCost",
            env_randomizer=None,
            exclude_reference_from_observation=True,
            exclude_x_velocity_from_observation=True,
        )
        yield env
        env.close()

    @pytest.fixture
    def env_cost_full(self):
        """Create MinitaurBulletCost environment with all observations."""
        env = gym.make(
            "MinitaurBulletCost",
            env_randomizer=None,
            exclude_reference_error_from_observation=False,
        )
        yield env
        env.close()

    def test_equal_reset(self, env_original, env_cost):
        """Test if reset behaves the same."""
        # Perform reset and check if observations are equal.
        observation, _ = env_original.reset(seed=42)
        observation_cost, _ = env_cost.reset(seed=42)
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
            assert np.allclose(
                observation, observation_cost
            ), f"{observation} != {observation_cost}"

    # NOTE: We decrease the test precision to 16 decimals to ignore numerical
    # differences due to hardware or library differences.
    def test_snapshot(self, snapshot, env_cost_full):
        """Test if the 'MinitaurBulletCost' environment is still equal to snapshot."""
        observation, info = env_cost_full.reset(seed=42)
        assert (change_precision(observation, precision=PRECISION) == snapshot).all()
        assert change_precision(info, precision=PRECISION) == snapshot
        env_cost_full.action_space.seed(42)
        for _ in range(5):
            action = env_cost_full.action_space.sample()
            observation, reward, terminated, truncated, info = env_cost_full.step(
                action
            )
            assert (
                change_precision(observation, precision=PRECISION) == snapshot
            ).all()
            assert change_precision(reward, precision=PRECISION) == snapshot
            assert terminated == snapshot
            assert truncated == snapshot
            assert change_precision(info, precision=PRECISION) == snapshot
