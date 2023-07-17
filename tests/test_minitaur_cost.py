"""Test if the MinitaurCost environment still behaves like the original Minitaur
environment when the same environment parameters are used.
"""
import gymnasium as gym
import numpy as np
from gymnasium.logger import ERROR
from gym.logger import ERROR as ERROR_ORG

gym.logger.set_level(ERROR)

import stable_gym  # noqa: F401, E402

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


class TestMinitaurCostEqual:
    # NOTE: The env randomizer is disabled because it is not deterministic.
    # Make original Minitaur environment.
    env = gym.make("MinitaurBulletEnv-v0", env_randomizer=None)
    # Make MinitaurCost environment.
    env_cost = gym.make(
        "MinitaurCost",
        env_randomizer=None,
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

    def test_snapshot(self, snapshot):
        """Test if the 'MinitaurCost' environment is still equal to snapshot."""
        self.env_cost = gym.make(
            "MinitaurCost",
            env_randomizer=None,
            exclude_reference_error_from_observation=False,
        )  # Check full observation.
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
