"""Test if the QuadXTrackingCost environment still behaves like the original QuadXHover
environment when the same environment parameters are used.
"""
import gymnasium as gym
import numpy as np
import pybullet
import pytest
from gymnasium.logger import ERROR
from stable_gym.common.utils import change_precision

gym.logger.set_level(ERROR)

PRECISION = 13


# TODO: Can be removed if https://github.com/jjshoots/PyFlyt/issues/1 is resolved.
@pytest.mark.skipif(
    not pybullet.isNumpyEnabled(),
    reason=(
        "pybullet was not built with numpy support. Please rebuild pybullet "
        "with numpy enabled."
    ),
)
class TestQuadXTrackingCostEqual:
    @pytest.fixture
    def env_original(self):
        """Create original QuadX-Hover environment."""
        return gym.make("PyFlyt.gym_envs:PyFlyt/QuadX-Hover")

    @pytest.fixture
    def env_cost(self):
        """Create QuadXTrackingCost environment."""
        return gym.make(
            "QuadXTrackingCost",
        )

    @pytest.fixture
    def env_cost_full(self):
        """Create QuadXTrackingCost environment with all observations."""
        return gym.make(
            "QuadXTrackingCost",
            exclude_reference_error_from_observation=False,
        )

    def test_equal_reset(self, env_original, env_cost):
        """Test if reset behaves the same."""
        # Perform reset and check if observations are equal.
        observation, _ = env_original.reset(seed=42)
        observation_cost, _ = env_cost.reset(seed=42)
        assert np.allclose(
            observation, observation_cost[:-3]
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
                observation, observation_cost[:-3]
            ), f"{observation} != {observation_cost}"

    # NOTE: We decrease the test precision to 16 decimals to ignore numerical
    # differences due to hardware or library differences.
    def test_snapshot(self, snapshot, env_cost_full):
        """Test if the 'QuadXTrackingCost' environment is still equal to snapshot."""
        observation, info = env_cost_full.reset(seed=42)
        sin_result = np.sin(
            (
                (2 * np.pi)
                * env_cost_full.unwrapped._reference_frequency[1]
                * env_cost_full.unwrapped.t
            )
            + env_cost_full.unwrapped._reference_phase_shift[1]
        )
        assert (change_precision(observation, precision=PRECISION) == snapshot).all()
        assert change_precision(info, precision=PRECISION) == snapshot
        env_cost_full.action_space.seed(42)
        for i in range(5):
            action = env_cost_full.action_space.sample()
            observation, reward, terminated, truncated, info = env_cost_full.step(
                action
            )
            sin_result = np.sin(
                (
                    (2 * np.pi)
                    * env_cost_full.unwrapped._reference_frequency[1]
                    * env_cost_full.unwrapped.t
                )
                + env_cost_full.unwrapped._reference_phase_shift[1]
            )
            print(f"two_pi{i}=", 2 * np.pi)
            print(f"sin_result{i}=", sin_result)
            print(f"obs{i}=", repr(observation))
            print(f"reward{i}=", reward)
            print(f"terminated{i}=", terminated)
            print(f"truncated{i}=", truncated)
            print(f"info{i}=", info)

            assert (
                change_precision(observation, precision=PRECISION) == snapshot
            ).all()
            assert change_precision(reward, precision=PRECISION) == snapshot
            assert terminated == snapshot
            assert truncated == snapshot
            assert change_precision(info, precision=PRECISION) == snapshot
