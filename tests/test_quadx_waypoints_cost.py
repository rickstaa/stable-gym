"""Test if the QuadXWaypointsCost environment still behaves like the original
QuadXWaypoints environment when the same environment parameters are used.
"""
import gymnasium as gym
import numpy as np
from gymnasium.logger import ERROR

from stable_gym.common.utils import change_precision

gym.logger.set_level(ERROR)

PRECISION = 15


class TestQuadXWaypointsCostEqual:
    # Make original QuadXWaypoints environment.
    env = gym.make("PyFlyt.gym_envs:PyFlyt/QuadX-Waypoints-v0")
    # Make QuadXWaypointsCost environment.
    env_cost = gym.make(
        "QuadXWaypointsCost",
        exclude_waypoint_targets_from_observation=True,
        only_observe_immediate_waypoint=False,
        exclude_waypoint_target_deltas_from_observation=False,
        only_observe_immediate_waypoint_target_delta=False,
    )

    def test_equal_reset(self):
        """Test if reset behaves the same."""
        # Perform reset and check if observations are equal.
        observation, _ = self.env.reset(seed=42)
        observation = np.concatenate(
            [observation["attitude"], observation["target_deltas"].flatten()]
        )  # Flatten and concatenate observation dictionary.
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
            observation = np.concatenate(
                [observation["attitude"], observation["target_deltas"].flatten()]
            )  # Flatten and concatenate observation dictionary.
            observation_cost, _, _, _, _ = self.env_cost.step(action)
            assert np.allclose(
                observation, observation_cost
            ), f"{observation} != {observation_cost}"

    # NOTE: We decrease the test precision to 16 decimals to ignore numerical
    # differences due to hardware or library differences.
    def test_snapshot(self, snapshot):
        """Test if the 'QuadXWaypointsCost' environment is still equal to snapshot."""
        env_cost = gym.make("QuadXWaypointsCost")  # Check full observation.
        observation, info = env_cost.reset(seed=42)
        assert (change_precision(observation, precision=PRECISION) == snapshot).all()
        assert change_precision(info, precision=PRECISION) == snapshot
        env_cost.action_space.seed(42)
        for _ in range(5):
            action = env_cost.action_space.sample()
            observation, reward, terminated, truncated, info = env_cost.step(action)
            assert (
                change_precision(observation, precision=PRECISION) == snapshot
            ).all()
            assert change_precision(reward, precision=PRECISION) == snapshot
            assert terminated == snapshot
            assert truncated == snapshot
            assert change_precision(info, precision=PRECISION) == snapshot
