"""Test if the Oscillator environment still behaves like the previous Oscillator
environment.
"""
import gymnasium as gym
from gymnasium.logger import ERROR

import stable_gym  # noqa: F401

gym.logger.set_level(ERROR)


class TestOscillatorComplicated:
    env = gym.make(
        "OscillatorComplicated", exclude_reference_error_from_observation=False
    )

    def test_reset(self, snapshot):
        """Test if reset is still equal to the last snapshot."""
        observation, info = self.env.reset(seed=42)
        assert (observation == snapshot).all()
        assert info == snapshot

    def test_step(self, snapshot):
        """Test if steps behave the same."""
        # Perform steps and check if observations are equal.
        self.env.action_space.seed(42)
        for _ in range(5):
            action = self.env.action_space.sample()
            observation, reward, terminated, truncated, info = self.env.step(action)
            assert (observation == snapshot).all()
            assert reward == snapshot
            assert terminated == snapshot
            assert truncated == snapshot
            assert info == snapshot

    def test_constant_snapshot(self, snapshot):
        """Test if the 'CartPoleCost' environment with 'constant' reference is still
        equal to snapshot.
        """
        env_cost_reference_tracking = gym.make(
            "CartPoleCost",
            reference_type="constant",
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
