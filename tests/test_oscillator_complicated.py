"""Test if the Oscillator environment still behaves like the previous Oscillator
environment.
"""
import gymnasium as gym
from gymnasium.logger import ERROR

import pytest
import stable_gym  # noqa: F401

gym.logger.set_level(ERROR)


class TestOscillatorComplicated:
    @pytest.fixture
    def env(self):
        """Create OscillatorComplicated environment."""
        return gym.make(
            "OscillatorComplicated", exclude_reference_error_from_observation=False
        )

    def test_reset(self, snapshot, env):
        """Test if reset is still equal to the last snapshot."""
        observation, info = env.reset(seed=42)
        assert (observation == snapshot).all()
        assert info == snapshot

    def test_step(self, snapshot, env):
        """Test if steps behave the same."""
        # Perform steps and check if observations are equal.
        _, _ = env.reset(seed=42)
        env.action_space.seed(42)
        for _ in range(5):
            action = env.action_space.sample()
            observation, reward, terminated, truncated, info = env.step(action)
            assert (observation == snapshot).all()
            assert reward == snapshot
            assert terminated == snapshot
            assert truncated == snapshot
            assert info == snapshot
