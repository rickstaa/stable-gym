"""Module that register the Stable Gym gymnasium environments.
"""
import gymnasium as gym
from gymnasium.envs.registration import register

# Make module version available.
from .version import __version__  # noqa: F401
from .version import __version_tuple__  # noqa: F401

# Available environments
ENVS = {
    "Oscillator-v1": {
        "module": "stable_gym.envs.biological.oscillator.oscillator:Oscillator",
        "max_step": 400,
        "reward_threshold": 300,
    },
    "Ex3EKF-v1": {
        "module": "stable_gym.envs.classic_control.ex3_ekf.ex3_ekf:Ex3EKF",
        "max_step": 400,
        "reward_threshold": 300,
    },
    "CartPoleCost-v1": {
        "module": "stable_gym.envs.classic_control.cartpole_cost.cartpole_cost:CartPoleCost",  # noqa: E501
        "max_step": 250,
        "reward_threshold": 300,
    },
}

for env, val in ENVS.items():
    if env not in gym.envs.registry:  # NOTE: Required because we use namespace packages
        register(
            id=env,
            entry_point=val["module"],
            max_episode_steps=val["max_step"],
            reward_threshold=val["reward_threshold"],
        )
