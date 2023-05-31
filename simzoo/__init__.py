"""Module that register the Simzoo gymnasium environments.
"""
import importlib

import gymnasium as gym
from gymnasium.envs.registration import register

# Create import prefix as stand-alone package or name_space package (mlc)
if importlib.util.find_spec("simzoo") is not None:
    namespace_prefix = ""
else:
    namespace_prefix = "bayesian_learning_control.simzoo."

# Available environments
ENVS = {
    "Oscillator-v1": {
        "module": "simzoo.envs.biological.oscillator.oscillator:Oscillator",
        "max_step": 400,
        "reward_threshold": 300,
    },
    "Ex3EKF-v1": {
        "module": "simzoo.envs.classic_control.ex3_ekf.ex3_ekf:Ex3EKF",
        "max_step": 400,
        "reward_threshold": 300,
    },
    "CartPoleCost-v0": {
        "module": "simzoo.envs.classic_control.cart_pole_cost.cart_pole_cost:CartPoleCost",  # noqa: E501
        "max_step": 400,
        "reward_threshold": 300,
    },
}

for env, val in ENVS.items():
    if env not in gym.envs.registry:  # NOTE: Required because we use namespace packages
        register(
            id=env,
            entry_point=namespace_prefix + val["module"],
            max_episode_steps=val["max_step"],
            reward_threshold=val["reward_threshold"],
        )
