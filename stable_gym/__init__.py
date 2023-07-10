"""Module that register the Stable Gym gymnasium environments."""
import gymnasium as gym
from gymnasium.envs.registration import register

# Make module version available.
from .version import __version__, __version_tuple__

# Available environments.
# TODO: Update reward thresholds.
ENVS = {
    "Oscillator-v1": {
        "module": "stable_gym.envs.biological.oscillator.oscillator:Oscillator",
        "max_step": 400,
        "reward_threshold": 300,
    },
    "OscillatorComplicated-v1": {
        "module": "stable_gym.envs.biological.oscillator_complicated.oscillator_complicated:OscillatorComplicated",
        "max_step": 400,
        "reward_threshold": 300,
    },
    "CartPoleCost-v1": {
        "module": "stable_gym.envs.classic_control.cartpole_cost.cartpole_cost:CartPoleCost",
        "max_step": 250,
        "reward_threshold": 300,
    },
    "Ex3EKF-v1": {
        "module": "stable_gym.envs.classic_control.ex3_ekf.ex3_ekf:Ex3EKF",
        "max_step": 400,
        "reward_threshold": 300,
    },
    "AntCost-v1": {
        "module": "stable_gym.envs.mujoco.ant_cost.ant_cost:AntCost",
        "max_step": 250,
        "reward_threshold": 300,
    },
    "HalfCheetahCost-v1": {
        "module": "stable_gym.envs.mujoco.half_cheetah_cost.half_cheetah_cost:HalfCheetahCost",
        "max_step": 200,
        "reward_threshold": 300,
    },
    "HopperCost-v1": {
        "module": "stable_gym.envs.mujoco.hopper_cost.hopper_cost:HopperCost",
        "max_step": 250,
        "reward_threshold": 300,
    },
    "SwimmerCost-v1": {
        "module": "stable_gym.envs.mujoco.swimmer_cost.swimmer_cost:SwimmerCost",
        "max_step": 250,
        "reward_threshold": 300,
    },
    "Walker2dCost-v1": {
        "module": "stable_gym.envs.mujoco.walker2d_cost.walker2d_cost:Walker2dCost",
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
