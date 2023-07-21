"""Entry_point that register the Stable Gym gymnasium environments."""
import gymnasium as gym
from gymnasium.envs.registration import register

from stable_gym.common.max_episode_steps_injection_wrapper import (
    MaxEpisodeStepsInjectionWrapper,
)

# Make entry_point version available.
from .version import __version__, __version_tuple__

# Available environments.
# TODO: Update reward thresholds.
ENVS = {
    "Oscillator-v1": {
        "entry_point": "stable_gym.envs.biological.oscillator.oscillator:Oscillator",
        "reward_threshold": 300,
        "max_episode_steps": 400,
    },
    "OscillatorComplicated-v1": {
        "entry_point": "stable_gym.envs.biological.oscillator_complicated.oscillator_complicated:OscillatorComplicated",
        "reward_threshold": 300,
        "max_episode_steps": 400,
    },
    "CartPoleCost-v1": {
        "entry_point": "stable_gym.envs.classic_control.cartpole_cost.cartpole_cost:CartPoleCost",
        "reward_threshold": 300,
        "max_episode_steps": 250,
    },
    "CartPoleTrackingCost-v1": {
        "entry_point": "stable_gym.envs.classic_control.cartpole_tracking_cost.cartpole_tracking_cost:CartPoleCost",
        "reward_threshold": 300,
        "max_episode_steps": 250,
    },
    "Ex3EKF-v1": {
        "entry_point": "stable_gym.envs.classic_control.ex3_ekf.ex3_ekf:Ex3EKF",
        "reward_threshold": 300,
        "max_episode_steps": 400,
    },
    "AntCost-v1": {
        "entry_point": "stable_gym.envs.mujoco.ant_cost.ant_cost:AntCost",
        "reward_threshold": 300,
        "max_episode_steps": 250,
    },
    "HalfCheetahCost-v1": {
        "entry_point": "stable_gym.envs.mujoco.half_cheetah_cost.half_cheetah_cost:HalfCheetahCost",
        "reward_threshold": 300,
        "max_episode_steps": 200,
    },
    "HopperCost-v1": {
        "entry_point": "stable_gym.envs.mujoco.hopper_cost.hopper_cost:HopperCost",
        "reward_threshold": 300,
        "max_episode_steps": 250,
    },
    "HumanoidCost-v1": {
        "entry_point": "stable_gym.envs.mujoco.humanoid_cost.humanoid_cost:HumanoidCost",
        "reward_threshold": 300,
        "max_episode_steps": 250,
    },
    "SwimmerCost-v1": {
        "entry_point": "stable_gym.envs.mujoco.swimmer_cost.swimmer_cost:SwimmerCost",
        "reward_threshold": 300,
        "max_episode_steps": 250,
    },
    "Walker2dCost-v1": {
        "entry_point": "stable_gym.envs.mujoco.walker2d_cost.walker2d_cost:Walker2dCost",
        "reward_threshold": 300,
        "max_episode_steps": 250,
    },
    "FetchReachCost-v1": {
        "entry_point": "stable_gym.envs.robotics.fetch.fetch_reach_cost.fetch_reach_cost:FetchReachCost",
        "reward_threshold": 300,
        "max_episode_steps": 50,
    },
    # NOTE: The Minitaur environment is not compatible with gymnasium. See
    # https://github.com/bulletphysics/bullet3/issues/4369 for more details.
    "MinitaurBulletCost-v1": {
        "entry_point": "stable_gym.envs.robotics.minitaur.minitaur_bullet_cost.minitaur_bullet_cost:MinitaurBulletCost",
        "reward_threshold": 300,
        "max_episode_steps": 500,
        "compatible": False,
        "additional_wrappers": (MaxEpisodeStepsInjectionWrapper.wrapper_spec(),),
    },
}

for env, val in ENVS.items():
    register(
        id=env,
        entry_point=val["entry_point"],
        reward_threshold=val["reward_threshold"],
        max_episode_steps=val["max_episode_steps"]
        if "max_episode_steps" in val
        else None,
        disable_env_checker=not val["compatible"] if "compatible" in val else False,
        apply_api_compatibility=not val["compatible"] if "compatible" in val else False,
        additional_wrappers=val["additional_wrappers"]
        if "additional_wrappers" in val
        else (),
    )
