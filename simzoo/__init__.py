"""Register the simzoo gym environments.
"""

import importlib

import gym
from gym.envs.registration import register

# Create import prefix as stand-alone package or name_space package (mlc)
if importlib.util.find_spec("simzoo") is not None:
    namespace_prefix = ""
else:
    namespace_prefix = "machine_learning_control.simzoo."

envs = {
    "name": ["Oscillator-v1", "Ex3_EKF-v0"],
    "module": ["simzoo.envs.oscillator:Oscillator", "simzoo.envs.Ex3_EKF:Ex3_EKF"],
    "max_step": [800, 800],
}

for idx, env in enumerate(envs["name"]):
    if (
        env not in gym.envs.registry.env_specs
    ):  # NOTE (rickstaa): Required because of namespace package
        register(
            id=env,
            entry_point=namespace_prefix + envs["module"][idx],
            max_episode_steps=envs["max_step"][idx],
        )
