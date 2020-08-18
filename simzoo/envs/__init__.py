"""Register the simzoo gym environments.
"""

import sys
import gym

# Import classes used for the gym environment registration
from gym.envs.registration import register

# Check if used as name_space package
namespace_prefix = "" if "simzoo" in sys.modules else "machine_learning_control.simzoo."

#################################################
# Register gym environments #####################
#################################################

# Environments
envs = {
    "name": ["Oscillator-v0"],
    "module": ["simzoo.envs.oscillator:Oscillator",],
}

# Register environments
for idx, env in enumerate(envs["name"]):
    if env not in gym.envs.registry.env_specs:  # Required because of namespace package
        register(
            id=env,
            entry_point=namespace_prefix + envs["module"][idx],
            max_episode_steps=50,
        )
