"""Module that contains the simzoo :gym:`gym <>` environments.
"""

# Import modules
from simzoo.envs.oscillator.oscillator import Oscillator

# Import classes used for the gym environment registration
from gym.envs.registration import register

#################################################
# Register gym environments #####################
#################################################

# Register oscillator environment
register(
    id="Oscillator-v0",
    entry_point="simzoo.envs.oscillator:Oscillator",
    max_episode_steps=50,
)
