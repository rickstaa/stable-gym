"""This script is used to test whether the gym environments were successfully
registered.
"""

# Main python imports
from gym import envs
import pytest

# Import panda openai sim task environments
import simzoo.envs

# Script Parameters
ENVS = ["Oscillator-v0"]


#################################################
# Test script ###################################
#################################################
@pytest.mark.parametrize("env_name", ENVS)
def test_env_reg(env_name):
    env = envs.make(env_name)
    assert env.spec.id == env_name
