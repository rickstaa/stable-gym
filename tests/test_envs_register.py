"""This script is used to test whether the gymnasium environments were successfully
registered.
"""
import pytest
from gymnasium import envs

# Import simzoo stand-alone package or name_space package (mlc)
from simzoo import ENVS


@pytest.mark.parametrize("env_name", ENVS.keys())
def test_env_reg(env_name):
    env = envs.make(env_name)
    assert env.spec.id == env_name
