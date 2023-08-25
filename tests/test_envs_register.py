"""This script is used to test whether the gymnasium environments were successfully
registered.
"""
from gymnasium import envs

import pytest
from stable_gym import ENVS


@pytest.mark.parametrize("env_name", ENVS.keys())
def test_env_reg(env_name):
    env = envs.make(env_name)
    assert env.spec.id == env_name
