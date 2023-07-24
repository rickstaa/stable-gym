"""This script is used to test whether the gymnasium environments were successfully
registered.
"""
import pytest
from gymnasium import envs
import pybullet

from stable_gym import ENVS

QUADX_ENVS = [env_name for env_name in ENVS.keys() if "Quad" in env_name]


@pytest.mark.parametrize("env_name", ENVS.keys())
def test_env_reg(env_name):
    # TODO: Can be removed if https://github.com/jjshoots/PyFlyt/issues/1 is resolved.
    # Skip if QuadX environment and pybullet has no numpy support.
    if env_name in QUADX_ENVS:
        if not pybullet.isNumpyEnabled():
            pytest.skip(
                "pybullet was not built with numpy support. Please rebuild pybullet "
                "with numpy enabled."
            )

    env = envs.make(env_name)
    assert env.spec.id == env_name
