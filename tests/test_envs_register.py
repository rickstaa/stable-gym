"""This script is used to test whether the gym environments were successfully
registered.
"""

import importlib
import sys

import pytest
from gym import envs
from simzoo import ENVS

# Import simzoo stand-alone package or name_space package (mlc)
if "simzoo" in sys.modules:
    pass
elif importlib.util.find_spec("simzoo") is not None:
    importlib.import_module("simzoo")
else:
    importlib.import_module("machine_learning_control.simzoo")


@pytest.mark.parametrize("env_name", ENVS["name"])
def test_env_reg(env_name):
    env = envs.make(env_name)
    assert env.spec.id == env_name
