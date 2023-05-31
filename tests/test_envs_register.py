"""This script is used to test whether the gymnasium environments were successfully
registered.
"""

import importlib
import sys

import pytest
from gymnasium import envs

# Import simzoo stand-alone package or name_space package (mlc)
if "simzoo" in sys.modules:
    from simzoo import ENVS
elif importlib.util.find_spec("simzoo") is not None:
    importlib.import_module("simzoo")
    from simzoo import ENVS
else:
    importlib.import_module("bayesian_learning_control.simzoo")
    from bayesian_learning_control.simzoo.simzoo import ENVS


@pytest.mark.parametrize("env_name", ENVS.keys())
def test_env_reg(env_name):
    env = envs.make(env_name)
    assert env.spec.id == env_name
