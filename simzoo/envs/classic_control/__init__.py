"""Environments for classical control theory problems.
"""
import importlib
import sys

# NOTE: Makes sure that it works both in the Simzoo stand-alone package and the
# name_space package (blc)
if "simzoo" in sys.modules:
    from simzoo.envs.classic_control.cartpole_cost.cartpole_cost import CartPoleCost
    from simzoo.envs.classic_control.ex3_ekf.ex3_ekf import Ex3EKF
elif importlib.util.find_spec("simzoo") is not None:
    Ex3EKF = getattr(
        importlib.import_module("simzoo.envs.classic_control.ex3_ekf.ex3_ekf"),
        "Ex3EKF",
    )
    CartPoleCost = getattr(
        importlib.import_module(
            "simzoo.envs.classic_control.cartpole_cost.cartpole_cost"
        ),
        "CartPoleCost",
    )
else:
    Ex3EKF = getattr(
        importlib.import_module(
            "bayesian_learning_control.simzoo.simzoo.envs.classic_control.ex3_ekf.ex3_ekf"  # noqa: E501
        ),
        "Ex3EKF",
    )
    CartPoleCost = getattr(
        importlib.import_module(
            "bayesian_learning_control.simzoo.simzoo.envs.classic_control.cartpole_cost.cartpole_cost"  # noqa: E501
        ),
        "CartPoleCost",
    )
