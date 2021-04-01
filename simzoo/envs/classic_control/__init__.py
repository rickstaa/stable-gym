"""Environments for classical control theory problems.
"""

import sys
import importlib

# NOTE: Makes sure that it works both in the Simzoo stand-alone package and the name_space package (blc)
if "simzoo" in sys.modules:
    from simzoo.envs.classic_control.ex3_ekf.ex3_ekf import Ex3EKF
    from simzoo.envs.classic_control.cart_pole_cost.cart_pole_cost import CartPoleCost
elif importlib.util.find_spec("simzoo") is not None:
    Ex3EKF = getattr(
        importlib.import_module("simzoo.envs.classic_control.ex3_ekf.ex3_ekf"),
        "Ex3EKF",
    )
    CartPoleCost = getattr(
        importlib.import_module(
            "simzoo.envs.classic_control.cart_pole_cost.cart_pole_cost"
        ),
        "CartPoleCost",
    )
else:
    Ex3EKF = getattr(
        importlib.import_module(
            "bayesian_learning_control.simzoo.simzoo.envs.classic_control.ex3_ekf.ex3_ekf"
        ),
        "Ex3EKF",
    )
    CartPoleCost = getattr(
        importlib.import_module(
            "bayesian_learning_control.simzoo.simzoo.envs.classic_control.cart_pole_cost.cart_pole_cost"
        ),
        "CartPoleCost",
    )