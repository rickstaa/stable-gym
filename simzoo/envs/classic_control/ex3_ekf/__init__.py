"""Noisy master slave system (Ex3EKF) gym environment."""
import sys
import importlib

# Import simzoo stand-alone package or name_space package (blc)
if "simzoo" in sys.modules:
    from simzoo.envs.classic_control.ex3_ekf.ex3_ekf import Ex3EKF
elif importlib.util.find_spec("simzoo") is not None:
    Ex3EKF = getattr(
        importlib.import_module("simzoo.envs.classic_control.ex3_ekf.ex3_ekf"), "Ex3EKF"
    )
else:
    Ex3EKF = getattr(
        importlib.import_module(
            "bayesian_learning_control.simzoo.simzoo.envs.classic_control.ex3_ekf.ex3_ekf"
        ),
        "Ex3EKF",
    )
