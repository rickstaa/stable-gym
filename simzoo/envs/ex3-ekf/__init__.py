"""Noisy master slave system (Ex3_EKF) gym environment."""
import sys
import importlib

# Import simzoo stand-alone package or name_space package (blc)
if "simzoo" in sys.modules:
    from simzoo.envs.ex3_ekf.ex3_ekf import Ex3_EKF
elif importlib.util.find_spec("simzoo") is not None:
    Ex3_EKF = getattr(importlib.import_module("simzoo.envs.ex3_ekf.ex3_ekf"), "Ex3_EKF")
else:
    Ex3_EKF = getattr(
        importlib.import_module(
            "bayesian_learning_control.simzoo.simzoo.envs.ex3_ekf.ex3_ekf"
        ),
        "Ex3_EKF",
    )
