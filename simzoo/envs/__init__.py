"""The Simzoo Gym environments."""
import importlib
import sys

# Import simzoo stand-alone package or name_space package (mlc)
if "simzoo" in sys.modules:
    from simzoo.envs.oscillator.oscillator import Oscillator
    from simzoo.envs.ex3_ekf.ex3_ekf import Ex3_EKF
    from simzoo.envs.cart_pole.CartPole import CartPoleCustom
elif importlib.util.find_spec("simzoo") is not None:
    Oscillator = getattr(
        importlib.import_module("simzoo.envs.oscillator.oscillator"), "Oscillator"
    )
    Ex3_EKF = getattr(importlib.import_module("simzoo.envs.ex3_ekf.ex3_ekf"), "Ex3_EKF")
    CartPole = getattr(importlib.import_module("simzoo.envs.ex3_ekf.cart_pole"), "CartPoleCustom")
else:
    Oscillator = getattr(
        importlib.import_module(
            "bayesian_learning_control.simzoo.simzoo.envs.oscillator.oscillator"
        ),
        "Oscillator",
    )
    Ex3_EKF = getattr(
        importlib.import_module(
            "bayesian_learning_control.simzoo.simzoo.envs.ex3_ekf.ex3_ekf"
        ),
        "Ex3_EKF",
    )
    CartPole = getattr(
        importlib.import_module(
            "bayesian_learning_control.simzoo.simzoo.envs.cart_pole.CartPole"
        ),
        "CartPoleCustom",
    )
