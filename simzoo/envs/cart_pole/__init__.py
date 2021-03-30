"""A synthetic oscillatory network of transcriptional regulators gym environment."""
import importlib
import sys

# Import simzoo stand-alone package or name_space package (blc)
if "simzoo" in sys.modules:
    from simzoo.envs.cart_pole.CartPole import CartPole
elif importlib.util.find_spec("simzoo") is not None:
    CartPole = getattr(
        importlib.import_module("simzoo.envs.cart_pole.CartPole"), "CartPole"
    )
else:
    CartPole = getattr(
        importlib.import_module(
            "bayesian_learning_control.simzoo.simzoo.envs.cart_pole.CartPole"
        ),
        "CartPole",
    )
