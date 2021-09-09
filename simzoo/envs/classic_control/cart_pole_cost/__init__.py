"""A synthetic oscillatory network of transcriptional regulators gym environment."""
import importlib
import sys

# Import simzoo stand-alone package or name_space package (blc)
if "simzoo" in sys.modules:
    from simzoo.envs.classic_control.cart_pole_cost.cart_pole_cost import CartPoleCost
elif importlib.util.find_spec("simzoo") is not None:
    CartPoleCost = getattr(
        importlib.import_module(
            "simzoo.envs.classic_control.cart_pole_cost.cart_pole_cost"
        ),
        "CartPoleCost",
    )
else:
    CartPoleCost = getattr(
        importlib.import_module(
            "bayesian_learning_control.simzoo.simzoo.envs.classic_control.cart_pole_cost.cart_pole_cost"  # noqa: E501
        ),
        "CartPoleCost",
    )
