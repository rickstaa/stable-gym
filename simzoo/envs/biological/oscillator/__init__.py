"""A synthetic oscillatory network of transcriptional regulators gym environment."""
import importlib
import sys

# Import simzoo stand-alone package or name_space package (blc)
if "simzoo" in sys.modules:
    from simzoo.envs.oscillator.oscillator import Oscillator
elif importlib.util.find_spec("simzoo") is not None:
    Oscillator = getattr(
        importlib.import_module("simzoo.envs.oscillator.oscillator"), "Oscillator"
    )
else:
    Oscillator = getattr(
        importlib.import_module(
            "bayesian_learning_control.simzoo.simzoo.envs.oscillator.oscillator"
        ),
        "Oscillator",
    )
