"""Import biological environments onto namespace
"""

import sys
import importlib

# NOTE: Makes sure that it works both in the Simzoo stand-alone package and the name_space package (blc)
if "simzoo" in sys.modules:
    from simzoo.envs.biological.oscillator.oscillator import Oscillator
elif importlib.util.find_spec("simzoo") is not None:
    Oscillator = getattr(
        importlib.import_module("simzoo.envs.biological.oscillator.oscillator"),
        "Oscillator",
    )
else:
    Oscillator = getattr(
        importlib.import_module(
            "bayesian_learning_control.simzoo.simzoo.envs.biological.oscillator.oscillator"
        ),
        "Oscillator",
    )
