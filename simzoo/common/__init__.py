"""Contains several functions that can be used across all the simzoo environments.
"""
import sys
import importlib

if "simzoo" in sys.modules:
    from simzoo.common.disturber import Disturber
elif importlib.util.find_spec("simzoo") is not None:
    Disturber = getattr(importlib.import_module("simzoo.common.disturber"), "Disturber")
else:
    Disturber = getattr(
        importlib.import_module(
            "bayesian_learning_control.simzoo.simzoo.common.disturber"
        ),
        "Disturber",
    )
