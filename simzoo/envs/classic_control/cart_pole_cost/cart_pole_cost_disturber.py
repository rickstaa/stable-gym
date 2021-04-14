"""A simple disturber class from which the CartPoleCost environment can inherit in order
to be able to use it with the Robustness Evaluation tool of the Bayesian Learning
Control package. For more information see the
`Robustness Evaluation <https://rickstaa.github.io/bayesian-learning-control/control/robustness_eval.html>`_
documentation.
"""  # noqa: E501


import importlib
import sys

import numpy as np

# Try to import the disturber class
# NOTE: Only works if the simzoo or bayesian learning control package is installed.
# fallback to object if not successfull.
if "simzoo" in sys.modules:
    from simzoo.common.disturber import Disturber
elif importlib.util.find_spec("simzoo") is not None:
    Disturber = getattr(importlib.import_module("simzoo.common.disturber"), "Disturber")
else:
    try:
        Disturber = getattr(
            importlib.import_module(
                "bayesian_learning_control.simzoo.simzoo.common.disturber"
            ),
            "Disturber",
        )
    except AttributeError:
        Disturber = object


# Disturber config used to overwrite the default config
# NOTE: Merged with the default config
DISTURBER_CFG = {
    # Disturbance type when no type has been given
    "default_type": "env_disturbance",
    # Disturbance applied to environment variables
    "env_disturbance": {
        "description": "Pole length disturbance",
        # The env variable which you want to disturb
        "variable": "length",
        # The range of values you want to use for each disturbance iteration
        "variable_range": np.linspace(0.1, 4.0, num=6, dtype=np.float32),
        # Label used in robustness plots.
        "label": "l: %s.3f",
    },
}


class CartPoleDisturber(Disturber):
    """Wrapper around the
    :meth:`~bayesian_learning_control.simzoo.simzoo.common.disturber.Disturber` that
    makes the disturber is compatible with the `CartPoleCost` environment.
    """

    def __init__(self, *args, **kwargs):
        """Initiate CartPoleDisturber object

        Args:
            *args: All args to pass to the parent :meth:`__init__` method.
            **kwargs: All kwargs to pass to the parent :meth:`__init__` method.
        """  # noqa E501
        kwargs["disturber_cfg"] = (
            {**DISTURBER_CFG, **kwargs["disturber_cfg"]}
            if "disturber_cfg" in kwargs.keys()
            else DISTURBER_CFG
        )
        super().__init__(*args, **kwargs)

    def init_disturber(self, *args, **kwargs):
        """Wrapper around the :meth:`~bayesian_learning_control.simzoo.simzoo.common.disturber.Disturber.init_disturber`
        method that makes sure an up to date version of the environment
        :obj:`DISTURBER_CFG` is used.

        Args:
            *args: All args to pass to the parent :meth:`init_disturber` method.
            **kwargs: All kwargs to pass to the parent :meth:`init_disturber` method.
        """  # noqa E501
        kwargs["disturber_cfg"] = (
            {**DISTURBER_CFG, **kwargs["disturber_cfg"]}
            if "disturber_cfg" in kwargs.keys()
            else DISTURBER_CFG
        )
        return super().init_disturber(*args, **kwargs)
