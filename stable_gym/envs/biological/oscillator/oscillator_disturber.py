"""A simple disturber class from which the Oscillator environment can inherit in order
to be able to use it with the Robustness Evaluation tool of the Bayesian Learning
Control package. For more information see the
`Robustness Evaluation <https://rickstaa.dev/stable-learning-control/control/robustness_eval.html>`_
documentation.
"""  # noqa: E501
# IMPROVE: File can be removed when a gymnasium wrapper is used.
from stable_gym.common.disturber import Disturber

# Disturber config used to overwrite the default config.
# NOTE: Merged with the default config
DISTURBER_CFG = {
    # Disturbance type when no type has been given.
    "default_type": "env",
    ##################################################
    # Environment disturbances #######################
    ##################################################
    # Disturbances applied to the *ENVIRONMENT* variables.
    # NOTE: The values below are meant as an example the environment disturbance config
    # needs to be implemented inside the environment.
    "env": {
        "description": "Lacl mRNA decay rate disturbance",
        # The env variable which you want to disturb.
        "variable": "K",  # Dissociation rate.
        # "variable": "c1",  # Promotor strength.
        # The range of values you want to use for each disturbance iteration.
        "variable_range": [5, 10],  # Dissociation rate.
        # "variable_range": [3.2, 4.8],  # Promotor strength.
        # Label used in robustness plots.
        "label": "r: %s",
    },
}


class OscillatorDisturber(Disturber):
    """Wrapper around the
    :meth:`~stable_gym.common.disturber.Disturber` that
    makes the disturber is compatible with the `Oscillator` environment.
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
        """Wrapper around the :meth:`~stable_gym.common.disturber.Disturber.init_disturber`
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
