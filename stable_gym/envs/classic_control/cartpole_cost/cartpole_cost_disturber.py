"""A simple disturber class from which the CartPoleCost environment can inherit in order
to be able to use it with the Robustness Evaluation tool of the Bayesian Learning
Control package. For more information see the
`Robustness Evaluation <https://rickstaa.dev/stable-learning-control/control/robustness_eval.html>`_
documentation.
"""  # noqa: E501
import numpy as np
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
        "description": "Pole length disturbance",
        # The env variable which you want to disturb.
        "variable": "length",
        # The range of values you want to use for each disturbance iteration.
        "variable_range": np.linspace(0.1, 4.0, num=6, dtype=np.float32),
        # Label used in robustness plots.
        "label": "l: %s",
    },
    "input": {
        # The disturbance variant used when no variant is given.
        "default_variant": "impulse",
        # Impulse disturbance applied opposite to the action at a given timestep.
        "impulse": {
            "description": "Impulse disturbance",
            # The step at which you want to first apply the impulse.
            "impulse_instant": 100,
            # The length of the impulse in seconds.
            "impulse_length": 1.0,
            # The frequency of the impulse in Hz. If you  specify 0.0 only one impulse
            # is given at the impulse instant).
            "impulse_frequency": 0.05,
            # The magnitudes you want to apply.
            "magnitude_range": np.linspace(80, 155, num=5, dtype=np.float32),
            # Label used in robustness plots.
            "label": "M: %s",
        },
        # Similar above but now the impulse force is continuously applied after the
        # impulse instant has been reached.
        "constant_impulse": {
            "description": "Constant impulse disturbance",
            # The step at which you want to apply the impulse.
            "impulse_instant": 100,
            # The magnitudes you want to apply.
            "magnitude_range": np.linspace(80, 155, num=3, dtype=np.int16),
            # Label that can be used in plots.
            "label": "M: %s",
        },
        # A periodic signal noise that is applied at every time step.
        # NOTE: Currently you can only uncomment one of the ranges.
        "periodic": {
            "description": "Periodic noise disturbance",
            # The magnitudes of the periodic signal.
            "amplitude_range": np.linspace(10, 80, num=3, dtype=np.int16),
            # The frequency of the periodic signal.
            # "frequency_range": np.linspace(0, 10, num=3, dtype=np.int16),
            # The phase of the periodic signal.
            # "phase_range": np.linspace(0.0, 90, num=3, dtype=np.int16),
            # Label used in robustness plots.
            "label": "A: %s",
        },
        # A random noise that is applied at every timestep.
        "noise": {
            "description": "Random noise disturbance",
            # The means and standards deviations of the random noise disturbance.
            "noise_range": {
                "mean": np.linspace(0.0, 0.0, num=3, dtype=np.float32),
                "std": np.linspace(1.0, 5.0, num=3, dtype=np.float32),
            },
            # Label used in robustness plots.
            "label": "x̅: %s, σ: %s",
        },
    },
    ##################################################
    # Output disturbances ############################
    ##################################################
    # Disturbance applied to the *OUTPUT* of the environment step function
    "output": {
        # The disturbance variant used when no variant is given.
        "default_variant": "impulse",
        # A random noise that is applied at every timestep.
        "noise": {
            "description": "Random noise disturbance",
            # The means and standards deviations of the random noise disturbance.
            "noise_range": {
                # "mean": np.linspace(80, 155, num=3, dtype=np.int16),  # All obs.
                "mean": np.vstack(
                    (
                        np.linspace(80, 155, num=3, dtype=np.int16),  # Obs 1
                        np.linspace(80, 155, num=3, dtype=np.int16),  # Obs 2
                        np.linspace(80, 155, num=3, dtype=np.int16),  # Obs 3
                        np.linspace(80, 155, num=3, dtype=np.int16),  # Obs 4
                    )
                ).T,
                # "std": np.linspace(1.0, 5.0, num=3, dtype=np.int16),  # All Obs.
                "std": np.vstack(
                    (
                        np.linspace(1.0, 5.0, num=3, dtype=np.int16),  # Obs 1
                        np.linspace(1.0, 5.0, num=3, dtype=np.int16),  # Obs 2
                        np.linspace(1.0, 5.0, num=3, dtype=np.int16),  # Obs 3
                        np.linspace(1.0, 5.0, num=3, dtype=np.int16),  # Obs 4
                    )
                ).T,
            },
            # Label used in robustness plots.
            "label": "x̅: %s, σ: %s",
        },
    },
}


class CartPoleDisturber(Disturber):
    """Wrapper around the
    :meth:`~stable_gym.common.disturber.Disturber` that
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
