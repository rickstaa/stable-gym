"""A disturber class from which a OpenAi Gym Environment can inherit in order to be able
to use it with the Robustness Evaluation tool of the Bayesian Learning Control package.
For more information see the
`Robustness Evaluation <https://rickstaa.dev/stable-learning-control/control/robustness_eval.html>`_
documentation.
"""  # noqa: E501
# IMPROVE: Replace with gymnasium wrappers https://alexandervandekleut.github.io/gym-wrappers/ # noqa: E501
import re

import gymnasium as gym
import numpy as np
from iteration_utilities import deepflatten

from .disturbances import impulse_disturbance, noise_disturbance, periodic_disturbance
from .utils import (
    abbreviate,
    colorize,
    friendly_list,
    get_flattened_keys,
    get_flattened_values,
    inject_value,
    strip_underscores,
)

# Default Disturber configuration variable.
# NOTE: You can also supply the disturber with your own disturbance configuration
# dictionary. When doing this you have to make sure it contains all the required keys.
# See https://rickstaa.dev/stable-learning-control/control/robustness_eval.html
# for more information.
DISTURBER_CFG = {
    # Disturbance type when no type has been given.
    "default_type": "input",
    ##################################################
    # Environment disturbances #######################
    ##################################################
    # Disturbances applied to the *ENVIRONMENT* variables.
    # NOTE: The values below are meant as an example the environment disturbance config
    # needs to be implemented inside the environment.
    "env": {
        "description": "Lacl mRNA decay rate disturbance",
        # The env variable which you want to disturb.
        "variable": "c1",
        # The range of values you want to use for each disturbance iteration.
        "variable_range": np.linspace(1.6, 3.0, num=5, dtype=np.float32),
        # Label used in robustness plots.
        "label": "r: %s",
    },
    ##################################################
    # Input disturbances #############################
    ##################################################
    # Disturbance applied to the *INPUT* of the environment step function
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
            # The frequency of the impulse. If you  specify 0.0 only one impulse is
            # given at the impulse instant).
            "impulse_frequency": 0.0,
            # The magnitudes you want to apply.
            "magnitude_range": np.linspace(80, 155, num=5, dtype=np.float32),
            # Label used in robustness plots.
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
        "default_variant": "noise",
        # Impulse disturbance applied opposite to the action at a given timestep.
        "impulse": {
            "description": "Impulse disturbance",
            # The step at which you want to first apply the impulse.
            "impulse_instant": 100,
            # The length of the impulse in seconds.
            "impulse_length": 1.0,
            # The frequency of the impulse. If you  specify 0.0 only one impulse is
            # given at the impulse instant).
            "impulse_frequency": 0.0,
            # The magnitudes you want to apply.
            "magnitude_range": np.linspace(0.0, 3.0, num=5, dtype=np.float32),
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
    # Input/Output disturbances ######################
    ##################################################
    # Disturbance applied to both the *INPUT* and *OUTPUT* of the environment step
    # function.
    "combined": {
        # The disturbance variant used when no variant is given.
        "default_variant": "noise",
        # Input and output impulse disturbance.
        "impulse": {
            "description": "Impulse disturbance",
            "input_impulse": {
                # The step at which you want to apply the impulse.
                "impulse_instant": 100,
                # The magnitudes you want to apply.
                "magnitude_range": np.linspace(0.0, 3.0, num=5, dtype=np.float32),
            },
            "output_impulse": {
                # The step at which you want to apply the impulse.
                "impulse_instant": 100,
                # The magnitudes you want to apply.
                "magnitude_range": np.linspace(0.0, 3.0, num=5, dtype=np.float32),
            },
            # Label used in robustness plots.
            "label": "M: (%s, %s)",
        },
        # Input and output noise disturbance.
        "noise": {
            "description": "Random input and output noise disturbance",
            "input_noise": {
                # The means and standards deviations of the random input noise
                # disturbance.
                "noise_range": {
                    "mean": np.linspace(0.0, 0.0, num=3, dtype=np.float32),
                    "std": np.linspace(1.0, 5.0, num=3, dtype=np.float32),
                },
            },
            "output_noise": {
                # The means and standards deviations of the random output noise
                # disturbance.
                "noise_range": {
                    "mean": np.linspace(0.0, 0.0, num=3, dtype=np.float32),
                    "std": np.linspace(1.0, 5.0, num=3, dtype=np.float32),
                },
            },
            # Label used in robustness plots.
            "label": "x̅: (%s, %s), σ: (%s, %s)",
        },
        # Input impulse disturbance and a output noise disturbance.
        "impulse_noise": {
            "description": "Input impulse disturbance and output noise disturbance",
            "input_impulse": {
                # The step at which you want to apply the impulse.
                "impulse_instant": 100,
                # The magnitudes you want to apply.
                "magnitude_range": np.linspace(0.0, 3.0, num=5, dtype=np.float32),
            },
            "output_noise": {
                # The means and standards deviations of the random output noise
                # disturbance.
                "noise_range": {
                    "mean": np.linspace(0.0, 0.0, num=3, dtype=np.float32),
                    "std": np.linspace(1.0, 5.0, num=3, dtype=np.float32),
                },
            },
            # Label used in robustness plots.
            "label": "M: %s - x̅:(%s), σ:(%s)",
        },
    },
}


class Disturber:
    """Environment disturbance class. This class adds additional methods to a OpenAi Gym
    Environemt in order to make it compatible with the Robustness Eval tool of the
    Bayesian Learning Control package.

    Attributes:
        disturber_done (bool): Whether the disturber has looped through all the
            available disturbances.
        disturbance_info (dict): Some additional information about the disturbances the
            Disturber applied. Useful for plotting.

    .. seealso::

        For more information see the
        `Robustness Evaluation <https://rickstaa.dev/stable-learning-control/control/robustness_eval.html>`_
        documentation.
    """  # noqa: E501

    def __init__(self, disturber_cfg=None):
        """Initiate disturber object.

        Args:
            disturber_cfg (dict, optional): A dictionary that describes the
                disturbances the :class:`Disturber` supports. This dictionary can be
                used to update values of the ``DISTURBANCE_CFG`` configuration which is
                present in the :class:`Disturber` class file.
        """  # noqa: E501
        assert any([issubclass(item, gym.Env) for item in self.__class__.__bases__]), (
            "Only classes that also inherit from the 'gymnasium.Env' class can inherit "
            "from the 'Disturber' class."
        )

        self.disturber_done = False
        self.disturbance_info = {}
        self._disturbance_done_warned = False
        self._disturber_cfg = (
            {**DISTURBER_CFG, **disturber_cfg}
            if disturber_cfg is not None
            else DISTURBER_CFG
        )  # Allow users to overwrite the default config.
        self._disturbance_cfg = None
        self._disturbance_type = None
        self._disturbance_variant = None
        self._disturbance_sub_variants = None
        self._disturbance_range_idx = 0
        self._disturbance_range_keys = None
        self._disturbance_range_length = None
        self._has_time_vars = None
        self._disturbance_significance = 2  # Disturbance range rounding significance.

    def _initiate_time_vars(self):
        """Initiates a time ``t`` and timestep ``dt`` variable if they are not present
        on the environment.
        """
        if not hasattr(self, "t"):
            self.t = 0.0
            self.dt = 1.0
            self._has_time_vars = False
        else:
            self._has_time_vars = True
        if not hasattr(self, "dt"):
            if hasattr(self, "tau"):
                self.dt = self.tau  # Some environments use tau instead of dt.
            else:
                self.dt = 1.0

    def _get_disturbance(  # noqa: C901
        self, input_signal, disturbance_variant, disturbance_cfg
    ):
        """Retrieves the right disturbance using the disturbance type and variant that
        were set using the :meth:`Disturber.init_disturber` method.

        Args:
            input_signal (numpy.ndarray): The signal to which the disturbance should be
                applied.
            disturbance_variant (str): Which disturbance variant you want to retrieve.
                Options are: 'impulse', 'constant_impulse', 'periodic' and 'noise'.
            disturbance_cfg (dict): The disturbance config used for createing the
                disturbance.

        Returns:
            numpy.ndarray: The disturbance array.
        """
        # Set the disturber state to done and return a zero disturbance if the user has
        # used all the specified disturbances.
        if (
            self._disturbance_range_length is not None
            and self._disturbance_range_idx > (self._disturbance_range_length - 1)
        ):
            if not self._disturbance_done_warned:
                self._disturbance_done_warned = True
                print(
                    colorize(
                        "WARNING: You are trying to apply a '%s' disturbance to the  "
                        "step while no disturbances are left in the disturber. As a  "
                        "result no disturbance will be added to the step.",
                        "yellow",
                        bold=True,
                    )
                    % self._disturbance_variant
                )
            return np.zeros_like(input_signal)

        # Retrieve the requested disturbance.
        # NOTE: New disturbances should be aded here.
        current_timestep = self.t / self.dt
        if "impulse" in disturbance_variant:
            impulse_magnitude = disturbance_cfg["magnitude_range"][
                self._disturbance_range_idx
            ]
            impulse_instant = disturbance_cfg["impulse_instant"]
            impulse_length = (
                disturbance_cfg["impulse_length"]
                if "impulse_length" in disturbance_cfg.keys()
                else None
            )
            impulse_frequency = (
                disturbance_cfg["impulse_frequency"]
                if "impulse_frequency" in disturbance_cfg.keys()
                else None
            )
            return impulse_disturbance(
                input_signal=input_signal,
                impulse_magnitude=impulse_magnitude,
                impulse_instant=impulse_instant,
                current_timestep=current_timestep,
                impulse_length=impulse_length,
                impulse_frequency=impulse_frequency,
            )
        elif disturbance_variant == "periodic":
            signal_kwargs = {}
            if any(
                ["amplitude" in item.lower() for item in self._disturbance_range_keys]
            ):
                signal_kwargs["amplitude"] = disturbance_cfg["amplitude_range"][
                    self._disturbance_range_idx
                ]
            elif any(
                ["frequency" in item.lower() for item in self._disturbance_range_keys]
            ):
                signal_kwargs["frequency"] = disturbance_cfg["frequency_range"][
                    self._disturbance_range_idx
                ]
            elif any(
                ["phase" in item.lower() for item in self._disturbance_range_keys]
            ):
                signal_kwargs["phase"] = disturbance_cfg["phase_range"][
                    self._disturbance_range_idx
                ]

            # Make sure the signal kwarg has the right shape.
            if not isinstance(signal_kwargs.values(), np.ndarray):
                signal_kwargs = {
                    k: np.repeat(v, input_signal.shape)
                    for k, v in signal_kwargs.items()
                }

            return periodic_disturbance(current_timestep, **signal_kwargs)
        elif disturbance_variant == "noise":
            mean = disturbance_cfg["noise_range"]["mean"][self._disturbance_range_idx]
            std = disturbance_cfg["noise_range"]["std"][self._disturbance_range_idx]

            # Make sure that the mean and std have the right shape.
            if not isinstance(mean, np.ndarray):
                mean = np.repeat(mean, input_signal.shape)
            if not isinstance(std, np.ndarray):
                std = np.repeat(std, input_signal.shape)

            return noise_disturbance(mean, std)
        else:
            raise NotImplementedError(
                f"Disturbance variant '{self._disturbance_variant}' not yet "
                "implemented."
            )

    def _set_disturber_type(self, disturbance_type=None):
        """Validates the input disturbance type and sets it when it exists.

        Args:
            disturbance_type (str, optional): The disturbance type you want to use. By
                default the disturbance type set under the `default_type` key will be
                used.

        Raises:
            ValueError: Thrown when the disturbance type does not exist on the
                disturber.
        """
        if disturbance_type is not None:
            disturbance_type_input = disturbance_type
            disturbance_type = [
                item
                for item in list(set([disturbance_type, disturbance_type.lower()]))
                if item in self._disturber_cfg.keys()
            ]  # Catch some common human writing errors.
            disturbance_type = disturbance_type[0] if disturbance_type else None

            # Throw warning if disturbance type was invalid.
            if not disturbance_type:
                try:
                    environment_name = self.unwrapped.spec.id
                except AttributeError:
                    environment_name = self.__class__.__name__.__str__()
                valid_type_keys = {
                    k for k in self._disturber_cfg.keys() if k not in ["default_type"]
                }
                d_type_info_msg = (
                    f"Disturbance type '{disturbance_type_input}' is not implemented "
                    f"for the '{environment_name}' environment. Please specify a "
                    f"valid disturbance type. Valid types are:"
                )
                for item in valid_type_keys:
                    d_type_info_msg += f"\t\n - {item}"
                raise ValueError(
                    d_type_info_msg,
                    "disturbance_type",
                )
        else:
            # Try to retrieve default disturbance type if type was not found.
            if "default_type" in self._disturber_cfg.keys():
                print(
                    colorize(
                        (
                            "INFO: No disturbance type given. Default type '"
                            + "{}' used instead.".format(
                                self._disturber_cfg["default_type"]
                            )
                        ),
                        "green",
                        bold=True,
                    )
                )
                disturbance_type = self._disturber_cfg["default_type"]
            else:  # Thrown warning if type could not be retrieved.
                valid_type_keys = {
                    k for k in self._disturber_cfg.keys() if k not in ["default_type"]
                }
                d_type_info_msg = (
                    "init_disturber(): is missing one required positional argument "
                    "'disturbance_type'. Please specify a valid disturbance type. "
                    "Valid types are:"
                )
                for item in valid_type_keys:
                    d_type_info_msg += f"\t\n - {item}"
                raise ValueError(
                    d_type_info_msg,
                    "disturbance_type",
                )
        self._disturbance_type = disturbance_type

    def _set_disturber_variant(self, disturbance_variant):
        """Validates the input disturbance variant and sets it when it exists.

        Args:
            disturbance_variant (str): The disturbance variant you want to use.

        Raises:
            ValueError: Thrown when the disturbance variant is not available for the set
                disturbance type.
            Exception: Thrown when the disturbance subvariants of the 'combined'
                disturbance could not be determined.
        """
        if self._disturbance_type == "env":
            if disturbance_variant is not None:
                print(
                    colorize(
                        (
                            f"WARNING: Disturbance variant '{disturbance_variant}' "
                            "ignored as it does not apply when using disturbance type "
                            "'env'."
                        ),
                        "yellow",
                        bold=True,
                    )
                )
            self._disturbance_variant = "environment"
        else:
            if disturbance_variant is not None:
                disturbance_variant_input = disturbance_variant
                disturbance_variant = [
                    item
                    for item in list(
                        set([disturbance_variant, disturbance_variant.lower()])
                    )
                    if item in self._disturber_cfg[self._disturbance_type].keys()
                ]  # Catch some common human writing errors.
                disturbance_variant = (
                    disturbance_variant[0] if disturbance_variant else None
                )

                # Throw warning if disturbance variant was invalid.
                if not disturbance_variant:
                    valid_variant_keys = {
                        k
                        for k in self._disturber_cfg[self._disturbance_type].keys()
                        if k not in ["default_type"]
                    }
                    d_variant_info_msg = (
                        f"Disturber variant '{disturbance_variant_input}' is not "
                        f"implemented for disturbance type '{self._disturbance_type}'. "
                        "Please specify a valid disturbance variant. Valid variants "
                        "are:"
                    )
                    for item in valid_variant_keys:
                        d_variant_info_msg += f"\t\n - {item}"
                    raise ValueError(
                        d_variant_info_msg,
                        "disturbance_variant",
                    )
            else:
                # Try to retrieve default disturbance variant if variant was not found.
                if (
                    "default_variant"
                    in self._disturber_cfg[self._disturbance_type].keys()
                ):
                    print(
                        colorize(
                            (
                                "INFO: No disturbance variant given. Default variant '"
                                + "{}' used instead.".format(
                                    self._disturber_cfg[self._disturbance_type][
                                        "default_variant"
                                    ]
                                )
                            ),
                            "green",
                            bold=True,
                        )
                    )
                    disturbance_variant = self._disturber_cfg[self._disturbance_type][
                        "default_variant"
                    ]
                else:
                    # Thrown warning if disturbance variant could not be retrieved.
                    valid_variant_keys = {
                        k
                        for k in self._disturber_cfg[self._disturbance_type].keys()
                        if k not in ["default_type"]
                    }
                    d_variant_info_msg = (
                        "init_disturber(): is missing one required positional argument "
                        "'disturbance_variant'. This argument is required for "
                        f"for disturbance type {self._disturbance_type}. Please "
                        "specify a valid disturbance variant. Valid variants are:"
                    )
                    for item in valid_variant_keys:
                        d_variant_info_msg += f"\t\n - {item}"
                    raise ValueError(
                        d_variant_info_msg,
                        "disturbance_variant",
                    )

            # Set disturbance variant.
            self._disturbance_variant = disturbance_variant

            # Set subvariants if type is combined.
            if self._disturbance_type in "combined":
                disturbance_variant_keys = self.disturber_cfg[self._disturbance_type][
                    self._disturbance_variant
                ].keys()
                self._disturbance_sub_variants_keys = [
                    key
                    for key in disturbance_variant_keys
                    if bool(re.search("(input_|output_)", key))
                ]
                self._disturbance_sub_variants = [
                    re.sub(r"(input_|output_)", "", key)
                    for key in disturbance_variant_keys
                    if bool(re.search("(input_|output_)", key))
                ]
                if not self._disturbance_sub_variants:
                    raise Exception(
                        "No disturbance subvariants could be found for the "
                        f"'{self._disturbance_variant}' combined disturbance. "
                        "Please ensure the disturbances defined in the disturbance "
                        "config of this disturbance variant contain keys with a "
                        "'input' and 'output' prefix. These prefixes are needed to "
                        "distinguish between the combined disturbances."
                    )

    def _reset_disturber(self):
        """Resets all disturber variables."""
        self.disturber_done = False
        self.disturbance_info = {}
        self._disturbance_done_warned = False
        self._disturbance_cfg = None
        self._disturbance_type = None
        self._disturbance_variant = None
        self._disturbance_sub_variants = None
        self._disturbance_range_idx = 0
        self._disturbance_range_keys = None
        self._disturbance_range_length = None
        self._has_time_vars = None

    def _validate_disturbance_cfg(self):
        """Validates the disturbance configuration dictionary to see if it contains the
        right information to apply the requested disturbance *type* and *variant*.

        Raises:
            Exception: Thrown when the disturbance config is invalid.
        """
        if self._disturbance_type == "env":
            req_keys = ["variable", "variable_range"]
            assert all(
                [req_key in self._disturbance_cfg.keys() for req_key in req_keys]
            ), (
                "The 'env' disturbance config is invalid. Please make sure it contains "
                f"the {friendly_list(req_keys, apostrophes=True)} keys."
            )
        else:
            if self._disturbance_type == "combined":
                vals_key_lengths = []
                for req_key in self._disturbance_sub_variants_keys:
                    disturbance_range_keys = [
                        key for key in self._disturbance_cfg[req_key] if "_range" in key
                    ]
                    disturbance_type = re.search("(input(?=_)|output(?=_))", req_key)[0]
                    try:
                        self._validate_disturbance_variant_cfg(
                            self._disturbance_cfg[req_key], disturbance_type
                        )
                        vals_key_lengths.extend(
                            [
                                inject_value(item, value=0.0)
                                for item in get_flattened_values(
                                    self._disturbance_cfg[req_key][
                                        disturbance_range_keys[0]
                                    ]
                                )
                            ]
                        )
                    except (AssertionError, ValueError) as e:
                        raise Exception(
                            "The 'combined_disturbance' disturbance config is invalid. "
                            "Please check the configuration and try again."
                        ) from e

                # Validate if all the disturbances have equal range lengths.
                if (
                    len(
                        set(
                            [
                                (
                                    len(item)
                                    if isinstance(item, (list, np.ndarray))
                                    else 1
                                )
                                for item in vals_key_lengths
                            ]
                        )
                    )
                    != 1
                ):
                    raise Exception(
                        "The 'combined_disturbance' disturbance config is invalid. It "
                        "looks like the length of the disturbance 'range' is different "
                        "between the 'input' and 'output' disturbances (note that the "
                        "zero disturbance is added automatically). Please check "
                        "the disturbance ranges and try again."
                    )
            else:
                try:
                    self._validate_disturbance_variant_cfg(
                        self._disturbance_cfg, self._disturbance_type
                    )
                except (AssertionError, ValueError) as e:
                    raise Exception(
                        f"The '{self._disturbance_variant}' disturbance config is "
                        "invalid. Please check the configuration and try again."
                    ) from e

    def _validate_disturbance_variant_cfg(  # noqa: C901
        self, disturbance_cfg, disturbance_type
    ):
        """Validates the disturbance variant configuration object to see if it is valid
        for the disturbances that are currently implemented.

        .. important::
            If you want to add validation for a newly added disturbance this is the
            method where you can add this validation.

        Args:
            disturbance_cfg (dict): The disturbance variant configuration object.

        Raises:
            ValueError: Thrown when the disturbance variant config is invalid for the
                currently implemented disturbances.
        """
        # Check if range key is present.
        # NOTE: All disturbance configuration objects should have at least one key with
        # the _range suffix.
        disturbance_range_keys = [
            key for key in disturbance_cfg.keys() if "_range" in key
        ]
        if len(disturbance_range_keys) > 1:
            raise ValueError(
                "Multiple keys with the '_range' suffix were found in the "
                f"'disturber_cfg' of the '{self._disturbance_variant}' disturbance. "
                "The  disturber uses this suffix as a identifier for retrieving the "
                "disturbance range. As a result currently only one key can have the "
                "'_range' suffix."
            )
        elif len(disturbance_range_keys) == 0:
            raise ValueError(
                "No keys with the '_range' suffix were found in the 'disturber_cfg' "
                f"of the '{self._disturbance_variant}' disturbance. The disturber uses "
                "this suffix as a identifier for retrieving the disturbance range. "
                "As a result one key with the '_range' suffix should be present in the "
                "'disturber_cfg'."
            )

        # Check if required keys are found and the range key has the right length.
        invalid_keys_string = (
            f"The '{self._disturbance_variant}' disturbance config is invalid. "
            "Please make sure it contains the "
            "%s keys."
        )
        if (
            self._disturbance_variant == "impulse"
            or self._disturbance_variant == "constant_impulse"
        ):
            req_keys = ["magnitude_range", "impulse_instant"]
            if not all([req_key in disturbance_cfg.keys() for req_key in req_keys]):
                raise ValueError(
                    invalid_keys_string % friendly_list(req_keys, apostrophes=True)
                )
        elif self._disturbance_variant == "periodic":
            possible_keys = ["amplitude_range", "frequency_range", "phase_range"]
            if (
                sum([req_key in disturbance_cfg.keys() for req_key in possible_keys])
                != 1
            ):
                invalid_keys_string = (
                    f"The '{self._disturbance_variant}' disturbance config is invalid. "
                    "Please make sure it contains ONE of the following keys:"
                )
                for item in possible_keys:
                    invalid_keys_string += f"\t\n - {item}"
                raise ValueError(invalid_keys_string)
        elif self._disturbance_variant == "noise":
            req_keys = ["noise_range"]
            if not all([req_key in disturbance_cfg.keys() for req_key in req_keys]):
                raise ValueError(
                    invalid_keys_string % friendly_list(req_keys, apostrophes=True)
                )
            req_sub_keys = ["mean", "std"]
            if not all(
                [
                    req_key in disturbance_cfg[req_keys[0]].keys()
                    for req_key in req_sub_keys
                ]
            ):
                raise ValueError(
                    invalid_keys_string % friendly_list(req_sub_keys, apostrophes=True)
                )
            if len(
                inject_value(disturbance_cfg[req_keys[0]]["mean"], value=0.0)
            ) != len(inject_value(disturbance_cfg[req_keys[0]]["std"], value=0.0)):
                raise ValueError(
                    "The 'noise' disturbance config is invalid. Please make sure the "
                    "length of the 'mean' and 'std' keys are equal (note that the zero "
                    "disturbance is added automatically)."
                )

        # Check if each observation/action has a disturbance range if it is a 2D array
        disturbance_range = disturbance_cfg[disturbance_range_keys[0]]
        disturbance_range_dict = (
            {"var_key": disturbance_range}
            if not isinstance(disturbance_range, dict)
            else disturbance_range
        )
        for key, val in disturbance_range_dict.items():
            if isinstance(val, np.ndarray) and val.ndim > 1:
                req_length = (
                    self.action_space.shape[0]
                    if disturbance_type.lower() == "input"
                    else self.observation_space.shape[0]
                )
                if val.shape[1] != req_length:
                    space_name = (
                        "action space"
                        if disturbance_type.lower() == "input"
                        else "observation space"
                    )
                    key_and_range_string = (
                        f"'{key}' key in the '{disturbance_range_keys[0]}'"
                        if isinstance(disturbance_range, dict)
                        else f"'{disturbance_range_keys[0]}'"
                    )
                    raise ValueError(
                        f"The '{self._disturbance_variant}' disturbance config is "
                        "invalid. Please make sure that the length of the "
                        f"{key_and_range_string} (i.e. {val.shape[0]}) is equal to "
                        f"the {space_name} size (i.e. {req_length})."
                    )

    def _parse_disturbance_cfg(self):  # noqa: C901
        """Parse the disturbance config to add determine the disturbance range and add
        the initial disturbance (0.0) if it is not yet present.

        Raises:
            TypeError: Thrown when the disturbance config has the wrong type.
        """
        self._disturbance_range_keys = []
        if self._disturbance_type == "combined":
            for sub_variant_key in self._disturbance_sub_variants_keys:
                self._disturbance_range_keys.append(
                    [
                        key
                        for key in self._disturbance_cfg[sub_variant_key].keys()
                        if "_range" in key
                    ][0]
                )

                # Thrown warning if the disturbance variant is invalid.
                if not isinstance(
                    self._disturbance_cfg[sub_variant_key][
                        self._disturbance_range_keys[-1]
                    ],
                    (dict, list, np.ndarray),
                ):
                    raise TypeError(
                        f"The '{sub_variant_key}' variable found in the "
                        "'disturber_cfg' has the wrong type. Please make sure it "
                        "contains a 'list' or a 'dictionary'."
                    )

                # Add zero disturbance and retrieve disturbance range length.
                disturbance_sub_variant_cfg = self._disturbance_cfg[sub_variant_key][
                    self._disturbance_range_keys[-1]
                ]
                if self._include_baseline:
                    disturbance_sub_variant_cfg = inject_value(
                        disturbance_sub_variant_cfg,
                        value=0.0,
                    )  # Add undisturbed state if not yet present.
                if isinstance(
                    self._disturbance_cfg[sub_variant_key][
                        self._disturbance_range_keys[-1]
                    ],
                    dict,
                ):
                    self._disturbance_range_length = len(
                        list(disturbance_sub_variant_cfg.values())[0]
                    )
                elif isinstance(
                    self._disturbance_cfg[sub_variant_key][
                        self._disturbance_range_keys[-1]
                    ],
                    (list, np.ndarray),
                ):
                    self._disturbance_range_length = len(disturbance_sub_variant_cfg)

                # Store disturbance subvariant config.
                self.disturbance_cfg[sub_variant_key][
                    self._disturbance_range_keys[-1]
                ] = disturbance_sub_variant_cfg
        elif self._disturbance_type == "env":
            variable = self._disturbance_cfg["variable"]
            self._disturbance_range_keys.append(
                [key for key in self._disturbance_cfg.keys() if "_range" in key][0]
            )
            disturbance_cfg = self._disturbance_cfg[self._disturbance_range_keys[-1]]
            if self._include_baseline:
                disturbance_cfg = inject_value(
                    disturbance_cfg,
                    value=getattr(self, variable),
                )  # Add undisturbed env if not yet present.
            self._disturbance_range_length = len(disturbance_cfg)
            self.disturbance_cfg[self._disturbance_range_keys[-1]] = disturbance_cfg
        else:
            self._disturbance_range_keys.append(
                [key for key in self._disturbance_cfg.keys() if "_range" in key][0]
            )

            # Thrown warning if the disturbance variant is invalid.
            if not isinstance(
                self._disturbance_cfg[self._disturbance_range_keys[-1]],
                (dict, list, np.ndarray),
            ):
                raise TypeError(
                    f"The '{self._disturbance_range_keys[-1]}' variable found in "
                    "the 'disturber_cfg' has the wrong type. Please make sure it "
                    "contains a 'list' or a 'dictionary'."
                )

            # Add zero disturbance and retrieve disturbance range length.
            disturbance_cfg = self._disturbance_cfg[self._disturbance_range_keys[-1]]
            if self._include_baseline:
                disturbance_cfg = inject_value(
                    disturbance_cfg, value=0.0
                )  # Add undisturbed state if not yet present.
            if isinstance(
                self._disturbance_cfg[self._disturbance_range_keys[-1]], dict
            ):
                self._disturbance_range_length = len(list(disturbance_cfg.values())[0])
            elif isinstance(
                self._disturbance_cfg[self._disturbance_range_keys[-1]],
                (list, np.ndarray),
            ):
                self._disturbance_range_length = len(disturbance_cfg)

            # Store disturbance config.
            self.disturbance_cfg[self._disturbance_range_keys[-1]] = disturbance_cfg

    def _set_disturbance_cfg(self):
        """Sets the disturbance configuration based on the set 'disturbance_type` and/or
        'disturbance_variant' and validates whether this configuration is valid given
        the disturbances that have been implemented.

        .. important::
            If you want to add validation for a newly added disturbance this should be
            added to the :meth:`Disturber._` method.
        """
        if self._disturbance_type == "env":
            self._disturbance_cfg = self._disturber_cfg[self._disturbance_type]
        else:
            self._disturbance_cfg = self._disturber_cfg[self._disturbance_type][
                self._disturbance_variant
            ]

        # Validate disturbance config, add initial (zero) disturbance and retrieve
        # disturbance range length.
        self._validate_disturbance_cfg()
        self._parse_disturbance_cfg()

    def _get_plot_labels(self):  # noqa: C901
        """Retrieves or generated plot labels which can be added to the
        :attr:`Disturber.disturbance_info` attribute. These labels can be used for
        plotting.

        Raises:
            Exception: Thrown when no labels were found in the disturbance config and
                something went wrong during automatic label creation.

        Returns:
            (tuple): tuple containing:

                - labels (:obj:`str`): List with all labels.
                - label (:obj:`str`): The label used for the current disturbance.
        """
        if self._disturbance_type != "env":
            if self._disturbance_type == "combined":
                # Retrieve disturbance values.
                label_values = [
                    get_flattened_values(self.disturbance_cfg[sub_var][range_key])
                    for sub_var, range_key in zip(
                        self._disturbance_sub_variants_keys,
                        self._disturbance_range_keys,
                    )
                ]

                # Make label_values compatible with the specified label.
                if (
                    "label" in self._disturbance_cfg.keys()
                    and len(set(self._disturbance_range_keys)) == 1
                ):
                    label_values = zip(*deepflatten(zip(*label_values), depth=1))
                else:
                    label_values = zip(*deepflatten(label_values, depth=1))

                    # Generate label literal if not supplied.
                    if "label" not in self._disturbance_cfg.keys():
                        label_keys = []
                        for idx, (sub_var, range_key) in enumerate(
                            zip(
                                self._disturbance_sub_variants_keys,
                                self._disturbance_range_keys,
                            )
                        ):
                            disturbance_range = self.disturbance_cfg[sub_var][range_key]
                            label_keys.extend(
                                get_flattened_keys(disturbance_range)
                                if isinstance(disturbance_range, dict)
                                else [self._disturbance_range_keys[idx]]
                            )
                        label_abbreviations = abbreviate(label_keys)
                        self._disturbance_cfg["label"] = (
                            ":%s, ".join(label_abbreviations) + ":%s"
                        )
            else:
                disturbance_range = self.disturbance_cfg[
                    self._disturbance_range_keys[0]
                ]

                # Retrieve values that should be used for the labels.
                if isinstance(disturbance_range, dict):
                    label_values = zip(*get_flattened_values(disturbance_range))
                else:
                    label_values = disturbance_range

                # Generate label literal if not supplied.
                if "label" not in self._disturbance_cfg.keys():
                    if isinstance(disturbance_range, dict):
                        label_keys = get_flattened_keys(disturbance_range)
                    else:
                        label_keys = self.disturbance_info["variable"]
                    label_abbreviations = abbreviate(label_keys)
                    self._disturbance_cfg["label"] = (
                        ": %s, ".join(label_abbreviations) + ": %s"
                    )

            # Create labels.
            try:
                labels = [
                    self._disturbance_cfg["label"] % item for item in label_values
                ]
                label = labels[self._disturbance_range_idx]
            except TypeError:
                req_vars = self._disturbance_cfg["label"].count("%")
                available_vars = [len(item) for item in label_values][0]
                raise Exception(
                    "Something went wrong while creating the 'plot_labels'. It "
                    "looks like the plot label that is specified in the disturber "
                    f"config requires {req_vars} values while {available_vars} values "
                    "could be retrieved from the disturbance config. Please check your "
                    "disturbance label and try again."
                )
        else:
            disturbance_range = self.disturbance_cfg[self._disturbance_range_keys[0]]
            disturbed_var = strip_underscores(self._disturbance_cfg["variable"])
            labels = [disturbed_var + ": %s" % item for item in disturbance_range]
            label = labels[self._disturbance_range_idx]

        return labels, label

    def _set_disturbance_info(self):
        """Puts information about the requested disturbance onto the
        :attr:`Disturber.disturbance_info` attribute. This info can for example be used
        to create a legend for a robustness evaluation plot.
        """
        # Store general disturbance info.
        self.disturbance_info["cfg"] = self._disturbance_cfg
        self.disturbance_info["type"] = self._disturbance_type
        self.disturbance_info["variant"] = self._disturbance_variant
        self.disturbance_info["variables"] = {}

        # Store disturbance range values and current value.
        if self._disturbance_type == "combined":
            sub_vars = [
                re.search("(input(?=_)|output(?=_))", key)[0]
                for key in self._disturbance_sub_variants_keys
            ]
            variables = [
                var + "_" + key.replace("_range", "")
                for var, key in zip(sub_vars, self._disturbance_range_keys)
            ]
            for var in variables:
                self.disturbance_info["variables"][var] = {}
            for var, sub_variant, range_key in zip(
                variables,
                self._disturbance_sub_variants_keys,
                self._disturbance_range_keys,
            ):
                disturbance_range = self.disturbance_cfg[sub_variant][range_key]
                self.disturbance_info["variables"][var]["values"] = disturbance_range
                if isinstance(self._disturbance_cfg[sub_variant][range_key], dict):
                    self.disturbance_info["variables"][var]["value"] = {
                        k: v[self._disturbance_range_idx]
                        for k, v in disturbance_range.items()
                    }
                else:
                    self.disturbance_info["variables"][var][
                        "value"
                    ] = disturbance_range[self._disturbance_range_idx]
        else:
            variable = self._disturbance_range_keys[0].replace("_range", "")
            self.disturbance_info["variables"][variable] = {}
            disturbance_range = self.disturbance_cfg[self._disturbance_range_keys[0]]
            self.disturbance_info["variables"][variable]["values"] = disturbance_range
            if isinstance(disturbance_range, dict):
                self.disturbance_info["variables"][variable]["value"] = {
                    k: v[self._disturbance_range_idx]
                    for k, v in disturbance_range.items()
                }
            else:
                self.disturbance_info["variables"][variable][
                    "value"
                ] = disturbance_range[self._disturbance_range_idx]

        # Retrieve disturbance description.
        if "description" in self._disturbance_cfg.keys():
            self.disturbance_info["description"] = self._disturbance_cfg["description"]
        else:
            if self._disturbance_type != "env":
                self.disturbance_info["description"] = (
                    self._disturbance_variant.capitalize() + " disturbance"
                )
            else:
                self.disturbance_info["description"] = "Environment disturbance"

        # Retrieve plot labels.
        (
            self.disturbance_info["labels"],
            self.disturbance_info["label"],
        ) = self._get_plot_labels()

    def init_disturber(  # noqa E901
        self,
        disturbance_type,
        disturbance_variant=None,
        disturber_cfg=None,
        include_baseline=True,
    ):  # IMPROVE: Can be removed when using gymnasium wrapper
        """Initializes the environment/step disturber.

        Args:
            disturbance_type (string): The disturbance type you want to use. Options are
                ``env``, ``input``, ``output`` or ``combined``.
            disturbance_variant (string, optional): The disturbance variant you want to
                use. Not required when you use a ``env`` disturbance.
            disturber_cfg (dict, optional): A dictionary that describes the disturbances
                the :class:`Disturber` supports. This dictionary can be used to update
                values of the ``DISTURBANCE_CFG`` configuration which is present in the
                :class:`Disturber` class file.
            include_baseline (bool): Whether you want to automatically add the baseline
                (i.e. zero disturbance) when it not present.

        Raises:
            ValueError: Thrown when the disturbance type or variant is not invalid.
        """
        self._include_baseline = include_baseline

        # Overwrite disturbance config if passed as a argument.
        if disturber_cfg is not None:
            self._disturber_cfg = {
                **DISTURBER_CFG,
                **disturber_cfg,
            }  # Allow users to update the disturber_cfg.
        else:
            # Update disturber config to the most recent default disturber config.
            self._disturber_cfg = DISTURBER_CFG

        # Reset disturber it was already initialized.
        if self._disturbance_type is not None:
            print(
                colorize(
                    "INFO: Disturber re-initialized. All disturber variables have been "
                    "reset.",
                    "green",
                    bold=True,
                )
            )
            self.reset_disturber()

        # Setup all required disturbance attributes.
        self._set_disturber_type(disturbance_type)
        self._set_disturber_variant(disturbance_variant)
        self._set_disturbance_cfg()
        self._initiate_time_vars()  # Make sure the env has a t variable.

        # Retrieve and store extra information about the disturbance.
        # NOTE: Useful for robustness evaluation plots
        self._set_disturbance_info()

        # Print information about the initial disturbance.
        print(
            colorize(
                "INFO: Disturber with disturbance type '{}'{}initialized.".format(
                    disturbance_type,
                    f" and variant '{disturbance_variant}' "
                    if disturbance_variant is not None
                    else " ",
                ),
                "green",
                bold=True,
            )
        )
        print(
            colorize(
                (
                    "INFO: Starting with the un-disturbed {} ".format(
                        "Environment"
                        if disturbance_type in ["env", "env_disturbance"]
                        else "Step"
                    )
                    + "({}).".format(
                        self.disturbance_info["labels"][self._disturbance_range_idx]
                    )
                ),
                "green",
                bold=True,
            )
        )

    def disturbed_step(self, action, *args, **kwargs):  # noqa: C901
        """Takes a action inside the gymnasium environment while applying the requested
        disturbance.

        Args:
            action (numpy.ndarray): The current action.

        Raises:
            RuntimeError: Thrown when this method is called before the
            :meth:`Disturber.init_disturber` method or when it is called when the
            disturbance type is ``env``.

        Returns:
            numpy.ndarray: The disturbed step.
        """  # IMPROVE: Change to action when using gymnasium wrapper
        if self._disturbance_type is None:
            raise RuntimeError(
                "You are trying to retrieve a disturbed step while the disturber has "
                "not yet been initialized using the 'init_disturber' method. Please "
                "initialize the disturber and try again."
            )
        if self._disturbance_type == "env":
            raise RuntimeError(
                "You are trying to retrieve a disturbed step while the disturbance "
                f"type is set to be '{self._disturbance_type}'. Please initialize the "
                "disturber with the 'input', 'output' or 'combine' type "
                "if you want to use this feature."
            )

        # Create time axis if not available.
        if not self._has_time_vars:
            self.t += self.dt  # Create time axis if not given by the environment.

        # Retrieve the disturbed step.
        if self._disturbance_type.split("_")[0] == "output":
            s, r, done, info = self.step(action, *args, **kwargs)
            try:
                s_dist = s + self._get_disturbance(
                    s, self._disturbance_variant, self._disturbance_cfg
                )
            except Exception as e:
                raise Exception(
                    "Something went wrong when trying to retrieve the disturbance. "
                    "Please check if the disturbance config is valid and try again."
                ) from e
            return s_dist, r, done, info
        elif self._disturbance_type.split("_")[0] == "input":
            try:
                return self.step(
                    action
                    + self._get_disturbance(
                        action, self._disturbance_variant, self._disturbance_cfg
                    ),
                    *args,
                    **kwargs,
                )
            except Exception as e:
                raise Exception(
                    "Something went wrong when trying to retrieve the disturbance. "
                    "Please check if the disturbance config is valid and try again."
                ) from e
        else:
            input_disturbance_variant = self._disturbance_sub_variants[0]
            input_disturbance_cfg = self._disturbance_cfg[
                self._disturbance_sub_variants_keys[0]
            ]
            output_disturbance_variant = self._disturbance_sub_variants[1]
            output_disturbance_cfg = self._disturbance_cfg[
                self._disturbance_sub_variants_keys[1]
            ]
            try:
                s, r, done, info = self.step(
                    action
                    + self._get_disturbance(
                        action, input_disturbance_variant, input_disturbance_cfg
                    ),
                    *args,
                    **kwargs,
                )
                s_dist = s + self._get_disturbance(
                    s, output_disturbance_variant, output_disturbance_cfg
                )
            except Exception as e:
                raise Exception(
                    "Something went wrong when trying to retrieve the disturbance. "
                    "Please check if the disturbance config is valid and try again."
                ) from e
            return s_dist, r, done, info

    def _apply_env_disturbance(self):
        """Function used to apply the next environment disturbance that is specified in
        the ``disturber_cfg``.

        Raises:
            RuntimeError: If something went wrong while trying to apply the disturbance.
        """
        var_name = (
            self._disturbance_cfg["variable"].strip("_")
            if self._disturbance_cfg["variable"].startswith("_")
            else self._disturbance_cfg["variable"]
        )
        var_description = self._disturbance_cfg["description"]
        var_value = self._disturbance_cfg[self._disturbance_range_keys[0]][
            self._disturbance_range_idx
        ]
        try:
            setattr(
                self,
                self._disturbance_cfg["variable"],
                var_value,
            )
            print(
                colorize(
                    (
                        "INFO: Environment disturbance applied. "
                        + f"Variable '{var_name}' ({var_description}) has been "
                        f"set to '{float(var_value):.3}'."
                    ),
                    "green",
                    bold=True,
                )
            )
        except (AttributeError, KeyError, IndexError, np.AxisError) as e:
            raise RuntimeError(
                "Something went wrong while trying to apply the environment "
                "disturbance. Please check your 'disturber_cfg' and try "
                "again."
            ) from e

    def next_disturbance(self):
        """Function used to request the next disturbance that is specified in the
        :obj:`~stable_gym.common.disturber.Disturber.disturbance_cfg`.

        Raises:
            RuntimeError: Thrown when this method is called before the
                :meth:`Disturber.init_disturber` method.

        Returns:
            bool: A boolean specifying whether the disturber has already used all
                specified disturbances.
        """  # noqa: E501
        if not self._disturbance_type:
            raise RuntimeError(
                "No disturbance found. Please call the 'init_disturber' method before "
                "calling the 'next_disturbance' method."
            )

        # Increment disturbance index and check if disturber is finished.
        self._disturbance_range_idx += 1

        # Check if disturber is finished.
        if self._disturbance_range_idx > (self._disturbance_range_length - 1):
            self.disturber_done = True
            return self.disturber_done

        # Add info about the current disturbance.
        self.disturbance_info["label"] = self.disturbance_info["labels"][
            self._disturbance_range_idx
        ]
        for key, val in self.disturbance_info["variables"].items():
            if isinstance(val["values"], dict):
                self.disturbance_info["variables"][key]["value"] = {
                    k: v[self._disturbance_range_idx] for k, v in val["values"].items()
                }
            else:
                self.disturbance_info["variables"][key]["value"] = val["values"][
                    self._disturbance_range_idx
                ]

        # Apply environment disturbance.
        if self._disturbance_type != "env":
            time_instant = [
                key for key in self._disturbance_cfg.keys() if "_instant" in key
            ]
            print(
                colorize(
                    (
                        "INFO: Apply {} disturbance ({}){}.".format(
                            self._disturbance_variant,
                            self.disturbance_info["labels"][
                                self._disturbance_range_idx
                            ],
                            f" at step {self._disturbance_cfg[time_instant[0]]}"
                            if time_instant
                            else "",
                        )
                    ),
                    "green",
                    bold=True,
                )
            )
        else:
            self._apply_env_disturbance()

        # Return disturber not finished boolean.
        return False

    @property
    def disturber_cfg(self):
        """The disturber configuration used by the disturber to generate the
        disturbances.
        """
        if self._disturber_cfg is None:
            error_msg = (
                f"'{self.__class__.__name__}' object does not yet have attribute "
                "'disturber_cfg'. Please make  sure you initialized the disturber "
                "using the 'init_disturber' method and try again."
            )
            raise AttributeError(error_msg)
        return self._disturber_cfg

    @disturber_cfg.setter
    def disturber_cfg(self, set_val):
        error_msg = (
            "Changing the 'disturber_cfg' value during runtime is not allowed. Please "
            "set your disturbance config before training or pass a disturbance config "
            "to the init_disturber method."
        )
        raise AttributeError(error_msg)

    @property
    def disturbance_cfg(self):
        """The disturbance config used to generate the currently selected disturber.
        This variable is retrieved from the
        :obj:`~stable_gym.common.disturber.Disturber.disturber_cfg`
        using the currently set ``disturbance_type`` and/or ``disturbance_variant``.
        """  # noqa: E501
        if self._disturbance_cfg is None:
            error_msg = (
                f"'{self.__class__.__name__}' object does not yet have attribute "
                "'disturbance_cfg'. Please make  sure you initialized the disturber "
                "using the 'init_disturber' method and try again."
            )
            raise AttributeError(error_msg)

        return self._disturbance_cfg

    @disturbance_cfg.setter
    def disturbance_cfg(self, set_val):
        error_msg = (
            "Changing the 'disturbance_cfg' value during runtime is not allowed. "
        )
        raise AttributeError(error_msg)
