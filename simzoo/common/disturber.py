"""A simple disturber class from which a OpenAi Gym Environment can inherit in order
to be able to use it with the Robustness Evaluation tool of the Bayesian Learning
Control package. For more information see the
`Robustness Evaluation <https://rickstaa.github.io/bayesian-learning-control/control/robustness_eval.html>`_
documentation.
"""  # noqa: E501

import gym
import numpy as np

from .helpers import colorize


def periodic_disturbance(time):
    """Returns the value of a periodic signal at a given timestep. This function is used
    in the periodic noise disturbance.

    Args:
        time (union[int, float]): The current time.

    Returns:
        np.float64: The current value of the periodic signal.

    .. note::
        You can make the periodic signal as difficult or simple as you want. The only
        requirement is that it you keep the return value the same. If you want to use a
        different return value you should modify the
        :meth:`Disturber._get_periodic_disturbance` method.
    """
    return np.sin(2 * np.pi * 10 * time)


# Default Disturber configuration variable.
# NOTE: You can also supply the disturber with your own disturbance configuration
# dictionary. When doing this you have to make sure it contains all the required keys.
# See https://rickstaa.github.io/bayesian-learning-control/control/robustness_eval.html
# for more information.
DISTURBER_CFG = {
    # Disturbance type when no type has been given
    "default_type": "step_disturbance",
    # Disturbance applied to environment variables
    "env_disturbance": {
        "description": "Lacl mRNA decay rate disturbance",
        # The env variable which you want to disturb
        "variable": "_c1",
        # The range of values you want to use for each disturbance iteration
        "variable_range": np.linspace(1.6, 3.0, num=5, dtype=np.float32),
        # Label used in robustness plots.
        "label": "r: %s",
    },
    # Disturbance applied to the environment step function
    "step_disturbance": {
        # The variant used when no variant is given by the user.
        "default_variant": "impulse",
        # Impulse disturbance applied in the opposite direction of the action at a given
        # timestep.
        "impulse": {
            "description": "Impulse disturbance",
            # The step at which you want to apply the impulse.
            "impulse_instant": 100,
            # The magnitudes you want to apply.
            "magnitude_range": np.linspace(0.0, 3.0, num=5, dtype=np.float),
            # Label used in robustness plots.
            "label": "M: %s",
        },
        # Similar to the impulse above but now the impulse force is continuously applied
        # against the action after the impulse instant has been reached.
        "constant_impulse": {
            "description": "Constant impulse disturbance",
            # The step at which you want to apply the impulse.
            "impulse_instant": 100,
            # The magnitudes you want to apply.
            "magnitude_range": np.linspace(80, 155, num=3, dtype=np.int),
            # Label that can be used in plots.
            "label": "M: %s",
        },
        # A periodic signal noise that is applied to the action at every time step.
        "periodic": {
            "description": "Periodic noise disturbance",
            # The magnitudes of the periodic signal.
            "amplitude_range": np.linspace(10, 80, num=3, dtype=np.int),
            # The function that describes the signal
            # NOTE: A amplitude between 0-1 is recommended.
            "periodic_function": periodic_disturbance,
            # Label used in robustness plots.
            "label": "A: %s",
        },
        # A random noise that is applied to the action at every timestep.
        "noise": {
            "description": "Random noise disturbance",
            # The means and standards deviations of the random noise disturbances.
            "noise_range": {
                "mean": np.linspace(80, 155, num=3, dtype=np.int),
                "std": np.linspace(1.0, 5.0, num=3, dtype=np.int),
            },
            # Label used in robustness plots.
            "label": "x̅:%s, σ:%s",
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
            Disturber applied. Usefull for plotting.
        disturber_cfg (dict): The disturber configuration used by the disturber to
            generate the disturbances. This configuration can be supplied as a argument
            to the :meth:`Disturber.init_disturber` method during the disturber
            initiation. It can not be changed during runtime. By default it uses the
            ``DISTURBANCE_CFG`` disturbance configuration that is present in the file
            of the :class:`Disturber` class.
        disturbance_cfg (dict): The disturbance config used to generate the currently
            selected disturber. This variable is retrieved from the
            :obj:`~bayesian_learning_control.simzoo.simzoo.common.disturber.Disturber.disturber_cfg`
            using the currently set ``disturbance_type`` and/or ``disturbance_variant``.

    .. seealso::

        For more information see the
        `Robustness Evaluation <https://rickstaa.github.io/bayesian-learning-control/control/robustness_eval.html>`_
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
            "Only classes that also inherit from the 'gym.Env' class can inherit from "
            "the 'Disturber' class."
        )
        self.disturber_done = False
        self.disturbance_info = {
            "type": None,
            "variant": None,
            "variable": None,
            "value": None,
            "values": [],
            "description": None,
            "label": [],
            "labels": [],
            "cfg": {},
        }
        self._disturbance_done_warned = False
        self._disturber_cfg = (
            {
                **DISTURBER_CFG,
                **disturber_cfg,
            }
            if disturber_cfg is not None
            else None
        )  # Allow users to overwrite the default config
        self._disturbance_cfg = None
        self._disturbance_type = None
        self._disturbance_variant = None
        self._disturbance_range = None
        self._disturbance_iter_idx = 0
        self._disturbance_iter_length = None

    def _get_disturbance(self, action):
        """Retrieves the right disturbance using the disturbance type and variant that
        were set using the :meth:`Disturber.init_disturber` method.

        Args:
            action (numpy.ndarray): The current action.

        Returns:
            numpy.ndarray: The disturbance array.
        """
        # Set the disturber state to done and return a zero disturbance if the user has
        # used all the specified disturbances
        if self._disturbance_iter_length is not None and self._disturbance_iter_idx > (
            self._disturbance_iter_length - 1
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
            return np.zeros_like(action)

        # Retrieve the requested disturbance
        if (
            self._disturbance_variant == "impulse"
            or self._disturbance_variant == "constant_impulse"
        ):
            return self._get_impulse_disturbance(action)
        elif self._disturbance_variant == "periodic":
            return self._get_periodic_disturbance(action)
        elif self._disturbance_variant == "noise":
            return self._get_noise_disturbance(action)
        else:
            raise NotImplementedError(
                f"Disturbance variant '{self._disturbance_variant}' not yet "
                "implemented."
            )

    def _get_impulse_disturbance(self, action):
        """Retrieves a impulse disturbance that can be applied to the action.

        Args:
            action (numpy.ndarray): The current action.

        Returns:
            numpy.ndarray: The disturbance array.

        .. note::
            The disturber currently implements two types of impulses: A regular
            (instant) impulse (applied at a single time step) and a constant impulse
            (applied at all steps following the set time instant). In both versions the
            direction of the impulse is opposite to the action that was applied at a
            given time instant.
        """
        cur_magnitude = self._disturbance_range[self._disturbance_iter_idx]
        if self._disturbance_variant == "constant_impulse":
            if (self.t / self.dt) == self._disturbance_cfg["impulse_instant"]:
                dist_val = cur_magnitude * (-np.sign(action))
            else:
                dist_val = np.zeros_like(action)
        else:
            if (self.t / self.dt) >= self._disturbance_cfg["impulse_instant"]:
                dist_val = cur_magnitude * (-np.sign(action))
            else:
                dist_val = np.zeros_like(action)
        return dist_val

    def _get_periodic_disturbance(self, action):
        """Returns a periodic disturbance signal that can be applied to the action.

        Args:
            action (numpy.ndarray): The current action.

        Returns:
            numpy.ndarray: The disturbance array.
        """
        cur_magnitude = self._disturbance_range[self._disturbance_iter_idx]
        return (
            cur_magnitude
            * self._disturbance_cfg["periodic_function"](self.t)
            * np.ones_like(action)
        )

    def _get_noise_disturbance(self, action):
        """Returns a random noise, with the a in the ``disturber_cfg`` specified mean
        and a standard deviation, which can be applied to the action.

        Args:
            action (numpy.ndarray): The current action.

        Returns:
            numpy.ndarray: The disturbance array.
        """
        mean_range = np.insert(
            self._disturbance_cfg["noise_range"]["mean"], 0, 0.0, axis=0
        )
        std_range = np.insert(
            self._disturbance_cfg["noise_range"]["std"], 0, 0.0, axis=0
        )
        return np.random.normal(
            mean_range[self._disturbance_iter_idx],
            std_range[self._disturbance_iter_idx],
            len(action),
        )

    def _validate_disturbance_variant_cfg(self):
        """Validates the disturbance configuration dictionary to see if it contains the
        right information to apply the requested disturbance *variant*.
        """
        # Check if a disturbance range key is present
        disturbance_range_keys = [
            key for key in self._disturbance_cfg.keys() if "_range" in key
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

        # Check if the required keys are present for the requested disturbance variant
        if (
            self._disturbance_variant == "impulse"
            or self._disturbance_variant == "constant_impulse"
        ):
            assert all(
                [
                    req_key in self._disturbance_cfg.keys()
                    for req_key in ["magnitude_range", "impulse_instant"]
                ]
            ), (
                "The 'impulse' disturbance config is invalid. Please make sure it "
                "contains a 'magnitude_range' and 'impulse_instant' key."
            )
        elif self._disturbance_variant == "periodic":
            assert all(
                [
                    req_key in self._disturbance_cfg.keys()
                    for req_key in ["amplitude_range", "periodic_function"]
                ]
            ), (
                "The 'impulse' disturbance config is invalid. Please make sure it "
                "contains a 'amplitude_range' and 'periodic_function' key."
            )
            assert callable(self._disturbance_cfg["periodic_function"]), (
                "The 'impulse' disturbance config is invalid. Please make sure the "
                "'periodic_function' key contains a callable function."
            )
        elif self._disturbance_variant == "noise":
            assert "noise_range" in self._disturbance_cfg.keys(), (
                "The 'noise' disturbance config is invalid. Please make sure it "
                "contains a 'noise_range' key."
            )
            assert len(self._disturbance_cfg["noise_range"]["mean"]) == len(
                self._disturbance_cfg["noise_range"]["std"]
            ), (
                "The 'noise' disturbance config is invalid. Please make sure the "
                " length of the 'mean' and 'std' keys are equal."
            )

    def _validate_disturbance_cfg(self):
        """Validates the disturbance configuration dictionary to see if it contains the
        right information to apply the requested disturbance *type* and *variant*.
        """
        if self._disturbance_type == "step_disturbance":
            self._validate_disturbance_variant_cfg()
        else:
            assert all(
                [
                    req_key in self._disturbance_cfg.keys()
                    for req_key in ["description", "variable", "variable_range"]
                ]
            ), (
                "The 'env_disturbance' config is invalid. Please make sure it contains "
                "a 'variable' and 'variable_range' key."
            )

    def _set_disturber_params(self, disturbance_type, disturbance_variant):
        """Validates the disturbance type and variant and sets the required variables.

        Args:
            disturbance_type (string): The disturbance type you want to use.
            disturbance_variant (string): The disturbance variant you want to use.

        Raises:
            TypeError: Thrown when a disturbance variable in the 'disturber_cfg' has the
                wrong type.
        """
        self._disturbance_type = disturbance_type
        self.disturbance_info["type"] = self._disturbance_type.replace(
            "_disturbance", ""
        )
        if disturbance_type == "step_disturbance":
            self._disturbance_variant = disturbance_variant
            self.disturbance_info["variant"] = self._disturbance_variant
            self._disturbance_cfg = self._disturber_cfg[disturbance_type][
                disturbance_variant
            ]

            # Validate disturbance config (Thrown warning if invalid)
            self._validate_disturbance_cfg()

            # Retrieve disturbance range and range length
            range_keys = [
                key for key in self._disturbance_cfg.keys() if "_range" in key
            ]
            self._disturbance_range_key = range_keys[0]
            if isinstance(self._disturbance_cfg[self._disturbance_range_key], dict):
                self._disturbance_range = {
                    k: np.insert(v, 0, 0.0, axis=0) if v[0] != 0.0 else v
                    for k, v in self._disturbance_cfg[
                        self._disturbance_range_key
                    ].items()
                }  # Add undisturbed state if not yet present
                self._disturbance_iter_length = len(
                    list(self._disturbance_range.values())[0]
                )
            elif isinstance(
                self._disturbance_cfg[self._disturbance_range_key], (list, np.ndarray)
            ):
                self._disturbance_range = (
                    np.insert(
                        self._disturbance_cfg[self._disturbance_range_key],
                        0,
                        0.0,
                        axis=0,
                    )
                    if self._disturbance_cfg[self._disturbance_range_key][0] != 0.0
                    else self._disturbance_cfg[self._disturbance_range_key]
                )  # Add undisturbed state if not yet present
                self._disturbance_iter_length = len(self._disturbance_range)
            else:
                raise TypeError(
                    f"The '{self._disturbance_range_key}' variable found in the "
                    "'disturber_cfg' has the wrong type. Please make sure it contains "
                    "a 'list' or a 'dictionary'."
                )
        else:
            if disturbance_variant is not None:
                print(
                    colorize(
                        (
                            f"WARNING: Disturbance variant '{disturbance_variant}' "
                            "ignored as it  does not apply when using disturbance type "
                            "'step_disturbance'."
                        ),
                        "yellow",
                        bold=True,
                    )
                )
            self._disturbance_variant = "environment"
            self.disturbance_info["variant"] = self._disturbance_variant
            self._disturbance_cfg = self._disturber_cfg[disturbance_type]

            # Retrieve disturbance range and range length
            range_keys = [
                key for key in self._disturbance_cfg.keys() if "_range" in key
            ]
            self._disturbance_range_key = range_keys[0]
            self._disturbance_range = (
                np.insert(
                    self._disturbance_cfg[self._disturbance_range_key],
                    0,
                    getattr(self, self._disturbance_cfg["variable"]),
                    axis=0,
                )
                if round(
                    float(self._disturbance_cfg[self._disturbance_range_key][0]), 3
                )
                != round(float(getattr(self, self._disturbance_cfg["variable"])), 3)
                else self._disturbance_cfg[self._disturbance_range_key]
            )  # Add undisturbed state if not yet present
            self._disturbance_iter_length = len(self._disturbance_range)

        # Store disturbance information
        self.disturbance_info["cfg"] = self._disturbance_cfg
        self.disturbance_info["variable"] = self._disturbance_range_key.replace(
            "_range", ""
        )
        if isinstance(self._disturbance_range, dict):
            self.disturbance_info["value"] = {
                k: v[self._disturbance_iter_idx]
                for k, v in self._disturbance_range.items()
            }
        else:
            self.disturbance_info["value"] = self._disturbance_range[
                self._disturbance_iter_idx
            ]
        self.disturbance_info["values"] = self._disturbance_range

    def _set_disturbance_info(self):
        """Puts information about the requested disturbance onto the
        :attr:`disturbance_info` attribute. This info can for example be used to create
        a legend for a robustness evaluation plot.
        """
        # Retrieve disturbance description
        if "description" in self._disturbance_cfg.keys():
            self.disturbance_info["description"] = self._disturbance_cfg["description"]
        else:
            if self._disturbance_type == "step_disturbance":
                self.disturbance_info["description"] = (
                    self._disturbance_variant.capitalize() + " disturbance"
                )
            else:
                self.disturbance_info["description"] = "Environment disturbance"

        # Retrieve plot labels
        if self._disturbance_type == "step_disturbance":
            if "label" in self._disturbance_cfg.keys():
                if isinstance(self._disturbance_range, dict):
                    self.disturbance_info["labels"] = [
                        self._disturbance_cfg["label"] % item
                        for item in zip(*self._disturbance_range.values())
                    ]
                    self.disturbance_info["label"] = self.disturbance_info["labels"][
                        self._disturbance_iter_idx
                    ]
                else:
                    self.disturbance_info["labels"] = [
                        self._disturbance_cfg["label"] % item
                        for item in self._disturbance_range
                    ]
                    self.disturbance_info["label"] = self.disturbance_info["labels"][
                        self._disturbance_iter_idx
                    ]
            else:  # If no label was specified use first letter of disturbance variant
                if isinstance(self._disturbance_range, dict):
                    identifiers = [
                        id[0].upper() for id in self._disturbance_range.keys()
                    ]
                    label_list = []
                    for idx, property in enumerate(self._disturbance_range.values()):
                        label_list.append(
                            [identifiers[idx] + ":" + str(item) for item in property]
                        )
                    self.disturbance_info["labels"] = [
                        ", ".join(item) for item in list(zip(*label_list))
                    ]
                    self.disturbance_info["label"] = self.disturbance_info["labels"][
                        self._disturbance_iter_idx
                    ]
                else:
                    self.disturbance_info["labels"] = [
                        self._disturbance_range_key[0].upper() + ": %s" % item
                        for item in self._disturbance_range
                    ]
                    self.disturbance_info["label"] = self.disturbance_info["labels"][
                        self._disturbance_iter_idx
                    ]
        else:
            disturbed_var = (
                self._disturbance_cfg["variable"].strip("_")
                if self._disturbance_cfg["variable"].startswith("_")
                else self._disturbance_cfg["variable"]
            )
            self.disturbance_info["labels"] = [
                disturbed_var + ": %s" % item for item in self._disturbance_range
            ]
            self.disturbance_info["label"] = self.disturbance_info["labels"][
                self._disturbance_iter_idx
            ]

    def init_disturber(  # noqa E901
        self, disturbance_type, disturbance_variant=None, disturber_cfg=None
    ):
        """Initializes the environment/step disturber.

        Args:
            disturbance_type (string): The disturbance type you want to use. Options are
                ``env_disturbance`` and ``step_disturbance``.
            disturbance_variant (string, optional): The disturbance variant you want to
                use. Only required when you use a ``step_disturbance``.
            disturber_cfg (dict, optional): A dictionary that describes the disturbances
                the :class:`Disturber` supports. This dictionary can be used to update
                values of the ``DISTURBANCE_CFG`` configuration which is present in the
                :class:`Disturber` class file.

        Raises:
            ValueError: Thrown when the disturbance type or variant is not supported by
                by the disturber.
            TypeError: Thrown when the disturbance variant is not specified but while
                required for the given disturbance_type.
        """
        # Overwrite disturbance config if passed as a argument
        if disturber_cfg is not None:
            self._disturber_cfg = {
                **DISTURBER_CFG,
                **disturber_cfg,
            }  # Allow users to update the disturber_cfg
        else:
            # Set default disturber if no disturber was given with the environment
            # initiation.
            if self._disturber_cfg is None:
                self._disturber_cfg = DISTURBER_CFG

        # Validate disturbance type and/or variant input arguments
        if disturbance_type is None:
            if "default_type" in self._disturber_cfg.keys():
                print(
                    colorize(
                        (
                            "INFO: No disturbance type given default type '{}' ".format(
                                self._disturber_cfg["default_type"]
                            )
                            + "used instead."
                        ),
                        "green",
                        bold=True,
                    )
                )
                disturbance_type = self._disturber_cfg["default_type"]
            else:
                raise TypeError(
                    "init_disturber(): is missing one required positional "
                    "argument: 'disturbance_type'.",
                    "disturbance_type",
                )
        disturbance_type = (
            disturbance_type.lower() + "_disturbance"
            if "_disturbance" not in disturbance_type.lower()
            else disturbance_type.lower()
        )
        if disturbance_type not in self._disturber_cfg.keys():
            try:
                environment_name = self.unwrapped.spec.id
            except AttributeError:
                environment_name = self.__class__.__name__.__str__()
                raise ValueError(
                    f"Disturbance type '{disturbance_type}' is not implemented for the "
                    f"'{environment_name}' environment. Please specify a valid "
                    f"disturbance type {self._disturber_cfg.keys()}."
                )
        if disturbance_type != "env_disturbance":
            if disturbance_variant is None:
                if "default_variant" in self._disturber_cfg[disturbance_type].keys():
                    print(
                        colorize(
                            (
                                "INFO: No disturbance variant given default variant '"
                                + "{}' used instead.".format(
                                    self._disturber_cfg[disturbance_type][
                                        "default_variant"
                                    ]
                                )
                            ),
                            "green",
                            bold=True,
                        )
                    )
                    disturbance_variant = self._disturber_cfg[disturbance_type][
                        "default_variant"
                    ]
                else:
                    raise TypeError(
                        "init_disturber(): is missing one required positional "
                        "argument: 'disturbance_variant'. This argument is required "
                        f"for disturbance type {disturbance_type}. Please specify a "
                        f"valid disturbance variant "
                        f"{list(self._disturber_cfg[disturbance_type].keys())}.",
                        "disturbance_variant",
                    )
            else:
                disturbance_variant = disturbance_variant.lower()
        if (
            disturbance_type == "step_disturbance"
            and disturbance_variant is not None
            and disturbance_variant not in self._disturber_cfg[disturbance_type].keys()
        ):
            raise ValueError(
                f"Disturber variant '{disturbance_variant}' is not implemented for "
                f"disturbance type '{self._disturbance_type}'. Please specify a valid "
                "disturbance variant {}.".format(
                    self._disturber_cfg[disturbance_type].keys()
                )
            )

        # Set the disturber parameters
        if self._disturbance_type is not None:
            print(
                colorize(
                    "INFO: Disturber re-initalized. All disturber variables have been "
                    "reset.",
                    "green",
                    bold=True,
                )
            )
            self._disturbance_type = None
            self._disturbance_variant = None
            self._disturbance_range = None
            self._disturbance_iter_idx = 0
            self._disturbance_iter_length = None
            self._disturbance_done_warned = False
            self.disturber_done = False
        self._set_disturber_params(disturbance_type, disturbance_variant)

        # Retrieve and store extra information about the disturbance
        # NOTE: Usefull for robustness evaluation plots
        self._set_disturbance_info()

        # Print information about the initial disturbance
        print(
            colorize(
                "INFO: Disturber with disturbance type '{}' {} initialized.".format(
                    disturbance_type,
                    f"and variant '{disturbance_variant}'"
                    if disturbance_variant is not None
                    else "",
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
                        if disturbance_type == "env_disturbance"
                        else "Step"
                    )
                    + "({}).".format(
                        self.disturbance_info["labels"][self._disturbance_iter_idx]
                    )
                ),
                "green",
                bold=True,
            )
        )

    def disturbed_step(self, action, *args, **kwargs):
        """Takes a action inside the gym environment while applying the requested
        disturbance.

        Args:
            action (numpy.ndarray): The current action.

        Raises:
            RuntimeError: Thrown when this method is called before the
                :meth:`Disturber.init_disturber` method.

        Returns:
            numpy.ndarray: The disturbed step.
        """
        if self._disturbance_type is None:
            raise RuntimeError(
                "You are trying to retrieve a disturbed step while the disturber has "
                "not yet been initialized using the 'init_disturber' method. Please "
                "initialize the disturber and try again."
            )

        # Retrieve the disturbed step
        if self._disturbance_type != "step_disturbance":
            raise RuntimeError(
                "You are trying to retrieve a disturbed step while the disturbance "
                f"type is set to be '{self._disturbance_type}'. Please initialize the "
                "disturber with the 'step_disturbance' type if you want to use this "
                "feature."
            )
        return self.step(action + self._get_disturbance(action), *args, **kwargs)

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
        var_value = self._disturbance_range[self._disturbance_iter_idx]
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
                        f"set to '{var_value}'."
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
        :obj:`~bayesian_learning_control.simzoo.simzoo.common.disturber.Disturber.disturbance_cfg`.

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

        # Increment disturbance index and check if disturber is finished
        self._disturbance_iter_idx += 1

        # Check if disturber is finished
        if self._disturbance_iter_idx > (self._disturbance_iter_length - 1):
            self.disturber_done = True
            return self.disturber_done

        # Add info about the current disturbance
        self.disturbance_info["label"] = self.disturbance_info["labels"][
            self._disturbance_iter_idx
        ]
        if isinstance(self._disturbance_range, dict):
            self.disturbance_info["value"] = {
                k: v[self._disturbance_iter_idx]
                for k, v in self._disturbance_range.items()
            }
        else:
            self.disturbance_info["value"] = self._disturbance_range[
                self._disturbance_iter_idx
            ]

        # Apply environment disturbance
        if self._disturbance_type == "env_disturbance":
            self._apply_env_disturbance()
        else:
            time_instant = [
                key for key in self._disturbance_cfg.keys() if "_instant" in key
            ]
            print(
                colorize(
                    (
                        "INFO: Apply {} disturbance ({}){}.".format(
                            self._disturbance_variant,
                            self.disturbance_info["labels"][self._disturbance_iter_idx],
                            f" at step {self._disturbance_cfg[time_instant[0]]}"
                            if time_instant
                            else "",
                        )
                    ),
                    "green",
                    bold=True,
                )
            )

        # Return disturber not finished boolean
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
        :obj:`~bayesian_learning_control.simzoo.simzoo.common.disturber.Disturber.disturber_cfg`
        using the currently set ``disturbance_type`` and/or ``disturbance_variant``.
        """  # noqa: E501
        if self._disturbance_cfg is None:
            error_msg = (
                f"'{self.__class__.__name__}' object does not yet have attribute "
                "'disturbance_cfg'. Please make  sure you initialized the disturber "
                "using the 'init_disturber' method and try again."
            )
            raise AttributeError(error_msg)

        return self._disturber_cfg

    @disturbance_cfg.setter
    def disturber_cfg(self, set_val):
        error_msg = (
            "Changing the 'disturbance_cfg' value during runtime is not allowed. "
        )
        raise AttributeError(error_msg)
