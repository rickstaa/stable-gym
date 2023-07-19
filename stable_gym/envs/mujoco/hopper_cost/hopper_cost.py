"""The HopperCost gymnasium environment."""
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from gymnasium import utils
from gymnasium.envs.mujoco.hopper_v4 import HopperEnv

import stable_gym  # NOTE: Ensure env is found in __main__. # noqa: F401

EPISODES = 10  # Number of env episodes to run when __main__ is called.
RANDOM_STEP = True  # Use random action in __main__. Zero action otherwise.


# TODO: Update solving criteria after training.
class HopperCost(HopperEnv, utils.EzPickle):
    r"""Custom Hopper gymnasium environment.

    .. note::
        Can also be used in a vectorized manner. See the
        :gymnasium:`gym.vector <api/vector>` documentation.

    Source:
        This is a modified version of the Hopper Mujoco environment found in the
        :gymnasium:`gymnasium library <environments/mujoco/hopper>`. This modification
        was first described by `Han et al. 2020 <https://arxiv.org/abs/2004.14288>`_.
        Compared to the original Hopper environment in this modified version:

        -   The objective was changed to a velocity-tracking task. To do this, the reward
            is replaced with a cost. This cost is the squared difference between the
            Hopper's forward velocity and a reference value (error). Additionally, also
            a control cost and health penalty can be included in the cost.
        -   Three **optional** variables were added to the observation space; The reference velocity, the reference error
            (i.e. the difference between the hopper's forward velocity and the reference) and the hopper's forward velocity.
            These variables can be enabled using the ``exclude_reference_from_observation``,
            ``exclude_reference_error_from_observation`` and ``exclude_velocity_from_observation`` environment arguments.

        The rest of the environment is the same as the original Hopper environment.
        Below, the modified cost is described. For more information about the environment
        (e.g. observation space, action space, episode termination, etc.), please refer
        to the :gymnasium:`gymnasium library <environments/mujoco/hopper>`.

    Modified cost:
        .. math::

            cost = w_{forward\_velocity} \times (x_{velocity} - x_{reference\_x\_velocity})^2 + w_{ctrl} \times c_{ctrl} + p_{health}

    Solved Requirements:
        Considered solved when the average cost is less than or equal to 50 over
        100 consecutive trials.

    How to use:
        .. code-block:: python

            import stable_gym
            import gymnasium as gym
            env = gym.make("HopperCost-v1")

    Attributes:
        state (numpy.ndarray): The current system state.
        dt (float): The environment step size. Also available as :attr:`.tau`.
        reference_forward_velocity (float): The forward velocity that the agent should
            try to track.
    """  # noqa: E501

    def __init__(
        self,
        reference_forward_velocity=1.0,
        randomise_reference_forward_velocity=False,
        randomise_reference_forward_velocity_range=(0.5, 1.5),
        forward_velocity_weight=1.0,
        include_ctrl_cost=False,
        include_health_penalty=True,
        health_penalty_size=10,
        ctrl_cost_weight=1e-3,
        terminate_when_unhealthy=True,
        healthy_state_range=(-100.0, 100.0),
        healthy_z_range=(0.7, float("inf")),
        healthy_angle_range=(-0.2, 0.2),
        reset_noise_scale=5e-3,
        exclude_current_positions_from_observation=True,
        exclude_reference_from_observation=False,
        exclude_reference_error_from_observation=True,
        exclude_x_velocity_from_observation=False,
        **kwargs,
    ):
        """Initialise a new HopperCost environment instance.

        Args:
            reference_forward_velocity (float, optional): The forward velocity that the
                agent should try to track. Defaults to ``1.0``.
            randomise_reference_forward_velocity (bool, optional): Whether to randomize
                the reference forward velocity. Defaults to ``False``.
            randomise_reference_forward_velocity_range (tuple, optional): The range of
                the random reference forward velocity. Defaults to ``(0.5, 1.5)``.
            forward_velocity_weight (float, optional): The weight used to scale the
                forward velocity error. Defaults to ``1.0``.
            include_ctrl_cost (bool, optional): Whether you also want to penalize the
                hopper if it takes actions that are too large. Defaults to ``False``.
            include_health_penalty (bool, optional): Whether to penalize the hopper if
                it becomes unhealthy (i.e. if it falls over). Defaults to ``True``.
            health_penalty_size (int, optional): The size of the unhealthy penalty.
                Defaults to ``10``.
            ctrl_cost_weight (float, optional): The weight used to scale the control
                cost. Defaults to ``1e-3``.
            terminate_when_unhealthy (bool, optional): Whether to terminate the episode
                when the hopper becomes unhealthy. Defaults to ``True``.
            healthy_state_range (tuple, optional): The range of healthy states. Defaults
                to ``(-100.0, 100.0)``.
            healthy_z_range (tuple, optional): The range of healthy z values. Defaults
                to ``(0.7, float("inf"))``.
            healthy_angle_range (tuple, optional): The range of healthy angles. Defaults
                to ``(-0.2, 0.2)``.
            reset_noise_scale (float, optional): Scale of random perturbations of the
                initial position and velocity. Defaults to ``5e-3``.
            exclude_current_positions_from_observation (bool, optional): Whether to omit
                the x- and y-coordinates of the front tip from observations. Excluding
                the position can serve as an inductive bias to induce position-agnostic
                behaviour in policies. Defaults to ``True``.
            exclude_reference_from_observation (bool, optional): Whether the reference
                should be excluded from the observation. Defaults to ``False``. Can only
                be set to ``True`` if ``randomise_reference_forward_velocity`` is set to
                ``False``.
            exclude_reference_error_from_observation (bool, optional): Whether the error
                should be excluded from the observation. Defaults to ``True``.
            exclude_x_velocity_from_observation (bool, optional): Whether to omit the
                x- component of the velocity from observations. Defaults to ``False``.
            **kwargs: Extra keyword arguments to pass to the
                :class:`~gymnasium.envs.mujoco.hopper_v4.HopperEnv` class.
        """
        self.reference_forward_velocity = reference_forward_velocity
        self._randomise_reference_forward_velocity = (
            randomise_reference_forward_velocity
        )
        self._randomise_reference_forward_velocity_range = (
            randomise_reference_forward_velocity_range
        )
        self._forward_velocity_weight = forward_velocity_weight
        self._include_ctrl_cost = include_ctrl_cost
        self._include_health_penalty = include_health_penalty
        self._health_penalty_size = health_penalty_size
        self._exclude_reference_from_observation = exclude_reference_from_observation
        self._exclude_reference_error_from_observation = (
            exclude_reference_error_from_observation
        )
        self._exclude_x_velocity_from_observation = exclude_x_velocity_from_observation

        # Validate input arguments.
        assert (
            not randomise_reference_forward_velocity
            or not exclude_reference_from_observation
        ), (
            "The reference can only be excluded from the observation if the forward "
            "velocity is not randomised."
        )

        self.state = None

        # Initialise the HopperEnv class.
        super().__init__(
            ctrl_cost_weight=ctrl_cost_weight,
            terminate_when_unhealthy=terminate_when_unhealthy,
            healthy_state_range=healthy_state_range,
            healthy_z_range=healthy_z_range,
            healthy_angle_range=healthy_angle_range,
            reset_noise_scale=reset_noise_scale,
            exclude_current_positions_from_observation=exclude_current_positions_from_observation,  # noqa: E501
            **kwargs,
        )

        # Extend observation space if necessary.
        low = self.observation_space.low
        high = self.observation_space.high
        if not self._exclude_reference_from_observation:
            low = np.append(low, -np.inf)
            high = np.append(high, np.inf)
        if not self._exclude_reference_error_from_observation:
            low = np.append(low, -np.inf)
            high = np.append(high, np.inf)
        if not self._exclude_x_velocity_from_observation:
            low = np.append(low, -np.inf)
            high = np.append(high, np.inf)
        self.observation_space = gym.spaces.Box(
            low,
            high,
            dtype=self.observation_space.dtype,
            seed=self.observation_space.np_random,
        )

        # Reinitialize the EzPickle class.
        # NOTE: Done to ensure the args of the HopperCost class are also pickled.
        # NOTE: Ensure that all args are passed to the EzPickle class!
        utils.EzPickle.__init__(
            self,
            reference_forward_velocity,
            randomise_reference_forward_velocity,
            randomise_reference_forward_velocity_range,
            forward_velocity_weight,
            include_ctrl_cost,
            include_health_penalty,
            health_penalty_size,
            ctrl_cost_weight,
            terminate_when_unhealthy,
            healthy_state_range,
            healthy_z_range,
            healthy_angle_range,
            reset_noise_scale,
            exclude_current_positions_from_observation,
            exclude_reference_from_observation,
            exclude_reference_error_from_observation,
            exclude_x_velocity_from_observation,
            **kwargs,
        )

    def cost(self, x_velocity, ctrl_cost):
        """Compute the cost of a given x velocity and control cost.

        Args:
            x_velocity (float): The Hopper's x velocity.
            ctrl_cost (float): The control cost.

        Returns:
            (tuple): tuple containing:

                -   cost (float): The cost of the action.
                -   info (:obj:`dict`): Additional information about the cost.
        """
        velocity_cost = self._forward_velocity_weight * np.square(
            x_velocity - self.reference_forward_velocity
        )
        cost = velocity_cost
        if self._include_ctrl_cost:
            cost += ctrl_cost

        # Add extra penalty if the hopper becomes unhealthy.
        health_penalty = 0.0
        if self._include_health_penalty:
            if not self.is_healthy:
                health_penalty = self._health_penalty_size
                cost += health_penalty

        return cost, {
            "cost_velocity": velocity_cost,
            "cost_ctrl": ctrl_cost,
            "penalty_health": health_penalty,
        }

    def step(self, action):
        """Take step into the environment.

        .. note::
            This method overrides the
            :meth:`~gymnasium.envs.mujoco.hopper_v4.HopperEnv.step` method
            such that the new cost function is used.

        Args:
            action (np.ndarray): Action to take in the environment.

        Returns:
            (tuple): tuple containing:

                -   obs (:obj:`np.ndarray`): Environment observation.
                -   cost (:obj:`float`): Cost of the action.
                -   terminated (:obj:`bool`): Whether the episode is terminated.
                -   truncated (:obj:`bool`): Whether the episode was truncated. This
                    value is set by wrappers when for example a time limit is reached or
                    the agent goes out of bounds.
                -   info (:obj:`dict`): Additional information about the environment.
        """
        obs, _, terminated, truncated, info = super().step(action)

        self.state = obs

        ctrl_cost = super().control_cost(action)
        cost, cost_info = self.cost(info["x_velocity"], ctrl_cost)

        # Add reference, x velocity and reference error to observation.
        if not self._exclude_reference_from_observation:
            obs = np.append(obs, self.reference_forward_velocity)
        if not self._exclude_reference_error_from_observation:
            obs = np.append(obs, info["x_velocity"] - self.reference_forward_velocity)
        if not self._exclude_x_velocity_from_observation:
            obs = np.append(obs, info["x_velocity"])

        # Update info.
        info.update(cost_info)

        return obs, cost, terminated, truncated, info

    def reset(self, seed=None, options=None):
        """Reset gymnasium environment.

        Args:
            seed (int, optional): A random seed for the environment. By default
                ``None``.
            options (dict, optional): A dictionary containing additional options for
                resetting the environment. By default ``None``. Not used in this
                environment.

        Returns:
            (tuple): tuple containing:

                -   obs (:obj:`numpy.ndarray`): Initial environment observation.
                -   info (:obj:`dict`): Dictionary containing additional information.
        """
        obs, info = super().reset(seed=seed, options=options)

        # Randomize the reference forward velocity if requested.
        if self._randomise_reference_forward_velocity:
            self.reference_forward_velocity = self.np_random.uniform(
                *self._randomise_reference_forward_velocity_range
            )

        # Add reference, x velocity and reference error to observation.
        if not self._exclude_reference_from_observation:
            obs = np.append(obs, self.reference_forward_velocity)
        if not self._exclude_reference_error_from_observation:
            obs = np.append(obs, 0.0 - self.reference_forward_velocity)
        if not self._exclude_x_velocity_from_observation:
            obs = np.append(obs, 0.0)

        self.state = obs

        return obs, info

    @property
    def tau(self):
        """Alias for the environment step size. Done for compatibility with the
        other gymnasium environments.
        """
        return self.dt

    @property
    def t(self):
        """Make simulation time available as a property."""
        return self.unwrapped.data.time


if __name__ == "__main__":
    print("Setting up 'HopperCost' environment.")
    env = gym.make("HopperCost", render_mode="human")

    # Run episodes.
    episode = 0
    path, paths = [], []
    s, _ = env.reset()
    path.append(s)
    print(f"\nPerforming '{EPISODES}' in the 'HopperCost' environment...\n")
    print(f"Episode: {episode}")
    while episode <= EPISODES:
        action = (
            env.action_space.sample()
            if RANDOM_STEP
            else np.zeros(env.action_space.shape)
        )
        s, r, terminated, truncated, _ = env.step(action)
        path.append(s)
        if terminated or truncated:
            paths.append(path)
            episode += 1
            path, reference = [], []
            s, _ = env.reset()
            path.append(s)
            print(f"Episode: {episode}")
    print("\nFinished 'HopperCost' environment simulation.")

    # Plot results per episode.
    print("\nPlotting episode data...")
    for i in range(len(paths)):
        path = paths[i]
        fig, ax = plt.subplots()
        print(f"\nEpisode: {i}")
        path = np.array(path)
        t = np.linspace(0, path.shape[0] * env.dt, path.shape[0])
        for j in range(path.shape[1]):  # NOTE: Change if you want to plot less states.
            ax.plot(t, path[:, j], label=f"State {j}")
        ax.set_xlabel("Time (s)")
        ax.set_title(f"HopperCost episode '{i}'")
        ax.legend()
        print("Close plot to see next episode...")
        plt.show()

    print("\nDone")
