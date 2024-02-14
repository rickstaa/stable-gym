"""The QuadXTrackingCost gymnasium environment."""

from pathlib import PurePath

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import PyFlyt
from gymnasium import logger, utils
from PyFlyt.gym_envs.quadx_envs.quadx_hover_env import QuadXHoverEnv

EPISODES = 10  # Number of env episodes to run when __main__ is called.
RANDOM_STEP = True  # Use random action in __main__. Zero action otherwise.


class QuadXTrackingCost(QuadXHoverEnv, utils.EzPickle):
    r"""Custom QuadX Bullet gymnasium environment.

    .. note::
        Can also be used in a vectorized manner. See the
        :gymnasium:`gym.vector <api/vector>` documentation.

    Source:
        Modified version of the `QuadXHover environment`_ found in the
        :PyFlyt:`PyFlyt package <>`. Compared to the original environment:

        -   The reward has been changed to a cost. This was done by negating the reward always
            to be positive definite.
        -   A health penalty has been added. This penalty is applied when the quadrotor moves
            outside the flight dome or crashes. The penalty equals the maximum episode steps
            minus the steps taken or a user-defined penalty.
        -   The ``max_duration_seconds`` has been removed. Instead, the ``max_episode_steps``
            parameter of the :class:`gym.wrappers.TimeLimit` wrapper is used to limit
            the episode duration.
        -   The objective has been changed to track a periodic reference trajectory.
        -   The info dictionary has been extended with the reference, state of interest
            (i.e. the state to track) and reference error.

        The rest of the environment is the same as the original QuadXHover environment.
        Please refer to the `original codebase <https://github.com/jjshoots/PyFlyt>`__,
        :PyFlyt:`the PyFlyt documentation <>` or the accompanying
        `article of Tai et al. 2023`_ for more information.

    .. _`QuadXHover environment`: https://jjshoots.github.io/PyFlyt/documentation/gym_envs/quadx_envs/quadx_hover_env.html
    .. _`Tai et al. 2023`: https://arxiv.org/abs/2304.01305
    .. _`article of Tai et al. 2023`: https://arxiv.org/abs/2304.01305

    Modified cost:
        A cost, computed using the :meth:`QuadXTrackingCost.cost` method, is given for each
        simulation step, including the terminal step. This cost is defined as the
        Euclidean distance error between the quadrotors' current position and a desired
        reference position (i.e. :math:`p=x_{x,y,z}=[0,0,1]`). A health penalty
        can also be included in the cost. This health penalty is added when the drone
        leaves the flight dome or crashes. It equals the ``max_episode_steps`` minus the
        number of steps taken in the episode or a fixed value. The cost is computed as:

        .. math::

            cost = \| p_{drone} - p_{reference} \| + p_{health}

    Solved Requirements:
        Considered solved when the average cost is less than or equal to 50 over
        100 consecutive trials.

    How to use:
        .. code-block:: python

            import stable_gym
            import gymnasium as gym
            env = gym.make("stable_gym:QuadXTrackingCost-v1")

    Attributes:
        state (numpy.ndarray): The current system state.
        agent_hz (int): The agent looprate.
        initial_physics_time (float): The simulation startup time. The physics time at
            the start of the episode after all the initialisation has been done.
    """  # noqa: E501

    def __init__(
        self,
        flight_dome_size=3.0,
        angle_representation="quaternion",
        agent_hz=40,
        render_mode=None,
        render_resolution=(480, 480),
        reference_target_position=(0.0, 0.0, 1.0),
        reference_amplitude=(1.0, 1.0, 0.25),
        reference_frequency=(0.25, 0.25, 0.10),
        reference_phase_shift=(0.0, -np.pi / 2.0, 0.0),
        include_health_penalty=True,
        health_penalty_size=None,
        exclude_reference_from_observation=False,
        exclude_reference_error_from_observation=True,
        action_space_dtype=np.float64,
        observation_space_dtype=np.float64,
        **kwargs,
    ):
        """Initialise a new QuadXTrackingCost environment instance.

        Args:
            flight_dome_size (float, optional): Size of the allowable flying area. By
                default ``3.0``.
            angle_representation (str, optional): The angle representation to use.
                Can be ``"euler"`` or ``"quaternion"``. By default ``"quaternion"``.
            agent_hz (int, optional): Looprate of the agent to environment interaction.
                By default ``40``.
            render_mode (None | str, optional): The render mode. Can be ``"human"`` or
                ``None``. By default ``None``.
            render_resolution (tuple[int, int], optional): The render resolution. By
                default ``(480, 480)``.
            reference_target_position (tuple[float, float, float], optional): The
                target position of the reference. Defaults to ``(0.0, 0.0, 1.0)``.
            reference_amplitude (tuple[float, float, float], optional): The amplitude
                of the reference. Defaults to ``(1.0, 1.0, 0.25)``.
            reference_frequency (tuple[float, float, float], optional): The frequency
                of the reference. Defaults to ``(0.25, 0.25, 0.10)``.
            reference_phase_shift (tuple[float, float, float], optional): The phase
                shift of the reference. Defaults to ``(0.0, -np.pi / 2, 0.0)``.
            include_health_penalty (bool, optional): Whether to penalize the quadrotor
                if it becomes unhealthy (i.e. if it falls over). Defaults to ``True``.
            health_penalty_size (int, optional): The size of the unhealthy penalty.
                Defaults to ``None``. Meaning the penalty is equal to the max episode
                steps and the steps taken.
            exclude_reference_from_observation (bool, optional): Whether the reference
                should be excluded from the observation. Defaults to ``False``.
            exclude_reference_error_from_observation (bool, optional): Whether the error
                should be excluded from the observation. Defaults to ``True``.
            action_space_dtype (union[numpy.dtype, str], optional): The data type of the
                action space. Defaults to ``np.float64``.
            observation_space_dtype (union[numpy.dtype, str], optional): The data type
                of the observation space. Defaults to ``np.float64``.
            **kwargs: Additional keyword arguments passed to the
                :class:`~PyFlyt.gym_envs.quadx_envs.quadx_hover_env.QuadXHoverEnv`
        """
        reference_target_position = np.array(reference_target_position)
        reference_amplitude = np.array(reference_amplitude)
        reference_frequency = np.array(reference_frequency)
        reference_phase_shift = np.array(reference_phase_shift)
        assert "sparse_reward" not in kwargs, (
            "'sparse_reward' should not be passed to the 'QuadXTrackingCost' "
            "environment as only 'dense' rewards are supported."
        )
        assert "max_duration_seconds" not in kwargs, (
            "'max_duration_seconds' should not be passed to the 'QuadXTrackingCost' "
            "as we use gymnasium's 'max_episode_steps' parameter together with "
            "the 'TimeLimit' wrapper to limit the episode duration."
        )
        assert reference_target_position.shape == (
            3,
        ), "The 'reference_target_position' must be a 3D vector."
        assert reference_amplitude.shape == (
            3,
        ), "The 'reference_amplitude' must be a 3D vector."
        assert reference_frequency.shape == (
            3,
        ), "The 'reference_frequency' must be a 3D vector."
        assert reference_phase_shift.shape == (
            3,
        ), "The 'reference_phase_shift' must be a 3D vector."
        assert np.all(
            reference_frequency >= 0
        ), "The x,y, z reference frequencies must be equal to or greater than zero."

        # Check if reference goes into the ground or above the flight dome.
        assert reference_target_position[2] > 0, (
            "The z component of the 'reference_target_position' must be greater than "
            "zero."
        )
        assert reference_target_position[2] - reference_amplitude[2] > 0, (
            "The z component of the 'reference_target_position' must be greater than "
            "the z component of the 'reference_amplitude' for the drone to be able to "
            "track the reference without going into the ground."
        )
        assert (
            reference_target_position[2] + reference_amplitude[2] < flight_dome_size
        ), (
            "The z component of the 'reference_target_position' plus the z component "
            "of the 'reference_amplitude' must be less than the 'flight_dome_size' "
            "for the drone to be able to track the reference without going above the "
            "flight dome."
        )
        assert (
            reference_target_position[0] + reference_amplitude[0] < flight_dome_size
        ), (
            "The x component of the 'reference_target_position' plus the x component "
            "of the 'reference_amplitude' must be less than the 'flight_dome_size' "
            "for the drone to be able to track the reference without going outside "
            "the flight dome."
        )
        assert (
            reference_target_position[1] + reference_amplitude[1] < flight_dome_size
        ), (
            "The y component of the 'reference_target_position' plus the y component "
            "of the 'reference_amplitude' must be less than the 'flight_dome_size' "
            "for the drone to be able to track the reference without going outside "
            "the flight dome."
        )
        assert (
            (reference_amplitude == 0).all() or (reference_frequency == 0).all()
        ) or not (
            exclude_reference_from_observation
            and exclude_reference_error_from_observation
        ), (
            "The agent needs to observe either the reference or the reference error "
            "for it to be able to learn when the reference is not constant."
        )

        self.state = None
        self.initial_physics_time = None
        self._max_episode_steps_applied = False
        self.agent_hz = agent_hz
        self._reference_target_pos = np.array(reference_target_position)
        self._reference_amplitude = reference_amplitude
        self._reference_frequency = reference_frequency
        self._reference_phase_shift = reference_phase_shift
        self._include_health_penalty = include_health_penalty
        self._health_penalty_size = health_penalty_size
        self._exclude_reference_from_observation = exclude_reference_from_observation
        self._exclude_reference_error_from_observation = (
            exclude_reference_error_from_observation
        )
        self._action_space_dtype = action_space_dtype
        self._observation_space_dtype = observation_space_dtype
        self._action_dtype_conversion_warning = False

        # Get reference target urdf file from PyFlyt.
        PyFlyt_dir = PyFlyt.__path__[0]
        self._reference_obj_dir = str(
            PurePath(PyFlyt_dir).joinpath("models", "target.urdf")
        )
        self._reference_visual = None

        super().__init__(
            flight_dome_size=flight_dome_size,
            angle_representation=angle_representation,
            agent_hz=agent_hz,
            render_mode=render_mode,
            render_resolution=render_resolution,
            **kwargs,
        )

        # Change action space dtype if necessary.
        if self._action_space_dtype != self.action_space.dtype:
            self.action_space = gym.spaces.Box(
                self.action_space.low,
                self.action_space.high,
                dtype=self._action_space_dtype,
                seed=self.action_space.np_random,
            )

        # Extend observation space if necessary.
        low = self.observation_space.low
        high = self.observation_space.high
        if not self._exclude_reference_from_observation:
            low = np.append(
                low, [-flight_dome_size, -flight_dome_size, -flight_dome_size]
            )
            high = np.append(
                high, [flight_dome_size, flight_dome_size, flight_dome_size]
            )
        if not self._exclude_reference_error_from_observation:
            low = np.append(
                low,
                [-2 * flight_dome_size, -2 * flight_dome_size, -2 * flight_dome_size],
            )
            high = np.append(
                high,
                [2 * flight_dome_size, 2 * flight_dome_size, 2 * flight_dome_size],
            )
        self.observation_space = gym.spaces.Box(
            low,
            high,
            dtype=self._observation_space_dtype,
            seed=self.observation_space.np_random,
        )

        # NOTE: Done to ensure the args of the QuadXTrackingCost class are also pickled.
        # NOTE: Ensure that all args are passed to the EzPickle class!
        utils.EzPickle.__init__(
            self,
            flight_dome_size,
            angle_representation,
            agent_hz,
            render_mode,
            render_resolution,
            reference_target_position,
            reference_amplitude,
            reference_frequency,
            reference_phase_shift,
            include_health_penalty,
            health_penalty_size,
            exclude_reference_from_observation,
            exclude_reference_error_from_observation,
            action_space_dtype=action_space_dtype,
            observation_space_dtype=observation_space_dtype,
            **kwargs,
        )

    def reference(self, t):
        """Returns the current value of the (periodic) drone (x, y, z) reference
        position that should be tracked.

        Args:
            t (float): The current time step.

        Returns:
            float: The current reference position.
        """
        return np.array(
            [
                self._reference_target_pos[0]
                + self._reference_amplitude[0]
                * np.sin(
                    ((2 * np.pi) * self._reference_frequency[0] * t)
                    + self._reference_phase_shift[0]
                ),
                self._reference_target_pos[1]
                + self._reference_amplitude[1]
                * np.sin(
                    ((2 * np.pi) * self._reference_frequency[1] * t)
                    + self._reference_phase_shift[1]
                ),
                self._reference_target_pos[2]
                + self._reference_amplitude[2]
                * np.sin(
                    ((2 * np.pi) * self._reference_frequency[2] * t)
                    + self._reference_phase_shift[2]
                ),
            ]
        )

    def cost(self):
        """Compute the cost of the current state.

        Returns:
            (float): The cost.
        """
        ref = self.reference(self.t)

        # Euclidean distance from reference point.
        linear_distance = np.linalg.norm(self.env.state(0)[-1] - ref)

        return linear_distance

    def step(self, action):
        """Take step into the environment.

        .. note::
            This method overrides the
            :meth:`~PyFlyt.gym_envs.quadx_envs.quadx_hover_env.QuadXHoverEnv.step`
            method such that the new cost function is used.

        Args:
            action (np.ndarray): Action to take in the environment.

        Returns:
            (tuple): tuple containing:

                -   obs (:obj:`np.ndarray`): Environment observation.
                -   cost (:obj:`float`): Cost of the action.
                -   terminated (:obj:`bool`): Whether the episode is terminated.
                -   truncated (:obj:`bool`): Whether the episode was truncated. This
                    value is set by wrappers when for example a time limit is reached
                    or the agent goes out of bounds.
                -   info (:obj:`dict`): Additional information about the environment.
        """
        # Convert action to correct data type if needed.
        if action.dtype != self._action_space_dtype:
            if not self._action_dtype_conversion_warning:
                logger.warn(
                    "The data type of the action that is supplied to the "
                    f"'ros_gazebo_gym:{self.spec.id}' environment ({action.dtype}) "
                    "does not match the data type of the action space "
                    f"({self._action_space_dtype.__name__}). The action data type will "
                    "be converted to the action space data type."
                )
                self._action_dtype_conversion_warning = True
            action = action.astype(self._action_space_dtype)

        obs, _, terminated, truncated, info = super().step(action)

        # Calculate the cost.
        cost = self.cost()

        # Add optional health penalty at the end of the episode if requested.
        if self._include_health_penalty:
            if terminated and info["collision"] or info["out_of_bounds"]:
                if self._health_penalty_size is not None:
                    cost += self._health_penalty_size
                else:  # If not set add unperformed steps to the cost.
                    cost += self.time_limit_max_episode_steps - self.step_count

        # Add reference and reference error to observation.
        ref = self.reference(self.t)
        if not self._exclude_reference_from_observation:
            obs = np.append(obs, ref)
        if not self._exclude_reference_error_from_observation:
            obs = np.append(obs, self.env.state(0)[-1] - ref)

        # If we are rendering, update the reference target.
        if self.render_mode is not None:
            self.visualize_reference()

        self.state = obs

        # Update info dictionary and change observation dtype.
        info_dict = dict(
            reference=ref,
            state_of_interest=self.env.state(0)[-1],
            reference_error=self.env.state(0)[-1] - ref,
        )
        info.update(info_dict)
        obs = obs.astype(self._observation_space_dtype)

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
        # Apply TimeLimit wrapper 'max_episode_steps' to the environment if exists.
        if not self._max_episode_steps_applied:
            self._max_episode_steps_applied = True
            time_limit_max_episode_steps = self.time_limit_max_episode_steps
            if time_limit_max_episode_steps is not None:
                self.max_steps = self.time_limit_max_episode_steps

        obs, info = super().reset(seed=seed, options=options)

        # Store simulation startup time.
        self.initial_physics_time = self.env.elapsed_time

        # Add reference and reference error to observation.
        ref = self.reference(self.t)
        if not self._exclude_reference_from_observation:
            obs = np.append(obs, ref)
        if not self._exclude_reference_error_from_observation:
            obs = np.append(obs, self.env.state(0)[-1] - ref)

        # If we are rendering, load in the reference target.
        self._reference_visual = None
        if self.render_mode is not None:
            self.visualize_reference()

        self.state = obs

        # Update info dictionary and change observation dtype.
        info_dict = dict(
            reference=ref,
            state_of_interest=self.env.state(0)[-1],
            reference_error=self.env.state(0)[-1] - ref,
        )
        info.update(info_dict)
        obs = obs.astype(self._observation_space_dtype)

        return obs, info

    def visualize_reference(self):
        """Visualize the reference target."""
        if self._reference_visual is None:
            self._reference_visual = self.env.loadURDF(
                self._reference_obj_dir,
                basePosition=self.reference(self.t),
                useFixedBase=True,
                globalScaling=0.01,
            )
            self._reference_visual_loaded = True
            self.env.changeVisualShape(
                self._reference_visual,
                linkIndex=-1,
                rgbaColor=(0, 1, 0, 1),
            )
        # Move the reference target.
        self.env.resetBasePositionAndOrientation(
            self._reference_visual,
            list(self.reference(self.t)),
            [0, 0, 0, 1],
        )

    @property
    def time_limit_max_episode_steps(self):
        """The maximum number of steps that the environment can take before it is
        truncated by the :class:`gymnasium.wrappers.TimeLimit` wrapper.
        """
        time_limit_max_episode_steps = (
            self._time_limit_max_episode_steps
            if hasattr(self, "_time_limit_max_episode_steps")
            and self._time_limit_max_episode_steps is not None
            else gym.registry[self.spec.id].max_episode_steps
        )
        return time_limit_max_episode_steps

    @property
    def time_limit(self):
        """The maximum duration of the episode in seconds."""
        return self.max_steps * self.agent_hz

    @property
    def dt(self):
        """The environment step size.

        Returns:
            (float): The simulation step size. Returns ``None`` if the environment is
                not yet initialized.
        """
        return 1 / self.agent_hz

    @property
    def tau(self):
        """Alias for the environment step size. Done for compatibility with the
        other gymnasium environments.

        Returns:
            (float): The simulation step size. Returns ``None`` if the environment is
                not yet initialized.
        """
        return self.dt

    @property
    def t(self):
        """Environment time."""
        return (
            self.env.elapsed_time - self.initial_physics_time
            if hasattr(self, "env")
            else 0.0
        )

    @property
    def physics_time(self):
        """Returns the physics time."""  # noqa: E501
        return self.env.elapsed_time if hasattr(self, "env") else 0.0


if __name__ == "__main__":
    print("Setting up 'QuadXTrackingCost' environment.")
    env = gym.make("stable_gym:QuadXTrackingCost", render_mode="human")

    # Run episodes.
    episode = 0
    path, paths = [], []
    s, _ = env.reset()
    path.append(s)
    print(f"\nPerforming '{EPISODES}' in the 'QuadXTrackingCost' environment...\n")
    print(f"Episode: {episode}")
    while episode + 1 <= EPISODES:
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
    print("\nFinished 'QuadXTrackingCost' environment simulation.")

    # Plot results per episode.
    print("\nPlotting episode data...")
    for i in range(len(paths)):
        path = paths[i]
        fig, ax = plt.subplots()
        print(f"\nEpisode: {i+1}")
        path = np.array(path)
        t = np.linspace(0, path.shape[0] * env.dt, path.shape[0])
        for j in range(path.shape[1]):  # NOTE: Change if you want to plot less states.
            ax.plot(t, path[:, j], label=f"State {j+1}")
        ax.set_xlabel("Time (s)")
        ax.set_title(f"QuadXTrackingCost episode '{i+1}'")
        ax.legend()
        print("Close plot to see next episode...")
        plt.show()

    print("\nDone")
    env.close()
