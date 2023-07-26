"""The QuadXHoverCost gymnasium environment."""
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from gymnasium import utils
from PyFlyt.gym_envs.quadx_envs.quadx_hover_env import QuadXHoverEnv

from stable_gym import ENVS  # noqa: F401

EPISODES = 10  # Number of env episodes to run when __main__ is called.
RANDOM_STEP = True  # Use random action in __main__. Zero action otherwise.


class QuadXHoverCost(QuadXHoverEnv, utils.EzPickle):
    r"""Custom QuadXHover Bullet gymnasium environment.

    .. note::
        Can also be used in a vectorized manner. See the
        :gymnasium:`gym.vector <api/vector>` documentation.

    Source:
        Modified version of the `QuadXHover environment`_ found in the
        :PyFlyt:`PyFlyt package <>`. This environment was first described by `Tai et al. 2023`_.
        In this modified version:

        -   The reward has been changed to a cost. This was done by negating the reward always
            to be positive definite.
        -   A health penalty has been added. This penalty is applied when the quadrotor moves
            outside the flight dome or crashes. The penalty equals the maximum episode steps
            minus the steps taken or a user-defined penalty.
        -   The ``max_duration_seconds`` has been removed. Instead, the ``max_episode_steps``
            parameter of the :class:`gym.wrappers.TimeLimit` wrapper is used to limit
            the episode duration.

        The rest of the environment is the same as the original QuadXHover environment.
        Please refer to the `original codebase <https://github.com/jjshoots/PyFlyt>`__,
        :PyFlyt:`the PyFlyt documentation <>` or the accompanying
        `article of Tai et al. 2023`_ for more information.

    .. _`QuadXHover environment`: https://jjshoots.github.io/PyFlyt/documentation/gym_envs/quadx_envs/quadx_hover_env.html
    .. _`Tai et al. 2023`: https://arxiv.org/abs/2304.01305
    .. _`article of Tai et al. 2023`: https://arxiv.org/abs/2304.01305

    Modified cost:
        A cost, computed using the :meth:`QuadXHoverCost.cost` method, is given for each
        simulation step, including the terminal step. This cost is defined as the
        Euclidean distance error between the quadrotors' current position and a desired
        hover position (i.e. :math:`p=x_{x,y,z}=[0,0,1]`) and the error between the
        quadrotors' current angular roll and pitch and their zero values. A health penalty
        can also be included in the cost. This health penalty is added when the drone
        leaves the flight dome or crashes. It equals the ``max_episode_steps`` minus the
        number of steps taken in the episode or a fixed value. The cost is computed as:

        .. math::

            cost = \| p_{drone} - p_{hover} \| + \| \theta_{roll,pitch} \| + p_{health}

    Solved Requirements:
        Considered solved when the average cost is less than or equal to 50 over
        100 consecutive trials.

    How to use:
        .. code-block:: python

            import stable_gym
            import gymnasium as gym
            env = gym.make("QuadXHoverCost-v1")

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
        include_health_penalty=True,
        health_penalty_size=None,
        **kwargs,
    ):
        """Initialise a new QuadXHoverCost environment instance.

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
            include_health_penalty (bool, optional): Whether to penalize the quadrotor
                if it becomes unhealthy (i.e. if it falls over). Defaults to ``True``.
            health_penalty_size (int, optional): The size of the unhealthy penalty.
                Defaults to ``None``. Meaning the penalty is equal to the max episode
                steps and the steps taken.
            **kwargs: Additional keyword arguments passed to the
                :class:`~PyFlyt.gym_envs.quadx_envs.quadx_hover_env.QuadXHoverEnv`
        """
        assert "sparse_reward" not in kwargs, (
            "'sparse_reward' should not be passed to the 'QuadXHoverCost' "
            "environment as only 'dense' rewards are supported."
        )
        assert "max_duration_seconds" not in kwargs, (
            "'max_duration_seconds' should not be passed to the 'QuadXHoverCost' "
            "as we use gymnasium's 'max_episode_steps' parameter together with "
            "the 'TimeLimit' wrapper to limit the episode duration."
        )
        self.state = None
        self.initial_physics_time = None
        self._max_episode_steps_applied = False
        self.agent_hz = agent_hz
        self._include_health_penalty = include_health_penalty
        self._health_penalty_size = health_penalty_size

        super().__init__(
            flight_dome_size=flight_dome_size,
            angle_representation=angle_representation,
            agent_hz=agent_hz,
            render_mode=render_mode,
            render_resolution=render_resolution,
            **kwargs,
        )

        # NOTE: Done to ensure the args of the QuadXHoverCost class are also pickled.
        # NOTE: Ensure that all args are passed to the EzPickle class!
        utils.EzPickle.__init__(
            self,
            flight_dome_size,
            angle_representation,
            agent_hz,
            render_mode,
            render_resolution,
            include_health_penalty,
            health_penalty_size,
            **kwargs,
        )

    def cost(self):
        """Compute the cost of the current state.

        Returns:
            (tuple): tuple containing:

                -   cost (:obj:`float`): The cost.
                -   info (:obj:`dict`): Dictionary containing additional information
                    about the cost.
        """
        # Euclidean distance from [0, 0, 1] hover point.
        linear_distance = np.linalg.norm(
            self.env.state(0)[-1] - np.array([0.0, 0.0, 1.0])
        )

        # How far are we from 0 roll and pitch.
        angular_distance = np.linalg.norm(self.env.state(0)[1][:2])

        return linear_distance + angular_distance, {
            "linear_distance": linear_distance,
            "angular_distance": angular_distance,
        }

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
        obs, _, terminated, truncated, info = super().step(action)

        # Calculate the cost.
        cost, cost_info = self.cost()

        # Add optional health penalty at the end of the episode if requested.
        if self._include_health_penalty:
            if terminated and info["collision"] or info["out_of_bounds"]:
                if self._health_penalty_size is not None:
                    cost += self._health_penalty_size
                else:  # If not set add unperformed steps to the cost.
                    cost += self.time_limit_max_episode_steps - self.step_count

        self.state = obs

        # Update info dictionary.
        info.update(cost_info)
        info.update(
            {
                "reference": np.array([0.0, 0.0, 1.0, 0.0, 0.0]),
                "state_of_interest": np.concatenate(
                    (self.env.state(0)[-1], self.env.state(0)[1][:2])
                ),
                "reference_error": np.concatenate(
                    (
                        self.env.state(0)[-1] - np.array([0.0, 0.0, 1.0]),
                        -self.env.state(0)[1][:2],
                    )
                ),
            }
        )

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

        _, cost_info = self.cost()

        # Store simulation startup time.
        self.initial_physics_time = self.env.elapsed_time

        self.state = obs

        # Update info dictionary.
        info.update(cost_info)
        info.update(
            {
                "reference": np.array([0.0, 0.0, 1.0, 0.0, 0.0]),
                "state_of_interest": np.concatenate(
                    (self.env.state(0)[-1], self.env.state(0)[1][:2])
                ),
                "reference_error": np.concatenate(
                    (
                        self.env.state(0)[-1] - np.array([0.0, 0.0, 1.0]),
                        -self.env.state(0)[1][:2],
                    )
                ),
            }
        )

        return obs, info

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
    print("Setting up 'QuadXHoverCost' environment.")
    env = gym.make("QuadXHoverCost", render_mode="human")

    # Run episodes.
    episode = 0
    path, paths = [], []
    s, _ = env.reset()
    path.append(s)
    print(f"\nPerforming '{EPISODES}' in the 'QuadXHoverCost' environment...\n")
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
    print("\nFinished 'QuadXHoverCost' environment simulation.")

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
        ax.set_title(f"QuadXHoverCost episode '{i+1}'")
        ax.legend()
        print("Close plot to see next episode...")
        plt.show()

    print("\nDone")
