"""The FetchReachCost gymnasium environment."""

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from gymnasium import logger, utils
from gymnasium_robotics.envs.fetch.reach import MujocoFetchReachEnv

EPISODES = 10  # Number of env episodes to run when __main__ is called.
RANDOM_STEP = True  # Use random action in __main__. Zero action otherwise.


# TODO: Update solving criteria after training.
class FetchReachCost(MujocoFetchReachEnv, utils.EzPickle):
    """Custom FetchReach gymnasium robotics environment.

    .. note::
        Can also be used in a vectorized manner. See the
        :gymnasium:`gym.vector <api/vector>` documentation.

    Source:
        Modified version of the FetchReach Mujoco environment found in the
        `Gymnasium Robotics library <https://robotics.farama.org/envs/fetch/>`_.
        This modification was first described by
        `Han et al. 2020 <https://arxiv.org/abs/2004.14288>`_. In this modified version:

        -   The reward was replaced with a cost. This was done by taking the absolute
            value of the reward.

        The rest of the environment is the same as the original FetchReach environment.
        Below, the modified cost is described. For more information about the
        environment (e.g. observation space, action space, episode termination, etc.),
        please refer to the
        :gymnasium-robotics:`gymnasium robotics library <envs/fetch/reach/>`.

    Modified cost:
        A cost, computed using the :meth:`FetchReachCost.cost` method, is given for each
        simulation step, including the terminal step. This cost is defined as the error
        between FetchReach's end-effector position and the desired goal position (i.e. Euclidean distance).
        The cost is computed as:

        .. math::

            cost = \\left | reward \\right |

    Solved Requirements:
        Considered solved when the average cost is less than or equal to 50 over
        100 consecutive trials.

    How to use:
        .. code-block:: python

            import stable_gyms
            import gymnasium as gym
            env = gym.make("stable_gym:FetchReachCost-v1")

    Attributes:
        state (numpy.ndarray): The current system state.
        dt (float): The environment step size. Also available as :attr:`.tau`.

    ..  attention::
        Accepts all arguments of the original :class:`~gymnasium_robotics.envs.fetch.reach.MujocoFetchReachEnv`
        class except for the ``reward_type`` argument. This is because we require dense
        rewards to calculate the cost.
    """  # noqa: E501

    def __init__(
        self,
        action_space_dtype=np.float32,
        observation_space_dtype=np.float64,
        **kwargs,
    ):
        """Initialise a new FetchReachCost environment instance.

        Args:
            action_space_dtype (union[numpy.dtype, str], optional): The data type of the
                action space. Defaults to ``np.float32``.
            observation_space_dtype (union[numpy.dtype, str], optional): The data type
                of the observation space. Defaults to ``np.float64``.
            **kwargs: Keyword arguments passed to the original
                :class:`~gymnasium_robotics.envs.fetch.reach.MujocoFetchReachEnv` class.
        """  # noqa: E501s
        assert "reward_type" not in kwargs, (
            "'reward_type' should not be passed to the 'FetchReachCost' environment as "
            "only 'dense' rewards are supported."
        )
        self.state = None
        self._action_space_dtype = action_space_dtype
        self._observation_space_dtype = observation_space_dtype
        self._action_dtype_conversion_warning = False

        # Initialise the FetchReachEnv class.
        super().__init__(
            reward_type="dense",  # NOTE: DONT CHANGE! This is required for the cost.
            **kwargs,
        )

        # Change action and observation space data types.
        # obs = self._get_obs()
        self.action_space = gym.spaces.Box(
            low=self.action_space.low.astype(action_space_dtype),
            high=self.action_space.high.astype(action_space_dtype),
            dtype=self._action_space_dtype,
            seed=self.action_space.np_random,
        )
        self.observation_space["observation"] = gym.spaces.Box(
            low=self.observation_space["observation"].low,
            high=self.observation_space["observation"].high,
            dtype=self._observation_space_dtype,
            seed=self.observation_space["observation"].np_random,
        )

        # Reinitialize the EzPickle class.
        # NOTE: Done to ensure the args of the FetchReachCost class are also pickled.
        # NOTE: Ensure that all args are passed to the EzPickle class!
        utils.EzPickle.__init__(
            self,
            action_space_dtype=action_space_dtype,
            observation_space_dtype=observation_space_dtype,
            **kwargs,
        )

    def cost(self, reward):
        """Calculate the cost.

        Args:
            reward (float): The reward returned from the FetchReach environment.

        Returns:
            float: The cost (i.e. negated reward).
        """
        return np.abs(reward)

    def step(self, action):
        """Take step into the environment.

        .. note::
            This method overrides the
            :meth:`~gymnasium_robotics.envs.fetch.fetch_env.MujocoFetchEnv.step` method
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

        obs, reward, terminated, truncated, info = super().step(action)

        self.state = obs

        # Update info dictionary and change observation dtype.
        obs["observation"] = obs["observation"].astype(self._observation_space_dtype)
        info.update(
            {
                "reference": obs["desired_goal"],
                "state_of_interest": obs["observation"][:3],
                "reference_error": obs["observation"][:3] - obs["desired_goal"],
            }
        )

        return obs, self.cost(reward), terminated, truncated, info

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

        self.state = obs["observation"]

        # Update info dictionary and change observation dtype.
        obs["observation"] = obs["observation"].astype(self._observation_space_dtype)
        info.update(
            {
                "reference": obs["desired_goal"],
                "state_of_interest": obs["observation"][:3],
                "reference_error": obs["observation"][:3] - obs["desired_goal"],
            }
        )

        return obs, info

    @property
    def tau(self):
        """Alias for the environment step size. Done for compatibility with the
        other gymnasium environments.
        """
        return self.dt

    @property
    def t(self):
        """Environment time."""
        return self.unwrapped.data.time - self.initial_time

    @property
    def physics_time(self):
        """Returns the physics time."""
        return self.unwrapped.data.time


if __name__ == "__main__":
    print("Setting up 'FetchReachCost' environment.")
    env = gym.make("stable_gym:FetchReachCost", render_mode="human")

    # Run episodes.
    episode = 0
    path, paths = [], []
    s, _ = env.reset()
    path.append(s)
    print(f"\nPerforming '{EPISODES}' in the 'FetchReachCost' environment...\n")
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
    print("\nFinished 'FetchReachCost' environment simulation.")

    # Plot results per episode.
    print("\nPlotting episode data...")
    for i in range(len(paths)):
        path = paths[i]
        fig, ax = plt.subplots()
        print(f"\nEpisode: {i+1}")
        path = np.array(
            [gym.spaces.flatten(env.observation_space, obs) for obs in path]
        )
        t = np.linspace(0, path.shape[0] * env.dt, path.shape[0])
        for j in range(path.shape[1]):  # NOTE: Change if you want to plot less states.
            ax.plot(t, path[:, j], label=f"State {j+1}")
        ax.set_xlabel("Time (s)")
        ax.set_title(f"FetchReachCost episode '{i+1}'")
        ax.legend()
        print("Close plot to see next episode...")
        plt.show()

    print("\nDone")
    env.close()
