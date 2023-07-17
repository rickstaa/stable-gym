"""The MinitaurCost gymnasium environment."""
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from gymnasium import utils
from pybullet_envs.bullet.minitaur_gym_env import MinitaurBulletEnv

from stable_gym import ENVS  # noqa: F401
from stable_gym.common.utils import change_dict_key, convert_gym_box_to_gymnasium_box

EPISODES = 10  # Number of env episodes to run when __main__ is called.
RANDOM_STEP = True  # Use random action in __main__. Zero action otherwise.

# Retrieve max episode steps from the ENVS dict.
# NOTE: Needed for default health penalty size.
MAX_EPISODE_STEPS = ENVS["MinitaurCost-v1"]["max_episode_steps"]


# TODO: Update solving criteria after training.
class MinitaurCost(MinitaurBulletEnv, utils.EzPickle):
    """Custom Minitaur gymnasium environment.

    .. note::
        Can also be used in a vectorized manner. See the
        :gymnasium:`gym.vector <api/vector>` documentation.

    Source:
        Modified version of the `Minitaur environment`_ in v3.2.5 of the
        `pybullet package`_. This modification was first described by
        `Han et al. 2020`_. In this modified version:

        -   The objective was changed to a velocity-tracking task. To do this, the
            reward is replaced with a cost. This cost is the squared difference between
            the Minitaur's forward velocity and a reference value (error). Additionally,
            also a energy cost and health penalty can be included in the cost.
        -   A minimal backward velocity bound is added to prevent the Minitaur from
            walking backwards.
        -   Users are given the option to modify the Minitaur fall criteria, and thus
            the episode termination criteria.

        The rest of the environment is the same as the original Minitaur environment.
        Please refer to the `original codebase`_ or `the article of Tan et al. 2018`_ on
        which the Minitaur environment is based for more information.

    .. _`Minitaur environment`: https://arxiv.org/abs/1804.10332
    .. _`pybullet package`:  https://pybullet.org/
    .. _`Han et al. 2020`: https://arxiv.org/abs/2004.14288
    .. _`original codebase`: https://github.com/bulletphysics/bullet3/blob/master/examples/pybullet/gym/pybullet_envs/bullet/minitaur_gym_env.py
    .. _`the article of Tan et al. 2018`: https://arxiv.org/abs/1804.10332

    Observation:
        **Type**: Box(28)

        The angles, velocities and torques of all motors.

    Actions:
        **Type**: Box(8)

        A list of desired motor angles for eight motors.

    Modified cost:
        .. math::

            cost = w_{forward\_velocity} \\times (x_{velocity} - x_{reference\_x\_velocity})^2 + w_{ctrl} \\times c_{ctrl} + p_{health}

    Starting State:
        The robot always starts at the same position and orientation, with zero
        velocity.

    Episode Termination:
        -   The episode is terminated if the Minitaur falls, meaning that the
            the orientation between the base and the world is greater than a threshold or
            the base is too close to the ground.
        -   Optionally, the episode can be terminated if the Minitaur walks backwards.

    Solved Requirements:
        Considered solved when the average cost is less than or equal to 50 over
        100 consecutive trials.

    How to use:
        .. code-block:: python

            import stable_gym
            import gymnasium as gym
            env = gym.make("MinitaurCost-v1")

    Attributes:
        state (numpy.ndarray): The current system state.
        reference_forward_velocity (float): The forward velocity that the agent should
            try to track.

    .. attention::
        Since the :meth:`~pybullet_envs.bullet.minitaur_gym_env.MinitaurBulletEnv`
        is not yet compatible with :gymnasium:`gymnasium v>=0.26.0 <>`, the
        :class:`gym.wrappers.EnvCompatibility` wrapper is used. This has the
        side effect that the ``render_mode`` argument is not working. Instead,
        the ``render`` argument should be used.
    """  # noqa: E501, W605

    # Replace deprecated metadata keys with new ones.
    # See https://github.com/openai/gym/pull/2654.
    # TODO: Can be removed when https://github.com/bulletphysics/bullet3/issues/4369 is resoled. # noqa: E501
    metadata = MinitaurBulletEnv.metadata
    change_dict_key(metadata, "render.modes", "render_modes")

    def __init__(
        self,
        reference_forward_velocity=1.0,
        randomise_reference_forward_velocity=False,
        randomise_reference_forward_velocity_range=(0.5, 1.5),
        forward_velocity_weight=1.0,
        include_energy_cost=False,
        energy_weight=0.005,
        include_shake_cost=False,
        shake_weight=0.01,  # NOTE: 0.0 in original environment.
        include_drift_cost=False,  # NOTE: 0.05 in original environment.
        drift_weight=0.01,
        distance_limit=float("inf"),
        render=False,
        include_health_penalty=True,
        health_penalty_size=None,
        backward_velocity_bound=-0.5,
        fall_criteria_up_rotation=0.85,
        fall_criteria_z_position=0.13,
        exclude_reference_from_observation=False,
        exclude_reference_error_from_observation=True,  # NOTE: False in Han et al. 2018. # noqa: E501
        exclude_x_velocity_from_observation=False,
        **kwargs,
    ):
        """Initialise a new MinitaurCost environment instance.

        Args:
            reference_forward_velocity (float, optional): The forward velocity that the
                agent should try to track. Defaults to ``1.0``.
            randomise_reference_forward_velocity (bool, optional): Whether to randomize
                the reference forward velocity. Defaults to ``False``.
            randomise_reference_forward_velocity_range (tuple, optional): The range of
                the random reference forward velocity. Defaults to ``(0.5, 1.5)``.
            forward_velocity_weight (float, optional): The weight used to scale the
                forward velocity error. Defaults to ``1.0``.
            include_energy_cost (bool, optional): Whether to include the energy cost in
                the cost function (i.e. energy of the motors). Defaults to ``False``.
            energy_weight (float, optional): The weight used to scale the energy cost.
                Defaults to ``0.005``.
            include_shake_cost (bool, optional): Whether to include the shake cost in
                the cost function (i.e. moving up and down). Defaults to ``False``.
            shake_weight (float, optional): The weight used to scale the shake cost.
                Defaults to ``0.01``.
            include_drift_cost (bool, optional): Whether to include the drift cost in
                the cost function (i.e. movement in the y direction). Defaults to
                ``False``.
            drift_weight (float, optional): The weight used to scale the drift cost.
                Defaults to ``0.01``.
            distance_limit (float, optional): The max distance (in meters) that the
                agent can travel before the episode is terminated. Defaults to
                ``float("inf")``.
            render (bool, optional): Whether to render the environment. Defaults to
                ``False``.
            include_health_penalty (bool, optional): Whether to penalize the Minitaur if
                it becomes unhealthy (i.e. if it falls over). Defaults to ``True``.
            health_penalty_size (int, optional): The size of the unhealthy penalty.
                Defaults to ``None``. Meaning the penalty is equal to the max episode
                steps and the steps taken.
            backward_velocity_bound (float): The max backward velocity (in meters per
                second) before the episode is terminated. Defaults to ``-0.5``.
            fall_criteria_up_rotation (float): The max up rotation (in radians) between
                the base and the world before the episode is terminated. Defaults to
                ``0.85``.
            fall_criteria_z_position (float): The max z position (in meters) before the
                episode is terminated. Defaults to ``0.13``.
            exclude_reference_from_observation (bool, optional): Whether the reference
                should be excluded from the observation. Defaults to ``False``. Can only
                be set to ``True`` if ``randomise_reference_forward_velocity`` is set to
                ``False``.
            exclude_reference_error_from_observation (bool, optional): Whether the error
                should be excluded from the observation. Defaults to ``True``.
            exclude_x_velocity_from_observation (bool, optional): Whether to omit the
                x- component of the velocity from observations. Defaults to ``False``.
            **kwargs: Extra keyword arguments to pass to the :class:`MinitaurBulletEnv`
                class.
        """  # noqa: E501
        self.state = None
        self.t = 0.0
        self.reference_forward_velocity = reference_forward_velocity
        self._randomise_reference_forward_velocity = (
            randomise_reference_forward_velocity
        )
        self._randomise_reference_forward_velocity_range = (
            randomise_reference_forward_velocity_range
        )
        self._forward_velocity_weight = forward_velocity_weight
        self._include_energy_cost = include_energy_cost
        self._energy_weight = energy_weight
        self._include_shake_cost = include_shake_cost
        self._shake_weight = shake_weight
        self._include_drift_cost = include_drift_cost
        self._drift_weight = drift_weight
        self._include_health_penalty = include_health_penalty
        self._health_penalty_size = health_penalty_size
        self._backward_velocity_bound = backward_velocity_bound
        self._fall_criteria_up_rotation = fall_criteria_up_rotation
        self._fall_criteria_z_position = fall_criteria_z_position
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

        # Initialise the MinitaurBulletEnv class.
        super().__init__(
            energy_weight=energy_weight,
            shake_weight=shake_weight,
            drift_weight=drift_weight,
            distance_limit=distance_limit,
            render=render,
            **kwargs,
        )

        # Convert gym spaces to gymnasium spaces.
        # TODO: Can be removed when https://github.com/bulletphysics/bullet3/issues/4369 is resoled. # noqa: E501
        self.observation_space = convert_gym_box_to_gymnasium_box(
            self.observation_space
        )
        self.action_space = convert_gym_box_to_gymnasium_box(self.action_space)

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
        # NOTE: Done to ensure the args of the MinitaurCost class are also pickled.
        # NOTE: Ensure that all args are passed to the EzPickle class!
        utils.EzPickle.__init__(
            self,
            reference_forward_velocity,
            forward_velocity_weight,
            include_energy_cost,
            energy_weight,
            include_shake_cost,
            shake_weight,
            include_drift_cost,
            drift_weight,
            distance_limit,
            render,
            include_health_penalty,
            health_penalty_size,
            backward_velocity_bound,
            fall_criteria_up_rotation,
            fall_criteria_z_position,
            exclude_reference_from_observation,
            exclude_reference_error_from_observation,
            exclude_x_velocity_from_observation,
            **kwargs,
        )

    def cost(self, x_velocity, energy_cost, drift_cost, shake_cost):
        """Compute the cost of a given base x velocity, energy cost, shake cost and
        drift cost.

        Args:
            x_velocity (float): The Minitaurs's base x velocity.
            energy_cost (float): The energy cost (i.e. motor cost).
            drift_cost (float): The drift (y movement) cost.
            shake_cost (float): The shake (z movement) cost.

        Returns:
            (tuple): tuple containing:

                -   cost (float): The cost of the action.
                -   info (:obj:`dict`): Additional information about the cost.
        """
        velocity_cost = self._forward_velocity_weight * np.square(
            x_velocity - self.reference_forward_velocity
        )
        cost = velocity_cost
        if self._include_energy_cost:
            cost += self._energy_weight * energy_cost
        if self._include_shake_cost:
            cost += self._shake_weight * shake_cost
        if self._include_drift_cost:
            cost += self._drift_weight * drift_cost

        return cost, {
            "cost_velocity": velocity_cost,
            "energy_cost": energy_cost,
            "cost_shake": shake_cost,
            "cost_drift": drift_cost,
        }

    def step(self, action):
        """Take step into the environment.

        .. note::
            This method overrides the
            :meth:`~pybullet_envs.bullet.minitaur_gym_env.MinitaurBulletEnv.step` method
            such that the new cost function is used.

        Args:
            action (np.ndarray): Action to take in the environment.
            render_mode (str, optional): The render mode to use. Defaults to ``None``.

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
        obs, _, terminated, info = super().step(action)

        # Add reference, x velocity and reference error to observation.
        base_velocity = self.base_velocity
        if not self._exclude_reference_from_observation:
            obs = np.append(obs, self.reference_forward_velocity)
        if not self._exclude_reference_error_from_observation:
            obs = np.append(obs, base_velocity - self.reference_forward_velocity)
        if not self._exclude_x_velocity_from_observation:
            obs = np.append(obs, base_velocity)

        self.state = obs
        self.t = self.t + self.dt

        # Retrieve original rew5ards and base velocity.
        # NOTE: Han et al. 2018 used the squared error for the drift reward. We use the
        # version found in the original Minitaur environment (i.e. absolute distance).
        objectives = self.get_objectives()
        last_rewards = objectives[-1]
        _, energy_reward, drift_reward, shake_reward = last_rewards
        drift_cost, shake_cost = -drift_reward, -shake_reward

        # Compute the cost and update the info dict.
        cost, cost_info = self.cost(
            self.base_velocity, energy_reward, drift_cost, shake_cost
        )
        info.update(cost_info)

        # Add optional health penalty at the end of the episode if requested.
        if self._include_health_penalty:
            if terminated:
                if self._health_penalty_size is not None:
                    cost += self._health_penalty_size
                else:  # If not set add unperformed steps to the cost.
                    cost += MAX_EPISODE_STEPS - self._env_step_counter

        return obs, cost, terminated, info

    def reset(self):
        """Reset gymnasium environment.

        Returns:
            (np.ndarray): Initial environment observation.
        """
        obs = super().reset()

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
        self.t = 0.0

        return obs

    def _termination(self):
        """Check whether the episode is terminated.

        .. note::
            This method overrides the :meth:`_termination` method of the original
            Minitaur environment so that we can also set a minimum velocity criteria.

        Returns:
            (bool): Boolean value that indicates whether the episode is terminated.
        """
        # NOTE: Han et al. 2018 returns `FALSE` here. We use the original termination
        # criteria from the Minitaur environment + a minimum velocity criteria.
        terminated = super()._termination()

        # Check if the minotaur is moved backwards.
        if self._backward_velocity_bound is not None:
            base_velocity = self.base_velocity
            if base_velocity <= self._backward_velocity_bound:
                terminated = True

        return terminated

    def is_fallen(self):
        """Check whether the minitaur has fallen.

        If the up directions (i.e. angle) between the base and the world are larger
        (the dot product is smaller than :attr:`._fall_criteria_up_rotation`) or the
        base is close to the ground (the height is smaller than
        :attr:`._fall_criteria_z_position`), the minitaur is considered fallen.

        .. note::
            This method overrides the :meth:`is_fallen` method of the original
            Minitaur environment to give users the ability to set the fall criteria.

        Returns:
            (bool): Boolean value that indicates whether the minitaur has fallen.
        """
        # NOTE: Han et al. 2018 doesn't use the z position criteria.
        orientation = self.minitaur.GetBaseOrientation()
        rot_mat = self._pybullet_client.getMatrixFromQuaternion(orientation)
        local_up = rot_mat[6:]
        pos = self.minitaur.GetBasePosition()
        return (
            np.dot(np.asarray([0, 0, 1]), np.asarray(local_up))
            < self._fall_criteria_up_rotation
            or pos[2] < self._fall_criteria_z_position
        )

    @property
    def base_velocity(self):
        """The base velocity of the minitaur."""
        objectives = self.get_objectives()
        forward_reward = objectives[-1][
            0
        ]  # NOTE: Forward_reward is x distance travelled in 1 time-step.
        base_velocity = forward_reward / self.dt
        return base_velocity

    @property
    def dt(self):
        """The environment step size."""
        return self._time_step

    @property
    def tau(self):
        """Alias for the environment step size. Done for compatibility with the
        other gymnasium environments.
        """
        return self._time_step


if __name__ == "__main__":
    print("Setting up 'MinitaurCost' environment.")
    env = gym.make("MinitaurCost", render=True)

    # Run episodes.
    episode = 0
    path, paths = [], []
    s, _ = env.reset()
    path.append(s)
    print(f"\nPerforming '{EPISODES}' in the 'MinitaurCost' environment...\n")
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
    print("\nFinished 'MinitaurCost' environment simulation.")

    # Plot results per episode.
    print("\nPlotting episode data...")
    for i in range(len(paths)):
        path = paths[i]
        fig, ax = plt.subplots()
        print(f"\nEpisode: {i}")
        path = np.array(path)
        t = np.linspace(0, path.shape[0] * env.unwrapped.env.dt, path.shape[0])
        for j in range(path.shape[1]):  # NOTE: Change if you want to plot less states.
            ax.plot(t, path[:, j], label=f"State {j}")
        ax.set_xlabel("Time (s)")
        ax.set_title(f"MinitaurCost episode '{i}'")
        ax.legend()
        print("Close plot to see next episode...")
        plt.show()

    print("\nDone")
