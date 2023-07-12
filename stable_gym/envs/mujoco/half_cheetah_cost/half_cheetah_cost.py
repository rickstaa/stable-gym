"""The HalfCheetahCost gymnasium environment."""

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from gymnasium import utils
from gymnasium.envs.mujoco.half_cheetah_v4 import HalfCheetahEnv

import stable_gym  # NOTE: Required to register environments. # noqa: F401

EPISODES = 10  # Number of env episodes to run when __main__ is called.
RANDOM_STEP = True  # Use random action in __main__. Zero action otherwise.


# TODO: Find correct control cost weight.
# TODO: Update solving criteria after training.
class HalfCheetahCost(HalfCheetahEnv, utils.EzPickle):
    """Custom HalfCheetah gymnasium environment.

    .. note::
        Can also be used in a vectorized manner. See the
        :gymnasium:`gym.vector <api/vector>` documentation.

    Source:
        This is a modified version of the HalfCheetah Mujoco environment in v0.28.1 of the
        :gymnasium:`gymnasium library <environments/mujoco/half_cheetah>`. This modification
        was first described by `Han et al. 2020 <https://arxiv.org/abs/2004.14288>`_.
        Compared to the original HalfCheetah environment in this modified version:

        -   The objective was changed to a velocity-tracking task. To do this, the reward
            is replaced with a cost. This cost is the squared difference between the
            HalfCheetah's forward velocity and a reference value (error). Additionally,
            also a control cost can be included in the cost.

        The rest of the environment is the same as the original HalfCheetah environment.
        Below, the modified cost is described. For more information about the environment
        (e.g. observation space, action space, episode termination, etc.), please refer
        to the :gymnasium:`gymnasium library <environments/mujoco/half_cheetah>`.

    Modified cost:
        .. math::

            cost = w_{forward} \\times (x_{velocity} - x_{reference\_x\_velocity})^2 + w_{ctrl} \\times c_{ctrl}

    Solved Requirements:
        Considered solved when the average cost is less than or equal to 50 over
        100 consecutive trials.

    How to use:
        .. code-block:: python

            import stable_gym
            import gymnasium as gym
            env = gym.make("HalfCheetahCost-v1")

    Attributes:
        state (numpy.ndarray): The current system state.
        t (float): The current time step.
        dt (float): The environment step size.
        reference_forward_velocity (float): The forward velocity that the agent should
            try to track.
    """  # noqa: E501, W605

    def __init__(
        self,
        reference_forward_velocity=1.0,
        forward_velocity_weight=1.0,
        include_ctrl_cost=False,
        ctrl_cost_weight=1e-4,  # NOTE: Lower than original because we use different cost. # noqa: E501
        reset_noise_scale=0.1,
        exclude_current_positions_from_observation=True,
        **kwargs,
    ):
        """Constructs all the necessary attributes for the HalfCheetahCost instance.

        Args:
            reference_forward_velocity (float, optional): The forward velocity that the
                agent should try to track. Defaults to ``1.0``.
            forward_velocity_weight (float, optional): The weight used to scale the
                forward velocity error. Defaults to ``1.0``.
            include_ctrl_cost (bool, optional): Whether you also want to penalize the
                half cheetah if it takes actions that are too large. Defaults to
                ``False``.
            ctrl_cost_weight (float, optional): The weight used to scale the control
                cost. Defaults to ``1e-4``.
            reset_noise_scale (float, optional): Scale of random perturbations of the
                initial position and velocity. Defaults to ``0.1``.
            exclude_current_positions_from_observation (bool, optional): Whether to omit
                the x- and y-coordinates of the front tip from observations. Excluding
                the position can serve as an inductive bias to induce position-agnostic
                behaviour in policies. Defaults to ``True``.
        """
        self.reference_forward_velocity = reference_forward_velocity
        self._forward_velocity_weight = forward_velocity_weight
        self._include_ctrl_cost = include_ctrl_cost

        self.state = None
        self.t = 0.0

        # Initialize the HalfCheetahEnv class.
        super().__init__(
            ctrl_cost_weight=ctrl_cost_weight,
            reset_noise_scale=reset_noise_scale,
            exclude_current_positions_from_observation=exclude_current_positions_from_observation,  # noqa: E501
            **kwargs,
        )

        # Reinitialize the EzPickle class.
        # NOTE: Done to ensure the args of the HalfCheetahCost class are also pickled.
        # NOTE: Ensure that all args are passed to the EzPickle class!
        utils.EzPickle.__init__(
            self,
            reference_forward_velocity,
            forward_velocity_weight,
            include_ctrl_cost,
            ctrl_cost_weight,
            reset_noise_scale,
            exclude_current_positions_from_observation,
            **kwargs,
        )

    def cost(self, x_velocity, ctrl_cost):
        """Compute the cost of a given x velocity and control cost.

        Args:
            x_velocity (float): The HalfCheetah's x velocity.
            ctrl_cost (float): The control cost.

        Returns:
            (tuple): tuple containing:

                - cost (float): The cost of the action.
                - info (:obj:`dict`): Additional information about the cost.
        """
        velocity_cost = self._forward_velocity_weight * np.square(
            x_velocity - self.reference_forward_velocity
        )
        cost = velocity_cost
        if self._include_ctrl_cost:
            cost += ctrl_cost
        return cost, {"cost_velocity": velocity_cost, "cost_ctrl": ctrl_cost}

    def step(self, action):
        """Take step into the environment.

        .. note::
            This method overrides the
            :meth:`~gymnasium.envs.mujoco.half_cheetah_v4.HalfCheetahEnv.step` method
            such that the new cost function is used.

        Args:
            action (np.ndarray): Action to take in the environment.

        Returns:
            (tuple): tuple containing:

                - obs (:obj:`np.ndarray`): Environment observation.
                - cost (:obj:`float`): Cost of the action.
                - terminated (:obj`bool`): Whether the episode is terminated.
                - truncated (:obj:`bool`): Whether the episode was truncated. This value
                    is set by wrappers when for example a time limit is reached or the
                    agent goes out of bounds.
                - info (:obj`dict`): Additional information about the environment.
        """
        obs, _, terminated, truncated, info = super().step(action)

        self.state = obs
        self.t = self.t + self.dt

        cost, cost_info = self.cost(info["x_velocity"], -info["reward_ctrl"])

        # Update info.
        del info["reward_run"], info["reward_ctrl"]
        info["cost_velocity"] = cost_info["cost_velocity"]
        info["cost_ctrl"] = cost_info["cost_ctrl"]

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

                - observation (:obj:`numpy.ndarray`): Array containing the current
                  observation.
                - info (:obj:`dict`): Dictionary containing additional information.
        """
        observation, info = super().reset(seed=seed, options=options)

        self.state = observation
        self.t = 0.0

        return observation, info

    @property
    def tau(self):
        """Alias for the environment step size. Done for compatibility with the
        other gymnasium environments.
        """
        return self.dt


if __name__ == "__main__":
    print("Setting up 'HalfCheetahCost' environment.")
    env = gym.make("HalfCheetahCost", render_mode="human")

    # Run episodes.
    episode = 0
    path, paths = [], []
    s, _ = env.reset()
    path.append(s)
    print(f"\nPerforming '{EPISODES}' in the 'HalfCheetahCost' environment...\n")
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
    print("\nFinished 'HalfCheetahCost' environment simulation.")

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
        ax.set_title(f"HalfCheetahCost episode '{i}'")
        ax.legend()
        print("Close plot to see next episode...")
        plt.show()

    print("\nDone")
