"""The HopperCost gymnasium environment."""

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from gymnasium import utils
from gymnasium.envs.mujoco.hopper_v4 import HopperEnv

import stable_gym  # NOTE: Required to register environments. # noqa: F401

RANDOM_STEP = True  # Use random action in __main__. Zero action otherwise.


# TODO: Update solving criteria after training.
class HopperCost(HopperEnv, utils.EzPickle):
    """Custom Hopper gymnasium environment.

    .. note::
        Can also be used in a vectorized manner. See the
        :gymnasium:`gym.vector <api/vector>` documentation.

    Source:
        This is a modified version of the Hopper Mujoco environment in v0.28.1 of the
        :gymnasium:`gymnasium library <environments/mujoco/hopper>`. This modification
        was first described by `Han et al. 2020 <https://arxiv.org/abs/2004.14288>`_.
        Compared to the original Hopper environment in this modified version:

        -   The objective was changed to a velocity-tracking task. To do this, the reward
            is replaced with a cost. This cost is the squared difference between the
            Hopper's forward velocity and a reference value (error). Additionally, also
            a control cost and health penalty can be included in the cost.

        The rest of the environment is the same as the original Hopper environment.
        Below, the modified cost is described. For more information about the environment
        (e.g. observation space, action space, episode termination, etc.), please refer
        to the :gymnasium:`gymnasium library <environments/mujoco/hopper>`.

    Modified cost:
        .. math::

            cost = w_{forward} \\times (x_{velocity} - x_{reference\_x\_velocity})^2 + w_{ctrl} \\times c_{ctrl} + p_{health}

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
        include_health_penalty=True,
        health_penalty_size=10,
        ctrl_cost_weight=1e-3,
        terminate_when_unhealthy=True,
        healthy_state_range=(-100.0, 100.0),
        healthy_z_range=(0.7, float("inf")),
        healthy_angle_range=(-0.2, 0.2),
        reset_noise_scale=5e-3,
        exclude_current_positions_from_observation=True,
        **kwargs,
    ):
        """Constructs all the necessary attributes for the HopperCost instance.

        Args:
            reference_forward_velocity (float, optional): The forward velocity that the
                agent should try to track. Defaults to ``1.0``.
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
        """
        self.reference_forward_velocity = reference_forward_velocity
        self._forward_velocity_weight = forward_velocity_weight
        self._include_ctrl_cost = include_ctrl_cost
        self._include_health_penalty = include_health_penalty
        self._health_penalty_size = health_penalty_size

        self.state = None
        self.t = 0.0

        # Initialize the HopperEnv class.
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

        # Reinitialize the EzPickle class.
        # NOTE: Done to ensure the args of the HopperCost class are also pickled.
        # NOTE: Ensure that all args are passed to the EzPickle class!
        utils.EzPickle.__init__(
            self,
            reference_forward_velocity,
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
            **kwargs,
        )

    def cost(self, x_velocity, ctrl_cost):
        """Compute the cost of a given x velocity and control cost.

        Args:
            x_velocity (float): The Hopper's x velocity.
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

        ctrl_cost = super().control_cost(action)
        cost, cost_info = self.cost(info["x_velocity"], ctrl_cost)

        # Update info.
        info["cost_velocity"] = cost_info["cost_velocity"]
        info["cost_ctrl"] = cost_info["cost_ctrl"]
        info["penalty_health"] = cost_info["penalty_health"]

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
    print("Setting up HopperCost environment.")
    env = gym.make("HopperCost", render_mode="human")

    # Take T steps in the environment.
    T = 1000
    path = []
    t1 = []
    s = env.reset(
        options={
            "low": [-2, -0.2, -0.2, -0.2],
            "high": [2, 0.2, 0.2, 0.2],
        }
    )
    print(f"Taking {T} steps in the HopperCost environment.")
    for i in range(int(T / env.dt)):
        action = (
            env.action_space.sample()
            if RANDOM_STEP
            else np.zeros(env.action_space.shape)
        )
        s, r, terminated, truncated, info = env.step(action)
        if terminated:
            env.reset()
        path.append(s)
        t1.append(i * env.dt)
    print("Finished HopperCost environment simulation.")

    # Plot results.
    print("Plot results.")
    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111)
    ax.plot(t1, np.array(path)[:, 0], color="orange", label="x")
    ax.plot(t1, np.array(path)[:, 1], color="magenta", label="x_dot")
    ax.plot(t1, np.array(path)[:, 2], color="sienna", label="theta")
    ax.plot(t1, np.array(path)[:, 3], color="blue", label=" theat_dot1")

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc=2, fancybox=False, shadow=False)
    plt.ioff()
    plt.show()

    print("done")
