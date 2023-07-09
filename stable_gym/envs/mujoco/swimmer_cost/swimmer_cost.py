"""The SwimmerCost gymnasium environment."""

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from gymnasium.envs.mujoco.swimmer_v4 import SwimmerEnv

import stable_gym  # NOTE: Required to register environments. # noqa: F401

RANDOM_STEP = True  # Use random action in __main__. Zero action otherwise.


# TODO: Add solving criteria after training.
class SwimmerCost(SwimmerEnv):
    """Custom Swimmer gymnasium environment.

    .. note::
        Can also be used in a vectorized manner. See the
        :gymnasium:`gym.vector <api/vector>` documentation.

    Source:
        This is a modified version of the swimmer Mujoco environment in v0.28.1 of the
        :gymnasium:`gymnasium library <environments/mujoco/swimmer>`. This modification was first
        described by `Han et al. 2020 <https://arxiv.org/abs/2004.14288>`_. Compared to the
        original Swimmer environment in this modified version:

        -   The objective was changed to a speed-tracking task. To do this, the reward
            is replaced with a cost. This cost is the squared difference between the
            swimmer's forward speed and a reference value (error).

        The rest of the environment is the same as the original Swimmer environment. Below,
        the modified cost is described. For more information about the environment
        (e.g. observation space, action space, episode termination, etc.), please refer
        to the :gymnasium:`gymnasium library <environments/mujoco/swimmer>`.

    Modified cost:
        .. math::

            cost = w_{forward} \times (x_{speed} - x_{reference\_speed})^2 + w_{ctrl} \times c_{ctrl}

    Solved Requirements:
        Considered solved when the average cost is less than or equal to 50 over
        100 consecutive trials.

    How to use:
        .. code-block:: python

            import stable_gym
            import gymnasium as gym
            env = gym.make("SwimmerCost-v1")

        On reset, the ``options`` parameter allows the user to change the bounds used to
        determine the new random state when ``random=True``.

    Attributes:
        reference_speed (float): The forward speed that the agent should try to track.
        include_ctrl_cost (bool): Whether you also want to penalize the swimmer if it
            takes actions that are too large.
        forward_speed_weight (float): The weight used to scale the forward speed error.
    """  # noqa: E501, W605

    def __init__(
        self,
        reference_forward_speed=1.0,
        include_ctrl_cost=True,
        forward_speed_weight=1.0,
        ctrl_cost_weight=None,
        **kwargs,
    ):
        """Constructs all the necessary attributes for the SwimmerCost instance.

        Args:
            reference_forward_speed (float, optional): The forward speed that the agent
                should try to track. Defaults to ``1.0``.
            include_ctrl_cost (bool, optional): Whether you also want to penalize the
                swimmer if it takes actions that are too large. Defaults to ``True``.
            forward_speed_weight (float, optional): The weight used to scale the forward
                speed error. Defaults to ``1.0``.
            ctrl_cost_weight (_type_, optional): The weight used to scale the control
                cost. Defaults to ``None`` meaning that the default value of the
                :attr:`~gymnasium.envs.mujoco.swimmer_v4.SwimmerEnv.ctrl_cost_weight`
                attribute is used.
        """
        super().__init__(**kwargs)
        self.reference_speed = reference_forward_speed
        self.include_ctrl_cost = include_ctrl_cost
        self._ctrl_cost_weight = (
            ctrl_cost_weight if ctrl_cost_weight else self._ctrl_cost_weight
        )
        self.forward_speed_weight = forward_speed_weight
        self.state = None

    def step(self, action):
        """Take step into the environment.

        .. note::
            This method overrides the
            :meth:`~gymnasium.envs.mujoco.swimmer_v4.SwimmerEnv.step` method such that
            the new cost function is used.

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
        cost, cost_info = self.cost(info["x_velocity"], -info["reward_ctrl"])

        # Update info.
        info["reward_fwd"] = cost_info["reward_fwd"]
        info["forward_reward"] = cost_info["reward_fwd"]

        return obs, cost, terminated, truncated, info

    def cost(self, x_velocity, ctrl_cost):
        """Compute the cost of the action.

        Args:
            x_velocity (float): The swimmer's x velocity.
            ctrl_cost (float): The control cost.

        Returns:
            (tuple): tuple containing:

                - cost (float): The cost of the action.
                - info (:obj:`dict`): Additional information about the cost.
        """
        reward_fwd = self.forward_speed_weight * np.square(
            x_velocity - self.reference_speed
        )
        cost = reward_fwd
        if self.include_ctrl_cost:
            cost += ctrl_cost
        return cost, {"reward_fwd": reward_fwd, "reward_ctrl": ctrl_cost}

    @property
    def ctrl_cost_weight(self):
        """Property that returns the control cost weight."""
        return self._ctrl_cost_weight

    @ctrl_cost_weight.setter
    def ctrl_cost_weight(self, value):
        """Setter for the control cost weight."""
        self._ctrl_cost_weight = value


if __name__ == "__main__":
    print("Setting up SwimmerCost environment.")
    env = gym.make("SwimmerCost", render_mode="human")

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
    print(f"Taking {T} steps in the SwimmerCost environment.")
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
    print("Finished SwimmerCost environment simulation.")

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
