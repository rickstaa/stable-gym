"""Modified version of the Hopper Mujoco environment in v0.28.1 of the
`gymnasium library <https://gymnasium.farama.org/environments/mujoco/hopper>`_.
This modification was first described by `Han et al. 2020 <https://arxiv.org/abs/2004.14288>`_.
In this modified version:

-   The objective was changed to a velocity-tracking task. To do this, the reward is replaced with a cost.
    This cost is the squared difference between the Hopper's forward velocity and a reference value (error).
"""  # noqa: E501
from stable_gym.envs.mujoco.hopper_cost.hopper_cost import HopperCost
