"""Modified version of the Walker2d Mujoco environment in v0.28.1 of the
`gymnasium library <https://gymnasium.farama.org/environments/mujoco/walker2d>`_.
This modification was first described by `Han et al. 2020 <https://arxiv.org/abs/2004.14288>`_.
In this modified version:

-   The objective was changed to a velocity-tracking task. To do this, the reward is replaced with a cost.
    This cost is the squared difference between the Walker2d's forward velocity and a reference value (error).
"""  # noqa: E501
from stable_gym.envs.mujoco.walker2d_cost.walker2d_cost import Walker2dCost
