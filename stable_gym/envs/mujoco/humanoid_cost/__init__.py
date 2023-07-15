"""Modified version of the Humanoid Mujoco environment in v0.28.1 of the
:gymnasium:`gymnasium library <environments/mujoco/humanoid>`. This modification was
first described by `Han et al. 2020`. In this
modified version:

-   The objective was changed to a velocity-tracking task. To do this, the reward is replaced with a cost.
    This cost is the squared difference between the Humanoid's forward velocity and a reference value (error).

.. _`Han et al. 2020`: https://arxiv.org/abs/2004.14288
"""  # noqa: E501
from stable_gym.envs.mujoco.humanoid_cost.humanoid_cost import HumanoidCost
