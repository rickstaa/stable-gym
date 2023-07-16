"""Modified version of the Ant Mujoco environment in v0.28.1 of the
:gymnasium:`gymnasium library <environments/mujoco/ant>`. This modification was first
described by `Han et al. 2020`_. In this modified version:

-   The objective was changed to a velocity-tracking task. To do this, the reward is replaced with a cost.
    This cost is the squared difference between the Ant's forward velocity and a reference value (error).
-   The reference velocity was added to the observation space.
-   Two **optional** variables were added to the observation space. These are
    the ant's forward velocity and the error (difference between the ant's
    forward velocity and the reference velocity). These variables can be enabled using 
    the ``exclude_velocity_from_observation`` and ``exclude_reference_error_from_observation``
    environment arguments.

.. _`Han et al. 2020`: https://arxiv.org/abs/2004.14288
"""  # noqa: E501
from stable_gym.envs.mujoco.ant_cost.ant_cost import AntCost
