"""Modified version of the Walker2d Mujoco environment found in the
:gymnasium:`gymnasium library <environments/mujoco/walker2d>`. This modification was
first described by `Han et al. 2020`_. In this modified version:

-   The objective was changed to a velocity-tracking task. To do this, the reward is replaced with a cost.
    This cost is the squared difference between the Walker2d's forward velocity and a reference value (error).
-   The reference velocity was added to the observation space.
-   Three **optional** variables were added to the observation space; The reference velocity, the reference error
    (i.e. the difference between the walker2d's forward velocity and the reference) and the walker2d's forward velocity.
    These variables can be enabled using the ``exclude_reference_from_observation``,
    ``exclude_reference_error_from_observation`` and ``exclude_velocity_from_observation`` environment arguments.

.. _`Han et al. 2020`: https://arxiv.org/abs/2004.14288
"""  # noqa: E501

from stable_gym.envs.mujoco.walker2d_cost.walker2d_cost import Walker2dCost
