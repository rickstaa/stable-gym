"""Modified version of the Humanoid Mujoco environment found in the
:gymnasium:`gymnasium library <environments/mujoco/humanoid>`. This modification was
first described by `Han et al. 2020`. In this modified version:

-   The objective was changed to a velocity-tracking task. To do this, the reward is replaced with a cost.
    This cost is the squared difference between the Humanoid's forward velocity and a reference value (error).
-   Three **optional** variables were added to the observation space; The reference velocity, the reference error
    (i.e. the difference between the humanoid's forward velocity and the reference) and the humanoid's forward velocity.
    These variables can be enabled using the ``exclude_reference_from_observation``,
    ``exclude_reference_error_from_observation`` and ``exclude_velocity_from_observation`` environment arguments.

.. _`Han et al. 2020`: https://arxiv.org/abs/2004.14288
"""  # noqa: E501
from stable_gym.envs.mujoco.humanoid_cost.humanoid_cost import HumanoidCost
