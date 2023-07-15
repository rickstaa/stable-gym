"""Modified version of the FetchReach Mujoco environment in v1.2.2 of the
:gymnasium-robotics:`Gymnasium Robotics library <envs/fetch/>`. This modification was
first described by `Han et al. 2020`_. In this modified version:

-   The reward was replaced with a cost. This was done by taking the absolute value of
    the reward.

.. _`Han et al. 2020`: https://arxiv.org/abs/2004.14288
"""  # noqa: E501
from stable_gym.envs.robotics.fetch.fetch_reach_cost.fetch_reach_cost import (
    FetchReachCost,
)
