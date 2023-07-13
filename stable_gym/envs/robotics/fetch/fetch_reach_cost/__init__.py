"""Modified version of the FetchReach Mujoco environment in v1.2.2 of the
`Gymnasium Robotics library <https://robotics.farama.org/envs/fetch/>`_.
This modification was first described by `Han et al. 2020 <https://arxiv.org/abs/2004.14288>`_.
In this modified version:

-   The reward was replaced with a cost. This was done by taking the absolute value of
    the reward.
"""  # noqa: E501
from stable_gym.envs.robotics.fetch.fetch_reach_cost.fetch_reach_cost import (
    FetchReachCost,
)
