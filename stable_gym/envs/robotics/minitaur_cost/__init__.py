"""Modified version of the `Minitaur environment`_ in v3.2.5 of the `pybullet package`_.
This modification was first described by `Han et al. 2020`_. In this modified version:

-   The objective was changed to a velocity-tracking task. To do this, the reward is replaced with a cost.
    This cost is the squared difference between the Minitaur's forward velocity and a reference value (error).
-   A minimal backward velocity bound is added to prevent the Minitaur from
    walking backwards.
-   Users are given the option to modify the Minitaur fall criteria and thus
    the episode termination criteria.

.. _`Minitaur environment`: https://arxiv.org/abs/1804.10332
.. _`pybullet package`:  https://pybullet.org/
.. _`Han et al. 2020`: https://arxiv.org/abs/2004.14288
"""  # noqa: E501
from stable_gym.envs.robotics.minitaur_cost.minitaur_cost import (
    MinitaurCost,
)
