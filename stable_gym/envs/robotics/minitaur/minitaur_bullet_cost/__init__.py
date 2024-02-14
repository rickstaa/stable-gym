"""This package contains a modified version of the `MinitaurBullet environment`_ found in
the :pybullet:`pybullet package <>`. This environment first appeared in a paper by `Tan et al. 2018`_.
The version found here is based on the modification given by `Han et al. 2020`_. In this modified version:

-   The objective was changed to a velocity-tracking task. To do this, the reward is replaced with a cost.
    This cost is the squared difference between the Minitaur's forward velocity and a reference value (error).
-   A minimal backward velocity bound is added to prevent the Minitaur from
    walking backwards.
-   Users are given the option to modify the Minitaur fall criteria and thus
    the episode termination criteria.

.. _`MinitaurBullet environment`: https://github.com/bulletphysics/bullet3/tree/master/examples/pybullet/gym/pybullet_envs/bullet/minitaur_gym_env.py
.. _`Tan et al. 2018`: https://arxiv.org/abs/1804.10332
.. _`Han et al. 2020`: https://arxiv.org/abs/2004.14288
"""  # noqa: E501

from stable_gym.envs.robotics.minitaur.minitaur_bullet_cost.minitaur_bullet_cost import (
    MinitaurBulletCost,
)
