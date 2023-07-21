"""Stable Gym gymnasium environments that are based on robotics environments found in
the :gymnasium-robotics:`Gymnasium Robotics <>` and `Pybullet`_ packages.

.. note::

    Some of these environments are based on the :class:`gym.GoalEnv` class. This means
    that the ``step`` method returns a dictionary with the following keys:

    -   ``observation``: The observation of the environment.
    -   ``achieved_goal``: The goal that was achieved during execution.
    -   ``desired_goal``: The desired goal that we asked the agent to attempt to achieve.

    If you want to use these environments with RL algorithms that expect the ``step``
    method to return a :obj:`np.ndarray` instead of a dictionary, you can use the
    :class:`gym.wrappers.FlattenObservation` wrapper to flatten the dictionary into a
    single :obj:`np.ndarray`.

.. _`Pybullet`: https://pybullet.org/
"""
from stable_gym.envs.robotics.fetch.fetch_reach_cost.fetch_reach_cost import (
    FetchReachCost,
)
from stable_gym.envs.robotics.minitaur.minitaur_bullet_cost.minitaur_bullet_cost import (
    MinitaurBulletCost,
)
