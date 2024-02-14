"""Modified version of the `QuadXHover environment`_ found in the :PyFlyt:`PyFlyt package <>`.
This environment was first described by `Tai et al. 2023`_. In this modified version:

-   The reward has been changed to a cost. This was done by negating the reward always
    to be positive definite.
-   A health penalty has been added. This penalty is applied when the quadrotor moves
    outside the flight dome or crashes. The penalty equals the maximum episode steps
    minus the steps taken or a user-defined penalty.
-   The ``max_duration_seconds`` has been removed. Instead, the ``max_episode_steps``
    parameter of the :class:`gym.wrappers.TimeLimit` wrapper is used to limit
    the episode duration.

The rest of the environment is the same as the original QuadXHover environment. For more
information about the original environment, please refer the
`original codebase <https://github.com/jjshoots/PyFlyt>`__,
:PyFlyt:`the PyFlyt documentation <>` or the accompanying` article of Tai et al. 2023`_ for more information.

.. _`QuadXHover environment`: https://jjshoots.github.io/PyFlyt/documentation/gym_envs/quadx_envs/quadx_hover_env.html
.. _`Tai et al. 2023`: https://arxiv.org/abs/2304.01305
.. _`article of Tai et al. 2023`: https://arxiv.org/abs/2304.01305
"""  # noqa: E501

from stable_gym.envs.robotics.quadrotor.quadx_hover_cost.quadx_hover_cost import (
    QuadXHoverCost,
)
