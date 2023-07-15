"""Modified version of the cart-pole environment in v0.28.1 of the `gymnasium library`_.
This modification was first described by `Han et al. 2020`_. In this modified version:

-   The action space is continuous, wherein the original version it is discrete.
-   The reward is replaced with a cost. This cost is defined as the difference between a
    state variable and a reference value (error).
-   A new ``reference_tracking`` task was added. This task can be enabled using the
    ``task_type`` environment argument. When this type is chosen, two extra observations
    are returned.
-   Some of the environment parameters were changed slightly.
-   The info dictionary returns extra information about the reference tracking task.

.. _`gymnasium library`: https://gymnasium.farama.org/environments/classic_control/cart_pole
.. _`Han et al. 2020`: https://arxiv.org/abs/2004.14288
"""
from stable_gym.envs.classic_control.cartpole_cost.cartpole_cost import CartPoleCost
