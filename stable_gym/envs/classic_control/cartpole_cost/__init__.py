"""Modified version of the cart-pole environment found in the `gymnasium library`_.
This modification was first described by `Han et al. 2020`_. In this modified version:

-   The action space is continuous, wherein the original version it is discrete.
-   The reward is replaced with a cost (i.e. negated reward).
-   Some of the environment parameters were changed slightly.

.. _`gymnasium library`: https://gymnasium.farama.org/environments/classic_control/cart_pole
.. _`Han et al. 2020`: https://arxiv.org/abs/2004.14288
"""

from stable_gym.envs.classic_control.cartpole_cost.cartpole_cost import CartPoleCost
