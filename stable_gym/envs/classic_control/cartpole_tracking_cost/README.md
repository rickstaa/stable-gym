# CartPoleTrackingCost gymnasium environment

<div align="center">
    <img src="https://github.com/rickstaa/stable-gym/assets/17570430/eb3d4f34-1429-4597-a51f-16aea0e7def2" alt="CartPole Cost environment" width="400px">
</div>

<!--alex ignore joint-->

An unactuated joint attaches a pole to a cart, which moves along a frictionless track. This environment is a modified version of the [CartPole-v1](https://gymnasium.farama.org/environments/classic_control/cart_pole/) found in the Gymnasium package, with several key alterations:

* The action space is **continuous**, contrasting with the original **discrete** setting.
* The **reward** function is replaced with a (positive definite) **cost** function (negated reward), in line with Lyapunov stability theory. This cost is the difference between a state variable and a reference value (error).
* Maximum cart force is increased from `10` to `20`.
* Episode length is reduced from `500` to `250`.
* A termination cost of `c=100` is introduced for early episode termination, to promote cost minimization.
* The terminal angle limit is expanded from the original `12` degrees to `20` degrees, enhancing recovery potential.
* The terminal position limit is extended from `2.4` meters to `10` meters, broadening the recovery range.
* Velocity limits are adjusted from ±∞ to ±50, accelerating training.
* Angular velocity termination threshold is lowered from ±∞ to ±50, likely for improved training efficiency.
* Random initial state range is modified from `[-0.05, 0.05]` to `[-5, 5]` for the cart position and `[-0.2, 0.2]` for all other states, allowing for expanded exploration.

Additional modifications in our implementation:

* Unlike the original environment's fixed cost threshold of `100`, this version allows users to adjust the maximum cost threshold improving training adaptability.
* An extra termination criterion for cumulative costs over `100` is added to hasten training.
* The gravity constant is adjusted back from `10` to the real-world value of `9.8`, aligning it closer with the original CartPole environment.
* The stabilization objective is replaced with a **reference tracking task** for enhanced control.
* Two additional observations are introduced, facilitating **reference tracking**.
* The info dictionary now provides **extra information** about the reference to be tracked.

These modifications were first described in [Han et al. 2019](https://arxiv.org/abs/2004.14288) and further adapted in our version for enhanced training and exploration.

## Observation space

By default, the environment returns the following observation:

* $x$ - Cart Position.
* $x_{dot}$ - Cart Velocity.
* $w$ - Pole angle.
* $w_{dot}$ - Pole angle velocity.
* $x_{ref}$ - The cart position reference.
* $x_{ref\_error}$ - The reference tracking error.

The last two variables can be excluded from the observation space by setting the `exclude_reference_from_observation` and `exclude_reference_error_from_observation` environment arguments to `True`. Please note that the environment needs the reference or the reference error to be included in the observation space when the reference signal is not constant to function correctly. If both are excluded, the environment will raise an error.

## Action space

* **u1:** The x-force applied on the cart.

## Episode Termination

An episode is terminated when:

* Pole Angle is more than 60 degrees.
* Cart Position is more than 10 m (center of the cart reaches the edge of the
  display).
* Episode length is greater than 200.
* The cost is greater than a set threshold (100 by default). This threshold can be changed using the `max_cost` environment argument.

## Environment goals

The goal is similar to the original `CartPole-v1` environment. The pendulum starts upright, and the goal is to prevent it from falling over by increasing and reducing the cart's control force. This version adds a secondary goal, which is to track a cart position reference signal.

## Cost function

The cost function of this environment is designed in such a way that it tries to minimize the error between the current cart position and a set reference position. Additionally, it also includes a penalty for the pole angle:

$$
cost = (x - x_{ref})^2 + (\theta / \theta_{threshold})^2
$$

The cost is between `0` and a set threshold value in both tasks, and the maximum cost is used when the episode is terminated.

## Environment step return

In addition to the observations, the cost and a termination and truncation boolean, the environment also returns an info dictionary:

```python
[observation, cost, termination, truncation, info_dict]
```

The info dictionary contains the following keys:

* **reference**: The set cart position reference.
* **state\_of\_interest**: The state that should track the reference (SOI).
* **reference\_error**: The error between SOI and the reference.

## How to use

This environment is part of the [Stable Gym package](https://github.com/rickstaa/stable-gym). It is therefore registered as the `stable_gym:CartPoleTrackingCost-v1` gymnasium environment when you import the Stable Gym package. If you want to use the environment in stand-alone mode, you can register it yourself.
