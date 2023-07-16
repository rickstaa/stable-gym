# CartPoleCost gymnasium environment

<div align="center">
    <img src="https://github.com/rickstaa/stable-gym/assets/17570430/eb3d4f34-1429-4597-a51f-16aea0e7def2" alt="CartPole Cost environment" width="400px">
</div>

<!--alex ignore joint-->

An unactuated joint attaches a pole to a cart, which moves along a frictionless track. This environment corresponds to the [CartPole-v1](https://gymnasium.farama.org/environments/classic_control/cart_pole/) environment included in the gymnasium package. It is different in the fact that:

*   The action space is continuous, wherein the original version it is discrete.
*   The reward is replaced with a cost. This cost is defined as the difference between a
    state variable and a reference value (error).
*   A new `reference_tracking` task was added. This task can be enabled using the
    `task_type` environment argument. When this type is chosen, two extra observations
    are returned.
*   Some of the environment parameters were changed slightly.
*   The info dictionary returns extra information about the reference tracking task.

This modification was first described in [Han et al. 2019](https://arxiv.org/abs/2004.14288).

## Observation space

## stabilisation task (original)

For the stabilisation task the environment returns the following observation:

*   $x$ - Cart Position.
*   $x_{dot}$ - Cart Velocity.
*   $w$ - Pole angle.
*   $w_{dot}$ - Pole angle velocity.

## Reference tracking task

By default, for the reference tracking task the environment returns the following observation:

*   $x$ - Cart Position.
*   $x_{dot}$ - Cart Velocity.
*   $w$ - Pole angle.
*   $w_{dot}$ - Pole angle velocity.
*   $x_{ref}$ - The cart position reference.

An extra variable will be returned if the `exclude_reference_error_from_observation` flag is set to `False`:

*   $x_{ref\_error}$ - The reference tracking error.

## Action space

*   **u1:** The x-force applied on the cart.

## Episode Termination

An episode is terminated when:

*   Pole Angle is more than 20 degrees in `stabilisation` task and 60 in `reference_tracking` task.
*   Cart Position is more than 10 m (center of the cart reaches the edge of the
    display).
*   Episode length is greater than 200.
*   The cost is greater than a set threshold (100 by default). This threshold can be changed using the `max_cost` environment argument.

## Environment goals

This environment has two task types (goals), which can be set using the `task_type` environment argument.

### stabilisation task

The stabilisation task is similar to the original `CartPole-v1` environment. The pendulum starts upright, and the goal is to prevent it from falling over by increasing and reducing the cart's control force. This must be done while the cart does not violate set position constraints. These constraints are defined in the cost function.

### Reference tracking task

Similar to the stabilisation task, now the card must also track a cart position reference signal.

## Cost function

The cost function of this environment is designed in such a way that it tries to minimize the error of a set of states and reference states. As stated above, this environment contains two types of tasks. Each task type has a different cost function:

**Stabilisation task**

$$
cost = (x / x_{threshold})^2 + 20 * (theta / theta_{threshold})^2
$$

**Reference tracking task**

$$
cost = ((x - x_{ref})/ x_{threshold})^2 + 20 * (theta / theta_{threshold})^2
$$

The exact definition of these tasks can be found in the environment's `stable_gym.envs.classical_control.cartpole_cost.cartpole_cost.CartPoleCost.cost` method. The cost is between `0` and a set threshold value in both tasks, and the maximum cost is used when the episode is terminated.

## Environment step return

In addition to the observations, the cost and a termination and truncation boolean, the environment also returns an info dictionary:

```python
[observation, cost, termination, truncation, info_dict]
```

The info dictionary contains the following keys:

*   **reference**: The set cart position reference.
*   **state\_of\_interest**: The state that should track the reference (SOI).
*   **reference\_error**: The error between SOI and the reference.
*   **reference\_constraint\_position**: A user-specified constraint they want to watch.
*   **reference\_constraint\_error**: The error between the SOI and the set reference constraint.
*   **reference\_constraint\_violated**: Whether the reference constraint was violated.

## How to use

This environment is part of the [Stable Gym package](https://github.com/rickstaa/stable-gym). It is therefore registered as the `stable_gym:CartPoleCost-v1` gymnasium environment when you import the Stable Gym package. If you want to use the environment in stand-alone mode, you can register it yourself.
