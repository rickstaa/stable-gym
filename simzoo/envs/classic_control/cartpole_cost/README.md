# CartPoleCost gymnasium environment

An un-actuated joint attaches a pole to a cart, which moves along a frictionless track. This environment
corresponds to the [CartPole-v1](https://gym.openai.com/envs/CartPole-v1/) environment that is included in the
openAi gymnasium package. It is different in the fact that:

*   The action space is continuous, wherein the original version it is discrete.
*   The reward is replaced with a cost. This cost is defined as the difference between a
    state variable and a reference value (error).
*   Some of the environment parameters were changed slightly.

This modification was first described in [Han et al. 2019](https://arxiv.org/abs/2004.14288).

## Observation space

*   **x**: Cart Position.
*   **x\_dot**: Cart Velocity.
*   **w**: Pole angle.
*   **w\_dot**: Pole angle velocity.

## Action space

*   **u1:** The x-force applied on the cart.

## Episode Termination:

An episode is terminated when:

*   Pole Angle is more than 20 degrees.
*   Cart Position is more than 10 m (center of the cart reaches the edge of the
    display).
*   Episode length is greater than 200.
*   The cost is greater than 100.

## Environment goal

The pendulum starts upright, and the goal is to prevent it from falling over by increasing and reducing the cart's
velocity. This needs to be done while the cart does not violate set position constraints. These constraints are defined
in the cost function.

## Cost function

The cost function of this environment is designed in such a way that it tries to minimize the error of a set of states and a set of reference
states. It contains two types of tasks:

*   A reference tracking task. In this task, the agent tries to make a state track a given reference.
*   A stabilization task. In this task, the agent attempts to stabilize a given state (e.g. keep the pole angle and or cart position zero)

The exact definition of these tasks can be found in the environment `cost()` method.

## Environment step return

In addition to the observations, the cost and a termination and truncation boolean the environment also returns a info dictionary:

```python
[(hat_x_1, hat_x_2, x_1, x_2), cost, termination, truncation, info_dict]
```

The info dictionary contains the following keys:

*   **cons\_pos**: The current x-position constraint.
*   **cons\_theta**: The current pole angle constraint.
*   **target**: The target position. Only present when performing a reference tracking task.
*   **violation\_of\_x\_threshold**: Whether the environment x-threshold was violated.
*   **violation\_of\_constraint**: Whether a certain x-constraint was voilated.
*   **reference**: The current reference (position and angles). Only present when performing a reference tracking task.
*   **state\_of\_interest**: The current state\_of\_interest which we try to minimize.

## How to use

This environment is part of the [simzoo package](https://github.com/rickstaa/simzoo). It is therefore registered as a gymnasium environment when you import the Simzoo package. If you want to use the environment in stand-alone mode, you can register it yourself.
