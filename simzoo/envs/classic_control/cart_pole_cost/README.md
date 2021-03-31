# Continuous action space CartPole gym environment

An un-actuated joint attaches a pole to a cart, which moves along a frictionless track. This environment
corresponds to the [CartPole-v1](https://gym.openai.com/envs/CartPole-v1/) environment that is included in the
openAi gym package. It is different in the fact that:

-   In this version, the action space is continuous, wherein the OpenAi version
    it is discrete.
-   The reward is replaced with a cost. This cost is defined as the difference between a state variable and a reference value (error).

## Observation space

-   **x**: Cart Position.
-   **x_dot**: Cart Velocity.
-   **w**: Pole angle.
-   **w_dot**: Pole angle velocity.

## Action space

-   **u1:** The x-force applied on the cart.

## Environment goal

The pendulum starts upright, and the goal is to prevent it from falling over by increasing and reducing the cart's
velocity. This needs to be done while the cart does not violate set position constraints. These constraints are defined
in the cost function.

## Cost function

The cost function of this environment is designed in such a way that it tries to minimize the error of a set of states and a set of reference
states. It contains two types of tasks:

-   A reference tracking task. In this task, the agent tries to make a state track a given reference.
-   A stabilization task. In this task, the agent attempts to stabilize a given state (e.g. keep the pole angle and or cart position zero)

The exact definition of these tasks can be found in the environment `cost()` method.

## Environment step return

In addition to the observations, the environment also returns a info dictionary:

```python
[hat_x_1, hat_x_2, x_1, x_2, info_dict]
```

This info dictionary contains the following keys:

-   **cons_pos**: The current x-position constraint.
-   **cons_theta**: The current pole angle constraint.
-   **target**: The target position. Only present when performing a reference tracking task.
-   **violation_of_x_threshold**: Whether the environment x-threshold was violated.
-   **reference**: The current reference (position and angles). Only present when performing a reference tracking task.
-   **state_of_interest**: The current state_of_interest which we try to minimize.

## How to use

This environment is part of the [simzoo package](https://github.com/rickstaa/simzoo). It is therefore registered as a gym environment when you import the Simzoo package. If you want to use the environment in stand-alone mode, you can register it yourself.
