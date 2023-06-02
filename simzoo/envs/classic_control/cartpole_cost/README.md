# CartPoleCost gymnasium environment

An un-actuated joint attaches a pole to a cart, which moves along a frictionless track. This environment
corresponds to the [CartPole-v1](https://gym.openai.com/envs/CartPole-v1/) environment that is included in the
openAi gymnasium package. It is different in the fact that:

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

## Stabilization task (original)

*   **x**: Cart Position.
*   **x\_dot**: Cart Velocity.
*   **w**: Pole angle.
*   **w\_dot**: Pole angle velocity.

## Reference tracking task

*   **x**: Cart Position.
*   **x\_dot**: Cart Velocity.
*   **w**: Pole angle.
*   **w\_dot**: Pole angle velocity.
*   **x\_ref**: The cart position reference.
*   **x\_ref\_error**: The reference tracking error.

## Action space

*   **u1:** The x-force applied on the cart.

## Episode Termination:

An episode is terminated when:

*   Pole Angle is more than 20 degrees.
*   Cart Position is more than 10 m (center of the cart reaches the edge of the
    display).
*   Episode length is greater than 200.
*   The cost is greater than 100.

## Environment goals

### Stabilization task

The stabilization task is similar to the one of the original `CartPole-v1` environment. The pendulum starts upright, and the goal is to prevent it from falling over by increasing and reducing the cart's control force. This must be done while the cart does not violate set position constraints. These constraints are defined in the cost function.

### Reference tracking task

Similar to the stabilization task but now the card also has to track a cart position reference signal.

## Cost function

The cost function of this environment is designed in such a way that it tries to minimize the error of a set of states and a set of reference
states. It contains two types of tasks:

*   A stabilization task. In this task, the agent attempts to stabilize a given state (e.g. keep the pole angle and or cart position zero)
*   A reference tracking task. In this task, the agent tries to make a state track a given reference.

The exact definition of these tasks can be found in the environment `cost()` method.

## Environment step return

In addition to the observations, the cost and a termination and truncation boolean the environment also returns a info dictionary:

```python
[observation, cost, termination, truncation, info_dict]
```

The info dictionary contains the following keys:

*   **reference**: The set cart position reference.
*   **state\_of\_interest**: The state that should track the reference (SOI).
*   **reference\_error**: The error between SOI and the reference.
*   **reference\_constraint\_position**: A user specified constraint they want to watch.
*   **reference\_constraint\_error**: The error between the SOI and the set reference constraint.
*   **reference\_constraint\_violated**: Whether the reference constraint was violated.

## How to use

This environment is part of the [simzoo package](https://github.com/rickstaa/simzoo). It is therefore registered as a gymnasium environment when you import the Simzoo package. If you want to use the environment in stand-alone mode, you can register it yourself.
