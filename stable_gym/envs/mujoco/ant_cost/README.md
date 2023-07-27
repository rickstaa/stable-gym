# AntCost gymnasium environment

<div align="center">
    <img src="https://github.com/rickstaa/stable-gym/assets/17570430/c9f6d7f9-586e-4236-91d3-fa2d0ce4aadc" alt="Ant Cost environment" width="200px">
</div>
</br>

An actuated 8-jointed ant. This environment corresponds to the [Ant-v4](https://gymnasium.farama.org/environments/mujoco/ant) environment included in the [gymnasium package](https://gymnasium.farama.org/). It is different in the fact that:

*   The objective was changed to a velocity-tracking task. To do this, the reward is replaced with a cost. This cost is the squared difference between the Ant's forward velocity and a reference value (error).
*   Three **optional** variables were added to the observation space; The reference velocity, the reference error (i.e. the difference between the ant's forward velocity and the reference) and the ant's forward velocity. These variables can be enabled using the `exclude_reference_from_observation`, `exclude_reference_error_from_observation` and `exclude_velocity_from_observation` environment arguments.

The rest of the environment is the same as the original Ant environment. Below, the modified cost is described. For more information about the environment (e.g. observation space, action space, episode termination, etc.), please refer to the [gymnasium library](https://gymnasium.farama.org/environments/mujoco/ant/).

## Observation space

The original observation space of the [Ant-v4](https://gymnasium.farama.org/environments/mujoco/ant) environment contains all eight motors' angles, velocities and torques. In this modified version, the observation space has been extended to add three additional observations:

*   $r$: The velocity reference signal that needs to be tracked.
*   $r_{error}$: The difference between the current and reference velocities.
*   $v_{x}$: The Ant's forward velocity.

These observations **optional** and can be excluded from the observation space by setting the `exclude_reference_from_observation`, `exclude_reference_error_from_observation` and `exclude_x_velocity_from_observation` environment arguments to `True`.

## Cost function

The cost function of this environment is designed in such a way that it tries to minimize the error between the Ant's forward velocity and a reference value. The cost function is defined as:

$$
cost = w_{forward\_velocity} \times (x_{velocity} - x_{reference\_x\_velocity})^2 + w_{ctrl} \times c_{ctrl} + p_{health}
$$

Where:

*   $w_{forward\_velocity}$ - is the weight of the forward velocity error.
*   $x_{velocity}$ - is the Hopper's forward velocity.
*   $x_{reference\_x\_velocity}$ is the reference forward velocity.
*   $w_{ctrl}$ is the weight of the control cost (**optional**).
*   $c_{ctrl}$ is the control cost (**optional**).
*   $p_{health}$ is a penalty for being unhealthy (i.e. if the ant falls over).

The control and health penalty are optional and can be disabled using the `include_ctrl_cost` and `include_health_penalty` environment arguments.

## Environment step return

In addition to the observations, the cost and a termination and truncation boolean, the environment also returns an info dictionary:

```python
[observation, cost, termination, truncation, info_dict]
```

Compared to the original [Ant-v4](https://gymnasium.farama.org/environments/mujoco/ant) environment, the following keys were added to this info dictionary:

*   **reference**: The reference velocity.
*   **state\_of\_interest**: The state that should track the reference (SOI).
*   **reference\_error**: The error between SOI and the reference.

## How to use

This environment is part of the [Stable Gym package](https://github.com/rickstaa/stable-gym). It is therefore registered as the `stable_gym:AntCost-v1` gymnasium environment when you import the Stable Gym package. If you want to use the environment in stand-alone mode, you can register it yourself.
