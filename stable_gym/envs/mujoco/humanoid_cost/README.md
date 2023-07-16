# HumanoidCost gymnasium environment

<div align="center">
    <img src="https://github.com/rickstaa/stable-gym/assets/17570430/7b3dd9c8-065c-40ab-bedd-d6c56633b4f7" alt="Humanoid Cost environment" width="200px">
</div>
</br>

An actuated 17-jointed humanoid. This environment corresponds to the [Humanoid-v4](https://gymnasium.farama.org/environments/mujoco/humanoid) environment included in the [gymnasium package](https://gymnasium.farama.org/). It is different in the fact that:

*   The objective was changed to a velocity-tracking task. To do this, the reward is replaced with a cost. This cost is the squared
    difference between the Humanoid's forward velocity and a reference value (error).
*   Three **optional** variables were added to the observation space; The reference velocity, the reference error (i.e. the difference between the humanoid's forward velocity and the reference) and the humanoid's forward velocity. These variables can be enabled using the `exclude_reference_from_observation`, `exclude_reference_error_from_observation` and `exclude_velocity_from_observation` environment arguments.

The rest of the environment is the same as the original Humanoid environment. Below, the modified cost is described. For more information about the environment (e.g. observation space, action space, episode termination, etc.), please refer to the [gymnasium library](https://gymnasium.farama.org/environments/mujoco/humanoid/).

## Cost function

The cost function of this environment is designed in such a way that it tries to minimize the error between the Humanoid's forward velocity and a reference value. The cost function is defined as:

$$
cost = w_{forward\_velocity} \times (x_{velocity} - x_{reference\_x\_velocity})^2 + w_{ctrl} \times c_{ctrl} + p_{health}
$$

Where:

*   $w_{forward\_velocity}$ - is the weight of the forward velocity error.
*   $x_{velocity}$ - is the 2D humanoid's forward velocity.
*   $x_{reference\_x\_velocity}$ is the reference forward velocity.
*   $w_{ctrl}$ is the weight of the control cost (**optional**).
*   $c_{ctrl}$ is the control cost (**optional**).
*   $p_{health}$ is a penalty for being unhealthy (i.e. if the 2D humanoid falls over).

The control and health penalty are optional and can be disabled using the `include_ctrl_cost` and `include_health_penalty` environment arguments.

## How to use

This environment is part of the [Stable Gym package](https://github.com/rickstaa/stable-gym). It is therefore registered as the `stable_gym:HumanoidCost-v1` gymnasium environment when you import the Stable Gym package. If you want to use the environment in stand-alone mode, you can register it yourself.
