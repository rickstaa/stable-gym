# SwimmerCost gymnasium environment

<div align="center">
    <img src="https://github.com/rickstaa/stable-gym/assets/17570430/dccd73b4-c97e-46ce-ba0d-4a1328c0aefe" alt="Swimmer environment" width="200px">
</div>
</br>

An actuated 2-jointed swimmer. This environment corresponds to the [Swimmer-v4](https://gymnasium.farama.org/environments/mujoco/swimmer) environment included in the [gymnasium package](https://gymnasium.farama.org/). It is different in the fact that:

*   The objective was changed to a velocity-tracking task. To do this, the reward is replaced with a cost. This cost is the squared difference between the Swimmer's forward velocity and a reference value (error).
*   Three **optional** variables were added to the observation space; The reference velocity, the reference error (i.e. the difference between the swimmer's forward velocity and the reference) and the swimmer's forward velocity. These variables can be enabled using the `exclude_reference_from_observation`, `exclude_reference_error_from_observation` and `exclude_velocity_from_observation` environment arguments.

The rest of the environment is the same as the original Swimmer environment. Below, the modified cost is described. For more information about the environment (e.g. observation space, action space, episode termination, etc.), please refer to the [gymnasium library](https://gymnasium.farama.org/environments/mujoco/swimmer/).

## Cost function

The cost function of this environment is designed in such a way that it tries to minimize the error between the Swimmer's forward velocity and a reference value. The cost function is defined as:

$$
cost = w_{forward\_velocity} \times (x_{velocity} - x_{reference\_x\_velocity})^2 + w_{ctrl} \times c_{ctrl}
$$

Where:

*   $w_{forward\_velocity}$ - is the weight of the forward velocity error.
*   $x_{velocity}$ - is the Swimmer's forward velocity.
*   $x_{reference\_x\_velocity}$ is the reference forward velocity.
*   $w_{ctrl}$ is the weight of the control cost (**optional**).
*   $c_{ctrl}$ is the control cost (**optional**).

The control cost is optional and can be disabled using the `include_ctrl_cost` environment arguments.

## How to use

This environment is part of the [Stable Gym package](https://github.com/rickstaa/stable-gym). It is therefore registered as the `stable_gym:SwimmerCost-v1` gymnasium environment when you import the Stable Gym package. If you want to use the environment in stand-alone mode, you can register it yourself.
