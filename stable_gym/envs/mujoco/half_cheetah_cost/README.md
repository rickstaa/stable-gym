# HalfCheetahCost gymnasium environment

<div align="center">
    <img src="https://github.com/rickstaa/stable-gym/assets/17570430/44360980-3ad1-40e9-863e-3417ed3aa4c8" alt="Half Cheetah Cost environment" width="200px">
</div>
</br>

An actuated 8-jointed half cheetah. This environment corresponds to the [HalfCheetah-v4](https://gymnasium.farama.org/environments/mujoco/half_cheetah) environment included in the [gymnasium package](https://gymnasium.farama.org/). It is different in the fact that:

*   The objective was changed to a velocity-tracking task. To do this, the reward is replaced with a cost. This cost is the squared
    difference between the HalfCheetah's forward velocity and a reference value (error).

The rest of the environment is the same as the original HalfCheetah environment. Below, the modified cost is described. For more information about the environment (e.g. observation space, action space, episode termination, etc.), please refer to the [gymnasium library](https://gymnasium.farama.org/environments/mujoco/half_cheetah/).

## Cost function

The cost function of this environment is designed in such a way that it tries to minimize the error between the HalfCheetah's forward velocity and a reference value. The cost function is defined as:

$$
cost = w_{forward\_velocity} \times (x_{velocity} - x_{reference\_x\_velocity})^2 + w_{ctrl} \times c_{ctrl}
$$

Where:

*   $w_{forward\_velocity}$ - is the weight of the forward velocity error.
*   $x_{velocity}$ - is the HalfCheetah's forward velocity.
*   $x_{reference\_x\_velocity}$ is the reference forward velocity.
*   $w_{ctrl}$ is the weight of the control cost (**optional**).
*   $c_{ctrl}$ is the control cost (**optional**).

The control cost is optional and can be disabled using the `include_ctrl_cost` environment arguments.

## How to use

This environment is part of the [Stable Gym package](https://github.com/rickstaa/stable-gym). It is therefore registered as the `stable_gym:HalfCheetahCost-v1` gymnasium environment when you import the Stable Gym package. If you want to use the environment in stand-alone mode, you can register it yourself.
