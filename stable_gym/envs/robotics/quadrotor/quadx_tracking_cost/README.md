# QuadXTrackingCost gymnasium environment

<div align="center">
    <img src="https://github.com/rickstaa/stable-gym/assets/17570430/24c894c3-a7df-430d-91d7-37a0517fbce4" alt="QuadX Tracking Cost environment" width="500px">
</div>
</br>

An actuated multirotor unmanned aerial vehicle (UAV) in the Quad-X configuration as described by [ArduPilot](https://ardupilot.org/copter/docs/connect-escs-and-motors.html) and [PX4](https://docs.px4.io/main/en/airframes/airframe_reference.html#quadrotor-x). It consists of four motors with implementations for cascaded PID controllers. This environment corresponds to the [QuadXHover-v1](https://jjshoots.github.io/PyFlyt/documentation/gym_envs/quadx_envs/quadx_hover_env.html) environment included in the [PyFlyt package](https://jjshoots.github.io/PyFlyt/index.html). It is different in the fact that:

*   The reward has been changed to a cost. This was done by negating the reward always to be positive definite.
*   A health penalty has been added. This penalty is applied when the quadrotor moves outside the flight dome or crashes. The penalty equals the maximum episode steps minus the steps taken or a user-defined penalty.
*   The `max_duration_seconds` has been removed. Instead, the `max_episode_steps` parameter of the [gym.wrappers.TimeLimit](https://gymnasium.farama.org/api/wrappers/misc_wrappers/#gymnasium.wrappers.TimeLimit) wrapper is used to limit the episode duration.
*   The objective has been changed to track a periodic reference trajectory.

The rest of the environment is the same as the original QuadXHover environment. Below, the modified cost and observation space is described. For more information about the environment (e.g. observation space, action space, episode termination, etc.), please refer to the [PyFlyt package documentation](https://jjshoots.github.io/PyFlyt/index.html).

## Observation space

Compared to the original QuadXWaypoints environment, the observation space has been flatted, and one additional observation has been added:

*   $r$: The reference signal's x, y, and z positions that need to be tracked.
*   $r_{error}$: The difference between the current position and the reference signal.

These observations are stacked with the other observations and are **optional**. They can be excluded from the observation space by setting the `exclude_reference_from_observation` and `exclude_reference_error_from_observation` environment arguments to `True`. Please note that the environment needs the reference or the reference error to be included in the observation space when the reference signal is not constant to function correctly.

## Cost function

The cost function of this environment is designed in such a way that it tries to minimize the error between the quadrotors' current position and a desired reference position (i.e. $p=x_{x,y,z}=[0,0,1]$). A health penalty can also be included in the cost. This health penalty is added when the drone leaves the flight dome or crashes. It equals the `max_episode_steps` minus the number of steps taken in the episode or a fixed value. The cost is computed as:

$$
cost = \| p_{drone} - p_{reference} \| + p_{health}
$$

Where:

*   $p_{drone}$ - is the current quadrotor position (i.e. x,y,z).
*   $p_{reference}$ - is the desired reference position (i.e. x,y,z).
*   $p_{health}$ is a penalty for being unhealthy (i.e. if the Quadrotor moves outside the flight dome or crashes).

The health penalty is optional and can be disabled using the `include_health_penalty` environment arguments.

## How to use

This environment is part of the [Stable Gym package](https://github.com/rickstaa/stable-gym). It is therefore registered as the `stable_gym:QuadXTrackingCost-v1` gymnasium environment when you import the Stable Gym package. If you want to use the environment in stand-alone mode, you can register it yourself.
