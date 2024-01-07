# QuadXWaypointsCost gymnasium environment

<div align="center">
    <img src="https://github.com/rickstaa/stable-gym/assets/17570430/4cf24737-24c6-49d1-b338-81f40979cbee" alt="QuadX Waypoints Cost environment" width="300px">
</div>
</br>

An actuated multirotor unmanned aerial vehicle (UAV) in the Quad-X configuration as described by [ArduPilot](https://ardupilot.org/copter/docs/connect-escs-and-motors.html) and [PX4](https://docs.px4.io/main/en/airframes/airframe_reference.html#quadrotor-x). It consists of four motors with implementations for cascaded PID controllers. This environment corresponds to the [QuadXWaypoints-v1](https://jjshoots.github.io/PyFlyt/documentation/gym_envs/quadx_envs/quadx_waypoints_env.html) environment included in the [PyFlyt package](https://jjshoots.github.io/PyFlyt/index.html). It is different in the fact that:

* The reward has been changed to a cost. This was done by negating the reward always to be positive definite.
* A health penalty has been added. This penalty is applied when the quadrotor moves outside the flight dome or crashes. The penalty equals the maximum episode steps minus the steps taken or a user-defined penalty.
* The `max_duration_seconds` has been removed. Instead, the `max_episode_steps` parameter of the [gym.wrappers.TimeLimit](https://gymnasium.farama.org/api/wrappers/misc_wrappers/#gymnasium.wrappers.TimeLimit) wrapper is used to limit the episode duration.

The rest of the environment is the same as the original QuadXWaypoints environment. Below, the modified cost and observation space is described. For more information about the environment (e.g. observation space, action space, episode termination, etc.), please refer to the [PyFlyt package documentation](https://jjshoots.github.io/PyFlyt/index.html).

## Observation space

Compared to the original QuadXWaypoints environment, the observation space has been flatted, and one additional observation has been added:

* $s_{waypoints}$: The x,y,z positions of the waypoint(s) that need to be reached.
* $s_{waypoints_\delta}$: The difference between the current position and the desired position of the waypoint(s).

These observations are stacked with the other observations and are **optional**. They can be excluded from the observation space by setting the `exclude_waypoint_targets_from_observation` and `exclude_waypoint_target_deltas_from_observation` environment arguments to `True`. Please note that the environment needs the reference or the reference error to be included in the observation for the agent to function correctly. If both are excluded, the environment will raise an error. Additionally, the
`only_observe_immediate_waypoint` and `only_observe_immediate_waypoint_delta` environment arguments can be used only to observe the immediate waypoint and its delta.

## Cost function

The cost function of this environment is designed in such a way that it tries to minimize the error between the quadrotors' current position and a desired waypoint position (i.e. $p=x_{x,y,z}=[0,0,1]$) and the error between the quadrotors' current angular roll and pitch and their zero values. Additionally, a penalty is given for moving away from the waypoint, and a health penalty can also be included in the cost. This health penalty is added when the drone leaves the flight dome or crashes. It equals the `max_episode_steps` minus the number of steps taken in the episode or a fixed value. The cost is computed as:

$$
cost = 10 \times \| p_{drone} - p_{waypoint} \| - \min(3.0 \times (p_{old} - p_{drone}), 0.0) + p_{health}
$$

Where:

* $p_{drone}$ - is the current quadrotor position (i.e. x,y,z).
* $p_{waypoint}$ - is the desired waypoint position (i.e. x,y,z).
* $p_{old}$ - is the previous quadrotor position.
* $p_{health}$ is a penalty for being unhealthy (i.e. if the Quadrotor moves outside the flight dome or crashes).

The health penalty is optional and can be disabled using the `include_health_penalty` environment arguments.

## Environment step return

In addition to the observations, the cost and a termination and truncation boolean, the environment also returns an info dictionary:

```python
[observation, cost, termination, truncation, info_dict]
```

Compared to the original [QuadXWaypoints-v1](https://jjshoots.github.io/PyFlyt/documentation/gym_envs/quadx_envs/quadx_waypoints_env.html) environment, the following keys were added to this info dictionary:

* **reference**: The reference that the quadrotor is tracking (i.e. the immediate waypoint).
* **state\_of\_interest**: The state that should track the reference (SOI).
* **reference\_error**: The error between SOI and the reference.

## How to use

This environment is part of the [Stable Gym package](https://github.com/rickstaa/stable-gym). It is therefore registered as the `stable_gym:QuadXWaypointsCost-v1` gymnasium environment when you import the Stable Gym package. If you want to use the environment in stand-alone mode, you can register it yourself.
