# FetchReachCost gymnasium environment

<div align="center">
    <img src="https://github.com/rickstaa/stable-gym/assets/17570430/d395ee04-a0e2-4320-9bd2-f248c207bf06" alt="Fetch Reach Cost environment" width="200px">
</div>
</br>

An actuated 7-DOF [Fetch Mobile manipulator](https://fetchrobotics.com/). This environment corresponds to the [FetchReach-v2](https://robotics.farama.org/envs/fetch/reach/) environment included in the [gymnasium robotics package](https://robotics.farama.org/). It is different in the fact that:

*   The reward was replaced with a cost. This was done by taking the absolute value of the reward.

The rest of the environment is the same as the original FetchReach environment. Below, the modified cost is described. For more information about the environment (e.g. observation space, action space, episode termination, etc.), please refer to the [gymnasium robotics library](https://robotics.farama.org/envs/fetch/reach/).

## Cost function

The cost function of this environment is designed in such a way that it tries to minimize the error between FetchReach's end-effector position and the desired goal position. It is defined as the Euclidean distance between the achieved goal position and the desired goal:

$$
cost = \left | r_{original} \right | = \left \| p - p_{goal} \right \| 
$$

Where:

*   $r{original}$ - is the original reward coming from the FetchReach environment.
*   $p$ - is the achieved goal position (i.e. the end-effector position in Cartesian space).
*   $p_{goal}$ - is the desired goal position in Cartesian space.

## Environment step return

In addition to the observations, the cost and a termination and truncation boolean, the environment also returns an info dictionary:

```python
[observation, cost, termination, truncation, info_dict]
```

Compared to the original [FetchReach-v2](https://robotics.farama.org/envs/fetch/reach/) environment, the following keys were added to this info dictionary:

*   **reference**: The reference position (x,y,z) that the FetchReach is tracking.
*   **state\_of\_interest**: The state that should track the reference (SOI).
*   **reference\_error**: The error between SOI and the reference.

## How to use

This environment is part of the [Stable Gym package](https://github.com/rickstaa/stable-gym). It is therefore registered as the `stable_gym:FetchReachCost-v1` gymnasium environment when you import the Stable Gym package. If you want to use the environment in stand-alone mode, you can register it yourself.
