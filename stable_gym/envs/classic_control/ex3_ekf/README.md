# Ex3EKF gymnasium environment

A gymnasium environment for a noisy master-slave system. This environment can be used to train an RL-based stationary Kalman filter.

## Observation space

* **hat\_x\_1:** The estimated angle.
* **hat\_x\_2:** The estimated frequency.
* **x\_1:** Actual angle.
* **x\_2:** Actual frequency.

## Action space

* **u1:** First action coming from the RL Kalman filter.
* **u2:** Second action coming from the RL Kalman filter.

## Episode termination

An episode is terminated when the maximum step limit is reached, or the step cost exceeds 100.

## Environment goal

The agent's goal in the Ex3EKF environment is to act so that the estimator estimates the original noisy system perfectly. By doing this, it serves as an RL-based stationary Kalman filter.

## Cost function

The Ex3EKF environment uses the following cost function:

$$
cost = (hat_x_1 - x_1)^2 + (hat_x_2 - x_2)^2
$$

## Environment step return

In addition to the observations, the environment returns an info dictionary containing the current reference and the error when a step is taken. This results in returning the following array:

```python
[hat_x_1, hat_x_2, x_1, x_2, info_dict]
```

## How to use

This environment is part of the [Stable Gym package](https://github.com/rickstaa/stable-gym). It is therefore registered as a gymnasium environment when you import the Stable Gym package. If you want to use the environment in stand-alone mode, you can register it yourself.
