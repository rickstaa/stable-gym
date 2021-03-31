# Ex3EKF gym environment

A gym environment for a noisy master-slave system. This environment can be used to train a
RL based stationary Kalman filter.

## Observation space

-   **hat_x_1:** The estimated angle.
-   **hat_x_2:** The estimated frequency.
-   **x_1:** Actual angle.
-   **x_2:** Actual frequency.

## Action space

-   **u1:** First action coming from the RL Kalman filter.
-   **u2:** Second action coming from the RL Kalman filter.

## Environment goal

The goal of the agent in the Ex3EKF environment is to act in such a way that
estimator perfectly estimated the original noisy system. By doing this, it serves
as an RL based stationary Kalman filter.

## Environment step return

In addition to the observations, the environment also returns an info dictionary that contains the current reference and
the error when a step is taken. This results in returning the following array:

```python
[hat_x_1, hat_x_2, x_1, x_2, info_dict]
```

## Cost function

The Ex3EKF environment uses the following cost function:

```python
cost = np.square(hat_x_1 - x_1) + np.square(hat_x_2 - x_2)
```

## How to use

This environment is part of the [simzoo package](https://github.com/rickstaa/simzoo). It is therefore registered as a gym environment when you import the Simzoo package. If you want to use the environment in stand-alone mode, you can register it yourself.
