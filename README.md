# Simzoo

A python package containing the [bayesian_learning_control](https://github.com/rickstaa/bayesian-learning-control) Openai gym environments. It currently contains the following environments:

*   [Oscillator:](https://github.com/rickstaa/oscillator) A gym environment for a synthetic oscillatory network of transcriptional regulators called a repressilator.
*   [Ex3EKF:](https://github.com/rickstaa/ex3\_ekf) A noisy master-slave environment that can be used to train a RL based stationary Kalman estimator.

## Clone the repository

Since the repository contains several git submodules to use all the features, it needs
to be cloned using the `--recurse-submodules` argument:

```bash
git clone --recurse-submodules https://github.com/rickstaa/simzoo.git
```

If you already cloned the repository and forgot the `--recurse-submodule` argument you
can pull the submodules using the following git command:

```bash
git submodule update --init --recursive
```
ness).
*   **u3:** Number of CI proteins produced during continuous growth under repressor saturation (Leakiness).

## Environment goal

The goal of the agent in the oscillator environment is to act in such a way that one
of the proteins of the synthetic oscillatory network follows a supplied reference
signal.

## Cost function

The Oscillator environment uses the absolute difference between the reference and the state of interest as the cost function:

```python
cost = np.square(p1 - r1)
```

## Environment step return

In addition to the observations, the environment also returns an info dictionary that contains the current reference and
the error when a step is taken. This results in returning the following array:

```python
[hat_x_1, hat_x_2, x_1, x_2, info_dict]
```

## How to use

This environment is part of the [simzoo package](https://github.com/rickstaa/simzoo). It is therefore registered as a gym environment when you import the Simzoo package. If you want to use the environment in the stand-alone mode, you can register it yourself.
hreshold was violated.
*   **reference**: The current reference (position and angles). Only present when performing a reference tracking task.
*   **state_of_interest**: The current state_of_interest which we try to minimize.

## How to use

This environment is part of the [simzoo package](https://github.com/rickstaa/simzoo). It is therefore registered as a gym environment when you import the Simzoo package. If you want to use the environment in stand-alone mode, you can register it yourself.
