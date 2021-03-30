# Simzoo

A python package containing the [bayesian_learning_control](https://github.com/rickstaa/bayesian-learning-control) Openai gym environments. It currently contains the following environments:

-   [Oscillator:](https://github.com/rickstaa/oscillator) A gym environment for a synthetic oscillatory network of transcriptional regulators called a repressilator.
-   [Ex3EKF:](https://github.com/rickstaa/ex3_ekf) A noisy master-slave environment that can be used to train a RL based stationary Kalman estimator.

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
