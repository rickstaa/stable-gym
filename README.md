# Simzoo

A python package containing the [bayesian_learning_control](https://github.com/rickstaa/bayesian-learning-control) Openai gym environments.

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

## Installation and Usage

Please see the [bayesian-learning-control docs](https://rickstaa.github.io/bayesian-learning-control/simzoo/simzoo.html) for installation and usage instructions.
