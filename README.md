# Simzoo

[![GitHub release (latest by date)](https://img.shields.io/github/v/release/rickstaa/simzoo)](https://github.com/rickstaa/simzoo/releases)
[![Python 3](https://img.shields.io/badge/Python-3.8%20%7C%203.7%20%7C%203.6-brightgreen)](https://www.python.org/)
[![Contributions](https://img.shields.io/badge/contributions-welcome-brightgreen.svg)](contributing.md)

A python package containing the non ROS-based [bayesian\_learning\_control](https://github.com/rickstaa/bayesian-learning-control) [Farama Foundation](https://farama.org/) gymnasium environments.

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
