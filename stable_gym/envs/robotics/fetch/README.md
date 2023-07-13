# FetchCost gymnasium environments

The [stable-gym package](https://github.com/rickstaa/stable-gym) contains modified versions of the [fetch environments](https://robotics.farama.org/envs/fetch/) found in the [gymnasium robotics package](https://robotics.farama.org). These environments are different because they return a (positive) cost instead of a (negative) reward, making them compatible with stable RL algorithms. Please check the [gymnasium robotics](https://robotics.farama.org/env/fetch) package for more information about these environments. The [stable-gym package](https://github.com/rickstaa/stable-gym) currently contains the following FetchCost environments:

*   [FetchReachCost-v1](https://github.com/rickstaa/stable-gym/stable_gym/envs/robotics/fetch/fetch_reach_cost/README.md): Fetch has to move its end-effector to the desired goal position.
