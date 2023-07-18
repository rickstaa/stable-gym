# MinitaurCost gymnasium environments

The [stable-gym package](https://github.com/rickstaa/stable-gym) contains modified versions of the [Minitaur environments](https://arxiv.org/abs/1804.10332) found in the [pybullet package](https://pybullet.org/). These environments are different because they return a (positive) cost instead of a (negative) reward, making them compatible with stable RL algorithms. Please refer to the [original codebase](https://github.com/bulletphysics/bullet3/tree/master/examples/pybullet/gym/pybullet_envs/minitaur) or [the article of Tan et al. 2018](https://arxiv.org/abs/1804.10332) on which the Minitaur environment is based for more information. The [stable-gym package](https://github.com/rickstaa/stable-gym) currently contains the following MinitaurCost environments:

*   [MinitaurBullet-v1](https://github.com/rickstaa/stable-gym/stable_gym/envs/robotics/minitaur/minitaur_bullet_cost/README.md): The minitaur has to track a reference velocity.
