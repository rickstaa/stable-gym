# QuadRotor gymnasium environments

The [stable-gym package](https://github.com/rickstaa/stable-gym) contains modified versions of the [quadrotor environments](https://jjshoots.github.io/PyFlyt/documentation/core/drones/quadx.html) found in the [PyFlyt package](https://jjshoots.github.io/PyFlyt/index.html). These environments are different because they return a (positive) cost instead of a (negative) reward, making them compatible with stable RL algorithms. Please check the [PyFlyt robotics](https://jjshoots.github.io/PyFlyt/index.html) package for more information about these environments. The [stable-gym package](https://github.com/rickstaa/stable-gym) currently contains the following quadrotor environments:

* [QuadXHoverCost-v1](https://github.com/rickstaa/stable-gym/stable_gym/envs/robotics/quadrotor/quadx_hover_cost/README.md): The quadrotor has to hover at a certain position.
* [QuadXTrackingCost-v1](https://github.com/rickstaa/stable-gym/stable_gym/envs/robotics/quadrotor/quadx_tracking_cost/README.md): The quadrotor has to track a certain position trajectory.
* [QuadXWaypointCost-v1](https://github.com/rickstaa/stable-gym/stable_gym/envs/robotics/quadrotor/quadx_waypoint_cost/README.md): The quadrotor has to fly through a series of waypoints.
