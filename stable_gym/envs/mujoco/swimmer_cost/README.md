# SwimmerCost gymnasium environment

<div align="center">
    <img src="https://github.com/rickstaa/stable-gym/assets/17570430/dccd73b4-c97e-46ce-ba0d-4a1328c0aefe" alt="swimmer" width="200px">
</div>
</br>

An actuated two-jointed swimmer. This environment corresponds to the [Swimmer-v2](https://gym.openai.com/envs/Swimmer-v2/) environment included in the OpenAI gym package. It is different in the fact that:

*   The objective was changed to a speed-tracking task. To do this, the reward is replaced with a cost. This cost is the squared difference between the swimmer's forward speed and a reference value (error).

The rest of the environment is the same as the original Swimmer environment. Below, the modified cost is described. For more information about the environment (e.g. observation space, action space, episode termination, etc.), please refer to the [gymnasium library](https://gymnasium.farama.org/environments/mujoco/swimmer/).

## Cost function

The cost function of this environment is designed in such a way that it tries to minimize the error between the swimmer's forward speed and a reference value. The cost function is defined as:

$$
cost = w\_{forward} \times (x\_{speed} - x\_{reference\_speed})^2 + w\_{ctrl} \times c\_{ctrl}
$$

## How to use

This environment is part of the [Stable Gym package](https://github.com/rickstaa/stable-gym). It is therefore registered as the `stable_gym:SwimmerCost-v1` gymnasium environment when you import the Stable Gym package. If you want to use the environment in stand-alone mode, you can register it yourself.
