"""A simple example on how to use the Stable Gym `gymnasium`_ environments.

.. _gymnasium: https://gymnasium.farama.org/
"""
import gymnasium as gym
import stable_gym  # noqa: F401

# ENV_NAME = "Oscillator-v1"
# ENV_NAME = "Ex3EKF-v1"
ENV_NAME = "CartPoleCost-v1"
# ENV_NAME = "CartPole-v1"

if __name__ == "__main__":
    env = gym.make(ENV_NAME, render_mode="human")

    # Define a policy function.
    # NOTE: Can be any function that takes an observation and returns an action.
    def policy(*args, **kwargs):
        """A simple policy that samples random actions."""
        return env.action_space.sample()

    # Run training loop.
    observation, info = env.reset(seed=42)
    for _ in range(1000):
        action = policy(observation)  # User-defined policy function
        observation, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            print("Environment terminated or truncated. Resetting.")
            observation, info = env.reset()
    env.close()
