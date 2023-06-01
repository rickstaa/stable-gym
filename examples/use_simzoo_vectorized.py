"""A simple example on how to use the Simzoo environments in a vectorized manner.

.. note::
    For more information on vectorized environments, see the `gym.vector`_
    documentation.

.. _gym.vector: https://gymnasium.farama.org/api/vector/
"""
import gymnasium as gym

import simzoo  # noqa: F401

# ENV_NAME = "Oscillator-v1"
# ENV_NAME = "Ex3EKF-v1"
# ENV_NAME = "CartPoleCost-v1"
ENV_NAME = "CartPole-v1"

if __name__ == "__main__":
    envs = gym.vector.make(
        ENV_NAME, render_mode="human", num_envs=3, asynchronous=False
    )

    # Define a policy function.
    # NOTE: Can be any function that takes an observation and returns an action.
    def policy(*args, **kwargs):
        """A simple policy that samples random actions."""
        return envs.action_space.sample()

    # Run training loop.
    observation, info = envs.reset(seed=42)
    for _ in range(1000):
        action = policy(observation)  # User-defined policy function
        observation, reward, terminated, truncated, info = envs.step(action)

        done = terminated | truncated
        if any(done):
            # Get indexes of environments that are done.
            done_idxs = [i for i, d in enumerate(done) if d]
            print(
                f"Environment(s) {done_idxs} terminated or truncated. These will be "
                "automatically reset."
            )  # NOTE: Reset is done automatically by the vectorized environment.
    envs.close()
