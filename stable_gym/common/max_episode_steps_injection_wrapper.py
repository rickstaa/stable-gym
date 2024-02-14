"""This file contains a small gymnasium wrapper that injects the `max_episode_steps`
argument of a potentially nested `TimeLimit` wrapper into the base environment under the
`_time_limit_max_episode_steps` attribute.
"""

import gymnasium as gym


def get_time_limit_wrapper_max_episode_steps(env):
    """Returns the ``max_episode_steps`` attribute of a potentially nested
    ``TimeLimit`` wrapper.

    Args:
        env (gym.Env): The gymnasium environment.

    Returns:
        int: The value of the ``max_episode_steps`` attribute of a potentially nested
            ``TimeLimit`` wrapper. If the environment is not wrapped in a ``TimeLimit``
            wrapper, then this function returns ``None``.
    """
    if hasattr(env, "env"):
        if isinstance(env, gym.wrappers.TimeLimit):
            return env._max_episode_steps
        get_time_limit_wrapper_max_episode_steps(env.env)
    return None


def inject_attribute_into_base_env(env, attribute_name, attribute_value):
    """Injects the ``max_episode_steps`` argument into the base environment under the
    `_time_limit_max_episode_steps` attribute.

    Args:
        env (gym.Env): The gymnasium environment.
        attribute_name (str): The attribute's name to inject into the base
            environment.
        attribute_value (object): The attribute's value to inject into the base
            environment.
    """
    if hasattr(env, "env"):
        return inject_attribute_into_base_env(env.env, attribute_name, attribute_value)
    setattr(env, attribute_name, attribute_value)


class MaxEpisodeStepsInjectionWrapper(gym.Wrapper):
    """A gymnasium wrapper that injects the ``max_episode_steps`` attribute of the
    ``TimeLimit`` wrapper into the base environment as the
    ``_time_limit_max_episode_steps`` attribute. If the environment is not wrapped in
    a ``TimeLimit`` wrapper, then the ``_time_limit_max_episode_steps`` attribute is
    set to ``None``.
    """

    def __init__(self, env):
        """Wrap a gymnasium environment.
        Args:
            env (gym.Env): The gymnasium environment.
        """
        super().__init__(env)

        # Retrieve max_episode_steps from potentially nested TimeLimit wrappers.
        max_episode_steps = get_time_limit_wrapper_max_episode_steps(self.env)

        # Inject the max_episode_steps attribute into the base environment.
        inject_attribute_into_base_env(
            self.env, "_time_limit_max_episode_steps", max_episode_steps
        )
