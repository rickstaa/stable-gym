"""A more challenging (i.e. complicated) version of the `Oscillator gymnasium environment`_
with an additional protein, mRNA transcription concentration variable and light input.

.. _`Oscillator gymnasium environment`: https://rickstaa.dev/stable-gym/envs/biological/oscillator.html
"""  # noqa: E501
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from gymnasium import logger, spaces

if __name__ == "__main__":
    from stable_gym.envs.biological.oscillator_complicated.oscillator_complicated_disturber import (  # noqa: E501, F401
        OscillatorComplicatedDisturber,
    )
else:
    from .oscillator_complicated_disturber import OscillatorComplicatedDisturber

EPISODES = 10  # Number of env episodes to run when __main__ is called.
RANDOM_STEP = True  # Use random action in __main__. Zero action otherwise.


# TODO: Update solving criteria after training.
class OscillatorComplicated(gym.Env, OscillatorComplicatedDisturber):
    """Challenging (i.e. complicated) oscillatory network environment. This environment
    class is based on the :class:`~stable_gym.envs.biological.oscillator.oscillator.Oscillator`
    environment class but has an additional protein, mRNA transcription and light input.

    .. Note::
        Can also be used in a vectorized manner. See the
        :gymnasium:`gym.vector <api/vector>` documentation.

    .. note::
        This gymnasium environment inherits from the
        :class:`~stable_gym.common.disturber.Disturber`
        in order to be able to use it with the Robustness Evaluation tool of the
        Stable Learning Control package (SLC). For more information see
        `the SLC documentation <https://rickstaa.dev/stable-learning-control/utils/tester.html#robustness-eval-utility>`_.

    Description:
        The goal of the agent in the oscillator environment is to act in such a way that
        one of the proteins of the synthetic oscillatory network follows a supplied
        reference signal.

    Source:
        This environment corresponds to the Oscillator environment used in the paper
        `Han et al. 2020`_.

    .. _`Han et al. 2020`: https://arxiv.org/abs/2004.14288

    Observation:
        **Type**: Box(7)

        +-----+-------------------------------------------------+-------------------+-------------------+
        | Num | Observation                                     | Min               | Max               |
        +=====+=================================================+===================+===================+
        | 0   | Lacl mRNA transcripts concentration             | 0                 | 100               |
        +-----+-------------------------------------------------+-------------------+-------------------+
        | 1   | tetR mRNA transcripts concentration             | 0                 | 100               |
        +-----+-------------------------------------------------+-------------------+-------------------+
        | 2   | CI mRNA transcripts concentration               | 0                 | 100               |
        +-----+-------------------------------------------------+-------------------+-------------------+
        | 3   | Extra protein mRNA transcripts concentration    | 0                 | 100               |
        +-----+-------------------------------------------------+-------------------+-------------------+
        | 4   || lacI (repressor) protein concentration         | 0                 | 100               |
        |     || (Inhibits transcription of the tetR gene)      |                   |                   |
        +-----+-------------------------------------------------+-------------------+-------------------+
        | 6   || tetR (repressor) protein concentration         | 0                 | 100               |
        |     || (Inhibits transcription of CI gene)            |                   |                   |
        +-----+-------------------------------------------------+-------------------+-------------------+
        | 7   || CI (repressor) protein concentration           | 0                 | 100               |
        |     || (Inhibits transcription of extra protein gene) |                   |                   |
        +-----+-------------------------------------------------+-------------------+-------------------+
        | 8   || Extra (repressor) protein concentration        | 0                 | 100               |
        |     || (Inhibits transcription of lacI gene)          |                   |                   |
        +-----+-------------------------------------------------+-------------------+-------------------+
        | 9   | The reference we want to follow                 | 0                 | 100               |
        +-----+-------------------------------------------------+-------------------+-------------------+
        | 10  || The error between the current value of         | -100              | 100               |
        |     || protein 1 and the reference                    |                   |                   |
        +-----+-------------------------------------------------+-------------------+-------------------+

    Actions:
        **Type**: Box(3)

        +-----+------------------------------------------------------------+---------+---------+
        | Num | Action                                                     | Min     |   Max   |
        +=====+============================================================+=========+=========+
        | 0   || Relative intensity of light signal that induce the        | -5      | 5       |
        |     || expression of the Lacl mRNA gene.                         |         |         |
        +-----+------------------------------------------------------------+---------+---------+
        | 1   || Relative intensity of light signal that induce the        | -5      | 5       |
        |     || expression of the tetR mRNA gene.                         |         |         |
        +-----+------------------------------------------------------------+---------+---------+
        | 2   || Relative intensity of light signal that induce the        | -5      | 5       |
        |     || expression of the CI mRNA gene.                           |         |         |
        +-----+------------------------------------------------------------+---------+---------+
        | 3   || Relative intensity of light signal that induce the        | -5      | 5       |
        |     || expression of the extra protein mRNA gene.                |         |         |
        +-----+------------------------------------------------------------+---------+---------+

    Cost:
        A cost, computed as the sum of the squared differences between the estimated and the actual states:

        .. math::

            C = {p_1 - r_1}^2

    Starting State:
        All observations are assigned a uniform random value in ``[0..5]``

    Episode Termination:
        -   An episode is terminated when the maximum step limit is reached.
        -   The step cost is greater than 100.

    Solved Requirements:
        Considered solved when the average cost is lower than 300.

    How to use:
        .. code-block:: python

            import stable_gym
            import gymnasium as gym
            env = gym.make("CartPoleCost-v1")

        On reset, the ``options`` parameter allows the user to change the bounds used to
        determine the new random state when ``random=True``.

    Attributes:
        state (numpy.ndarray): The current system state.
        t (float): The current time step.
        dt (float): The environment step size. Also available as :attr:`.tau`.
        sigma (float): The variance of the system noise.
    """  # noqa: E501

    instances = 0  # Number of instances created.

    def __init__(
        self,
        render_mode=None,
        reference_type="periodic",
        reference_target_position=8.0,
        reference_constraint_position=20.0,
        clip_action=True,
    ):
        """Initialise a new OscillatorComplicated environment instance.

        Args:
            render_mode (str, optional): The render mode you want to use. Defaults to
                ``None``. Not used in this environment.
            reference_type (str, optional): The type of reference you want to use
                (``constant`` or ``periodic``), by default ``periodic``.
            reference_target_position: The reference target position, by default
                ``8.0`` (i.e. the mean of the reference signal).
            reference_constraint_position: The reference constraint position, by
                default ``20.0``. Not used in the environment but used for the info
                dict.
            clip_action (str, optional): Whether the actions should be clipped if
                they are greater than the set action limit. Defaults to ``True``.
        """
        super().__init__()  # Setup disturber.
        self._action_clip_warning = False
        self._clip_action = clip_action

        # Validate input arguments.
        if reference_type.lower() not in ["constant", "periodic"]:
            raise ValueError(
                "The reference type must be either 'constant' or 'periodic'."
            )

        self.reference_type = reference_type
        self.t = 0.0
        self.dt = 1.0
        self._init_state = np.array(
            [0.8, 1.5, 0.5, 0.3, 3.3, 3, 3, 2.8]
        )  # Used when random is disabled in reset.
        self._init_state_range = {
            "low": [0, 0, 0, 0, 0, 0, 0, 0],
            "high": [5, 5, 5, 5, 5, 5, 5, 5],
        }  # Used when random is enabled in reset.

        # Set oscillator network parameters.
        self.K1 = 1.0  # mRNA dissociation constants m1.
        self.K2 = 1.0  # mRNA dissociation constant m2.
        self.K3 = 1.0  # mRNA dissociation constant m3.
        self.K4 = 1.0  # mRNA dissociation constant m4.
        self.a1 = 1.6  # Maximum promoter strength m1.
        self.a2 = 1.6  # Maximum promoter strength m2.
        self.a3 = 1.6  # Maximum promoter strength m3.
        self.a4 = 1.6  # Maximum promoter strength m4.
        self.gamma1 = 0.16  # mRNA degradation rate m1.
        self.gamma2 = 0.16  # mRNA degradation rate m2.
        self.gamma3 = 0.16  # mRNA degradation rate m3.
        self.gamma4 = 0.16  # mRNA degradation rate m4.
        self.beta1 = 0.16  # Protein production rate p1.
        self.beta2 = 0.16  # Protein production rate p2.
        self.beta3 = 0.16  # Protein production rate p3.
        self.beta4 = 0.16  # Protein production rate p4.
        self.c1 = 0.06  # Protein degradation rate p1.
        self.c2 = 0.06  # Protein degradation rate p2.
        self.c3 = 0.06  # Protein degradation rate p3.
        self.c4 = 0.06  # Protein degradation rate p4.
        self.b1 = 1.0  # Control input gain u1.
        self.b2 = 1.0  # Control input gain u2.
        self.b3 = 1.0  # Control input gain u3.
        self.b4 = 1.0  # Control input gain u4.

        # Set noise parameters.
        # NOTE: Zero during training.
        self.delta1 = 0.0  # m1 noise.
        self.delta2 = 0.0  # m2 noise.
        self.delta3 = 0.0  # m3 noise.
        self.delta4 = 0.0  # m4 noise.
        self.delta5 = 0.0  # p1 noise.
        self.delta6 = 0.0  # p2 noise.
        self.delta7 = 0.0  # p3 noise.
        self.delta8 = 0.0  # p4 noise.

        obs_low = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, -100], dtype=np.float32)
        obs_high = np.array(
            [100, 100, 100, 100, 100, 100, 100, 100, 100, 100], dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=np.array([-5.0, -5.0, -5.0, -5.0], dtype=np.float32),
            high=np.array([5.0, 5.0, 5.0, 5.0], dtype=np.float32),
            dtype=np.float32,
        )  # QUESTION: Should we use a absolute action space (i.e. 0-10)?
        self.observation_space = spaces.Box(obs_low, obs_high, dtype=np.float32)
        self.cost_range = spaces.Box(
            np.array([0.0], dtype=np.float32),
            np.array([100], dtype=np.float32),
            dtype=np.float32,
        )

        self.viewer = None
        self.state = None
        self.steps_beyond_done = None

        # Reference target and constraint positions.
        self.reference_target_pos = reference_target_position  # Reference target.
        self.reference_constraint_pos = (
            reference_constraint_position  # Reference constraint.
        )

        # Print vectorization debug info.
        self.__class__.instances += 1
        self.instance_id = self.__class__.instances
        logger.debug(f"Oscillator instance '{self.instance_id}' created.")
        logger.debug(
            f"Oscillator instance '{self.instance_id}' uses a '{reference_type}' "
            "reference."
        )

    def step(self, action):
        """Take step into the environment.

        Args:
            action (numpy.ndarray): The action we want to perform in the environment.

        Returns:
            (tuple): tuple containing:

                -   obs (:obj:`np.ndarray`): Environment observation.
                -   cost (:obj:`float`): Cost of the action.
                -   terminated (:obj:`bool`): Whether the episode is terminated.
                -   truncated (:obj:`bool`): Whether the episode was truncated. This
                    value is set by wrappers when for example a time limit is reached or
                    the agent goes out of bounds.
                -   info (:obj:`dict`): Additional information about the environment.
        """
        # Clip action if needed.
        if self._clip_action:
            # Throw warning if clipped and not already thrown.
            if not self.action_space.contains(action) and not self._action_clip_warning:
                logger.warn(
                    f"Action '{action}' was clipped as it is not in the action_space "
                    f"'high: {self.action_space.high}, low: {self.action_space.low}'."
                )
                self._action_clip_warning = True

            u1, u2, u3, u4 = np.clip(
                action, self.action_space.low, self.action_space.high
            )
        else:
            assert self.action_space.contains(
                action
            ), f"{action!r} ({type(action)}) invalid"
            u1, u2, u3, u4 = action
        assert self.state is not None, "Call reset before using step method."

        # Perform action in the environment and return the new state.
        # NOTE: The new state is found by solving 3 first-order differential equations.
        (
            m1,
            m2,
            m3,
            m4,
            p1,
            p2,
            p3,
            p4,
        ) = self.state  # NOTE: [x1, x2, x3, x4, x5, x6] in paper.
        m1_dot = -self.gamma1 * m1 + self.a1 / (self.K1 + np.square(p4)) + self.b1 * u1
        m2_dot = -self.gamma2 * m2 + self.a2 / (self.K2 + np.square(p1)) + self.b2 * u2
        m3_dot = -self.gamma3 * m3 + self.a3 / (self.K3 + np.square(p2)) + self.b3 * u3
        m4_dot = -self.gamma4 * m4 + self.a4 / (self.K4 + np.square(p3)) + self.b4 * u4
        p1_dot = -self.c1 * p1 + self.beta1 * m1
        p2_dot = -self.c2 * p2 + self.beta2 * m2
        p3_dot = -self.c3 * p3 + self.beta3 * m3
        p4_dot = -self.c4 * p4 + self.beta4 * m4

        # Calculate mRNA concentrations.
        # Note: Use max to make sure concentrations can not be negative.
        m1 = np.max(
            [
                m1
                + m1_dot * self.dt
                + self.np_random.uniform(-self.delta1, self.delta1, 1),
                np.zeros([1]),
            ]
        )
        m2 = np.max(
            [
                m2
                + m2_dot * self.dt
                + self.np_random.uniform(-self.delta2, self.delta2, 1),
                np.zeros([1]),
            ]
        )
        m3 = np.max(
            [
                m3
                + m3_dot * self.dt
                + self.np_random.uniform(-self.delta3, self.delta3, 1),
                np.zeros([1]),
            ]
        )
        m4 = np.max(
            [
                m4
                + m4_dot * self.dt
                + self.np_random.uniform(-self.delta4, self.delta4, 1),
                np.zeros([1]),
            ]
        )

        # Calculate protein concentrations.
        # Note: Use max to make sure concentrations can not be negative.
        p1 = np.max(
            [
                p1
                + p1_dot * self.dt
                + self.np_random.uniform(-self.delta5, self.delta5, 1),
                np.zeros([1]),
            ]
        )
        p2 = np.max(
            [
                p2
                + p2_dot * self.dt
                + self.np_random.uniform(-self.delta6, self.delta6, 1),
                np.zeros([1]),
            ]
        )
        p3 = np.max(
            [
                p3
                + p3_dot * self.dt
                + self.np_random.uniform(-self.delta7, self.delta7, 1),
                np.zeros([1]),
            ]
        )
        p4 = np.max(
            [
                p4
                + p4_dot * self.dt
                + self.np_random.uniform(-self.delta8, self.delta8, 1),
                np.zeros([1]),
            ]
        )

        # Retrieve state.
        self.state = np.array([m1, m2, m3, m4, p1, p2, p3, p4])
        self.t = self.t + self.dt

        # Calculate cost.
        r1 = self.reference(self.t)
        cost = np.square(p1 - r1)

        # Define stopping criteria.
        terminated = bool(cost > self.cost_range.high or cost < self.cost_range.low)

        # Return state, cost, terminated, truncated and info_dict
        return (
            np.array([m1, m2, m3, m4, p1, p2, p3, p4, r1, p1 - r1], dtype=np.float32),
            cost,
            terminated,
            False,
            dict(
                reference=r1,
                state_of_interest=p1,
                reference_error=p1 - r1,
                reference_constraint_position=self.reference_constraint_pos,
                reference_constraint_error=p1 - self.reference_constraint_pos,
                reference_constraint_violated=bool(
                    abs(p1) > self.reference_constraint_pos
                ),
            ),
        )

    def reset(
        self,
        seed=None,
        options=None,
        random=True,
    ):
        """Reset gymnasium environment.

        Args:
            seed (int, optional): A random seed for the environment. By default
                ``None``.
            options (dict, optional): A dictionary containing additional options for
                resetting the environment. By default ``None``. Not used in this
                environment.
            random (bool, optional): Whether we want to randomly initialise the
                environment. By default True.

        Returns:
            (tuple): tuple containing:

                -   obs (:obj:`numpy.ndarray`): Initial environment observation.
                -   info (:obj:`dict`): Dictionary containing additional information.
        """
        super().reset(seed=seed)

        # Initialise custom bounds while ensuring that the bounds are valid.
        # NOTE: If you use custom reset bounds, it may lead to out-of-bound
        # state/observations.
        low = np.array(
            options["low"]
            if options is not None and "low" in options
            else self._init_state_range["low"],
            dtype=np.float32,
        )
        high = np.array(
            options["high"]
            if options is not None and "high" in options
            else self._init_state_range["high"],
            dtype=np.float32,
        )
        assert (
            self.observation_space.contains(
                np.append(low, np.zeros(2, dtype=np.float32))
            )
        ) and (
            self.observation_space.contains(
                np.append(high, np.zeros(2, dtype=np.float32))
            )
        ), (
            "Reset bounds must be within the observation space bounds "
            f"({self.observation_space})."
        )

        # Set initial state, reset time and return initial observation.
        self.state = (
            self.np_random.uniform(low=low, high=high, size=(8,))
            if random
            else self._init_state
        )
        self.t = 0.0
        m1, m2, m3, m4, p1, p2, p3, p4 = self.state
        r1 = self.reference(self.t)
        return np.array(
            [m1, m2, m3, m4, p1, p2, p3, p4, r1, p1 - r1], dtype=np.float32
        ), dict(
            reference=r1,
            state_of_interest=p1,
            reference_error=p1 - r1,
            reference_constraint_position=self.reference_constraint_pos,
            reference_constraint_error=p1 - self.reference_constraint_pos,
            reference_constraint_violated=bool(abs(p1) > self.reference_constraint_pos),
        )

    def reference(self, t):
        """Returns the current value of the periodic reference signal that is tracked by
        the Synthetic oscillatory network.

        Args:
            t (float): The current time step.

        Returns:
            float: The current reference value.
        """
        if self.reference_type == "periodic":
            return self.reference_target_pos + 7 * np.sin((2 * np.pi) * t / 200)
        else:
            return self.reference_target_pos

    def render(self, mode="human"):
        """Render one frame of the environment.

        Args:
            mode (str, optional): Gym rendering mode. The default mode will do something
                human friendly, such as pop up a window.

        NotImplementedError: Will throw a NotImplimented error since the render method
            has not yet been implemented.

        Note:
            This currently is not yet implemented.
        """
        raise NotImplementedError(
            "No render method was implemented yet for the Oscillator environment."
        )

    @property
    def tau(self):
        """Alias for the environment step size. Done for compatibility with the
        other gymnasium environments.
        """
        return self.dt


if __name__ == "__main__":
    print("Setting up 'OscillatorComplicated' environment.")
    env = gym.make("OscillatorComplicated")

    # Run episodes.
    episode = 0
    path, paths = [], []
    reference, references = [], []
    s, info = env.reset()
    path.append(s)
    reference.append(info["reference"])
    print(f"\nPerforming '{EPISODES}' in the 'OscillatorComplicated' environment...\n")
    print(f"Episode: {episode}")
    while episode <= EPISODES:
        action = (
            env.action_space.sample()
            if RANDOM_STEP
            else np.zeros(env.action_space.shape)
        )
        s, r, terminated, truncated, info = env.step(action)
        path.append(s)
        reference.append(info["reference"])
        if terminated or truncated:
            paths.append(path)
            references.append(reference)
            episode += 1
            path, reference = [], []
            s, info = env.reset()
            path.append(s)
            reference.append(info["reference"])
            print(f"Episode: {episode}")
    print("\nFinished 'OscillatorComplicated' environment simulation.")

    # Plot results per episode.
    print("\nPlotting episode data...")
    for i in range(len(paths)):
        path = paths[i]
        fig, ax = plt.subplots()
        print(f"\nEpisode: {i}")
        path = np.array(path)
        t = np.linspace(0, path.shape[0] * env.dt, path.shape[0])
        for j in range(path.shape[1]):  # NOTE: Change if you want to plot less states.
            ax.plot(t, path[:, j], label=f"State {j}")
        ax.set_xlabel("Time (s)")
        ax.set_title(f"OscillatorComplicated episode '{i}'")

        # Plot reference signal.
        ax.plot(
            t,
            np.array(references[i]),
            color="black",
            linestyle="--",
            label="Reference",
        )
        ax.legend()
        print("Close plot to see next episode...")
        plt.show()

    print("\nDone")
