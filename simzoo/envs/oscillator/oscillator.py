"""Synthetic oscillatory network of transcriptional regulators

A gym environment for a synthetic oscillatory network of transcriptional regulators
called a repressilator. A repressilator is a three-gene regulatory network where the
dynamics of mRNA and proteins follow an oscillatory behavior
(see https://www-nature-com.tudelft.idm.oclc.org/articles/35002125).
"""

import argparse

import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
import matplotlib.pyplot as plt


class Oscillator(gym.Env):
    """Synthetic oscillatory network
    The goal of the agent in the oscillator environment is to act in such a way that
    one of the proteins of the synthetic oscillatory network follows a supplied
    reference signal.

    Observations:
        - m1: Lacl mRNA concentration.
        - m2: tetR mRNA concentration.
        - m3: CI mRNA concentration.
        - p1: lacI (repressor) protein concentration (Inhibits transcription tetR gene).
        - p2: tetR (repressor) protein concentration (Inhibits transcription CI).
        - p3: CI (repressor) protein concentration (Inhibits transcription of lacI).

    Action space: # QUESTION: Is this alright? The formulas are different than the once in the article.
        - u1: Number of Lacl proteins produced during continuous growth under repressor
          saturation (Leakiness).
        - u1: Number of tetR proteins produced during continuous growth under repressor
          saturation (Leakiness).
        - u1: Number of CI proteins produced during continuous growth under repressor
          saturation (Leakiness).
    """

    def __init__(self, p_ref=1, amp=7.0, freq=5e-3, h_shift=0.0, v_shift=8.0):
        """Constructs all the necessary attributes for the oscillator object.

        Args:
            p_ref (int, optional): The protein which you want to track the
                reference signal (1: lacI, 2: tetR or 3: CI). Defaults to 1: lacI.
            amp (float, optional): The amplitude of the reference signal we want to
                track. Defaults to 7.
            freq (tuple, optional): The frequency of the reference signal we want to
                track. Defaults to (1.0 / 200).
            h_shift (float, optional): The horizontal phase shift of the reference
                signal we want to track. Defaults to 0.0.
            v_shift (float, optional): The vertical shift of the reference signal
                we want to track.
        """

        # Set oscillator network parameters
        # QUESTION: What do the variables represent
        self.K = 1  # QUESTION: In the article this 1 was hardcoded what does it represent? some mean protein value?
        self.c1 = 1.6  # Ratio protein decay rate to the mRNA decay rate (ALPHA)
        self.c2 = 0.16
        self.c3 = 0.16  # QUESTION: Has something to do with beta but now it is split in c3 and c4
        self.c4 = 0.06  # QUESTION: Has something to do with beta but now it is split in c3 and c4
        self.b1 = 1
        self.b2 = 1
        self.b3 = 1

        # QUESTION: Rewriten variables
        self.n = 2  # Hill coefficient
        self.alpha = 1.6  # Ratio protein decay rate to the mRNA decay rate (ALPHA)
        self.sigma = 0.0  # Random noise variance
        self.dt = 1.0  # Time step size [s]
        self.t = 0  # Start time [s]

        # Set reference signal parameters
        if p_ref not in range(1, 3):
            raise TypeError(
                f"Protein {p_ref} does not exists and can, therefore, not be set to "
                "track the reference. Please specify a number between 1 and 3."
            )
        self.p_ref = p_ref
        self.amp = amp  # amplitude
        self.freq = freq  # Frequency
        self.h_shift = h_shift  # Horizontal shift
        self.v_shift = v_shift  # Vertical shift

        # Set angle limit set to 2 * theta_threshold_radians so failing observation
        # is still within bounds # Question?
        high = np.array([100, 100, 100, 100, 100, 100, 100, 100])

        # Setup gym action and observation space
        self.action_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0]),
            high=np.array([1.0, 1.0, 1.0]),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        # Create random seed and set gym environment parameters
        self.seed()
        self.viewer = None
        self.state = None
        self.steps_beyond_done = None

    def seed(self, seed=None):
        """Return random seed."""
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        """Take step into the environment.

        Args:
            action ([type]): The action we want to perform in the environment.

        Returns:
            tuple: [description]
        """

        # Perform action in the environment and return the new state
        # NOTE: The new state is found by solving 3 first-order differential equations.
        u1, u2, u3 = action
        m1, m2, m3, p1, p2, p3 = self.state
        m1_dot = self.c1 / (self.K + np.square(p3)) - self.c2 * m1 + self.b1 * u1
        p1_dot = self.c3 * m1 - self.c4 * p1
        m2_dot = self.c1 / (self.K + np.square(p1)) - self.c2 * m2 + self.b2 * u2
        p2_dot = self.c3 * m2 - self.c4 * p2
        m3_dot = self.c1 / (self.K + np.square(p2)) - self.c2 * m3 + self.b3 * u3
        p3_dot = self.c3 * m3 - self.c4 * p3

        # Question: I rewrote the formulas to better represent the article can you quickly check if they are correct
        m1_dot = (
            self.alpha / (self.K + np.power(p3, self.n)) - self.c2 * m1 + self.b1 * u1
        )  # Question: Is u1 a0? and c2 a scaling of the m1 variable
        p1_dot = self.c3 * m1 - self.c4 * p1
        m2_dot = (
            self.alpha / (self.K + np.power(p3, self.n)) - self.c2 * m2 + self.b2 * u2
        )
        p2_dot = self.c3 * m2 - self.c4 * p2  # QUESTION: Why is this split?
        m3_dot = (
            self.alpha / (self.K + np.power(p3, self.n)) - self.c2 * m3 + self.b3 * u3
        )
        p3_dot = self.c3 * m3 - self.c4 * p3

        # Calculate mRNA concentrations
        # Note: Use max to make sure concentrations can not be negative.
        m1 = np.max(
            [
                m1 + m1_dot * self.dt + np.random.uniform(-self.sigma, self.sigma, 1),
                np.zeros([1]),
            ]
        )
        m2 = np.max(
            [
                m2 + m2_dot * self.dt + np.random.uniform(-self.sigma, self.sigma, 1),
                np.zeros([1]),
            ]
        )
        m3 = np.max(
            [
                m3 + m3_dot * self.dt + np.random.uniform(-self.sigma, self.sigma, 1),
                np.zeros([1]),
            ]
        )

        # Calculate protein concentrations
        # Note: Use max to make sure concentrations can not be negative.
        p1 = np.max(
            [
                p1 + p1_dot * self.dt + np.random.uniform(-self.sigma, self.sigma, 1),
                np.zeros([1]),
            ]
        )
        p2 = np.max(
            [
                p2 + p2_dot * self.dt + np.random.uniform(-self.sigma, self.sigma, 1),
                np.zeros([1]),
            ]
        )
        p3 = np.max(
            [
                p3 + p3_dot * self.dt + np.random.uniform(-self.sigma, self.sigma, 1),
                np.zeros([1]),
            ]
        )

        # Retrieve state, cost and check if episode is done
        self.state = np.array([m1, m2, m3, p1, p2, p3])
        self.t = self.t + self.dt  # Increment time step # Question: This used to be 1
        ref = self.reference(self.t)
        cost = np.square(p1 - ref)
        if cost > 100:
            done = True
        else:
            done = False

        # Return state, cost, done and reference
        p_ref = [p1, p2, p3][self.p_ref - 1]  # Retrieve value of tracking protein
        return (
            np.array([m1, m2, m3, p1, p2, p3, ref, p_ref - ref]),
            cost,
            done,
            dict(reference=ref, state_of_interest=p_ref),
        )

    def reset(self):
        """Reset gym environment.

        Returns:
            numpy.ndarray: Array containing the current observations.
        """

        # Return random initial state
        self.state = self.np_random.uniform(low=0, high=5, size=(6,))
        self.t = 0
        m1, m2, m3, p1, p2, p3 = self.state
        p_ref = [p1, p2, p3][self.p_ref - 1]  # Retrieve value of tracking protein
        ref = self.reference(self.t)
        return np.array([m1, m2, m3, p1, p2, p3, ref, p_ref - ref])

    def reference(self, t):
        """Returns the current value of the periodic reference signal that is tracked by
        the Synthetic oscillatory network.

        Args:
            t (float): The current time step.

        Returns:
            float: The current reference value.
        """
        return (
            self.amp * np.sin((2 * np.pi) * self.freq * (t - self.h_shift))
            + self.v_shift
        )

    def render(self, mode="human"):
        """Render one frame of the environment.

        Args:
            mode (str, optional): Gym rendering mode. The default mode will do something
            human friendly, such as pop up a window.

        Note:
            This currently is not yet implemented.
        """
        return


if __name__ == "__main__":

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--p-ref",
        type=int,
        default=1,
        help="Which protein you want to track the reference.",
    )
    parser.add_argument(
        "--amp", type=float, default=7.0, help="Reference signal amplitude."
    )
    parser.add_argument(
        "--freq", type=float, default=5e-3, help="Reference signal frequency."
    )
    parser.add_argument(
        "--h-shift",
        type=float,
        default=0.0,
        help="Reference signal horizontal phase shift.",
    )
    parser.add_argument(
        "--v-shift", type=float, default=8.0, help="Reference signal vertical shift."
    )
    args = parser.parse_args()

    # Setup oscillator gym environment
    print("Settting up oscillator environment.")
    env = Oscillator(
        p_ref=args.p_ref,
        amp=args.amp,
        freq=args.freq,
        h_shift=args.h_shift,
        v_shift=args.v_shift,
    )

    # Take T steps in the environment
    T = 600
    path = []
    t1 = []
    s = env.reset()
    print(f"Taking {T} steps in the oscillator environment.")
    for i in range(int(T / env.dt)):
        s, r, done, info = env.step(np.array([0, 0, 0]))
        path.append(s)
        t1.append(i * env.dt)
    print("Finished oscillator environment simulation.")

    # Plot results
    print("Plot results.")
    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111)
    ax.plot(t1, np.array(path)[:, 3], color="blue", label="protein1")
    ax.plot(t1, np.array(path)[:, 6], color="yellow", label="error")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc=2, fancybox=False, shadow=False)
    plt.show()
    print("Done")
