"""
Classic cart-pole system implemented by Rich Sutton et al.
Copied from http://incompleteideas.net/sutton/book/code/pole.c
permalink: https://perma.cc/C9ZM-652R
"""
__VERSION__ = "1.4.3"  # Cartpole version

import importlib
import sys

import math
import gym
import matplotlib.pyplot as plt
import numpy as np
from gym import spaces
from gym.utils import seeding

# Try to import the disturber class
# NOTE: Only works if the simzoo or bayesian learning control package is installed.
# fallback to object if not successfull.
if "simzoo" in sys.modules:
    from simzoo.common.disturber import Disturber
elif importlib.util.find_spec("simzoo") is not None:
    Disturber = getattr(importlib.import_module("simzoo.common.disturber"), "Disturber")
else:
    try:
        Disturber = getattr(
            importlib.import_module(
                "bayesian_learning_control.simzoo.simzoo.common.disturber"
            ),
            "Disturber",
        )
    except AttributeError:
        Disturber = object


RANDOM_STEP = False  # Use random steps in __main__


# # Disturber config used to overwrite the default config
# # NOTE: Merged with the default config
# DISTURBER_CFG = {
#     # Disturbance applied to environment variables
#     "env_disturbance": {
#         "description": "Lacl mRNA decay rate disturbance",
#         # The env variable which you want to disturb
#         "variable": "_c1",
#         # The range of values you want to use for each disturbance iteration
#         "variable_range": np.linspace(1.0, 3.0, num=5, dtype=np.float32),
#         # Label used in robustness plots.
#         "label": "r: %s",
#     },
# }


class CartPoleCustom(gym.Env, Disturber):
    """
    Description:
        A pole is attached by an un-actuated joint to a cart, which moves along a 
        frictionless track. The pendulun starts upright, and the goal is to prevent it
        from failing over by increasing and reducing the cart's velocity.

    Source:
        This environment corresponds to the version of cart-pole problem described 
        by Barto, Sutton, and Anderson.

    Observations:
        Type: Box(4)
        Num     Observation                     Min         Max
        0	          Cart Position                   -4.8            4.8
        1	          Cart Velocity                   -Inf            Inf
        2	          Pole Angle                        -24°           24°
        3	          Pole Velocity At Tip      -Inf            Inf

    Actions:
        Type: Discrete(2)
        Num	    Action
        0	         Push cart to the left
        1	         Push cart to the right

    Note: The amount the velocity is reduced or increased 
        is not fixed as it depends on the angle the pole is pointing. 
        This is because the center of gravity of the pole increases 
        the amount of energy needed to move the cart underneath it

 Reward:
        Reward is 1 for every step taken, including the termination step

    Starting State:
        All observations are assigned a uniform random value between ±0.05

    Episode Termination:
        Pole Angle is more than ±12°
        Cart Position is more than ±2.4 (center of the cart reaches the edge of the display)
        Episode length is greater than 200
        Solved Requirements
        Considered solved when the average reward is greater than or equal to 195.0 over 100 consecutive trials.

    Attributes:
        state (numpy.ndarray): The current system state.
        t (float): The current time step.
        dt (float): The environment step size.
        sigma (float): The variance of the system noise.

    .. note::
        This gym environment inherits from the
        :class:`~bayesian_learning_control.simzoo.simzoo.common.disturber.Disturber`
        in order to be able to use it with the Robustness Evaluation tool of the
        Bayesian Learning Control package (BLC). For more information see
        `the BLC documentation <https://rickstaa.github.io/bayesian-learning-control/control/robustness_eval.html>`_.
    """  # noqa: E501

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self, seed=None):
        super().__init__()

        # 1 0.1 0.5 original
        self.masscart = 1
        self.masspole = 0.1
        self.gravity = 10
        self.total_mass = (self.masspole + self.masscart)
        self.length = 0.5 + 0  # actually half the pole's length
        self.polemass_length = (self.masspole * self.length)
        self.force_mag = 20
        self.tau = 0.02  # seconds between state updates
        self.dt = self.tau
        self.kinematics_integrator = 'euler'
        self.cons_pos = 4
        self.target_pos = 0
        # Angle at which to fail the episode
        self.theta_threshold_radians = 20 * 2 * math.pi / 360
        # self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 10
        # self.max_v=1.5
        # self.max_w=1
        # FOR DATA
        self.max_v = 50
        self.max_w = 50

        # Set angle limit set to 2 * theta_threshold_radians so failing observation
        # is still within bounds
        high = np.array([
            self.x_threshold * 2,
            self.max_v,
            self.theta_threshold_radians * 2,
            self.max_w])

        self.action_space = spaces.Box(low=-self.force_mag, high=self.force_mag, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        # self.reward_range = spaces.Box(
        #     np.array([0.0], dtype=np.float32),
        #     np.array([100], dtype=np.float32),
        #     dtype=np.float32,
        # )

        # Create random seed and set gym environment parameters
        self.seed(seed)
        self.viewer = None
        self.state = None
        self.steps_beyond_done = None

    def seed(self, seed=None):
        """Return random seed."""
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def set_params(self, length, mass_of_cart, mass_of_pole, gravity):
        self.gravity = gravity
        self.length = length
        self.masspole = mass_of_pole
        self.masscart = mass_of_cart
        self.total_mass = (self.masspole + self.masscart)
        self.polemass_length = (self.masspole * self.length)

    def get_params(self):
        return self.length, self.masspole, self.masscart, self.gravity

    def reset_params(self):
        self.gravity = 10
        self.masscart = 1
        self.masspole = 0.1
        self.length = 0.5
        self.total_mass = (self.masspole + self.masscart)
        self.polemass_length = (self.masspole * self.length)

    def step(self, action, impulse=0, process_noise=np.zeros([5])):
        """Take step into the environment.

        Args:
            action (numpy.ndarray): The action we want to perform in the environment.

        Returns:
            (tuple): tuple containing:

                - obs (:obj:`numpy.ndarray`): The current state
                - cost (:obj:`numpy.float64`): The current cost.
                - done (:obj:`bool`): Whether the episode was done.
                - info_dict (:obj:`dict`): Dictionary with additional information.
        """

        # Perform action in the environment and return the new state
        # NOTE: The new state is found by solving 3 first-order differential equations.
        a = 0
        action = np.clip(action, self.action_space.low, self.action_space.high)
        # self.gravity = np.random.normal(10, 2)
        # self.masscart = np.random.normal(1, 0.2)
        # self.masspole = np.random.normal(0.1, 0.02)
        self.total_mass = (self.masspole + self.masscart)

        state = self.state
        x, x_dot, theta, theta_dot = state
        force = np.random.normal(action, 0)# wind
        force = force + process_noise[0] + impulse
        # force = action

        costheta = math.cos(theta)
        sintheta = math.sin(theta)
        temp = (force + self.polemass_length * theta_dot * theta_dot * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
                self.length * (4.0 / 3.0 - self.masspole * costheta * costheta / self.total_mass))
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        if self.kinematics_integrator == 'euler':
            x = x + self.tau * x_dot+ process_noise[2]
            x_dot = x_dot + self.tau * xacc + process_noise[4]
            # x_dot = np.clip(x_dot, -self.max_v, self.max_v)
            theta = theta + self.tau * theta_dot + process_noise[1]
            theta_dot = theta_dot + self.tau * thetaacc + process_noise[3]

            # theta_dot = np.clip(theta_dot, -self.max_w, self.max_w)
        elif self.kinematics_integrator == 'friction':
            xacc = -0.1 * x_dot / self.total_mass + temp - self.polemass_length * thetaacc * costheta / self.total_mass
            x = x + self.tau * x_dot + process_noise[2]
            x_dot = x_dot + self.tau * xacc + process_noise[4]
            # x_dot = np.clip(x_dot, -self.max_v, self.max_v)
            theta = theta + self.tau * theta_dot + process_noise[1]
            theta_dot = theta_dot + self.tau * thetaacc+ process_noise[3]
            # theta_dot = np.clip(theta_dot, -self.max_w, self.max_w):
        else:  # semi-implicit euler
            x_dot = x_dot + self.tau * xacc + process_noise[4]
            x = x + self.tau * x_dot  + process_noise[2]
            theta_dot = theta_dot + self.tau * thetaacc+ process_noise[3]
            theta = theta + self.tau * theta_dot + process_noise[1]
        self.state = np.array([x, x_dot[0], theta, theta_dot[0]])
        done = abs(x) > self.x_threshold \
               or theta < -self.theta_threshold_radians \
               or theta > self.theta_threshold_radians
        done = bool(done)
        # done = False
        if x < -self.x_threshold \
                or x > self.x_threshold:
            a = 1
        r1 = ((self.x_threshold/10 - abs(x-self.target_pos))) / (self.x_threshold/10)  # -4-----1
        r2 = ((self.theta_threshold_radians / 4) - abs(theta)) / (self.theta_threshold_radians / 4)  # -3--------1
        # r1 = max(10 * (1 - ((x-self.target_pos)/self.x_threshold) **2), 1)
        # r2 = max(10 * (1 - np.abs((theta)/self.theta_threshold_radians)), 1)
        # cost1=(self.x_threshold - abs(x))/self.x_threshold
        e1 = (abs(x)) / self.x_threshold
        e2 = (abs(theta)) / self.theta_threshold_radians
        cost = COST_V1(r1, r2, e1, e2, x, x_dot, theta, theta_dot)
        # cost = 0.1+10*max(0, (self.theta_threshold_radians - abs(theta))/self.theta_threshold_radians) \
        #     #+ 5*max(0, (self.x_threshold - abs(x-self.target_pos))/self.x_threshold)\
        cost = 1* x**2/100 + 20 *(theta/ self.theta_threshold_radians)**2
        l_rewards = 0
        if done:
            cost = 100.
        if abs(x)>self.cons_pos:
            violation_of_constraint = 1
        else:
            violation_of_constraint = 0

        # Return state, cost, done and reference
        return self.state, cost, done, dict(hit=a,
                                            l_rewards=l_rewards,
                                            cons_pos=self.cons_pos,
                                            cons_theta=self.theta_threshold_radians,
                                            target=self.target_pos,
                                            violation_of_constraint=violation_of_constraint,
                                            reference=0,
                                            state_of_interest=theta,
                                            )

    def reset(self, random=True):
        """Reset gym environment.

        Args:
            random (bool, optional): Whether we want to randomly initialise the
                environment. By default True.

        Returns:
            numpy.ndarray: Array containing the current observations.
        """

        # Return random initial state
        self.state = self.np_random.uniform(low=-0.2, high=0.2, size=(4,))
        # self.state[0] = self.np_random.uniform(low=5, high=6)
        self.state[0] = self.np_random.uniform(low=-5, high=5)
        self.steps_beyond_done = None
        return np.array(self.state)

    def render(self, mode='human'):
        screen_width = 800
        screen_height = 400

        world_width = self.x_threshold * 2
        scale = screen_width / world_width
        carty = 100  # TOP OF CART
        polewidth = 10.0
        polelen = scale * 2.0
        cartwidth = 50.0
        cartheight = 30.0

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
            axleoffset = cartheight / 4.0
            cart = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            self.carttrans = rendering.Transform()
            cart.add_attr(self.carttrans)
            self.viewer.add_geom(cart)
            l, r, t, b = -polewidth / 2, polewidth / 2, polelen - polewidth / 2, -polewidth / 2
            pole = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            pole.set_color(.8, .6, .4)
            self.poletrans = rendering.Transform(translation=(0, axleoffset))
            pole.add_attr(self.poletrans)
            pole.add_attr(self.carttrans)
            self.viewer.add_geom(pole)
            self.axle = rendering.make_circle(polewidth / 2)
            self.axle.add_attr(self.poletrans)
            self.axle.add_attr(self.carttrans)
            self.axle.set_color(.5, .5, .8)
            self.viewer.add_geom(self.axle)
            self.track = rendering.Line((0, carty), (screen_width, carty))
            self.track.set_color(0, 0, 0)
            self.viewer.add_geom(self.track)

            # Render the target position
            self.target = rendering.Line((self.target_pos * scale + screen_width / 2.0, 0),
                                         (self.target_pos * scale + screen_width / 2.0, screen_height))
            self.target.set_color(1, 0, 0)
            self.viewer.add_geom(self.target)


            # # Render the constrain position
            # self.cons = rendering.Line((self.cons_pos * scale + screen_width / 2.0, 0),
            #                              (self.cons_pos * scale + screen_width / 2.0, screen_height))
            # self.cons.set_color(0, 0, 1)
            # self.viewer.add_geom(self.cons)

        if self.state is None: return None

        x = self.state
        cartx = x[0] * scale + screen_width / 2.0  # MIDDLE OF CART
        self.carttrans.set_translation(cartx, carty)
        self.poletrans.set_rotation(-x[2])

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

def COST_V1(r1, r2, e1, e2, x, x_dot, theta, theta_dot):
    cost = 20*np.sign(r2) * ((r2) ** 2)+ 1* np.sign(r1) * (( r1) ** 2)
    return cost


if __name__ == "__main__":

    print("Settting up Cartpole environment.")
    env = CartPoleCustom()

    # Take T steps in the environment
    T = 1000
    path = []
    t1 = []
    s = env.reset()
    print(f"Taking {T} steps in the Cartpole environment.")
    for i in range(int(T / env.dt)):
        action = (
            env.np_random.uniform(
                env.action_space.low, env.action_space.high, env.action_space.shape
            )
            if RANDOM_STEP
            else np.zeros(env.action_space.shape)
        )
        s, r, done, info = env.step(action)
        env.render()
        path.append(s)
        t1.append(i * env.dt)
    print("Finished Cartpole environment simulation.")

    # Plot results
    print("Plot results.")
    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111)
    ax.plot(t1, np.array(path)[:, 0], color="orange", label="x")
    ax.plot(t1, np.array(path)[:, 1], color="magenta", label="x_dot")
    ax.plot(t1, np.array(path)[:, 2], color="sienna", label="theta")
    ax.plot(t1, np.array(path)[:, 3], color="blue", label=" theat_dot1")

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc=2, fancybox=False, shadow=False)
    plt.show()
    print("Done")
