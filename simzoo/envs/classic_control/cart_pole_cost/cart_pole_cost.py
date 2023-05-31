"""Modified version of the classic
`CartPole-v1 <https://gym.OpenAi.com/envs/CartPole-v1/>`_ OpenAi Gym environment. In
this version two things are different compared to the original:

-   In this version, the action space is continuous, wherein the OpenAi version
    it is discrete.
-   The reward is replaced with a cost. This cost is defined as the difference between a
    state variable and a reference value (error).
-   Some pendulum parameters are changed to make the environment more realistic. See the
    notes next to the parameters for more information.

.. note::
    See the :meth:`CartPoleCost.cost` method for the exact implementation of the cost.
"""

import math

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from gymnasium import logger, spaces
from gymnasium.error import DependencyNotInstalled
from gymnasium.utils import colorize, seeding

if __name__ == "__main__":
    from cart_pole_cost_disturber import CartPoleDisturber
else:
    from .cart_pole_cost_disturber import CartPoleDisturber

RANDOM_STEP = False  # Use random steps in __main__


class CartPoleCost(gym.Env, CartPoleDisturber):
    """Continuous action space CartPole gymnasium environment

    Description:
        A pole is attached by an un-actuated joint to a cart, which moves along
        a frictionless track. The pendulum starts upright, and the goal is to
        prevent it from falling over by increasing and reducing the cart's
        velocity.

    Source:
        This environment corresponds to the version that is included in the Farama
        Foundation gymnasium package. It is different in the fact that:

        -   In this version, the action space is continuous, wherein the OpenAi version
            it is discrete.
        -   The reward is replaced with a cost. This cost is defined as the difference
            between a state variable and a reference value (error).

    Observation:
        **Type**: Box(4)

        +-----+-----------------------+----------------------+--------------------+
        | Num | Observation           | Min                  | Max                |
        +=====+=======================+======================+====================+
        | 0   | Cart Position         | -4.8                 | 4.8                |
        +-----+-----------------------+----------------------+--------------------+
        | 1   | Cart Velocity         | -Inf                 | Inf                |
        +-----+-----------------------+----------------------+--------------------+
        | 2   | Pole Angle            | -0.418 rad (-24 deg) | 0.418 rad (24 deg) |
        +-----+-----------------------+----------------------+--------------------+
        | 3   | Pole Angular Velocity | -20rad               | 20rad              |
        +-----+-----------------------+----------------------+--------------------+

    Actions:
        **Type**: Box(1)

        +-----+----------------------+
        | Num | Action               |
        +=====+======================+
        | 0   | The card x velocity. |
        +-----+----------------------+

        .. Note::
            The amount the velocity that is reduced or increased is not
            fixed; it depends on the angle the pole is pointing. This is because
            the center of gravity of the pole increases the amount of energy needed
            to move the cart underneath it
    Cost:
        A cost, computed using the :meth:`CartPoleCost.cost` method, is given for each
        simulation step including the terminal step.

    Starting State:
        All observations are assigned a uniform random value in ``[-0.2..0.2]``

    Episode Termination:
        -   Pole Angle is more than 12 degrees.
        -   Cart Position is more than 5 m (center of the cart reaches the edge of the
            display).
        -   Episode length is greater than 200.

    Solved Requirements:
        Considered solved when the average return is greater than or equal to 195.0 over
        100 consecutive trials.

    Attributes:
        state (numpy.ndarray): Array containing the current state.
        t (float): Current time step.
        dt (float): Seconds between state updates.
    """

    metadata = {
        "render.modes": ["human", "rgb_array"],
        "render_fps": 50,
    }  # Not used during training but in other gymnasium utilities
    instances = []

    def __init__(
        self,
        render_mode=None,
        seed=None,
        task_type="stabilization",
        reference_type="constant",
        kinematics_integrator="euler",
        clipped_action=True,
    ):
        """Constructs all the necessary attributes for the CartPoleCost instance.

        Args:
            render_mode (str, optional): Gym rendering mode. By default ``None``.
            seed (int, optional): A random seed for the environment. By default
                ``None``.
            task_type (str, optional): The task you want the agent to perform (options
                are "reference_tracking" and "stabilization"). Defaults to
                "stabilization".
            reference_type (str, optional): The type of reference you want to use
                (``constant`` or ``periodic``), by default ``periodic``.
            kinematics_integrator (str, optional): Solver used for the kinematics
                intergration (options are "euler", "friction", "semi-implicit euler").
                Defaults to "euler".
            clipped_action (str, optional): Whether the actions should be clipped if
                they are greater than the set action limit. Defaults to ``True``.
        """
        super().__init__()  # Setup disturber
        self.__class__.instances.append(self)
        self._instance_nr = len(self.__class__.instances)
        self._action_clip_warning = False
        self.render_mode = render_mode

        self.t = 0
        self.dt = 0.02  # seconds between state updates
        self.task_type = task_type
        self.reference_type = reference_type
        self.length = self._length_init = 1.0
        self.mass_cart = self._mass_cart_init = 1.0
        self.mass_pole = self._mass_pole_init = 0.1
        self.gravity = self._gravity_init = 9.8
        # self.force_mag = 10  # NOTE: OpenAI values
        self.force_mag = 20  # Maximum force magnitude
        self._kinematics_integrator = kinematics_integrator
        self._clipped_action = clipped_action
        self._init_state = np.array(
            [0.1, 0.2, 0.3, 0.1]
        )  # Initial state when random is disabled
        # self._init_state_range = {
        #     "low": [-0.2, -0.05, -0.05, -0.05],
        #     "high": [0.2, 0.05, 0.05, 0.05],
        # }  # NOTE: OpenAI values
        self._init_state_range = {
            "low": [-5, -0.2, -0.2, -0.2],
            "high": [5, 0.2, 0.2, 0.2],
        }  # Initial state range when random is enabled

        # Print environment information
        print(
            colorize(
                f"INFO: CartPoleCost environment {self._instance_nr} is initiated for "
                f"a '{task_type}' task.",
                "green",
                bold=True,
            )
        )
        print(
            colorize(
                (
                    f"INFO: CartPoleCost environment {self._instance_nr} is using a "
                    f"'{reference_type}' reference."
                ),
                "green",
                bold=True,
            )
        )

        # Set the lyapunov constraint and target positions
        self.const_pos = 4.0
        self.target_pos = 0.0

        # Thresholds
        # self.theta_threshold_radians = 12 * 2 * math.pi / 360  # NOTE: OpenAi value
        self.theta_threshold_radians = (
            20 * 2 * math.pi / 360
        )  # Angle at which to fail the episode
        # self.x_threshold = 2.4  # NOTE: OpenAi value
        self.x_threshold = 10
        self.y_threshold = (
            5  # NOTE: Defines real world window height (not used as threshold)
        )
        # self.max_v = np.finfo(np.float32).max  # NOTE: OpenAi value
        # self.max_w = np.finfo(np.float32).max  # NOTE: OpenAi value
        self.max_v = 50
        self.max_w = 50

        # Set angle limit set to 2 * theta_threshold_radians so failing observation
        # is still within bounds
        obs_high = np.array(
            [
                self.x_threshold * 2,
                self.max_v,
                self.theta_threshold_radians * 2,
                self.max_w,
            ]
        )
        self.action_space = spaces.Box(
            low=-self.force_mag, high=self.force_mag, shape=(1,), dtype=np.float32
        )  # NOTE: Discrete in OpenAI version.
        self.observation_space = spaces.Box(-obs_high, obs_high, dtype=np.float32)
        self.reward_range = spaces.Box(
            np.array([0.0], dtype=np.float32),
            np.array([100], dtype=np.float32),
            dtype=np.float32,
        )

        # Create random seed and set gymnasium environment parameters
        self.seed(seed)  # TODO: remove

        self.screen_width = 600
        self.screen_height = 400
        self.screen = None
        self.clock = None
        self.is_open = True
        self.state = None

        self.steps_beyond_terminated = None

    def seed(self, seed=None):
        """Return random seed."""
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def set_params(self, length, mass_of_cart, mass_of_pole, gravity):
        """Sets the most important system parameters.

        Args:
            length (float): The pole length.
            mass_of_cart (float): Cart mass.
            mass_of_pole (float): Pole mass.
            gravity (float): The gravity constant.
        """
        self.length = length
        self.mass_pole = mass_of_pole
        self.mass_cart = mass_of_cart
        self.gravity = gravity

    def get_params(self):
        """Retrieves the most important system parameters.

        Returns:
            (tuple): tuple containing:

                - length(:obj:`float`): The pole length.
                - pole_mass (:obj:`float`): The pole mass.
                - pole_mass (:obj:`float`): The cart mass.
                - gravity (:obj:`float`): The gravity constant.
        """
        return self.length, self.mass_pole, self.mass_cart, self.gravity

    def reset_params(self):
        """Resets the most important system parameters."""
        self.length = self._length_init
        self.mass_pole = self._mass_pole_init
        self.mass_cart = self._mass_cart_init
        self.gravity = self._gravity_init

    def reference(self, t):
        """Returns the current value of the periodic reference signal that is tracked by
        the Synthetic oscillatory network.

        Args:
            t (float): The current time step.

        Returns:
            float: The current reference value.
        """
        if self.reference_type == "periodic":
            return self.target_pos + 7 * np.sin((2 * np.pi) * t / 200)
        else:
            return self.target_pos

    def cost(self, x, theta):
        """Returns the cost for a given cart position (x) and a pole angle (theta).

            Args:
                x (float): The current cart position.
                theta (float): The current pole angle (rads).

        Returns:
            (tuple): tuple containing:

                - cost (float): The current cost.
                - r_1 (float): The current position reference.
                - r_2 (float): The cart_pole angle reference.
        """
        if self.task_type.lower() == "reference_tracking":
            # Calculate cost (reference tracking)
            stab_cost = x**2 / 100 + 20 * (theta / self.theta_threshold_radians) ** 2
            ref = [self.reference(self.t), 0.0]
            ref_cost = abs(x - ref[0])
            # ref_cost = np.square(x - ref[0])
            cost = stab_cost + ref_cost
        else:
            # Calculate cost (stabilization task)
            cost = (
                x**2 / 100 + 20 * (theta / self.theta_threshold_radians) ** 2
            )  # Stabilization task
            ref = np.array([0.0, 0.0])

        return cost, ref

    def step(self, action):
        """Take step into the environment.

        Args:
            action (numpy.ndarray): The action we want to perform in the environment.

        Returns:
            (tuple): tuple containing:

                - obs (:obj:`numpy.ndarray`): The current state
                - cost (:obj:`numpy.float64`): The current cost.
                - terminated (:obj:`bool`): Whether the episode was done.
                - truncated (:obj:`bool`): Whether the episode was truncated. This value
                    is set by wrappers when for example a time limit is reached or the
                    agent goes out of bounds.
                - info_dict (:obj:`dict`): Dictionary with additional information.
        """
        # Clip action if needed
        if self._clipped_action:
            if (
                (action < self.action_space.low).any()
                or (action > self.action_space.high).any()
                and not self._action_clip_warning
            ):
                print(
                    colorize(
                        (
                            f"WARNING: Action '{action}' was clipped as it is not in "
                            f"the action_space 'high: {self.action_space.high}, "
                            f"low: {self.action_space.low}'."
                        ),
                        "yellow",
                        bold=True,
                    )
                )
                self._action_clip_warning = True
            force = np.clip(action, self.action_space.low, self.action_space.high)
        else:
            force = action

        # For the interested reader:
        # https://coneural.org/florian/papers/05_cart_pole.pdf
        # NOTE: The new state is found by solving 3 first-order differential equations.
        x, x_dot, theta, theta_dot = self.state
        cos_theta = math.cos(theta)
        sin_theta = math.sin(theta)
        temp = (
            force + self._pole_mass_length * theta_dot**2 * sin_theta
        ) / self.total_mass
        theta_acc = (self.gravity * sin_theta - cos_theta * temp) / (
            self._com_length
            * (4.0 / 3.0 - self.mass_pole * cos_theta * cos_theta / self.total_mass)
        )  # TODO: Looks wrong
        x_acc = temp - self._pole_mass_length * theta_acc * cos_theta / self.total_mass

        # TODO: looks incorrect.
        if self._kinematics_integrator == "euler":
            x = x + self.dt * x_dot
            x_dot = x_dot + self.dt * x_acc
            theta = theta + self.dt * theta_dot
            theta_dot = theta_dot + self.dt * theta_acc
        elif self._kinematics_integrator == "friction":
            x_acc = (
                -0.1 * x_dot / self.total_mass
                + temp
                - self._pole_mass_length * theta_acc * cos_theta / self.total_mass
            )
            x = x + self.dt * x_dot
            x_dot = x_dot + self.dt * x_acc
            theta = theta + self.dt * theta_dot
            theta_dot = theta_dot + self.dt * theta_acc
        else:  # Semi-implicit euler
            x_dot = x_dot + self.dt * x_acc
            x = x + self.dt * x_dot
            theta_dot = theta_dot + self.dt * theta_acc
            theta = theta + self.dt * theta_dot
        self.state = np.array([x, x_dot[0], theta, theta_dot[0]])
        self.t = self.t + self.dt  # Increment time step

        # Calculate cost
        cost, ref = self.cost(x, theta)

        # Define stopping criteria
        terminated = bool(
            abs(x) > self.x_threshold
            or abs(theta) > self.theta_threshold_radians
            or cost > self.reward_range.high
            or cost < self.reward_range.low
        )
        if terminated:
            cost = 100.0

            # Throw warning if already done
            if self.steps_beyond_terminated is None:
                # Pole just fell!
                self.steps_beyond_terminated = 0
            else:
                if self.steps_beyond_terminated == 0:
                    logger.warn(
                        "You are calling 'step()' even though this "
                        "environment has already returned done = True. You "
                        "should always call 'reset()' once you receive 'done = "
                        "True' -- any further steps are undefined behavior."
                    )
                self.steps_beyond_terminated += 1

        # Render environment if requested
        if self.render_mode == "human":
            self.render()

        # Return state, cost, terminated, truncated and info_dict
        violation_of_constraint = bool(abs(x) > self.const_pos)
        violation_of_x_threshold = bool(x < -self.x_threshold or x > self.x_threshold)
        return (
            self.state,
            cost,
            terminated,
            False,
            dict(
                cons_pos=self.const_pos,
                cons_theta=self.theta_threshold_radians,
                target=self.target_pos,
                violation_of_x_threshold=violation_of_x_threshold,
                violation_of_constraint=violation_of_constraint,
                reference=ref,
                state_of_interest=theta,
            ),
        )

    def reset(self, seed=None, random=True):
        """Reset gymnasium environment.

        Args:
            seed (int, optional): A random seed for the environment. By default
                ``None``.
            random (bool, optional): Whether we want to randomly initialise the
                environment. By default True.

        Returns:
            numpy.ndarray: Array containing the current observations.
            info_dict (:obj:`dict`): Dictionary with additional information.
        """
        if seed is not None:
            self.seed(seed)

        # Return random initial state
        self.state = (
            self.np_random.uniform(
                low=self._init_state_range["low"], high=self._init_state_range["high"]
            )
            if random
            else self._init_state
        )
        self.steps_beyond_terminated = None
        self.t = 0.0

        # Return state and info_dict
        x, _, theta, _ = self.state
        _, ref = self.cost(x, theta)
        violation_of_constraint = bool(abs(x) > self.const_pos)
        violation_of_x_threshold = bool(x < -self.x_threshold or x > self.x_threshold)
        return np.array(self.state), dict(
            cons_pos=self.const_pos,
            cons_theta=self.theta_threshold_radians,
            target=self.target_pos,
            violation_of_x_threshold=violation_of_x_threshold,
            violation_of_constraint=violation_of_constraint,
            reference=ref,
            state_of_interest=theta,
        )

    def render(self):
        """Render one frame of the environment."""
        if self.render_mode is None:
            env_command = (
                f"gymnasium.make('{self.spec.id}', render_mode='rgb_array')"
                if self.spec
                else f'{self.__class__.__name__}(render_mode="rgb_array")'
            )
            logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f"e.g. {env_command}"
            )
            return  # TODO: Check render mode.

        try:
            import pygame
            from pygame import gfxdraw
        except ImportError:
            raise DependencyNotInstalled(
                "pygame is not installed, run `pip install gymnasium[classic_control]`"
            )

        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode(
                    (self.screen_width, self.screen_height)
                )
            else:  # mode == "rgb_array"
                self.screen = pygame.Surface((self.screen_width, self.screen_height))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        world_width = self.x_threshold * 2
        scale = self.screen_width / world_width
        polewidth = 10.0
        polelen = scale * (2 * self.length)
        cartwidth = 50.0
        cartheight = 30.0

        if self.state is None:
            return None

        x = self.state

        self.surf = pygame.Surface((self.screen_width, self.screen_height))
        self.surf.fill((255, 255, 255))

        l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
        axleoffset = cartheight / 4.0
        cartx = x[0] * scale + self.screen_width / 2.0  # MIDDLE OF CART
        carty = 100  # TOP OF CART
        cart_coords = [(l, b), (l, t), (r, t), (r, b)]
        cart_coords = [(c[0] + cartx, c[1] + carty) for c in cart_coords]
        gfxdraw.aapolygon(self.surf, cart_coords, (0, 0, 0))
        gfxdraw.filled_polygon(self.surf, cart_coords, (0, 0, 0))

        l, r, t, b = (
            -polewidth / 2,
            polewidth / 2,
            polelen - polewidth / 2,
            -polewidth / 2,
        )

        pole_coords = []
        for coord in [(l, b), (l, t), (r, t), (r, b)]:
            coord = pygame.math.Vector2(coord).rotate_rad(-x[2])
            coord = (coord[0] + cartx, coord[1] + carty + axleoffset)
            pole_coords.append(coord)
        gfxdraw.aapolygon(self.surf, pole_coords, (202, 152, 101))
        gfxdraw.filled_polygon(self.surf, pole_coords, (202, 152, 101))

        gfxdraw.aacircle(
            self.surf,
            int(cartx),
            int(carty + axleoffset),
            int(polewidth / 2),
            (129, 132, 203),
        )
        gfxdraw.filled_circle(
            self.surf,
            int(cartx),
            int(carty + axleoffset),
            int(polewidth / 2),
            (129, 132, 203),
        )

        gfxdraw.hline(self.surf, 0, self.screen_width, carty, (0, 0, 0))

        self.surf = pygame.transform.flip(self.surf, False, True)
        self.screen.blit(self.surf, (0, 0))
        if self.render_mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()

        elif self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )

    def close(self):
        """Close down the viewer"""
        if self.screen is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.is_open = False

    @property
    def total_mass(self):
        """Property that returns the full mass of the system."""
        return self.mass_pole + self.mass_cart

    @property
    def _com_length(self):
        """Property that returns the position of the center of mass."""
        return self.length * 0.5  # half the pole's length

    @property
    def _pole_mass_length(self):
        """Property that returns the pole mass times the COM length."""
        return self.mass_pole * self._com_length

    @property
    def tau(self):
        """Property that also makes the timestep available under the :attr:`tau`
        attribute. This was done to keep this environment consistent with the
        original gymnasium environment.
        """
        return self.dt


if __name__ == "__main__":
    print("Settting up CartpoleCost environment.")
    env = gym.make("CartPoleCost")

    # Take T steps in the environment
    T = 1000
    path = []
    t1 = []
    s = env.reset()
    print(f"Taking {T} steps in the Cartpole environment.")
    for i in range(int(T / env.dt)):
        action = (
            env.action_space.sample()
            if RANDOM_STEP
            else np.zeros(env.action_space.shape)
        )
        s, r, terminated, truncated, info = env.step(action)
        env.render()
        path.append(s)
        t1.append(i * env.dt)
    print("Finished CartpoleCost environment simulation.")

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
    plt.ioff()
    plt.show()

    print("done")
