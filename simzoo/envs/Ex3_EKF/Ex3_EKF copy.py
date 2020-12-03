import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
import csv
import matplotlib.pyplot as plt

# This example is the RL based stationary Kalman filter

# the dynamic system whose state is to be estimated:
# x(k+1)=Ax(k)+w(k)
# x_1: angle
# x_2: frequency
# x_3: amplitude
# y(k)=x_3(k)*sin(x_1(k))+v(k)
# A=[1,dt,0;0,1,0;0,0,1]
# x(0)~N([0;10;1],[3,0,0;0,3,0;0,0,3])
# w(k)~N([0;0;0],[1/3*(dt)^3*q_1,1/2*(dt)^2*q_1,0;1/2*(dt)^2*q_1,dt*q_1,0;0,0,dt*q_2])
# v(k)~N(0,1)

# estimator design
# \hat(x)(k+1)=A\hat(x)(k)+u
# where u=[u1,u2,u3]', u=l(\hat(x)(k),y(k)) come from the policy network l(.,.)


class Ex3_EKF(gym.Env):
    def __init__(self):

        self.t = 0
        self.dt = 0.1
        self.q1 = 0.01
        self.g = 9.81
        # self.l = max(0.5,1 + np.random.normal(0,0.5))
        self.l = 1
        # self.mean0 = [1.5, 0]
        # self.cov0_1 = 0.1
        # self.cov0_2 = 0.1
        # self.cov0_1 = 0
        # self.cov0_2 = 0
        self.mean1 = [0, 0]
        self.cov1 = np.array(
            [
                [1 / 3 * (self.dt) ** 3 * self.q1, 1 / 2 * (self.dt) ** 2 * self.q1],
                [1 / 2 * (self.dt) ** 2 * self.q1, self.dt * self.q1],
            ]
        )
        # self.cov1 = np.array([[0,0],[0,0]])
        self.mean2 = 0
        self.cov2 = 1e-2
        # self.cov2 = 0
        self.missing_rate = 0
        self.sigma = 0
        # displacement limit set to be [-high, high]
        high = np.array([10000, 10000])

        self.action_space = spaces.Box(
            low=np.array([-10.0, -10.0]), high=np.array([10.0, 10.0]), dtype=np.float32
        )
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.seed()
        self.viewer = None
        self.state = None
        self.output = None

        self.steps_beyond_done = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        u1, u2 = action
        t = self.t
        input = 0 * np.cos(t) * self.dt
        # Slave
        hat_x_1, hat_x_2, x_1, x_2 = self.state
        # y_1 = self.output
        # hat_y_1 = np.sin(hat_x_1)
        #
        # hat_x_1 = hat_x_1 + self.dt * hat_x_2 + self.dt * u1*(y_1-hat_y_1)
        # hat_x_2 = hat_x_2 - self.g*np.sin(hat_x_1)*self.dt + self.dt * u2*(y_1-hat_y_1) + input

        # Master

        x_1 = x_1 + self.dt * x_2
        x_2 = x_2 - self.g * self.l * np.sin(x_1) * self.dt + input

        state = np.array([x_1, x_2])
        # add process noise
        state = state + np.random.multivariate_normal(self.mean1, self.cov1).flatten()
        x_1, x_2 = state

        y_1 = np.sin(x_1) + np.random.normal(self.mean2, np.sqrt(self.cov2))
        hat_y_1 = np.sin(hat_x_1 + self.dt * hat_x_2)

        # flag=1: received
        # flag=0: dropout
        (flag,) = np.random.binomial(1, 1 - self.missing_rate, 1)
        # drop_rate = 1
        # to construct cost
        if flag == 1:
            hat_x_1 = hat_x_1 + self.dt * hat_x_2 + self.dt * u1 * (y_1 - hat_y_1)
            hat_x_2 = (
                hat_x_2
                - self.g * np.sin(hat_x_1) * self.dt
                + self.dt * u2 * (y_1 - hat_y_1)
                + input
            )
        else:
            hat_x_1 = hat_x_1 + self.dt * hat_x_2
            hat_x_2 = hat_x_2 - self.g * np.sin(hat_x_1) * self.dt + input

        # hat_x_1 = hat_x_1 + self.dt * hat_x_2 + self.dt * u1 * (y_1 - hat_y_1)
        # hat_x_2 = hat_x_2 - self.g * np.sin(hat_x_1) * self.dt + self.dt * u2 * (y_1 - hat_y_1) + input

        cost_u = (
            np.square(u1 * (y_1 - hat_y_1)) * self.dt
            + np.square(u2 * (y_1 - hat_y_1)) * self.dt
        )
        # cost_y = np.abs(hat_y_1 - y_1) * self.dt
        # cost = cost_y
        cost = np.square(hat_x_1 - x_1) + np.square(hat_x_2 - x_2)
        # cost = np.abs(hat_x_1 - x_1)**1 + np.abs(hat_x_2 - x_2)**1
        # print('cost',cost)
        if cost > 100:
            done = True
        else:
            done = False

        # update new for next round
        self.state = np.array([hat_x_1, hat_x_2, x_1, x_2])
        self.output = y_1
        self.t = self.t + self.dt

        # return np.array([hat_x_1,hat_x_2,y_1, y_2]), cost, done, dict(reference=y_1, state_of_interest=np.array([hat_y_1,hat_y_2]))

        return (
            np.array([hat_x_1, hat_x_2]),
            cost,
            done,
            dict(
                reference=y_1,
                state_of_interest=np.array([hat_x_1 - x_1, hat_x_2 - x_2]),
            ),
        )

    def reset(self):
        x_1 = np.random.uniform(-np.pi / 2, np.pi / 2)
        x_2 = np.random.uniform(-np.pi / 2, np.pi / 2)
        hat_x_1 = x_1 + np.random.uniform(-np.pi / 4, np.pi / 4)
        hat_x_2 = x_2 + np.random.uniform(-np.pi / 4, np.pi / 4)
        self.state = np.array([hat_x_1, hat_x_2, x_1, x_2])
        self.output = np.sin(x_1) + np.random.normal(self.mean2, np.sqrt(self.cov2))
        y_1 = self.output
        y_2 = np.sin(x_2) + np.random.normal(self.mean2, np.sqrt(self.cov2))
        return np.array([hat_x_1, hat_x_2])

    def render(self, mode="human"):

        return


if __name__ == "__main__":
    env = Ex3_EKF()
    T = 10
    path = []
    t1 = []
    s = env.reset()
    for i in range(int(T / env.dt)):
        s, r, info, done = env.step(np.array([0, 0]))
        path.append(s)
        t1.append(i * env.dt)

    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111)

    ax.plot(t1, np.array(path)[:, 0], color="yellow", label="x1")
    ax.plot(t1, np.array(path)[:, 1], color="green", label="x2")
    # ax.plot(t1, np.array(path)[:, 2], color='black', label='measurement')

    handles, labels = ax.get_legend_handles_labels()
    #
    ax.legend(handles, labels, loc=2, fancybox=False, shadow=False)
    plt.show()
    print("done")
