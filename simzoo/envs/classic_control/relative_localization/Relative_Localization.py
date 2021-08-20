import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
from scipy.stats import truncnorm
import csv
from itertools import cycle
import matplotlib.pyplot as plt


# np.random.seed(565620) # seed the random number generator for reproducibility
def calcInput_FlyIn1m(step, velocity, maxVel, numRob):
    # Calculate control inputs [vx, vy, yaw_rate]' of all robots such that
    # all robots fly randomly within 1m range
    # print('step', step)
    if (step % 100) == 0:
        if (step % 200) == 0:  # why only be negative when step % 200 ==0 ? then it may be fly out the border!
            # understood! one time forward, onetime backward!
            velocity = -velocity
        else:
            velocity[0:2, :] = np.random.uniform(0, maxVel * 2, (2, numRob)) - maxVel
            velocity[2, :] = np.random.uniform(0, 1, (1, numRob)) - 0.5
        # print('velocity: ', velocity)
    return velocity


def motion_model(x, u, numRob, dt):
    # Robot model for state prediction
    xPred = np.zeros((3, numRob))
    for i in range(numRob):
        # X_{k+1} = X_k + Ve * dt; e means earth, b means body
        # Ve = [[c(psi), -s(psi)],[s(psi),c(psi)]] * Vb
        F = np.array([[1.0, 0, 0],
                      [0, 1.0, 0],
                      [0, 0, 1.0]])
        B = np.array([[np.cos(x[2, i]), -np.sin(x[2, i]), 0],
                      [np.sin(x[2, i]), np.cos(x[2, i]), 0],
                      [0.0, 0.0, 1]]) * dt
        xPred[:, i] = F @ x[:, i] + B @ u[:, i]
        # print("u[:,i]", u[:,i])
    return xPred


def update(xTrue, u, numRob, devObser, devInput, dt):
    # Calculate the updated groundTruth(xTrue), noised observation(zNoise), and noised input(uNoise)
    xTrue = motion_model(xTrue, u, numRob, dt)
    zTrue = np.zeros((numRob, numRob))  # distances
    for i in range(numRob):
        for j in range(numRob):
            dx = xTrue[0, i] - xTrue[0, j]
            dy = xTrue[1, i] - xTrue[1, j]
            zTrue[i, j] = np.sqrt(dx ** 2 + dy ** 2)
    randNxN = np.random.randn(numRob, numRob)  # standard normal distribution.
    np.fill_diagonal(randNxN, 0)  # self distance is zero
    zNois = zTrue + randNxN * devObser  # add noise
    rand3xN = np.random.randn(3, numRob)
    uNois = u + rand3xN * devInput  # add noise
    return xTrue, zNois, uNois


def EKF(uNois, zNois, relativeState, Pmatrix, ekfStride, QR, numRob):
    # Calculate relative position between robot i and j in i's body-frame
    Q = np.diag([QR[2], QR[2], QR[3], QR[2], QR[2], QR[3]]) ** 2  ########## Should it be Pxy, Pxy?????
    R = np.diag([QR[4]]) ** 2  # observation covariance
    dtEKF = ekfStride * 0.01
    for i in range(1):
        for j in [jj for jj in range(numRob) if jj != i]:
            # the relative state Xij = Xj - Xi
            uVix, uViy, uRi = uNois[:, i]
            uVjx, uVjy, uRj = uNois[:, j]
            xij, yij, yawij = relativeState[:, i, j]
            dotXij = np.array([np.cos(yawij) * uVjx - np.sin(yawij) * uVjy - uVix + uRi * yij,
                               np.sin(yawij) * uVjx + np.cos(yawij) * uVjy - uViy - uRi * xij,
                               uRj - uRi])
            statPred = relativeState[:, i, j] + dotXij * dtEKF
            # print('statPred', statPred)

            jacoF = np.array([[1, uRi * dtEKF, (-np.sin(yawij) * uVjx - np.cos(yawij) * uVjy) * dtEKF],
                              [-uRi * dtEKF, 1, (np.cos(yawij) * uVjx - np.sin(yawij) * uVjy) * dtEKF],
                              [0, 0, 1]])
            jacoB = np.array([[-1, 0, yij, np.cos(yawij), -np.sin(yawij), 0],
                              [0, -1, -xij, np.sin(yawij), np.cos(yawij), 0],
                              [0, 0, -1, 0, 0, 1]]) * dtEKF
            PPred = jacoF @ Pmatrix[:, :, i, j] @ jacoF.T + jacoB @ Q @ jacoB.T

            # print(self.Pmatrix[:, :, i, j])

            xij, yij, yawij = statPred
            zPred = dist = np.sqrt(xij ** 2 + yij ** 2)
            jacoH = np.array([[xij / dist, yij / dist, 0]])
            resErr = zNois[i, j] - zPred
            S = jacoH @ PPred @ jacoH.T + R
            K = PPred @ jacoH.T @ np.linalg.inv(S)
            relativeState[:, [i], [j]] = statPred.reshape((3, 1)) + K @ np.array([[resErr]])
            Pmatrix[:, :, i, j] = (np.eye(len(statPred)) - K @ jacoH) @ PPred
    # print(np.trace(self.Pmatrix[:, :, i, j])) # for tuning the filter
    return relativeState


def calcAbsPosUseRelaPosWRTRob0(posRob0, relaPos, xTrue, numRob):
    # Calculate the world-frame position of all robots by using the relative states with respect to robot 0
    xEsti = np.zeros((3, numRob))
    for i in range(numRob):
        # xj = R*x0j+x0
        xEsti[0, i] = relaPos[0, 0, i] * np.cos(xTrue[2, 0]) - relaPos[1, 0, i] * np.sin(xTrue[2, 0])
        xEsti[1, i] = relaPos[0, 0, i] * np.sin(xTrue[2, 0]) + relaPos[1, 0, i] * np.cos(xTrue[2, 0])
        xEsti[2, i] = relaPos[2, 0, i]
        xEsti[:, i] = xEsti[:, i] + posRob0
    return xEsti


def calcRelaState(xTrue, numRob):
    # Calculate the relative states by using the position in world-frame
    xRelaGT = np.zeros((3, numRob))  # ground truth
    x0 = xTrue[0, 0]
    y0 = xTrue[1, 0]
    yaw0 = xTrue[2, 0]
    for i in range(numRob):
        # xj = R*x0j+x0
        x0i = xTrue[0, i] - x0
        y0i = xTrue[1, i] - y0
        yaw0i = xTrue[2, i] - yaw0
        xRelaGT[0, i] = x0i * np.cos(yaw0) + y0i * np.sin(yaw0)
        xRelaGT[1, i] = -x0i * np.sin(yaw0) + y0i * np.cos(yaw0)
        xRelaGT[2, i] = yaw0i
    return xRelaGT


class Relative_Localization(gym.Env):
    def __init__(self):
        # self.choice = 'runInMain'
        self.choice = ''

        self.numRob = 2  # number of robots
        self.dt = 0.01  # time interval [s]
        self.simTime = 70.0  # simulation time [s]
        self.maxVel = 0.5  # maximum velocity [m/s]
        self.devInput = np.array(
            [[0.25, 0.25, 0.01]]).T  # input deviation in simulation, Vx[m/s], Vy[m/s], yawRate[rad/s]
        self.devObser = 0.1  # observation deviation of distance[m]
        self.ekfStride = 1  # update interval of EKF is simStride*0.01[s]

        self.Pxy = 10
        self.Pr = 0.1
        self.Qxy = 0.25
        self.Qr = 0.4
        self.Rd = 0.1

        self.t = 0.
        self.velocity = np.zeros((3, self.numRob))
        self.xTrue = np.random.uniform(-3, 3, (
            3, self.numRob))  # random initial groundTruth of state [x, y, yaw]' of numRob robots
        self.relativeState_ekf = np.zeros(
            (3, self.numRob, self.numRob))  # [x_ij, y_ij, yaw_ij]' of the second robot in the first robot's view
        self.Pmatrix = np.zeros((3, 3, self.numRob, self.numRob))

        self.xEsti_rl = np.zeros((1, 3))
        self.xEsti_ekf = self.relativeState_ekf[:, 0, :]  # relative states in robot0's body-frame (EKF)
        self.xTrueRL = calcRelaState(self.xTrue, self.numRob)  # groundTruth relative states

        # displacement limit set to be [-high, high]
        high_obs = np.array([0.1, 0.1, 0.1]) * 5
        high_act = np.array([0.1, 0.1, 0.1]) * 0.2

        self.action_space = spaces.Box(low=-high_act,
                                       high=high_act,
                                       dtype=np.float32)
        self.observation_space = spaces.Box(-high_obs, high_obs, dtype=np.float32)

        self.seed()
        self.viewer = None
        self.state = None
        self.output = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):  # here u1,u2=measurement, which is a result of the action

        u_11, u_21, u_31 = action

        step = int(self.t * 100)

        # Simulation
        self.velocity = calcInput_FlyIn1m(step, self.velocity, self.maxVel, self.numRob)
        self.xTrue, zNois, uNois = update(self.xTrue, self.velocity, self.numRob, self.devObser, self.devInput, self.dt)
        self.xTrueRL = calcRelaState(self.xTrue, self.numRob)  # groundTruth relative states

        # RL update
        # the relative state Xij = Xj - Xi
        i = 0
        j = 1
        # # test
        # self.xEsti_rl = np.array(
        #      [self.relativeState_ekf[:, 0, 1]])
        uVix, uViy, uRi = uNois[:, i]
        uVjx, uVjy, uRj = uNois[:, j]
        xij = self.xEsti_rl[0][0]
        yij = self.xEsti_rl[0][1]
        yawij = self.xEsti_rl[0][2]
        dotXij = np.array([np.cos(yawij) * uVjx - np.sin(yawij) * uVjy - uVix + uRi * yij,
                           np.sin(yawij) * uVjx + np.cos(yawij) * uVjy - uViy - uRi * xij,
                           uRj - uRi])
        xEsti_rl_pred = self.xEsti_rl + dotXij * self.dt
        # print('xEsti_rl_pred', xEsti_rl_pred)

        xij = self.xEsti_rl[0][0]
        yij = self.xEsti_rl[0][1]
        yawij = self.xEsti_rl[0][2]
        zPred = np.sqrt(xij ** 2 + yij ** 2)
        resErr = zNois[i, j] - zPred
        hat_res_xEsti_rl = resErr * np.array([u_11, u_21, u_31])  # state of RL

        self.xEsti_rl = xEsti_rl_pred + hat_res_xEsti_rl
        # hat_res_xEsti_rl = np.expand_dims(hat_res_xEsti_rl, axis=-1)

        # cost = np.linalg.norm(self.xEsti_rl - self.xTrueRL[:,1]) ** 2
        resi_yawij = (self.xEsti_rl[0][2] - self.xTrueRL[2][1]) % (2 * np.pi)
        if resi_yawij > np.pi:
            resi_yawij = resi_yawij - np.pi * 2
        cost = (self.xEsti_rl[0][0] - self.xTrueRL[0][1]) ** 2 + (self.xEsti_rl[0][1] - self.xTrueRL[1][1]) ** 2 \
               + resi_yawij ** 2

        cost = - cost
        # print('cost', cost)
        if cost < -100:
            done = True
        else:
            done = False

        # EKF update
        if step % self.ekfStride == 0:
            QR = np.array([self.Pxy, self.Pr, self.Qxy, self.Qr, self.Rd])
            if self.choice == 'runInMain':
                self.relativeState_ekf = EKF(uNois, zNois, self.relativeState_ekf, self.Pmatrix, self.ekfStride, QR,
                                             self.numRob)
            else:
                self.relativeState_ekf = np.zeros(
                    (3, self.numRob, self.numRob))

        self.xEsti_ekf = self.relativeState_ekf[:, 0, :]  # relative states in robot0's body-frame

        self.t = self.t + self.dt

        if self.choice == 'runInMain':
            return self.xEsti_ekf, self.xTrue, self.xTrueRL, self.Pmatrix, self.resErr, self.xEsti_rl
        else:
            return hat_res_xEsti_rl, cost, done, dict(
                reference=np.array(np.hstack(
                    [self.xTrueRL[:, 1], self.xEsti_ekf[:, 1]])),
                state_of_interest=np.array(np.hstack(
                    [self.xEsti_rl[0]])))
            # return -1

    def reset(self, eval=False):
        self.t = 0.
        self.velocity = np.zeros((3, self.numRob))
        self.xTrue = np.random.uniform(-3, 3, (
            3, self.numRob))  # random initial groundTruth of state [x, y, yaw]' of numRob robots
        self.relativeState_ekf = np.zeros(
            (3, self.numRob, self.numRob))  # [x_ij, y_ij, yaw_ij]' of the second robot in the first robot's view
        self.Pmatrix = np.zeros((3, 3, self.numRob, self.numRob))

        self.xEsti_rl = np.zeros((1, 3))  # np.random.uniform(-3, 3, (1,3))
        self.xEsti_ekf = self.relativeState_ekf[:, 0, :]  # relative states in robot0's body-frame
        self.xTrueRL = calcRelaState(self.xTrue, self.numRob)  # groundTruth relative states

        hat_res_xEsti_rl = np.random.normal(-1., 1., 3)
        # hat_res_xEsti_rl = np.expand_dims(hat_res_xEsti_rl, axis=-1)

        # test
        self.resErr = np.random.randn(1)

        if self.choice == 'runInMain':
            return self.xEsti_ekf, self.xTrue, self.xTrueRL, self.Pmatrix, self.resErr, self.xEsti_rl
        else:
            return hat_res_xEsti_rl

    def render(self, mode='human'):

        return

    def saveChoice(self, choiceIn):
        self.choice = choiceIn
        return self.choice


if __name__ == '__main__':
    env = Relative_Localization()
    T = 70

    choice = 'runInMain'
    # choice = []
    if env.saveChoice(choice) == 'runInMain':
        estiPath_ekf = []
        estiPath_noUpdate = []
        truePath = []
        steps = []
        xEsti_ekf, xTrue, xTrueRL, Pmatrix, resErr, xEsti_noUpdate = env.reset(np.array([0, 0, 0]))
        xE_ekf = xEsti_ekf[:, 1]
        xT = xTrueRL[:, 1]
        estiPath_ekf.append(xE_ekf)
        estiPath_noUpdate.append(xEsti_noUpdate[0])
        truePath.append(xT)

        # test
        resiPath = []
        resiPath.append(resErr)
        PPath = []
        PPath.append(Pmatrix.diagonal())

        steps.append(0)

        for i in range(int(T / env.dt)):
            if i > 6000:
                aa = 0
            xEsti_ekf, xTrue, xTrueRL, Pmatrix, resErr, xEsti_noUpdate = env.step(np.array([0, 0, 0]))
            # print(xTrue)
            xE_ekf = xEsti_ekf[:, 1] + np.array([0., 0., 0.])
            xT = xTrueRL[:, 1] + np.array([0., 0., 0.])
            estiPath_ekf.append(xE_ekf)
            estiPath_noUpdate.append(xEsti_noUpdate[0])
            truePath.append(xT)
            # test
            resiPath.append(resErr)
            PPath.append(Pmatrix.diagonal())

            steps.append(i / 100.0 + 0.01)

        #     np.savetxt('5.csv', path, delimiter=',')
        colors = "bgrcmk"
        cycol = cycle(colors)
        fig = plt.figure(figsize=(9, 6))
        ax = fig.add_subplot(111)
        color1 = next(cycol)
        ax.plot(steps, np.array(estiPath_ekf)[:, 0], color=color1, label='x0_ekf', linestyle="dotted")
        ax.plot(steps, np.array(truePath)[:, 0], color=color1, label='x0_groundtruth')
        ax.plot(steps, np.array(estiPath_noUpdate)[:, 0], color=color1, label='x0_nofusion', linestyle="dashed")
        color1 = next(cycol)
        ax.plot(steps, np.array(estiPath_ekf)[:, 1], color=color1, label='x1_ekf', linestyle="dotted")
        ax.plot(steps, np.array(truePath)[:, 1], color=color1, label='x1_groundtruth')
        ax.plot(steps, np.array(estiPath_noUpdate)[:, 1], color=color1, label='x1_nofusion', linestyle="dashed")
        color1 = next(cycol)
        ax.plot(steps, np.array(estiPath_ekf)[:, 2], color=color1, label='x2_ekf', linestyle="dotted")
        ax.plot(steps, np.array(truePath)[:, 2], color=color1, label='x2_groundtruth')
        ax.plot(steps, np.array(estiPath_noUpdate)[:, 1], color=color1, label='x2_nofusion', linestyle="dashed")
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, loc=2, fancybox=False, shadow=False)
        #     # xlim0 = 38.5
        #     # xlim1 = 40.1
        #     # ylim0 = -0.9
        #     # ylim1 = 0.9
        #     # ax.set_xlim(xlim0, xlim1)
        #     # ax.set_ylim(ylim0, ylim1)
        plt.show()

        f, (ax1, ax2, ax3) = plt.subplots(3, sharex=True)
        plt.margins(x=0)
        ax1.plot(steps, np.array(estiPath_ekf)[:, 0], linestyle="dashed")
        ax1.plot(steps, np.array(truePath)[:, 0])
        ax1.plot(steps, np.array(estiPath_noUpdate)[:, 0])
        ax1.set_ylabel(r"$x_{ij}$ (m)", fontsize=12)
        ax1.grid(True)
        ax2.plot(steps, np.array(estiPath_ekf)[:, 1], linestyle="dashed")
        ax2.plot(steps, np.array(truePath)[:, 1])
        ax2.plot(steps, np.array(estiPath_noUpdate)[:, 1])
        ax2.set_ylabel(r"$y_{ij}$ (m)", fontsize=12)
        ax2.grid(True)
        ax3.plot(steps, np.array(estiPath_ekf)[:, 2], linestyle="dashed")
        ax3.plot(steps, np.array(truePath)[:, 2])
        ax3.plot(steps, np.array(estiPath_noUpdate)[:, 1])
        ax3.set_ylabel(r"$\mathrm{\psi_{ij}}$ (rad)", fontsize=12)
        ax3.set_xlabel("Time (s)", fontsize=12)
        ax3.grid(True)
        ax3.legend(loc='upper center', bbox_to_anchor=(0.8, 0.6), shadow=True, ncol=1, fontsize=12)
        # Fine-tune figure; make subplots close to each other and hide x ticks for all but bottom plot.
        f.subplots_adjust(hspace=0)
        plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)
        plt.show()