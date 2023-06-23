"""Noisy master slave system (Ex3EKF) gymnasium environment.

The dynamic system whose state is to be estimated:

.. math::

    x(k+1)=Ax(k)+w(k)
    x_1: angle
    x_2: frequency
    x_3: amplitude

    y(k)=x_3(k)*sin(x_1(k))+v(k)
    A=[1,dt,0;0,1,0;0,0,1]
    x(0)~N([0;10;1],[3,0,0;0,3,0;0,0,3])
    w(k)~N([0;0;0],[1/3*(dt)^3*q_1,1/2*(dt)^2*q_1,0;1/2*(dt)^2*q_1,dt*q_1,0;0,0,dt*q_2])
    v(k)~N(0,1)

Estimator design:

.. math::

    \\hat(x)(k+1)=A\\hat(x)(k)+u
    where u=[u1,u2,u3]', u=l(\\hat(x)(k),y(k)) come from the policy network l(.,.)
"""
from stable_gym.envs.classic_control.ex3_ekf.ex3_ekf import Ex3EKF  # noqa: F401
