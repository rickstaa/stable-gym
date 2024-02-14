r"""Noisy master slave system (Ex3EKF) gymnasium environment.

Dynamics
========

The dynamics of the system whose state is to be estimated are given by:

.. math::
   :nowrap:

   \[
   \begin{split}
   x(k+1) &= A x(k) + w(k) \\
   \end{split}
   \]

In which the state vector :math:`(x(k)` is given by:

.. math::
   :nowrap:

   \[
   \begin{align*}
   x_1 &: \text{angle} \\
   x_2 &: \text{frequency} \\
   x_3 &: \text{amplitude}
   \end{align*}
   \]

and the measurement vector :math:`(y(k))` is given by:

.. math::
   :nowrap:

   \[
   \begin{split}
   y(k) &= x_3(k) \cdot \sin(x_1(k)) + v(k) \\
   A &= \begin{bmatrix}
       1 & dt & 0 \\
       0 & 1 & 0 \\
       0 & 0 & 1
       \end{bmatrix} \\
   x(0) &\sim \mathcal{N}\left(\begin{bmatrix}0 \\ 10 \\ 1\end{bmatrix}, \begin{bmatrix}
       3 & 0 & 0 \\
       0 & 3 & 0 \\
       0 & 0 & 3
       \end{bmatrix}\right) \\
   w(k) &\sim \mathcal{N}\left(\begin{bmatrix}0 \\ 0 \\ 0\end{bmatrix}, \begin{bmatrix}
       \frac{1}{3}dt^3 q_1 & \frac{1}{2}dt^2 q_1 & 0 \\
       \frac{1}{2}dt^2 q_1 & dt q_1 & 0 \\
       0 & 0 & dt q_2
       \end{bmatrix}\right) \\
   v(k) &\sim \mathcal{N}(0, 1)
   \end{split}
   \]

Estimator design:

.. math::
   :nowrap:

   \[
   \begin{split}
   \hat{x}(k+1) &= A \hat{x}(k) + u \\
   \text{where } u &= [u1, u2, u3]', \ u = l(\hat{x}(k), y(k)) \text{ come from the policy network } l(.,.).
   \end{split}
   \]
"""

from stable_gym.envs.classic_control.ex3_ekf.ex3_ekf import Ex3EKF
