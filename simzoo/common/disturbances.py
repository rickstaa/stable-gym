"""Common disturbances used in the disturber.
"""

import numpy as np


def impulse_disturbance(
    input_signal, impulse_magnitude, impulse_instant, impulse_type, current_timestep,
):
    """Retrieves a impulse disturbance that acts in the opposite direction of the input
    signal.

    Args:
        input_signal (numpy.ndarray): The signal to which the disturbance should be
            applied. Used for determining the direction of the disturbance.
        impulse_magnitude (union[float, :obj:`numpy.ndarray`]): The magnitude of the
            impulse disturbance.
        impulse_instant (float): The time step at which you want to apply the impulse
            disturbance.
        impulse_type (str): The type of impulse disturbance you want to use. Options
            are: 'regular' and 'constant'. A regular (instant) impulse (applied at a
            single time step) and a constant impulse (applied at all steps following the
            set time instant)
        current_timestep (int): The current time step.

    Returns:
        numpy.ndarray: The disturbance array.
    """
    if impulse_type.lower() == "constant":
        if current_timestep >= impulse_instant:
            dist_val = impulse_magnitude * (-np.sign(input_signal))
        else:
            dist_val = np.zeros_like(input_signal)
    else:
        # FIXME: Quick experiment
        # if (
        #     round(current_timestep) == impulse_instant
        # ):  # FIXME: round is Quickfix make more clear fix!
        #     dist_val = impulse_magnitude * (-np.sign(input_signal))
        # else:
        #     dist_val = np.zeros_like(input_signal)
        if (
            round(current_timestep) % 20 == 0 and round(current_timestep) != 0
        ):  # FIXME: round is Quickfix make more clear fix!
            dist_val = impulse_magnitude * (-np.sign(input_signal))
        else:
            dist_val = np.zeros_like(input_signal)
    return dist_val


def periodic_disturbance(current_timestep, amplitude=1, frequency=10, phase_shift=0):
    """Returns a periodic disturbance signal that has the same shape as the input signal.

    Args:
        current_timestep(int): The current time step.
        amplitude (union[float, numpy.ndarray], optional): The periodic signal
            amplitude. Defaults to ``1``.
        frequency (union[float, numpy.ndarray], optional): The periodic signal
            frequency. Defaults to ``10``.
        phase_shift (union[float, numpy.ndarray), optional): The periodic signal phase
            shift. Defaults to ``0``.

    Returns:
        numpy.ndarray: The disturbance array.
    """
    return amplitude * np.sin(2 * np.pi * frequency * current_timestep + phase_shift)


def noise_disturbance(mean, std):
    """Returns a random noise specified mean and a standard deviation.

    Args:
        mean (union[float, :obj:`numpy.ndarray`]): The mean value of the noise.
        std (union[float, :obj:`numpy.ndarray`]): The standard deviation of the noise.

    Returns:
        numpy.ndarray: The disturbance array.
    """
    return np.random.normal(mean, std)
