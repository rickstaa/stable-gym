"""Common disturbances used in the disturber.
"""
import math

import numpy as np


def impulse_disturbance(
    input_signal,
    impulse_magnitude,
    impulse_instant,
    current_timestep,
    impulse_length=1.0,
    impulse_frequency=0.0,
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
        current_timestep (int): The current time step.
        impulse_length (float): The length of the supplied impulse disturbance. Defaults
            to `1.0` s.
        impulse_frequency (float): The frequency that is used for supplying the impulse.
            Defaults to `0.0` meaning only one impulse is supplied.
        impulse_type (str): The type of impulse disturbance you want to use. Options
            are: 'instant' and 'constant'. A instant impulse (applied at a
            single time step) and a constant impulse (applied at all steps following the
            set time instant). Defaults to 'instant`.

    Returns:
        numpy.ndarray: The disturbance array.
    """
    impulse_length = impulse_length if impulse_length is not None else 1.0
    if current_timestep >= impulse_instant:
        if impulse_frequency == 0.0 or impulse_frequency is None:
            if current_timestep <= impulse_instant + impulse_length:
                return impulse_magnitude * (-np.sign(input_signal))
        else:
            if math.modf(
                (current_timestep - impulse_instant) / (1 / impulse_frequency)
            )[0] >= 0.0 and math.modf(
                (current_timestep - impulse_instant) / (1 / impulse_frequency)
            )[
                0
            ] <= impulse_length / (
                1 / impulse_frequency
            ):
                return impulse_magnitude * (-np.sign(input_signal))

    # Return undisturbed state
    return np.zeros_like(input_signal)


def periodic_disturbance(current_timestep, amplitude=1, frequency=10, phase_shift=0):
    """Returns a periodic disturbance signal that has the same shape as the input
    signal.

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
