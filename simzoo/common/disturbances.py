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
        impulse_magnitude (float): The magnitude of the impulse disturbance.
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
        if current_timestep == impulse_instant:
            dist_val = impulse_magnitude * (-np.sign(input_signal))
        else:
            dist_val = np.zeros_like(input_signal)
    return dist_val


def periodic_disturbance(
    input_signal, current_timestep, amplitude=1, frequency=10, phase_shift=0
):
    """Returns a periodic disturbance signal that has the same shape as the input signal.

    Args:
        input_signal (numpy.ndarray): The signal to which the disturbance should be
            applied. Used for determining the direction of the disturbance.
        current_timestep(int): The current time step.
        amplitude (float, optional): The periodic signal amplitude. Defaults to ``1``.
        frequency (float, optional): The periodic signal frequency. Defaults to ``10``.
        phase_shift (float, optional): The periodic signal phase shift. Defaults to
            ``0``.

    Returns:
        numpy.ndarray: The disturbance array.
    """
    return (
        amplitude
        * np.sin(2 * np.pi * frequency * current_timestep + phase_shift)
        * np.ones_like(input_signal)
    )


def noise_disturbance(input_signal, mean, std):
    """Returns a random noise specified mean and a standard deviation.

    Args:
        input_signal (numpy.ndarray): The signal to which the disturbance should be
            applied. Used for determining the direction of the disturbance.
        mean

    Returns:
        numpy.ndarray: The disturbance array.
    """
    return np.random.normal(mean, std, len(input_signal),)
