"""A more challenging (i.e. complicated) version of the `Oscillator gymnasium environment`_.
This version adds an extra 4th protein and its accompanying mRNA transcription concentration
to the environment. It also contains an additional action input (i.e. light signal) that can
be used to induce the mRNA transcription of this extra protein.

.. _`Oscillator gymnasium environment`: https://rickstaa.dev/stable-gym/envs/biological/oscillator.html
"""
from stable_gym.envs.biological.oscillator_complicated.oscillator_complicated import (
    OscillatorComp,
)
