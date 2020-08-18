import sys

# Check if used as name_space package
if "simzoo" in sys.modules:
    from simzoo.envs.oscillator.oscillator import Oscillator
else:
    from machine_learning_control.simzoo.simzoo.envs.oscillator.oscillator import (
        Oscillator,
    )
