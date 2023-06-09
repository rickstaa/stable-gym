"""A gymnasium environment for a synthetic oscillatory network of transcriptional
regulators called a repressilator. A repressilator is a three-gene regulatory network
where the dynamics of mRNA and proteins follow an oscillatory behaviour (see
`Elowitch et al. 2000`_ and `Han et al. 2020`_).

.. _`Elowitch et al. 2000`: https://www-nature-com.tudelft.idm.oclc.org/articles/35002125
.. _`Han et al. 2020`: https://arxiv.org/abs/2004.14288
"""  # noqa: E501
from stable_gym.envs.biological.oscillator.oscillator import Oscillator  # noqa: F401
