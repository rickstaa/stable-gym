# oscillator

A gym environment for a synthetic oscillatory network of transcriptional regulators
called a repressilator. A repressilator is a three-gene regulatory network where the
dynamics of mRNA and proteins follow an oscillatory behavior
([see Elowitch et al. 2000](https://www-nature-com.tudelft.idm.oclc.org/articles/35002125)
).

## Observation space

    - m1: Lacl mRNA concentration.
    - m2: tetR mRNA concentration.
    - m3: CI mRNA concentration.
    - p1: lacI (repressor) protein concentration (Inhibits transcription tetR gene).
    - p2: tetR (repressor) protein concentration (Inhibits transcription CI).
    - p3: CI (repressor) protein concentration (Inhibits transcription of lacI).

## Action space

    - u1: Number of Lacl proteins produced during continuous growth under repressor
      saturation (Leakiness).
    - u1: Number of tetR proteins produced during continuous growth under repressor
      saturation (Leakiness).
    - u1: Number of CI proteins produced during continuous growth under repressor
      saturation (Leakiness).

## Goal

The goal of the agent in the oscillator environment is to act in such a way that one
of the proteins of the synthetic oscillatory network follows a supplied reference
signal.
