# Oscillator gymnasium environment

A gymnasium environment for a synthetic oscillatory network of transcriptional regulators
called a repressilator. A repressilator is a three-gene regulatory network where the
dynamics of mRNA and proteins follow an oscillatory behaviour
([see Elowitch et al. 2000](https://www-nature-com.tudelft.idm.oclc.org/articles/35002125)
).

## Observation space

*   **m1:** Lacl mRNA concentration.
*   **m2:** tetR mRNA concentration.
*   **m3:** CI mRNA concentration.
*   **p1:** lacI (repressor) protein concentration (Inhibits transcription tetR gene).
*   **p2:** tetR (repressor) protein concentration (Inhibits transcription CI).
*   **p3:** CI (repressor) protein concentration (Inhibits transcription of lacI).

## Action space

*   **u1:** Number of Lacl proteins produced during continuous growth under repressor saturation (Leakiness).
*   **u2:** Number of tetR proteins produced during continuous growth under repressor saturation (Leakiness).
*   **u3:** Number of CI proteins produced during continuous growth under repressor saturation (Leakiness).

## Environment goal

The goal of the agent in the oscillator environment is to act in such a way that one
of the proteins of the synthetic oscillatory network follows a supplied reference
signal.

## Cost function

The Oscillator environment uses the absolute difference between the reference and the state of interest as the cost function:

```python
cost = np.square(p1 - r1)
```

## Environment step return

In addition to the observations, the environment also returns an info dictionary that contains the current reference and
the error when a step is taken. This results in returning the following array:

```python
[hat_x_1, hat_x_2, x_1, x_2, info_dict]
```

## How to use

This environment is part of the [simzoo package](https://github.com/rickstaa/simzoo). It is therefore registered as a gymnasium environment when you import the Simzoo package. If you want to use the environment in the stand-alone mode, you can register it yourself.
