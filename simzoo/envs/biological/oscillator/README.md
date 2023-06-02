# Oscillator gymnasium environment

A gymnasium environment for a synthetic oscillatory network of transcriptional regulators
called a repressilator. A repressilator is a three-gene regulatory network where the
dynamics of mRNA and proteins follow an oscillatory behaviour
([see Elowitch et al. 2000](https://www-nature-com.tudelft.idm.oclc.org/articles/35002125)
and [Han et al. 2020](https://arxiv.org/abs/2004.14288)).

## Observation space

<<<<<<< Updated upstream

*   **m1:** Lacl mRNA transcripts concentration.
*   **m2:** tetR mRNA transcripts concentration.
*   **m3:** CI mRNA transcripts concentration.
*   **p1:** lacI (repressor) protein concentration (Inhibits transcription of tetR gene).
*   **p2:** tetR (repressor) protein concentration (Inhibits transcription of CI gene).
*   **p3:** CI (repressor) protein concentration (Inhibits transcription of lacI gene).
*   **error:** The error between the reference and the state of interest (i.e. p1).
    \=======
*   **m1:** Lacl mRNA concentration.
*   **m2:** tetR mRNA concentration.
*   **m3:** CI mRNA concentration.
*   **p1:** lacI (repressor) protein concentration (Inhibits transcription tetR gene).
*   **p2:** tetR (repressor) protein concentration (Inhibits transcription CI).
*   **p3:** CI (repressor) protein concentration (Inhibits transcription of lacI).
*   **ref:** The reference we want to follow.
*   **ref\_error:** The error between the state of interest (i.e. p1) and the reference.

> > > > > > > Stashed changes

## Action space

*   **u1:** Relative intensity of light signal that induce the expression of the Lacl mRNA gene.
*   **u2:** Relative intensity of light signal that induce the expression of the tetR mRNA gene.
*   **u3:** Relative intensity of light signal that induce the expression of the CI mRNA gene.

## Environment episode stop criteria

An episode is terminated when the maximum step limit is reached, or the cost is greater than 100.

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

In addition to the observations, the cost and a termination and truncation boolean the environment also returns a info dictionary:

```python
[observation, info_dict]
```

The info dictionary contains the following keys:

*   **reference**: The set cart position reference.
*   **state\_of\_interest**: The state that should track the reference (SOI).
*   **reference\_error**: The error between SOI and the reference.
*   **reference\_constraint\_position**: A user specified constraint they want to watch.
*   **reference\_constraint\_error**: The error between the SOI and the set reference constraint.
*   **reference\_constraint\_violated**: Whether the reference constraint was violated.

## How to use

This environment is part of the [simzoo package](https://github.com/rickstaa/simzoo). It is therefore registered as a gymnasium environment when you import the Simzoo package. If you want to use the environment in the stand-alone mode, you can register it yourself.
