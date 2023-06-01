# Oscillator gymnasium environment

A gymnasium environment for a synthetic oscillatory network of transcriptional regulators
called a repressilator. A repressilator is a three-gene regulatory network where the
dynamics of mRNA and proteins follow an oscillatory behaviour
([see Elowitch et al. 2000](https://www-nature-com.tudelft.idm.oclc.org/articles/35002125)
and [Han et al. 2020](https://arxiv.org/abs/2004.14288)).

## Observation space

*   **m1:** Lacl mRNA transcripts concentration.
*   **m2:** tetR mRNA transcripts concentration.
*   **m3:** CI mRNA transcripts concentration.
*   **p1:** lacI (repressor) protein concentration (Inhibits transcription of tetR gene).
*   **p2:** tetR (repressor) protein concentration (Inhibits transcription of CI gene).
*   **p3:** CI (repressor) protein concentration (Inhibits transcription of lacI gene).
*   **error:** The error between the reference and the state of interest (i.e. p1).

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
[(hat_x_1, hat_x_2, x_1, x_2), cost, termination, truncation, info_dict]
```

The info dictionary contains the following keys:

*   **reference**: The current reference (position and angles). Only present when performing a reference tracking task.
*   **state\_of\_interest**: The state of the variable that is meant to track the reference (i.e. p1).
*   **violation\_of\_constraint**: Whether the protein concentration of the state of interest (i.e. p1) is higher than a certain constraint level (i.e 20).

## How to use

This environment is part of the [simzoo package](https://github.com/rickstaa/simzoo). It is therefore registered as a gymnasium environment when you import the Simzoo package. If you want to use the environment in the stand-alone mode, you can register it yourself.
