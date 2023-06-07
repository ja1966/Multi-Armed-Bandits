"""
An implementation of the multi-armed bandit problem including the
Epsilon-Greedy and Upper Confidence Bound algorithms.

Classes
-------
Game :    Contains algorithms used to solve the multi-armed bandit problem.

Functions
---------
bernoulli_arms :    Produces a list of 'randomly' generated bernoulli arms.
normal_arms :       Produces a list of 'randomly' generated normal arms.
uniform_arms :      Produces a list of 'randomly' generated uniform arms.
"""

from .bandits import Game, bernoulli_arms, normal_arms, uniform_arms  # noqa F401
