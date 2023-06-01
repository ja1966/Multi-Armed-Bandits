"""
An implementation of the multi-armed bandits problem including the
epsilon-greedy algorithm and the upper confidence bound algorithm.

Classes
-------
Game :    Contains algorithms used to solve the multi-armed bandits problem.

Functions
---------
arms :    Produces a list of 'randomly' generated arms.
bernoulli_arms :    Produces a list of 'randomly' generated bernoulli arms.
normal_arms :    Produces a list of 'randomly' generated normal arms.
uniform_arms :    Produces a list of 'randomly' generated uniform arms.
"""

from .bandits import Game, arms, bernoulli_arms, normal_arms, uniform_arms # noqa F401
