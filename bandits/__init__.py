"""
An implementation of the multi-armed bandits problem including the
epsilon-greedy algorithm and the upper confidence bound algorithm.

Classes
-------
Game:    Contains algorithms used to solve the multi-armed bandits problem.

Functions
---------
arms:    Produces a list of 'randomly' generated arms.
"""

from .bandits import Game, arms, bernoulli_arms, Arm, Bandits # noqa F401
