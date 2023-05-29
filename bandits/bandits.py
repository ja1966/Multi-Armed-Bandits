"""An implementation of the multi-armed bandits problem."""

from numpy.random import rand, normal, randint, binomial, uniform
from numpy import log, argmax

distributions = ["Bernoulli", "Uniform", "Normal"]


def arms(number_of_arms, bernoulli_min_reward=1, bernoulli_max_reward=100,
         normal_min_mean=25, normal_max_mean=75, normal_min_var=10,
         normal_max_var=25, uniform_min=0, uniform_max=100):
    """
    Create a list of 'randomly' distributed arms.

    Parameters
    ----------
    number_of_arms: int
                    Number of arms used in the multi-armed bandits problem.

    Other Parameters
    ----------------
    bernoulli_min_reward: float, optional
                          Lower bound of the reward given by a bernoulli
                          distributed arm.
    bernoulli_max_reward: float, optional
                          Upper bound of the reward given by a bernoulli
                          distributed arm.
    normal_min_mean: float, optional
                     Lower bound of the possible mean for a normally
                     distributed arm.
    normal_max_mean: float, optional
                     Upper bound of the possible mean for a normally
                     distributed arm.
    normal_min_var: float, optional
                    Lower bound of the possible variance for a normally
                    distributed arm.
    normal_max_var: float, optional
                    Upper bound of the possible variance for a normally
                    distributed arm.
    uniform_min: float, optional
                 Minimal value for a uniformly distributed arm.
    uniform_max: float, optional
                 Maximal value for a uniformly distributed arm.

    Returns
    -------
    list
        Contains the randomly generated arms.

    Notes
    -----
    Creates arms using the Bernoulli, Uniform and Normal distributions.
    """
    arms = []
    for arm in range(number_of_arms):
        dist = distributions[randint(0, 3)]
        if dist == "Bernoulli":
            parameter = rand()
            reward = randint(bernoulli_min_reward, bernoulli_max_reward)
            arms.append(Arm(dist, parameter, reward=reward))
        elif dist == "Normal":
            mean = randint(normal_min_mean, normal_max_mean)
            variance = randint(normal_min_var, normal_max_var)
            arms.append(Arm(dist, mean, variance))
        elif dist == "Uniform":
            parameters = sorted((randint(uniform_min, uniform_max), randint(
                uniform_min, uniform_max)))
            arms.append(Arm(dist, parameters[0], parameters[1]))

    return arms


def bernoulli_arms(number_of_arms, bernoulli_min_reward=1,
                   bernoulli_max_reward=100):
    """Create a list of Bernoulli distributed arms.

    Parameters
    ----------
    number_of_arms: int
                    Number of arms used in the multi-bandits problem.

    Other Parameters
    ----------------
    bernoulli_min_reward: float, optional
                          Lower bound of the reward given by a bernoulli
                          distributed arm.
    bernoulli_max_reward: float, optional
                          Upper bound of the reward given by a bernoulli
                          distributed arm.

    Returns
    -------
    list
        Contains the randomly generated Bernoulli arms.
    """
    arms = []
    for arm in range(number_of_arms):
        parameter = rand()
        reward = randint(bernoulli_min_reward, bernoulli_max_reward)
        arms.append(Arm("Bernoulli", parameter, reward=reward))

    return arms


class Arm:
    """Create an arm object in a multi-armed bandits problem."""

    def __init__(self, distribution, *parameters, reward=0):
        """
        Parameters
        ----------
        distribution : string
                    Distribution of the arm.
        parameters: tuple
                Parameters of the given distribution.
        reward: float, optional
                Reward of a parametric (e.g. Bernoulli) arm.

        Attributes
        ----------
        times_played: int
                      Number of times the arm has been pulled.
        mean_reward: float
                     Mean reward of the arm.
        lcb: float
             Lower confidence bound of the arm.
        ucb: float
             Upper confidence bound of the arm.
        """
        self._distribution = distribution
        self.reward = reward
        self.times_played = 0
        self.parameters = parameters
        self.mean_reward = 0

        if distribution == "Bernoulli":
            self.variance = parameters[0] * (1 - parameters[0])
        elif distribution == "Normal":
            self.mean = parameters[0]
            self.variance = parameters[1]
        elif distribution == "Uniform":
            self.min = parameters[0]
            self.max = parameters[1]
            self.variance = ((self.max - self.min) ** 2) / 12

        self.lcb = 0
        self.ucb = 0

    def __repr__(self):
        return f"{self._distribution}_{type(self).__name__}{self.parameters}"

    def _play(self):
        """Compute the reward from playing the arm."""
        self.times_played += 1

        if self._distribution == "Bernoulli":
            if binomial(1, self.parameters):
                return self.reward
            else:
                return 0
        elif self._distribution == "Normal":
            reward = max(normal(self.mean, self.variance), 0)
            return reward
        elif self._distribution == "Uniform":
            reward = max(uniform(self.min, self.max), 0)
            return reward

    def _update_lcb(self, delta):
        """Update the lower confidence bound."""
        self.lcb = self.mean_reward - (((2 * self.variance * log(1/delta)) /
                                       self.times_played) ** (1/2))

    def _update_ucb(self, delta):
        """Update the upper confidence bound."""
        self.ucb = self.mean_reward + (((2 * self.variance * log(1/delta)) /
                                       (self.times_played) ** (1/2)))


class Bandits:
    """An implementation of the multi-armed bandits problem.

    Methods
    ----------
    reset:  Reset the game.
    regret: Compute the regret for a given algorithm and history.
    """

    def __init__(self, arms):
        """
        Parameters
        ----------
        arms: list
              Arms used in the multi-armed bandits problem.

        Attributes
        ----------
        history: list
                 History of the previous rounds.
        """
        self.arms = tuple(arms)
        self._number_of_arms = len(arms)
        self.history = []

    def __repr__(self):
        return f"{self._number_of_arms}-Arm {type(self).__name__}"

    def _round(self, arm, arm_index):
        """Update the history and the mean reward of the current arm."""
        reward = arm._play()
        self.history.append((arm_index, reward))
        arm.mean_reward = (((arm.times_played - 1) * arm.mean_reward +
                            reward) / arm.times_played)

    # Used in the epsilon-greedy algorithm
    def _pull_uniform_explore(self):
        """Pull an arm uniformly at random."""
        random_int = randint(0, self._number_of_arms)
        arm = self.arms[random_int]
        self._round(arm, random_int)

    # Used in the round robin, follow the leader, epsilon greedy, and upper
    # confidence bound algorithms
    def _pull_cycle(self):
        """Pull every arm once."""
        for round in range(self._number_of_arms):
            arm = self.arms[round]
            self._round(arm, round)

    # Used in the follow the leader and epsilon greedy algorithms
    def _avg_reward(self):
        """Find the arm with the highest average reward, and its index."""
        avg_reward = []
        for arm in self.arms:
            avg_reward.append(arm.mean_reward)
        arm_index = argmax(avg_reward)
        # Note to self - how to deal with any tie cases? (could choose
        # uniformly; max() chooses the 1st max it finds)
        return self.arms[arm_index], arm_index

    # Used in the follow the leader and epsilon greedy algorithms
    def _pull_exploit(self):
        """Pull the arm with the highest mean reward."""
        arm, arm_index = self._avg_reward()
        self._round(arm, arm_index)

    # Used in the upper confidence bound algorithm
    def _greatest_ucb(self):
        """Find the arm with the greatest UCB, and its index."""
        ucb = []
        for arm in self.arms:
            ucb.append(arm.ucb)
        ucb_index = argmax(ucb)

        return self.arms[ucb_index], ucb_index

    # Used in the upper confidence bound algorithm
    def _pull_ucb(self, delta):
        """Pull the arm with the greatest UCB, and update the mean reward, LCB
        and UCB.
        """
        arm, arm_index = self._greatest_ucb()
        self._round(arm, arm_index)
        arm._update_lcb(delta)
        arm._update_ucb(delta)

    def reset(self):
        """Return the multi-armed bandits problem to its initial state."""
        self.history = []
        for arm in self.arms:
            arm.times_played = 0
            arm.mean_reward = 0
            arm.lcb = 0
            arm.ucb = 0

    # Uses the second definition of regret in the notes
    def regret(self):
        """Compute the regret of any algorithm."""
        regret = 0
        max_expected_reward = self._avg_reward()[0].mean_reward
        for arm in self.arms:
            regret += arm.times_played * (max_expected_reward -
                                          arm.mean_reward)

        return regret


class Game(Bandits):
    """
    An implementation of the multi-armed bandits problem.

    Methods
    -------
    round_robin:              Round Robin algorithm.

    follow_the_leader:        Follow the Leader algorithm.

    epsilon_greedy:           Epsilon-greedy algorithm.

    upper_confidence_bound:   Upper Confidence Bound algorithm.

    reset:                    Reset the game.

    regret:                   Compute the regret of a game using any algorithm.

    Notes
    -----
    reset should be used after using an algorithm and analysing a game.
    """

    def round_robin(self, cycles):
        """Apply the Round Robin algorithm."""
        cycle = 0
        while cycle < cycles:
            self._pull_cycle()
            cycle += 1
        return self.history

    def follow_the_leader(self, rounds):
        """Apply the Follow the Leader algorithm."""
        self._pull_cycle()
        for round in range(rounds - self._number_of_arms):
            self._pull_exploit()

        return self.history

    def epsilon_greedy(self, rounds):
        """Apply the epsilon-greedy algorithm."""
        self._pull_cycle()
        for round in range(rounds - self._number_of_arms):
            if binomial(1, rand()):
                self._pull_uniform_explore()
            else:
                self._pull_exploit()

        return self.history

    def upper_confidence_bound(self, rounds, delta):
        """Apply the Upper Confidence Bound algorithm."""
        self._pull_cycle()
        for round in range(self._number_of_arms):
            self.arms[round]._update_lcb(delta)
            self.arms[round]._update_ucb(delta)

        for round in range(rounds - self._number_of_arms):
            self._pull_ucb(delta)

        return self.history
