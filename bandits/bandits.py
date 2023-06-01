"""An implementation of the multi-armed bandits problem."""

from numpy.random import rand, normal, randint, binomial, uniform, beta
from numpy import log, argmax, arange
from itertools import accumulate
import matplotlib.pyplot as plt


class Arm:
    """Create an arm object in a multi-armed bandits problem."""

    def __init__(self, distribution, *parameters):
        """.

        Parameters
        ----------
        distribution : string
                    Distribution of the arm.
        parameters : tuple
                Parameters of the given distribution.

        Attributes
        ----------
        times_played : int
                      Number of times the arm has been pulled.
        mean_reward : float
                     Current mean reward of the arm.
        lcb : float
             Current lower confidence bound of the arm.
        ucb : float
             Current upper confidence bound of the arm.
        post_params : list
                     Posterior parameters for Thompson sampling.
        """
        self._distribution = distribution
        self.times_played = 0
        self.parameters = parameters
        self.mean_reward = 0

        if distribution == "Bernoulli":
            self.variance = parameters[0] * (1 - parameters[0])
            self.post_params = [1, 1]
        elif distribution == "Normal":
            self.mean = parameters[0]
            self.variance = parameters[1]
            self.post_params = [0, 1]
        elif distribution == "Uniform":
            self.min = parameters[0]
            self.max = parameters[1]
            # Variance of a uniform distribution
            self.variance = ((self.max - self.min) ** 2) / 12

        self.lcb = 0
        self.ucb = 0

        # Used in the Thompson sampling algorithm
        self._total_reward = 0

    def __repr__(self):
        """Canonical String Representation."""
        return f"{self._distribution}_{type(self).__name__}{self.parameters}"

    def _play(self):
        """Compute the reward from playing an arm."""
        self.times_played += 1

        if self._distribution == "Bernoulli":
            # Draw a random sample from a bernoulli distribution
            if binomial(1, self.parameters):
                return 1
            else:
                return 0
        elif self._distribution == "Normal":
            # Draw a random sample from a normal distribution
            reward = max(normal(self.mean, self.variance), 0)
            return reward
        elif self._distribution == "Uniform":
            # Draw a random sample from a uniform distribution
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
    -------
    reset :  Reset the game.
    regret : Compute the regret for a given algorithm and history.
    """

    def __init__(self, arms):
        """
        Parameters
        ----------
        arms : list
              Arms used in the multi-armed bandits problem.

        Attributes
        ----------
        history : list
                 History of the previous rounds.
        cumulative_regret : list
                           Cumulative regret of a game.
        """
        self.arms = tuple(arms)
        self._number_of_arms = len(arms)
        self.history = []
        self.cumulative_regret = []

    def __repr__(self):
        """Canonical String Representation."""
        return f"{self._number_of_arms}-Arm {type(self).__name__}"

    def _cumulative_regret(self):
        """Update the cumulative regret of an algorithm."""
        self.cumulative_regret.append(self.regret())

    # Used in the round robin, follow the leader, epsilon greedy, and upper
    # confidence bound algorithms
    def _round(self, arm, arm_index):
        """Update history, the mean reward of the current arm, and cumulative
        regret."""
        reward = arm._play()
        self.history.append((arm_index, reward))
        arm.mean_reward = (((arm.times_played - 1) * arm.mean_reward +
                            reward) / arm.times_played)
        self._cumulative_regret()

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

    # Used in the follow the leader, epsilon greedy and thompson sampling
    # algorithms
    def _avg_reward(self):
        """Find the arm with the highest average reward, and its index."""
        arm_index = argmax([arm.mean_reward for arm in self.arms])
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
        """.

        Pull the arm with the greatest UCB, and update the mean reward, LCB
        and UCB.
        """
        arm, arm_index = self._greatest_ucb()
        self._round(arm, arm_index)
        arm._update_lcb(delta)
        arm._update_ucb(delta)

    def _thompson_sample(self):
        """Sample from the posterior distribution of each arm."""
        for arm in self.arms:
            if arm._distribution == "Bernoulli":
                arm.mean_reward = beta(arm.post_params[0],
                                       arm.post_params[1])
            elif arm._distribution == "Normal":
                arm.mean_reward = normal(arm.post_params[0],
                                         arm.post_params[1])

    def _thompson_round(self, arm, arm_index):
        """Play current arm; Update history, posterior of current arm, and
        cumulative regret."""
        reward = arm._play()
        self.history.append((arm_index, reward))
        if arm._distribution == "Bernoulli":
            arm.post_params[0] += reward
            arm._total_reward += reward
            arm.post_params[1] = arm._total_reward - arm.post_params[0] + 2
        elif arm._distribution == "Normal":
            mu = arm.post_params[0]
            var = arm.post_params[1]
            M = (mu * arm.variance) / (var + arm.variance)
            V_squared = (arm.variance * var) / (arm.variance + var)

            arm.post_params[0] = M
            arm.post_params[1] = V_squared

        self._cumulative_regret()

    def reset(self):
        """Return the multi-armed bandits problem to its initial state."""
        self.history = []
        self.cumulative_regret = []
        for arm in self.arms:
            arm.times_played = 0
            arm.mean_reward = 0
            arm.lcb = 0
            arm.ucb = 0

    # Uses the second definition of regret in the notes
    def regret(self):
        """Compute the regret over any rounds."""
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
    round_robin :             Round Robin algorithm.

    follow_the_leader :       Follow the Leader algorithm.

    epsilon_greedy :          Epsilon-greedy algorithm.

    upper_confidence_bound :  Upper Confidence Bound algorithm.

    thompson_sampling :       Thompson sampling algorithm.

    reset :                   Reset the game.

    regret :                  Compute the regret of a game using any algorithm.

    Notes
    -----
    reset should be used after using an algorithm and analysing a game.
    """

    def round_robin(self, cycles):
        """Apply the Round Robin algorithm.

        Parameters
        ----------
        cycles : int
                Number of cycles of the round robin.

        Returns
        -------
        cumulative_regret : list
                           Cumulative regret of the game.
        """
        cycle = 0
        while cycle < cycles:
            self._pull_cycle()
            cycle += 1
        return self.cumulative_regret

    def follow_the_leader(self, rounds):
        """Apply the Follow the Leader algorithm.

        Parameters
        ----------
        rounds : int
                Number of rounds.

        Returns
        -------
        cumulative_regret : list
                           Cumulative regret of the game.
        """
        self._pull_cycle()
        for round in range(rounds - self._number_of_arms):
            self._pull_exploit()

        return self.cumulative_regret

    def epsilon_greedy(self, rounds):
        """Apply the epsilon-greedy algorithm.

        Parameters
        ----------
        rounds : int
                Number of rounds.

        Returns
        -------
        cumulative_regret : list
                           Cumulative regret of the game.
        """
        self._pull_cycle()
        for round in range(rounds - self._number_of_arms):
            # If epsilon gives heads
            if binomial(1, rand()):
                self._pull_uniform_explore()
            else:
                self._pull_exploit()

        return self.cumulative_regret

    def upper_confidence_bound(self, rounds, delta):
        """Apply the Upper Confidence Bound algorithm.

        Parameters
        ----------
        rounds : int
                Number of rounds.
        delta : float
               Confidence parameter for the upper confidence bound.

        Returns
        -------
        cumulative_regret : list
                           Cumulative regret of the game.
        """
        self._pull_cycle()
        for round in range(self._number_of_arms):
            self.arms[round]._update_lcb(delta)
            self.arms[round]._update_ucb(delta)

        for round in range(rounds - self._number_of_arms):
            self._pull_ucb(delta)

        return self.cumulative_regret

    def thompson_sampling(self, rounds):
        """Apply the Thompson sampling algorithm.

        Parameters
        ----------
        rounds : int
                Number of rounds.

        Returns
        -------
        cumulative_regret : list
                           Cumulative regret of the game.
        """
        round = 0
        while round < rounds:
            self._thompson_sample()
            max_arm, arm_index = self._avg_reward()
            self._thompson_round(max_arm, arm_index)
            round += 1

        return self.cumulative_regret

    @staticmethod  # Note : Decorator. Do not call this when plotting
    def regret_graph(self, algorithm, rounds, graph=None):  # TOBEUPDATED W CONFIDENCE INTERVALS
        """Plot a graph of cumulative regret versus Rounds Played."""
        round_array = arange(0, rounds, 1)
        if algorithm == "Round Robin":
            history = self.round_robin(rounds)
            regrets_rr = [t[1] for t in history]
            cum_regrets_rr = list(accumulate(regrets_rr))
            graph.plot(round_array, cum_regrets_rr, 'b.',
                       label='Round Robin')
        elif algorithm == "Follow the Leader":
            history = self.follow_the_leader(rounds)
            regrets_ftl = [t[1] for t in history]
            cum_regrets_ftl = list(accumulate(regrets_ftl))
            graph.plot(round_array, cum_regrets_ftl, 'r.',
                       label='Follow the Leader')
        elif algorithm == "Epsilon Greedy":
            history = self.epsilon_greedy(rounds)
            regrets_eg = [t[1] for t in history]
            cum_regrets_eg = list(accumulate(regrets_eg))
            graph.plot(round_array, cum_regrets_eg, 'g.',
                       label='Epsilon Greedy')
        elif algorithm == "Thompson Sampling":
            history = self.thompson_sampling(rounds)
            regrets_ts = [t[1] for t in history]
            cum_regrets_ts = list(accumulate(regrets_ts))
            graph.plot(round_array, cum_regrets_ts, 'o.',
                       label='Thompson Sampling')

    def plot_cumregret_graph(self, algorithms, rounds):  # call this
        """Plot all regret graphs for each algorithm on the same figure."""
        fig, axs = plt.subplots(figsize=(14, 7))
        for algorithm in algorithms:
            self.regret_graph(algorithm, rounds, graph=fig)
        axs.set_xlabel('Number of Pulls')
        axs.set_ylabel('Cumulative Regret')
        axs.legend()
        plt.show()


distributions = ["Bernoulli", "Uniform", "Normal"]


def arms(number_of_arms, normal_min_mean=3, normal_max_mean=7,
         normal_min_var=1, normal_max_var=3, uniform_min=0, uniform_max=10):
    """
    Create a list of 'randomly' distributed arms.

    Creates arms using the Bernoulli, Uniform and Normal distributions.

    Parameters
    ----------
    number_of_arms : int
                    Number of arms used in the multi-armed bandits problem.

    Other Parameters
    ----------------
    normal_min_mean : float, optional
                     Lower bound of the possible mean for a normally
                     distributed arm.
    normal_max_mean : float, optional
                     Upper bound of the possible mean for a normally
                     distributed arm.
    normal_min_var : float, optional
                    Lower bound of the possible variance for a normally
                    distributed arm.
    normal_max_var : float, optional
                    Upper bound of the possible variance for a normally
                    distributed arm.
    uniform_min : float, optional
                 Minimal value for a uniformly distributed arm.
    uniform_max : float, optional
                 Maximal value for a uniformly distributed arm.

    Returns
    -------
    list
        Contains the randomly generated arms.
    """
    arms = []
    for arm in range(number_of_arms):
        dist = distributions[randint(0, 3)]
        if dist == "Bernoulli":
            parameter = rand()
            arms.append(Arm(dist, parameter))
        elif dist == "Normal":
            mean = randint(normal_min_mean, normal_max_mean)
            variance = randint(normal_min_var, normal_max_var)
            arms.append(Arm(dist, mean, variance))
        elif dist == "Uniform":
            parameters = sorted((randint(uniform_min, uniform_max), randint(
                uniform_min, uniform_max)))
            arms.append(Arm(dist, parameters[0], parameters[1]))

    return arms


def bernoulli_arms(number_of_arms):
    """Create a list of Bernoulli distributed arms.

    Parameters
    ----------
    number_of_arms : int
                    Number of arms used in the multi-bandits problem.

    Returns
    -------
    list
        Contains the randomly generated Bernoulli arms.
    """
    arms = []
    for arm in range(number_of_arms):
        parameter = rand()
        arms.append(Arm("Bernoulli", parameter))

    return arms


def normal_arms(number_of_arms, normal_min_mean=3, normal_max_mean=7,
                normal_min_var=1, normal_max_var=3):
    """Create a list of Normally distributed arms.

    Parameters
    ----------
    number_of_arms : int
                    Number of arms used in the multi-bandits problem.

    Other Parameters
    ----------------
    normal_min_mean : float, optional
                     Lower bound of the possible mean for a normally
                     distributed arm.
    normal_max_mean : float, optional
                     Upper bound of the possible mean for a normally
                     distributed arm.
    normal_min_var : float, optional
                    Lower bound of the possible variance for a normally
                    distributed arm.
    normal_max_var : float, optional
                    Upper bound of the possible variance for a normally
                    distributed arm.

    Returns
    -------
    list
        Contains the randomly generated Normal arms.
    """
    arms = []
    for arm in range(number_of_arms):
        mean = randint(normal_min_mean, normal_max_mean)
        variance = randint(normal_min_var, normal_max_var)
        arms.append(Arm("Normal", mean, variance))

    return arms


def uniform_arms(number_of_arms, uniform_min=0, uniform_max=10):
    """Create a list of Uniformly distributed arms.

    Parameters
    ----------
    number_of_arms : int
                    Number of arms used in the multi-bandits problem.

    Other Parameters
    ----------------
    uniform_min : float, optional
                 Minimal value for a uniformly distributed arm.
    uniform_max : float, optional
                 Maximal value for a uniformly distributed arm.

    Returns
    -------
    list
        Contains the randomly generated Uniform arms.
    """
    arms = []
    for arm in range(number_of_arms):
        parameters = sorted((randint(uniform_min, uniform_max),
                             randint(uniform_min, uniform_max)))
        arms.append(Arm("Uniform", parameters[0], parameters[1]))

    return arms
