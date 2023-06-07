"""An implementation of the multi-armed bandit problem."""

from numpy import log, argmax, linspace
from numpy.random import rand, normal, randint, binomial, uniform, beta, \
    pareto
import matplotlib.pyplot as plt


class Arm:
    """Create an arm object in the multi-armed bandit problem."""

    def __init__(self, distribution, *parameters):
        """
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
        mean : float
               Mean of the arm's distribution.
        variance : float
                   Variance of the arm's distribution.
        ucb : float
              Current upper confidence bound of the arm.
        post_params : list
                      Posterior parameters for Thompson Sampling.
        """
        self._distribution = distribution
        self.times_played = 0
        self.parameters = parameters
        self.mean_reward = 0

        if distribution == "Bernoulli":
            self.mean = parameters[0]
            self.variance = parameters[0] * (1 - parameters[0])
            self.post_params = [1, 1]
        elif distribution == "Normal":
            self.mean = parameters[0]
            self.variance = parameters[1]
            self.post_params = [0, 1]
        elif distribution == "Uniform":
            self.min = parameters[0]
            self.max = parameters[1]
            self.post_params = [1, 1]
            self.mean = (self.min + self.max) / 2
            # Variance of a uniform distribution
            self.variance = ((self.max - self.min) ** 2) / 12

        self.lcb = 0
        self.ucb = 0

        # Used in the Thompson Sampling algorithm
        self._total_reward = 0

    def __repr__(self):
        if len(self.parameters) == 1:
            repr = (f"{self._distribution}_{type(self).__name__}"
                    f"({self.parameters[0]})")
            return repr
        else:
            repr = (f"{self._distribution}_{type(self).__name__}"
                    f"{self.parameters}")
            return repr

    def _play(self):
        """Compute the reward from playing an arm."""
        self.times_played += 1

        if self._distribution == "Bernoulli":
            # Draw a random sample from a bernoulli distribution
            reward = float(binomial(1, self.parameters))
            return reward
        elif self._distribution == "Normal":
            # Draw a random sample from a normal distribution
            reward = normal(self.mean, self.variance)
            return reward
        elif self._distribution == "Uniform":
            # Draw a random sample from a uniform distribution
            reward = float(uniform(self.min, self.max))
            return reward

    def _update_ucb(self, delta):
        """Update the upper confidence bound for the UCB algorithm."""
        self.ucb = self.mean_reward + (((2 * self.variance * log(1/delta)) /
                                       (self.times_played) ** (1/2)))


class Bandits:
    """
    Implement the multi-armed bandit problem.

    Methods
    -------
    reset :  Reset a game.
    regret : Compute the regret of an round in a multi-armed bandit game.
    """

    def __init__(self, arms):
        """
        Parameters
        ----------
        arms : list
               Arms used in the multi-armed bandit problem.

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

        # Used to compute regret
        self.optimal_mean = max([arm.mean for arm in arms])

    def __repr__(self):
        return f"{self._number_of_arms}-Arm {type(self).__name__}"

    def regret(self):
        """Compute the regret of a round."""
        arm_index = self.history[-1][0]
        mean_current_arm = self.arms[arm_index].mean
        return self.optimal_mean - mean_current_arm

    def _cumulative_regret(self):
        """Update the cumulative regret for an algorithm."""
        if self.cumulative_regret:
            cum_regret = self.cumulative_regret[-1] + self.regret()
            self.cumulative_regret.append(cum_regret)
        else:
            self.cumulative_regret.append(self.regret())

    # Used in the Round Robin, Follow the Leader, Epsilon-Greedy, and Upper
    # Confidence Bound algorithms
    def _round(self, arm, arm_index):
        """
        Update history, the mean reward of the current arm, and cumulative
        regret.
        """
        reward = arm._play()
        self.history.append((arm_index, reward))
        arm.mean_reward = (((arm.times_played - 1) * arm.mean_reward +
                            reward) / arm.times_played)
        self._cumulative_regret()

    # Used in the Round Robin, Follow the Leader, Epsilon-Greedy, and Upper
    # Confidence Bound algorithms
    def _pull_cycle(self):
        """Pull every arm once."""
        for round in range(self._number_of_arms):
            arm = self.arms[round]
            self._round(arm, round)

    # Used in the Follow the Leader, Epsilon-Greedy and Thompson Sampling
    # algorithms
    def _max_mean_reward(self):
        """Find the arm with the highest mean reward, and its index."""
        arm_index = argmax([arm.mean_reward for arm in self.arms])
        return self.arms[arm_index], arm_index

    # Used in the Follow the Leader and Epsilon-Greedy algorithms
    def _pull_exploit(self):
        """Pull the arm with the highest mean reward."""
        arm, arm_index = self._max_mean_reward()
        self._round(arm, arm_index)

    # Used in the Epsilon-Greedy algorithm
    def _pull_uniform_explore(self):
        """Pull an arm uniformly at random."""
        random_int = randint(0, self._number_of_arms)
        arm = self.arms[random_int]
        self._round(arm, random_int)

    # Used in the Upper Confidence Bound algorithm
    def _greatest_ucb(self):
        """Find the arm with the greatest UCB, and its index."""
        ucb = []
        for arm in self.arms:
            ucb.append(arm.ucb)
        ucb_index = argmax(ucb)
        return self.arms[ucb_index], ucb_index

    # Used in the Upper Confidence Bound algorithm
    def _pull_ucb(self, delta):
        """
        Pull the arm with the greatest UCB, and update the mean reward and UCB.
        """
        arm, arm_index = self._greatest_ucb()
        self._round(arm, arm_index)
        arm._update_ucb(delta)

    # Used in the Thompson Sampling algorithm
    def _thompson_sample(self):
        """Sample from the posterior distribution of each arm."""
        for arm in self.arms:
            if arm._distribution == "Bernoulli":
                arm.mean_reward = beta(arm.post_params[0],
                                       arm.post_params[1])
            elif arm._distribution == "Normal":
                arm.mean_reward = normal(arm.post_params[0],
                                         arm.post_params[1])
            elif arm._distribution == "Uniform":
                # Using Pareto Type I
                arm.mean_reward = (arm.post_params[1] *
                                   (pareto(arm.post_params[0]) + 1))

    # Used in the Thompson Sampling algorithm
    def _round_thompson(self, arm, arm_index):
        """
        Play current arm; Update history, posterior of current arm, and
        cumulative regret.

        Notes
        -----
        Bernoulli distributed arms have a Uniform prior;
        Normally distributed arms have a Gaussian prior;
        Uniformly distributed arms have a Pareto Type I prior.
        """
        reward = arm._play()
        self.history.append((arm_index, reward))
        if arm._distribution == "Bernoulli":
            arm._total_reward += reward  # Number of sucesses
            arm.post_params[0] = arm._total_reward + 1
            arm.post_params[1] = arm.times_played - arm._total_reward + 1
        elif arm._distribution == "Normal":
            arm.post_params[0] = arm.mean_reward
            arm.post_params[1] = 1 / arm.times_played
        elif arm._distribution == "Uniform":
            arm.post_params[0] += 1
            arm.post_params[1] = max(reward, arm.post_params[1])
        self._cumulative_regret()

    def reset(self):
        """Reset the multi-armed bandit problem to its initial state."""
        self.history = []
        self.cumulative_regret = []
        for arm in self.arms:
            arm.times_played = 0
            arm.mean_reward = 0
            arm.lcb = 0
            arm.ucb = 0
            arm._total_reward = 0


class Game(Bandits):
    """
    An implementation of the multi-armed bandit problem.

    Methods
    -------
    round_robin :             Round Robin algorithm.

    follow_the_leader :       Follow the Leader algorithm.

    epsilon_greedy :          Epsilon-greedy algorithm.

    upper_confidence_bound :  Upper Confidence Bound algorithm.

    thompson_sampling :       Thompson Sampling algorithm.

    reset :                   Reset a game.

    regret :                  Compute the regret of a round in a game.

    Notes
    -----
    reset should be used after using an algorithm and analysing a game.
    """

    def round_robin(self, cycles):
        """
        Apply the Round Robin algorithm.

        Parameters
        ----------
        cycles : int
                 Number of cycles of the Round Robin.

        Returns
        -------
        cumulative_regret : list
                            Cumulative regret of a game.
        """
        cycle = 0
        while cycle < cycles:
            self._pull_cycle()
            cycle += 1
        return self.cumulative_regret

    def follow_the_leader(self, rounds):
        """
        Apply the Follow the Leader algorithm.

        Parameters
        ----------
        rounds : int
                 Number of rounds.

        Returns
        -------
        cumulative_regret : list
                            Cumulative regret of a game.
        """
        self._pull_cycle()
        for round in range(rounds - self._number_of_arms):
            self._pull_exploit()
        return self.cumulative_regret

    def epsilon_greedy(self, rounds):
        """
        Apply the epsilon-greedy algorithm.

        Epsilon depends on the rounds in a game, with
        epsilon = number of arms / the current round of a game.

        Parameters
        ----------
        rounds : int
                 Number of rounds.

        Returns
        -------
        cumulative_regret : list
                            Cumulative regret of a game.
        """
        self._pull_cycle()
        for round in range(rounds - self._number_of_arms):
            epsilon = self._number_of_arms / (round + self._number_of_arms)
            if binomial(1, epsilon):
                self._pull_uniform_explore()
            else:
                self._pull_exploit()
        return self.cumulative_regret

    def upper_confidence_bound(self, rounds, delta):
        """
        Apply the Upper Confidence Bound algorithm.

        The delta typically used is 1 / (number of arms *
        (number of rounds) ** 2).

        Parameters
        ----------
        rounds : int
                 Number of rounds.
        delta : float
                Confidence parameter for the Upper Confidence Bound algorithm.

        Returns
        -------
        cumulative_regret : list
                            Cumulative regret of a game.
        """
        self._pull_cycle()
        for round in range(self._number_of_arms):
            # Update UCB after pulling each arm once
            self.arms[round]._update_ucb(delta)
        for round in range(rounds - self._number_of_arms):
            self._pull_ucb(delta)
        return self.cumulative_regret

    def thompson_sampling(self, rounds):
        """
        Apply the Thompson Sampling algorithm.

        Parameters
        ----------
        rounds : int
                 Number of rounds.

        Returns
        -------
        cumulative_regret : list
                            Cumulative regret of a game.
        """
        round = 0
        while round < rounds:
            self._thompson_sample()
            max_arm, arm_index = self._max_mean_reward()
            self._round_thompson(max_arm, arm_index)
            round += 1
        return self.cumulative_regret

    @staticmethod  # Note : Decorator. Do not call this when plotting
    def rr_regret_graph(history, rounds):  # TOBEUPDATED W CONFIDENCE INTERVALS
        """Plot a graph of cumulative regret vs rounds, for Round Robin."""
        round_list = list(range(1, rounds + 1))
        cum_regrets_rr = history[:rounds]
        plt.plot(round_list, cum_regrets_rr, 'b',
                 label='Round Robin')

    @staticmethod
    def ftl_regret_graph(history, rounds):
        """Plot graph of cumulative regret vs round, for Follow the Leader."""
        round_list = list(range(1, rounds + 1))
        cum_regrets_ftl = history[:rounds]
        plt.plot(round_list, cum_regrets_ftl, 'r',
                 label='Follow the Leader')

    @staticmethod
    def eg_regret_graph(history, rounds):
        """Plot cumulative regret vs rounds, for Epsilon-Greedy."""
        round_list = list(range(1, rounds + 1))
        cum_regrets_eg = history[:rounds]
        plt.plot(round_list, cum_regrets_eg, 'g',
                 label='Epsilon-Greedy')

    @staticmethod
    def ts_regret_graph(history, rounds):
        """Plot cumulative regret vs rounds, for Thompson Sampling."""
        round_list = list(range(1, rounds + 1))
        cum_regrets_ts = history[:rounds]
        plt.plot(round_list, cum_regrets_ts, color='orange',
                 label='Thompson Sampling')

    def plot_cumregret_graph(self, algorithms, rounds):  # call this
        """Plot all regret graphs for each algorithm on the same figure."""
        plt.figure(figsize=(14, 7))
        for algorithm in algorithms:
            if algorithm == "Round Robin":
                self.rr_regret_graph(self.round_robin(rounds), rounds)
            elif algorithm == "Follow the Leader":
                self.ftl_regret_graph(self.follow_the_leader(rounds),
                                      rounds)
            elif algorithm == "Epsilon-Greedy":
                self.eg_regret_graph(self.epsilon_greedy(rounds),
                                     rounds)
            elif algorithm == "Thompson Sampling":
                self.ts_regret_graph(self.thompson_sampling(rounds),
                                     rounds)
            else:
                raise TypeError("Must enter a valid algorithm name.")
        plt.xlabel('Number of Pulls')
        plt.ylabel('Cumulative Regret')
        plt.legend()
        plt.show()


distributions = ["Bernoulli", "Uniform", "Normal"]


def bernoulli_arms(number_of_arms, uniform_means=0):
    """
    Create a list of Bernoulli distributed arms.

    Parameters
    ----------
    number_of_arms : int
                     Number of arms used in the multi-armed bandit problem.
    uniform_means: int, default=0
                   If set to 1, the means of the arms are uniformly
                   distributed between 0 and 1.

    Returns
    -------
    list
        Contains the randomly generated Bernoulli arms.
    """
    arms = []
    if uniform_means:
        means = linspace(0, 1, number_of_arms)
        for parameter in means:
            arms.append(Arm("Bernoulli", parameter))
    else:
        for arm in range(number_of_arms):
            parameter = rand()
            arms.append(Arm("Bernoulli", parameter))
    return arms


def normal_arms(number_of_arms, uniform_means=0):
    """
    Create a list of normally distributed arms.

    The unknown mean value lies between 0 and 1; the variance is equal to 1.

    Parameters
    ----------
    number_of_arms : int
                     Number of arms used in the multi-armed bandit problem.
    uniform_means: int, default=0
                   If set to 1, the means of the arms are uniformly
                   distributed between 0 and 1.

    Returns
    -------
    list
        Contains the randomly generated normal arms.
    """
    arms = []
    variance = 1
    if uniform_means:
        means = linspace(0, 1, number_of_arms)
        for mean in means:
            arms.append(Arm("Normal", mean, variance))
    else:
        for arm in range(number_of_arms):
            mean = rand()
            arms.append(Arm("Normal", mean, variance))
    return arms


def uniform_arms(number_of_arms, uniform_means=0):
    """
    Create a list of Uniformly distributed arms.

    The minimum value is set to be 0; the maximum value lies
    between 0 and 2 such that the mean lies between 0 and 1.

    Parameters
    ----------
    number_of_arms : int
                     Number of arms used in the multi-armed bandit problem.
    uniform_means: int, default=0
                   If set to 1, the means of the arms are uniformly
                   distributed between 0 and 1.

    Returns
    -------
    list
        Contains the randomly generated Uniform arms.
    """
    arms = []
    min_value = 0
    if uniform_means:
        max = linspace(0, 2, number_of_arms)
        for max_value in max:
            arms.append(Arm("Uniform", min_value, max_value))
    else:
        for arm in range(number_of_arms):
            max = uniform(0, 2)
            arms.append(Arm("Uniform", min_value, max))
    return arms
