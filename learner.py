"""
Implements various learning algorithms for Multi-arm Bandits
"""

import numpy

def _greedy_choose(running_average, n_bandits, arms, eps):
    # Choose random bandits for exploration
    explore = numpy.array([numpy.random.binomial(1, eps) for _ in range(n_bandits)])
    random_arms = numpy.random.randint(0, arms, size=n_bandits)
    # Exploitation : Choose argmax of the existing estimated value
    # Exploration : Choose a random arm
    actions = (1 - explore) * numpy.argmax(running_average, axis=0) + \
              explore * random_arms
    return actions

def _softmax_choose(running_average, n_bandits, arms, tau):
    num = numpy.exp(running_average / tau)
    soft_output = num / numpy.sum(num, axis=0)
    actions = [numpy.random.choice(arms, 1, p=soft_output[:, i]) for i in range(n_bandits)]
    actions = numpy.array(actions)
    return actions

def learner(bandits, plays, alg="greedy", **kwargs):
    # pylint: disable=R0914
    """
    Generic method for learning 'optimal' actions for a multi-arm bandit
    """
    if alg not in {"greedy", "softmax"}:
        raise NotImplementedError
    if alg == "greedy":
        eps = 0.
        if 'eps' in kwargs:
            eps = kwargs['eps']
        if eps < 0:
            raise Exception("eps should be positive")

    if alg == "softmax":
        if 'tau' not in kwargs:
            raise Exception('The parameter tau should be provided with the softmax alg.')
        tau = kwargs['tau']
        if tau <= 0:
            raise Exception("The parameter tau should be positive")

    arms = bandits[0].arms
    n_bandits = len(bandits)
    running_average = numpy.zeros((arms, n_bandits))
    running_samples = numpy.zeros((arms, n_bandits))
    history_avg_value = []
    history_optimal_count = []

    for _ in range(plays):
        # Get next actions
        if alg == "greedy":
            actions = _greedy_choose(running_average, n_bandits, arms, eps)
        if alg == "softmax":
            actions = _softmax_choose(running_average, n_bandits, arms, tau)

        # Get feedback from the bandits for the actions taken
        true_value = [bandits[i].get_value(actions[i]) for i in range(n_bandits)]
        optimal_counts = [bandits[i].is_oracle(actions[i]) for i in range(n_bandits)]
        # Current estimate for the decisions
        estimated_value = running_average[actions, range(n_bandits)]
        # Update our estimate of value
        running_samples[actions, range(n_bandits)] += 1.
        step_size = (1. / running_samples[actions, range(n_bandits)])
        running_average[actions, range(n_bandits)] += step_size * (true_value - estimated_value)
        # Record the average value gained across bandits wrt our decisions
        history_avg_value.append(numpy.average(true_value))
        history_optimal_count.append(numpy.average(optimal_counts))

    return history_avg_value, history_optimal_count
