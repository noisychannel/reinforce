"""
Comparison of learning algorithms for reinforcement learning
"""

import numpy
import matplotlib.pyplot as plt

from bandit import Bandit
from learner import learner

ARMS = 10
#N_BANDITS = 2000
N_BANDITS=1
PLAYS = 100000

# pylint: disable=C0103
bandits = [Bandit() for _ in range(N_BANDITS)]
average_oracle = [numpy.average([bandits[i].oracle_value for i in range(N_BANDITS)])] * PLAYS

history_value, history_optimal = learner(bandits, PLAYS, 'greedy')
history_value_0_01, history_optimal_0_01 = learner(bandits, PLAYS, 'greedy', eps=0.01)
history_value_0_1, history_optimal_0_1 = learner(bandits, PLAYS, 'greedy', eps=0.1)

#t_history_value, t_history_optimal = softmax_learn(bandits, plays, 1.)
#t_history_value_10, t_history_optimal_10 = softmax_learn(bandits, plays, 10.)
#t_history_value_100, t_history_optimal_100 = softmax_learn(bandits, plays, 100.)
#print history_value
plt.plot([i for i in range(len(history_value))], average_oracle, label="oracle")
plt.plot([i for i in range(len(history_value))], history_value, label="eps=0 (greedy)")
plt.plot([i for i in range(len(history_value))], history_value_0_01, label="eps=0.01")
plt.plot([i for i in range(len(history_value))], history_value_0_1, label="eps=0.1")
#plt.plot([i for i in range(len(history_value))], t_history_value, label="tau=1")
#plt.plot([i for i in range(len(history_value))], t_history_value_10, label="tau=10")
#plt.plot([i for i in range(len(history_value))], t_history_value_100, label="tau=100")
#plt.plot([i for i in range(len(history_value))], history_optimal, label="eps=0 (greedy)")
#plt.plot([i for i in range(len(history_value))], history_optimal_0_01, label="eps=0.01")
#plt.plot([i for i in range(len(history_value))], history_optimal_0_1, label="eps=0.1")
plt.legend(loc=4)
plt.show()
