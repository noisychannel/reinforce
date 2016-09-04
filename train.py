"""
Comparison of learning algorithms for reinforcement learning
"""

import numpy
import matplotlib.pyplot as plt

from bandit import Bandit
from learner import learner

ARMS = 10
N_BANDITS = 2000
PLAYS = 1000

# pylint: disable=C0103
bandits = [Bandit() for _ in range(N_BANDITS)]
average_oracle = [numpy.average([bandits[i].oracle_value for i in range(N_BANDITS)])] * PLAYS

history_value, history_optimal = learner(bandits, PLAYS, 'greedy')
history_value_0_01, history_optimal_0_01 = learner(bandits, PLAYS, 'greedy', eps=0.01)
history_value_0_1, history_optimal_0_1 = learner(bandits, PLAYS, 'greedy', eps=0.1)

average_oracle = [numpy.average([bandits[i].oracle_value for i in range(N_BANDITS)])] * PLAYS
plt.plot([i for i in range(len(history_value))], average_oracle, label="oracle")
plt.plot([i for i in range(len(history_value))], history_value, label="eps=0 (greedy)")
plt.plot([i for i in range(len(history_value))], history_value_0_01, label="eps=0.01")
plt.plot([i for i in range(len(history_value))], history_value_0_1, label="eps=0.1")
#plt.plot([i for i in range(len(history_value))], history_optimal, label="eps=0 (greedy)")
#plt.plot([i for i in range(len(history_value))], history_optimal_0_01, label="eps=0.01")
#plt.plot([i for i in range(len(history_value))], history_optimal_0_1, label="eps=0.1")

N_BANDITS = 1
bandits = [Bandit() for _ in range(N_BANDITS)]
t_history_value, t_history_optimal = learner(bandits, PLAYS, 'softmax', tau=1.)
t_history_value_0_1, t_history_optimal_0_1 = learner(bandits, PLAYS, 'softmax', tau=0.1)
t_history_value_0_01, t_history_optimal_0_01 = learner(bandits, PLAYS, 'softmax', tau=0.01)
plt.plot([i for i in range(len(history_value))], t_history_value, label="tau=1")
plt.plot([i for i in range(len(history_value))], t_history_value_0_1, label="tau=0.1")
plt.plot([i for i in range(len(history_value))], t_history_value_0_01, label="tau=0.01")

plt.legend(loc=4)
plt.show()
