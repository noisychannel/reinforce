"""
Implements a Multi-arm Bandit
"""

import numpy

class Bandit(object):
    """
    A multi-arm bandit
    """
    def __init__(self, arms=10):
        self.arms = arms
        self.actual_value = numpy.random.normal(
            numpy.random.normal(0., 1., 1),
            1., arms
        )
        self.oracle_value = numpy.max(self.actual_value)
        self.oracle_action = numpy.argmax(self.actual_value)


    def get_value(self, action):
        """
        Action is 0-indexed
        """
        if action >= self.arms:
            raise Exception("The action cannot be greater than the number of arms")
        return self.actual_value[action]

    def is_oracle(self, action):
        """
        Checks if the action is the oracle action
        """
        if action == self.oracle_action:
            return 1
        return 0
