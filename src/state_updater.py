import numpy as np
from enums import *
from rewarder import Rewarder

class StateUpdater:
    def __init__(self, high_to_low_prob, deplete_prob, rewarder: Rewarder) -> None:
        self.high_to_low_prob = high_to_low_prob
        self.deplete_prob = deplete_prob
        self.rewarder = rewarder

    @property
    def high_to_low_prob(self):
        return self._high_to_low_prob

    @high_to_low_prob.setter
    def high_to_low_prob(self, value):
        if value < 0:
            raise ValueError("high_to_low_prob must be non-negative.")
        self._high_to_low_prob = value

    @property
    def deplete_prob(self):
        return self._deplete_prob

    @deplete_prob.setter
    def deplete_prob(self, value):
        if value < 0:
            raise ValueError("deplete_prob must be non-negative.")
        self._deplete_prob = value
    
    def get_new_state_and_reward(self, action, seed=None):
        param = np.random.rand(seed=seed) if seed else np.random.rand()
        match action:
            case HighActions.SEARCH:
                if param < self.high_to_low_prob:
                    return (RobotStates.LOW, self.rewarder.r_search)
                return (RobotStates.HIGH, self.rewarder.r_search)

            case HighActions.WAIT:
                return (RobotStates.HIGH, self.rewarder.r_wait)

            case LowActions.SEARCH:
                if param < self.deplete_prob:
                    return (RobotStates.LOW, self.rewarder.r_search)
                return (RobotStates.HIGH, -3)
            
            case LowActions.WAIT:
                return (RobotStates.LOW, self.rewarder.r_wait)
            
            case LowActions.RECHARGE:
                return (RobotStates.HIGH, 0)