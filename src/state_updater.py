import numpy as np
from src.enums import *
from src.rewarder import Rewarder

class StateUpdater:
    """
    Class that manages the environment's dynamics for the Recycling Robot problem.

    This class determines the next state and the resulting reward based on the
    robot's current state and chosen action. It encapsulates the probabilistic
    rules of the environment.
    """

    def __init__(self, high_to_low_prob, deplete_prob, rewarder: Rewarder) -> None:
        """
        Initializes the state updater with the environment's probabilities.

        Args:
            high_to_low_prob (float): The probability of transitioning from a high to a
                                      low battery state when searching (α in the textbook).
            deplete_prob (float): The probability of the battery being depleted when
                                  searching in a low state (1-β in the textbook).
            rewarder (Rewarder): An object that provides the reward values.
        """
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
        """
        Calculates the next state and reward based on the given action.

        This method implements the transition function T(s, a, s') and the
        reward function R(s, a).

        Args:
            action: The action taken by the robot.
            seed (int, optional): A seed for the random number generator for reproducibility.

        Returns:
            A tuple containing the new RobotState and the received reward.
        """        
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
                    return (RobotStates.HIGH, -3)    
                return (RobotStates.LOW, self.rewarder.r_search)
            
            case LowActions.WAIT:
                return (RobotStates.LOW, self.rewarder.r_wait)
            
            case LowActions.RECHARGE:
                return (RobotStates.HIGH, 0)