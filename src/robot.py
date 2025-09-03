import numpy as np
from src.enums import *
from rich.progress import track

class Robot:
    """
    Class that implements a reinforcement learning agent that learns to solve the recycling problem.
    
    This specific implementation uses a trajectory-based, backward-pass update
    at the end of each epoch to update its policy estimations.
    """

    def __init__(self) -> None:
        """Initializes the robot agent."""
        self.state = RobotStates.HIGH
        self.estimations = self.__initial_estimations()
        self.epsilon = 0.1          # Exploration rate 
        self.learning_rate = 0.1    # alpha
        self.gamma = 0.9            #discount factor
        self.states = [self.state]
        self.actions = []
        self.greedy = []

    
    def __initial_estimations(self):
        """Helper method to create the initial Q-table with small default values."""
        
        return {
            RobotStates.HIGH: {
                HighActions.SEARCH: 0.01,
                HighActions.WAIT: 0.01
            },
            RobotStates.LOW: {
                LowActions.SEARCH: 0.01,
                LowActions.WAIT: 0.01,
                LowActions.RECHARGE: 0.01
            }
        }

    def act(self):
        """
        Selects an action using an epsilon-greedy policy and records it.

        Returns:
            The chosen action (enum member).
        """
   
        # Exploration: chooses a random action with probability epsilon
        if np.random.rand() <= self.epsilon:
            if self.state == RobotStates.HIGH:
                action = (np.random.choice(list(HighActions)), False)
            else:
                action = (np.random.choice(list(LowActions)), False)
        # Exploitation: choses the best known action on Q-table
        else:
            if self.state == RobotStates.HIGH:
                action = (max(self.estimations[RobotStates.HIGH], key=self.estimations[RobotStates.HIGH].get), True)
            else:
                action = (max(self.estimations[RobotStates.LOW], key=self.estimations[RobotStates.LOW].get), True)
        # Records both the action and whether it was a greedy choice.
        self.actions.append(action[0])  
        self.greedy.append(action[1])
        return action[0]

    def update_state(self, new_state):
        """
        Updates the robot's current state and records it in the history.

        Args:
            new_state (RobotStates): The new state to transition to.
        """

        self.state = new_state
        self.states.append(new_state)

    
    def update_policy(self, total_reward: float):
        """
        Updates the Q-table (estimations) by processing the recorded trajectory
        in reverse. This is called at the end of an epoch.

        Args:
            total_reward (float): The total reward accumulated during the entire epoch.
        """    
        if self.actions:
            last_action_index = len(self.actions) - 1
            
            if self.greedy[last_action_index]:
                last_state = self.states[last_action_index]
                last_action = self.actions[last_action_index]

                td_error_end = total_reward - self.estimations[last_state][last_action]
                self.estimations[last_state][last_action] += self.learning_rate * td_error_end

        for i in track(reversed(range(len(self.actions)-1)), "Updating Robot Policy"):
            if self.greedy[i]:
                current_state = self.states[i]
                current_action = self.actions[i]

                next_state = self.states[i+1]
                next_action = self.actions[i+1]

                immediate_reward = 0
                td_target = immediate_reward + self.gamma * self.estimations[next_state][next_action]
                td_error = td_target - self.estimations[current_state][current_action]

                self.estimations[current_state][current_action] += self.learning_rate * td_error

    def end_of_epoch_update(self):
        """Performs end-of-epoch maintenance, like decaying epsilon."""

        self.epsilon = max(0.01, self.epsilon * 0.99)

    def reset(self):
        """Resets the robot to its initial state for a new training run."""
        
        self.state = RobotStates.HIGH
        self.estimations = self.__initial_estimations()
        self.epsilon = 0.1          
        self.gamma = 0.9
        self.actions = []
        self.greedy = []
        self.states = [self.state]