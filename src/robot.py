import numpy as np
from src.enums import *
from rich.progress import track

class Robot:
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=0.1) -> None:
        self.learning_rate = alpha      # Taxa de aprendizado (alpha)
        self.initial_gamma = gamma
        self.initial_epsilon = epsilon  # Taxa de exploração (exploration rate)
        self.reset()
    
    def __initial_estimations(self):
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
        # Exploração: escolhe uma ação aleatória com probabilidade epsilon
        if np.random.rand() <= self.epsilon:
            if self.state == RobotStates.HIGH:
                action = (np.random.choice(list(HighActions)), False)
            else:
                action = (np.random.choice(list(LowActions)), False)
        # Explotação: escolhe a melhor ação conhecida com base na tabela Q
        else:
            if self.state == RobotStates.HIGH:
                action = (max(self.estimations[RobotStates.HIGH], key=self.estimations[RobotStates.HIGH].get), True)
            else:
                action = (max(self.estimations[RobotStates.LOW], key=self.estimations[RobotStates.LOW].get), True)
        
        self.actions.append(action[0])  # Guarda a ação tomada para o aprendizado
        self.greedy.append(action[1])
        return action[0]

    def update_state(self, new_state):
        self.state = new_state
        self.states.append(new_state)

    
    def update_policy(self, total_reward):
        if self.actions:
            last_action_index = len(self.actions) - 1
            
            if self.greedy[last_action_index]:
                last_state = self.states[last_action_index]
                last_action = self.actions[last_action_index]

                td_error_end = total_reward - self.estimations[last_state][last_action]
                self.estimations[last_state][last_action] += self.learning_rate * td_error_end

        for i in reversed(range(len(self.actions)-1)):
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
        self.epsilon = max(0.01, self.epsilon * 0.99)

    def reset(self):
        self.state = RobotStates.HIGH
        self.estimations = self.__initial_estimations()
        self.epsilon = self.initial_epsilon          # Taxa de exploração (exploration rate)
        self.gamma = self.initial_gamma
        self.actions = []
        self.greedy = []
        self.states = [self.state]