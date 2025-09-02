import numpy as np
from src.enums import *
from src.state_updater import StateUpdater

class Robot:
    def __init__(self, state_updater: StateUpdater) -> None:
        """
        Inicializa o Robô Agente.
        """
        self.state = RobotStates.HIGH   
        self.state_updater = state_updater
        self.total_reward = 0
        self.step_count = 0
        
        # A tabela Q armazena os valores das ações em cada estado.
        # Inicializar com 0.0 é uma prática comum.
        self.q_table = {
            RobotStates.HIGH: {
                HighActions.SEARCH: 0.0,
                HighActions.WAIT: 0.0
            },
            RobotStates.LOW: {
                LowActions.SEARCH: 0.0,
                LowActions.WAIT: 0.0,
                LowActions.RECHARGE: 0.0
            }
        }
        
        # Parâmetros do algoritmo de Q-learning
        self.epsilon = 0.1          # Taxa de exploração (exploration rate)
        self.learning_rate = 0.1    # Taxa de aprendizado (alpha)
        self.discount_factor = 0.9  # Fator de desconto (gamma)
        
        # Armazena a última ação tomada para a atualização do TD
        self.last_action = None

    def act(self):
        """
        Decide a próxima ação usando uma política epsilon-greedy.
        """
        # Exploração: escolhe uma ação aleatória com probabilidade epsilon
        if np.random.rand() <= self.epsilon:
            if self.state == RobotStates.HIGH:
                action = np.random.choice(list(HighActions))
            else:
                action = np.random.choice(list(LowActions))
        # Explotação: escolhe a melhor ação conhecida com base na tabela Q
        else:
            if self.state == RobotStates.HIGH:
                action = max(self.q_table[RobotStates.HIGH], key=self.q_table[RobotStates.HIGH].get)
            else:
                action = max(self.q_table[RobotStates.LOW], key=self.q_table[RobotStates.LOW].get)
        
        self.last_action = action  # Guarda a ação tomada para o aprendizado
        return action
    
    def update_q_value(self, reward: float, new_state: RobotStates):
        """
        Atualiza o valor Q para o par estado-ação anterior usando a regra de atualização TD(0).
        """
        # A atualização só ocorre se uma ação já tiver sido tomada.
        if self.last_action is not None:
            # Obtém o valor Q do par estado-ação que acabou de ocorrer (s, a)
            current_q = self._get_q_value(self.state, self.last_action)
            
            # Obtém o valor máximo de Q para o novo estado (s')
            next_max_q = self._get_max_q_value(new_state)
            
            # Calcula o "alvo" do TD: R + γ * max_a'(Q(s', a'))
            td_target = reward + self.discount_factor * next_max_q
            
            # Calcula o erro do TD: (Alvo TD) - Q(s, a)
            td_error = td_target - current_q
            
            # Atualiza o valor Q usando a taxa de aprendizado: Q(s, a) <- Q(s, a) + α * [erro TD]
            new_q = current_q + self.learning_rate * td_error
            self._set_q_value(self.state, self.last_action, new_q)

        # Transita para o novo estado para a próxima iteração
        self.state = new_state
        self.total_reward += reward
        self.step_count += 1
    
    def end_of_epoch(self):
        """
        Procedimentos de final de época, como a diminuição do epsilon.
        """
        # Diminui a taxa de exploração (epsilon) ao longo do tempo
        self.epsilon = max(0.01, self.epsilon * 0.99)
        
    def restart(self):
        """
        Reinicia o estado do robô para uma nova série de treinamento.
        """
        self.state = RobotStates.HIGH
        self.total_reward = 0
        self.step_count = 0
        self.last_action = None
        self.q_table = {
            RobotStates.HIGH: {
                HighActions.SEARCH: 0.0,
                HighActions.WAIT: 0.0
            },
            RobotStates.LOW: {
                LowActions.SEARCH: 0.0,
                LowActions.WAIT: 0.0,
                LowActions.RECHARGE: 0.0
            }
        }
        
    def _get_q_value(self, state: RobotStates, action) -> float:
        """Retorna o valor Q para um dado estado e ação."""
        return self.q_table[state][action]
    
    def _set_q_value(self, state: RobotStates, action, value: float):
        """Define o valor Q para um dado estado e ação."""
        self.q_table[state][action] = value
    
    def _get_max_q_value(self, state: RobotStates) -> float:
        """Encontra o valor Q máximo para um determinado estado."""
        return max(self.q_table[state].values())