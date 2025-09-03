from rich.progress import track
from src.enums import *
import os

class Recycling:
    """ Class that manages the training process of the Recycling Robot

    This class orchestrates the simulation, running epochs and multiple training sessions,
    and handles the logging of results like rewards and learned policies to text files.

    """
    def __init__(self, state_updater, robot, num_runs_per_epoch: int=1000) -> None:
        """
        Initializes the training environment.

        Args:
            state_updater: An object that handles state transitions and rewards.
            robot (Robot): The learning agent.
            num_runs_per_epoch (int): The number of steps (actions) to run in each epoch.
        """    
        self.state_updater = state_updater
        self.robot = robot
        self.training_counter = 0
        self.__num_runs_per_epoch = num_runs_per_epoch
        self.data_dir = "data" # Directory data

        os.makedirs(self.data_dir, exist_ok=True)
        self.rewards_path = os.path.join(self.data_dir, "rewards.txt")
        self.policy_path = os.path.join(self.data_dir, "optimal_policy.txt")

        with open(self.rewards_path, "w") as f:
            f.write(f"training,epoch,total_reward\n")

        with open(self.policy_path, "w") as f:
            f.write(f"training,low_recharge,low_search,low_wait,high_search,high_wait\n")

    def run_epoch(self, epoch_index) -> None:
        """
        Executes a single training epoch consisting of multiple simulation steps.
        
        During each epoch, the robot performs actions, receives rewards, and updates
        its policy. Total reward for the epoch is recorded to file.
        
        Args:
            epoch_index: The current epoch number for progress tracking
        """
        total_reward = 0
        for _ in range(self.__num_runs_per_epoch):
            action = self.robot.act()
            new_state, reward = self.state_updater.get_new_state_and_reward(action)
            self.robot.update_state(new_state)
            total_reward += reward
        self.robot.update_policy(total_reward) # Robot receives a list of dictionaries {'action': action, 'reward': reward}
        self.robot.end_of_epoch_update()
        with open(self.rewards_path, "a") as f:
            f.write(f"{self.training_counter},{epoch_index},{total_reward}\n")
    
    def run_training(self, epochs):
                """
        Runs a full training session for a specified number of epochs.

        Args:
            epochs (int): The total number of epochs to run in this training session.
        """
        for i in track(range(epochs), description=f'Running Training {self.training_counter}'):
            self.run_epoch(i)
        self.training_counter+=1

    def run_multiple_training(self, num_train, epochs):
                """
        Runs multiple independent training sessions.

        This is useful for averaging results to get a smoother learning curve
        and see how robust the learning process is.

        Args:
            num_train (int): The number of independent training sessions to run.
            epochs (int): The number of epochs per training session.
        """

        for _ in track(range(num_train), description='Running Multiple Training'):
            self.run_training(epochs)
            with open(self.policy_path, "a") as f:
                f.write(f"{self.training_counter},\
                        {self.robot.estimations[RobotStates.LOW][LowActions.RECHARGE]},\
                        {self.robot.estimations[RobotStates.LOW][LowActions.SEARCH]},\
                        {self.robot.estimations[RobotStates.LOW][LowActions.WAIT]},\
                        {self.robot.estimations[RobotStates.HIGH][HighActions.SEARCH]},\
                        {self.robot.estimations[RobotStates.HIGH][HighActions.WAIT]}\n")
            self.robot.reset()