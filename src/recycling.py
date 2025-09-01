import numpy as np
from rich.progress import track
from enums import *
from rewarder import Rewarder
from state_updater import StateUpdater
from robot import Robot
import os

class Recycling:
    def __init__(self, state_updater, robot, num_runs_per_epoch=1000) -> None:
        self.state_updater = state_updater
        self.robot = robot
        self.training_counter = 0
        self.__num_runs_per_epoch = num_runs_per_epoch
        self.data_dir = "data" # Directory data

        os.makedirs(self.data_dir, exist_ok=True)
        self.rewards_path = os.path.join(self.data_dir, "rewards.txt")
        self.policy_path = os.path.join(self.data_dir, "optimal_policy.txt")

        with open(self.rewards_path, "a") as f:
            f.write(f"training,epoch,total_reward\n")

        with open(self.policy_path, "a") as f:
            f.write(f"training,low_recharge,low_search,low_wait,high_search,high_wait\n")

    def run_epoch(self, epoch_index) -> None:
        total_reward = 0
        for _ in track(range(self.__num_runs_per_epoch), description=f'Epoch {epoch_index}'):
            action = self.robot.act()
            new_state, reward = self.state_updater.get_new_state_and_reward(action)
            self.robot.update_state(new_state)
            total_reward += reward
        self.robot.update_policy(total_reward) # Robot receives a list of dictionaries {'action': action, 'reward': reward}
        with open(self.rewards_path, "a") as f:
            f.write(f"{self.training_counter},{epoch_index},{total_reward}\n")
    
    def run_training(self, epochs):
        for i in track(range(epochs), description=f'Running Training {self.training_counter}'):
            self.run_epoch(i)

    def run_multiple_training(self, num_train, epochs):
        for _ in track(range(num_train), description='Running Multiple Training'):
            self.run_training(epochs)
            with open(self.policy_path, "a") as f:
                f.write(f"{self.training_counter},\
                        {self.robot.q_table[RobotStates.HIGH[LowActions.RECHARGE]]},\
                        {self.robot.q_table[RobotStates.HIGH[LowActions.SEARCH]]},\
                        {self.robot.q_table[RobotStates.LOW[LowActions.WAIT]]},\
                        {self.robot.q_table[RobotStates.LOW[LowActions.SEARCH]]},\
                        {self.robot.q_table[RobotStates.LOW[LowActions.WAIT]]}\n")
            self.robot.restart()