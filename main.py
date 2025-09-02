from src.rewarder import Rewarder
from src.state_updater import StateUpdater
from src.robot import Robot
from src.recycling import Recycling
from src.plotter import plot_rewards, plot_policy_heatmap

if __name__ == '__main__':
	# Simulation parameters
	NUM_TRAIN = 3
	NUM_EPOCHS = 50
	STEPS_PER_EPOCH = 1000

	# Robot Hyperparameters
	ALPHA = 0.1
	GAMMA = 0.9
	EPSILON = 0.1

	# Environment parameters
	R_SEARCH = 0.5
	R_WAIT = 0.5
	HIGH_TO_LOW_PROB = 0.3
	DEPLETE_PROB = 0.3

	rewarder = Rewarder(R_SEARCH, R_WAIT)
	state_updater = StateUpdater( HIGH_TO_LOW_PROB, DEPLETE_PROB, rewarder)

	robot = Robot(ALPHA, GAMMA, EPSILON)
	recycling = Recycling(state_updater, robot, num_runs_per_epoch=STEPS_PER_EPOCH)

	recycling.run_multiple_training(NUM_TRAIN, NUM_EPOCHS)

	plot_rewards()
	plot_policy_heatmap()