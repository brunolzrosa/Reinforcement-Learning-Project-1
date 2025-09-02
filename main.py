from src.rewarder import Rewarder
from src.state_updater import StateUpdater
from src.robot import Robot
from src.recycling import Recycling

if __name__ == '__main__':
    NUM_EPOCHS = 50
    STEPS_PER_EPOCH = 1000

    rewarder = Rewarder(r_search=0.5, r_wait=0.2)
    state_updater = StateUpdater(high_to_low_prob=0.3, deplete_prob=0.3, rewarder=rewarder)
    recycling = Recycling(state_updater, Robot(), num_runs_per_epoch=STEPS_PER_EPOCH)
    recycling.run_multiple_training(3, NUM_EPOCHS)