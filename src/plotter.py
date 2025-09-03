import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

DATA_DIR = os.path.join(os.path.dirname(__file__), '../data/')
PLOTS_DIR = os.path.join(os.path.dirname(__file__), '../data/plots')
REWARDS_FILE = os.path.join(DATA_DIR, 'rewards.txt')
POLICY_FILE = os.path.join(DATA_DIR, 'optimal_policy.txt')

def plot_rewards():
    """
    Reads the rewards data from rewards.txt and plots the learning curve.
    
    The plot shows the average total reward per epoch across multiple training
    runs, with the standard deviation shown as an error band.
    """
    
    try:
        df = pd.read_csv(REWARDS_FILE)
    except FileExistsError:
        print(f"Rewards file not found at: {REWARDS_FILE}")
        return
    
    plt.figure(figsize=(12,7))
    sns.lineplot(data=df, x='epoch', y='total_reward', errorbar='sd')

    plt.title('Total reward accumulated by epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Total Reward')
    plt.grid(True)

    output_path = os.path.join(PLOTS_DIR, 'reward_plot.png')
    plt.savefig(output_path)
    print(f"Rewards graphic saved in: {output_path}")
    # plt.show()


def plot_policy_heatmap():
    """
    Reads the final learned policy from optimal_policy.txt and visualizes it
    as a heatmap, showing the final Q-values for each state-action pair.
    """
        
    try:
        df = pd.read_csv(POLICY_FILE)
    except FileExistsError:
        print(f"Policy file not found at: {POLICY_FILE}")
        return
    
    finally_policy = df.iloc[-1]

    policy_matrix = pd.DataFrame({
        'Search': [finally_policy['high_search'], finally_policy['low_search']],
        'Wait': [finally_policy['high_wait'], finally_policy['low_wait']],
        'Recharge': [np.nan, finally_policy['low_recharge']], # Can't recharge while HIGH
    }, index=['High', 'Low'])

    plt.figure(figsize=(8,5))
    sns.heatmap(policy_matrix, annot=True, cmap='coolwarm_r')

    plt.title('Optimal policy - Final Q values')
    plt.xlabel('Action')
    plt.ylabel('Battery Status')

    output_path = os.path.join(PLOTS_DIR, 'policy_heatmap.png')
    plt.savefig(output_path)
    print(f"Policy heatmap saved in: {output_path}")
    # plt.show()

if __name__ == '__main__':
    if not os.path.exists(PLOTS_DIR):
        os.makedirs(PLOTS_DIR)
    
    plot_rewards()
    plot_policy_heatmap()

