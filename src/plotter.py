import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

PlOTS_DIR = os.path.join(os.path.dirname(__file__), '../data/plots')
REWARDS_FILE = os.path.join(PlOTS_DIR, 'rewards.txt')
POLICY_FILE = os.path.join(PlOTS_DIR, 'optimal_policy.txt')

def plot_rewards():
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
    plt.show()


def plot_policy_heatmap():
    try:
        df = pd.read_csv(POLICY_FILE)
    except FileExistsError:
        print(f"Policy file not found at: {REWARDS_FILE}")
        return
    
    finally_policy = df.iloc[-1]

    policy_matrix = pd.DataFrame({
        'Search': [finally_policy['high_search'], finally_policy['low_search']],
        'Wait': [finally_policy['high_wait'], finally_policy['low_wait']],
        'Recharge': [finally_policy['high_recharge'], finally_policy['low_recharge']],
    }, index=['High', 'Low'])

    plt.figure(figsize=(8,5))
    sns.heatmap(policy_matrix, annot=True, cmap='viridis', fmt='.2f')

    plt.title('Optimal policy - Final Q values')
    plt.xlabel('Action')
    plt.ylabel('Battery Status')

    output_path = os.path.join(PlOTS_DIR, 'policy_heatmap.png')
    plt.savefig(output_path)
    print(f"Policy heatmap saved in: {output_path}")
    plt.show()

if __name__ == '__main__':
    if not os.path.exists(PlOTS_DIR):
        os.makedirs(PLOTS_DIR)
    
    plot_rewards()
    plot_policy_heatmap()

