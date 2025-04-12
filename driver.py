import os
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from tqdm import tqdm

from models.DQN import DQNAgent
from models.PQC import PQCAgent, train_pqc, evaluate_pqc

def plot_benchmarks(dqn_runs, pqc_runs, filename="outputs/benchmark_rewards.png"):
    plt.figure(figsize=(10, 6))
    avg_dqn = np.mean(np.array(dqn_runs), axis=0)
    std_dqn = np.std(np.array(dqn_runs), axis=0)
    avg_pqc = np.mean(np.array(pqc_runs), axis=0)
    std_pqc = np.std(np.array(pqc_runs), axis=0)
    episodes = np.arange(1, len(avg_dqn)+1)
    plt.plot(episodes, avg_dqn, label="DQN")
    plt.fill_between(episodes, avg_dqn-std_dqn, avg_dqn+std_dqn, alpha=0.3)
    plt.plot(episodes, avg_pqc, label="PQC")
    plt.fill_between(episodes, avg_pqc-std_pqc, avg_pqc+std_pqc, alpha=0.3)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Benchmark: DQN vs PQC on CartPole-v1")
    plt.legend()
    plt.grid(True)
    os.makedirs("outputs", exist_ok=True)
    plt.savefig(filename)
    plt.show()

def main():
    env_name = "CartPole-v1"
    # Get environment dimensions for DQN
    temp_env = gym.make(env_name)
    n_observations = temp_env.observation_space.shape[0]
    n_actions = temp_env.action_space.n
    temp_env.close()
    
    num_runs = 20
    num_episodes = 500
    
    dqn_run_rewards = []
    for run in tqdm(range(num_runs), desc="[DQN] Run Counter"):
        agent = DQNAgent(n_observations, n_actions)
        rewards = agent.train(env_name, num_episodes=num_episodes)
        dqn_run_rewards.append(rewards)
    
    pqc_run_rewards = []
    for run in tqdm(range(num_runs), desc="[PQC] Run Counter"):
        agent = PQCAgent(n_qubits=4, n_layers=10, n_actions=n_actions, beta=1.0)
        _, rewards = train_pqc(agent, env_name, n_episodes=num_episodes)
        pqc_run_rewards.append(rewards)
    
    # Evaluate final agents (optional)
    dqn_eval_reward = DQNAgent(n_observations, n_actions).evaluate(env_name)
    pqc_eval_reward = PQCAgent(n_qubits=4, n_layers=10, n_actions=n_actions, beta=1.0).evaluate(env_name)
    print(f"[DQN] Evaluation Reward: {dqn_eval_reward}")
    print(f"[PQC] Evaluation Reward: {pqc_eval_reward}")
    
    plot_benchmarks(dqn_run_rewards, pqc_run_rewards)

if __name__ == "__main__":
    main()