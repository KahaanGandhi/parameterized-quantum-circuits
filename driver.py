import os
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from tqdm import tqdm
import argparse
import torch

from models.DQN import DQNAgent
from models.PQC import PQCAgent, train_pqc, evaluate_pqc

# TODO: put a nested TQDM with desc="{:40}"
# TODO: add early stopping to ArgParse
# TODO: add model.train() and model.evaluate() functions (utils...?)
# TODO: Batch of 64 or 128, LR of 1e-3 or 1e-4 -- tune this

def plot_benchmarks(dqn_runs, pqc_runs, filename="outputs/benchmark_rewards.png"):
    plt.figure(figsize=(10, 6))
    if dqn_runs:
        avg_dqn = np.mean(np.array(dqn_runs), axis=0)
        std_dqn = np.std(np.array(dqn_runs), axis=0)
    if pqc_runs:
        avg_pqc = np.mean(np.array(pqc_runs), axis=0)
        std_pqc = np.std(np.array(pqc_runs), axis=0)
    
    # Use episode count from available data
    if dqn_runs:
        episodes = np.arange(1, len(avg_dqn) + 1)
    elif pqc_runs:
        episodes = np.arange(1, len(avg_pqc) + 1)
    else:
        episodes = np.array([])

    if dqn_runs:
        plt.plot(episodes, avg_dqn, label="DQN")
        plt.fill_between(episodes, avg_dqn - std_dqn, avg_dqn + std_dqn, alpha=0.3)
    if pqc_runs:
        plt.plot(episodes, avg_pqc, label="PQC")
        plt.fill_between(episodes, avg_pqc - std_pqc, avg_pqc + std_pqc, alpha=0.3)
    
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Benchmark: DQN vs PQC on CartPole-v1")
    plt.legend()
    plt.grid(True)
    os.makedirs("outputs", exist_ok=True)
    plt.savefig(filename)
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Benchmark DQN and/or PQC on CartPole-v1.")
    parser.add_argument(
        "--mode",
        type=str,
        default="both",
        choices=["dqn", "pqc", "both"],
        help='Run "dqn" only, "pqc" only, or "both" (default).'
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")

    env_name = "CartPole-v1"
    temp_env = gym.make(env_name)
    n_observations = temp_env.observation_space.shape[0]
    n_actions = temp_env.action_space.n
    temp_env.close()

    num_runs = 1
    num_episodes = 500

    dqn_run_rewards = []
    pqc_run_rewards = []

    if args.mode in ["dqn", "both"]:
        for run in tqdm(range(num_runs), desc="[DQN] Run"):
            agent = DQNAgent(n_observations, n_actions, device)
            rewards = agent.train(env_name, num_episodes=num_episodes)
            dqn_run_rewards.append(rewards)
    
    if args.mode in ["pqc", "both"]:
        for run in tqdm(range(num_runs), desc="[PQC] Run"):
            agent = PQCAgent(n_qubits=4, n_layers=10, n_actions=n_actions, beta=1.0)
            _, rewards = train_pqc(agent, env_name, n_episodes=num_episodes)
            pqc_run_rewards.append(rewards)
    
    # Optional evaluation of trained agents
    if args.mode in ["dqn", "both"]:
        dqn_eval_reward = DQNAgent(n_observations, n_actions, device).evaluate(env_name)
        print(f"[DQN] Evaluation Reward: {dqn_eval_reward}")
    if args.mode in ["pqc", "both"]:
        pqc_eval_reward = PQCAgent(n_qubits=4, n_layers=10, n_actions=n_actions, beta=1.0).evaluate(env_name)
        print(f"[PQC] Evaluation Reward: {pqc_eval_reward}")

    # Plot benchmarking results
    if args.mode == "both":
        plot_benchmarks(dqn_run_rewards, pqc_run_rewards)
    elif args.mode == "dqn":
        plot_benchmarks(dqn_run_rewards, [])
    elif args.mode == "pqc":
        plot_benchmarks([], pqc_run_rewards)

if __name__ == "__main__":
    main()