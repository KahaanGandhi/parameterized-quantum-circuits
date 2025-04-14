import os
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
import argparse
import torch
from tqdm import tqdm

from models.DQN import DQNAgent
from models.PQC import PQCAgent, train_pqc, evaluate_pqc

# TODO: add early stopping to ArgParse
# TODO: add model.train() and model.evaluate() functions (utils...?)
# TODO: for official run, be able to vary multiple depths

def plot_benchmarks(results_dict, title="Benchmark", filename="outputs/benchmark_rewards.png"):
    """
    Plot benchmark curves from a dictionary mapping configuration names to a list of reward curves.
    """
    plt.figure(figsize=(10, 6))
    for label, runs in results_dict.items():
        # Compute the average and std of rewards per episode across runs
        avg_rewards = np.mean(np.array(runs), axis=0)
        std_rewards = np.std(np.array(runs), axis=0)
        episodes = np.arange(1, len(avg_rewards) + 1)
        plt.plot(episodes, avg_rewards, label=label)
        plt.fill_between(episodes, avg_rewards - std_rewards, avg_rewards + std_rewards, alpha=0.3)

    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    os.makedirs("outputs", exist_ok=True)
    plt.savefig(filename)
    plt.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="both", choices=["dqn", "pqc", "both"])
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    env_name = "CartPole-v1"
    temp_env = gym.make(env_name)
    n_observations = temp_env.observation_space.shape[0]
    n_actions = temp_env.action_space.n
    temp_env.close()

    num_runs = 20
    num_episodes = 500

    # Store runs from each configuration (for plotting later)
    dqn_results = {}  # key: configuration label -> list of reward curves (one per run)
    pqc_results = {}

    if args.mode in ["dqn", "both"]:
        dqn_configs = {
            "DQN (PyTorch docs)": lambda: DQNAgent(n_observations, n_actions, device, batch_size=128, lr=1e-4),
            "DQN (fine-tuned)":   lambda: DQNAgent(n_observations, n_actions, device, batch_size=64, lr=1e-3),
        }

        for label, agent_func in dqn_configs.items():
            dqn_results[label] = []
            
            for run in tqdm(range(num_runs), desc="{:18}".format(label)):
                agent = agent_func()
                rewards = agent.train(env_name, num_episodes=num_episodes)
                dqn_results[label].append(rewards)
        
        # # Evaluate each configuration (using a fresh agent per config)
        # for label, agent_func in dqn_configs.items():
        #     eval_reward = agent_func().evaluate(env_name)
        #     print(f"{label} Evaluation Reward: {eval_reward}")

    if args.mode in ["pqc", "both"]:
        pqc_configs = {
            "PQC (10 layers)": lambda: PQCAgent(n_qubits=4, n_layers=10, n_actions=n_actions, beta=1.0),
        }

        for label, agent_func in pqc_configs.items():
            pqc_results[label] = []
            for run in tqdm(range(num_runs), desc=f"{label:18}"):
                agent = agent_func()
                _, rewards = train_pqc(agent, env_name, n_episodes=num_episodes)
                pqc_results[label].append(rewards)
        
        # # Evaluate each configuration
        # for label, agent_func in pqc_configs.items():
        #     eval_reward = agent_func().evaluate(env_name)
        #     print(f"{label} Evaluation Reward: {eval_reward}")

    # If both types run, combine into one dictionary and set title accordingly
    if args.mode == "both":
        all_results = {}
        all_results.update(dqn_results)
        all_results.update(pqc_results)
        plot_benchmarks(all_results, title="Benchmark: DQN vs PQC on CartPole-v1")
    elif args.mode == "dqn":
        plot_benchmarks(dqn_results, title="Benchmark: DQN on CartPole-v1")
    elif args.mode == "pqc":
        plot_benchmarks(pqc_results, title="Benchmark: PQC on CartPole-v1")

if __name__ == "__main__":
    main()