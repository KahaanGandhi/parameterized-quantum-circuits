import numpy as np
np.bool8 = np.bool_  # Patch so Gym sees np.bool8

import os
import matplotlib.pyplot as plt
from PIL import Image
import imageio

from models.quantum import QuantumAgent, train_quantum, evaluate_quantum
from models.classical import train_dqn_simple, train_dqn_advanced, evaluate_dqn


def ensure_output_dir():
    """
    Helper function to create the outputs folder.
    """
    if not os.path.exists("outputs"):
        os.makedirs("outputs")


def save_circuit_configuration():
    """
    Creates a dummy quantum agent and saves its circuit visualization.
    """
    n_qubits = 4
    n_layers = 5
    n_actions = 2
    dummy_agent = QuantumAgent(n_qubits, n_layers, n_actions, beta=1.0)
    dummy_agent.pqc.quantum_circuit.visualize_circuit()
    print("Saved quantum circuit configuration to outputs/quantum_circuit_structure.png")


# ====== Benchmarking Functions ====== #

def benchmark_quantum(n_runs=10, n_episodes=1000, batch_size=10, solved_threshold=500.0, window=10):
    """
    Runs multiple training runs for the quantum agent.
    """
    quantum_rewards_runs = []
    quantum_agents = []
    for run in range(n_runs):
        print(f"Quantum Run {run+1}/{n_runs}")
        env_name = "CartPole-v1"
        state_bounds = np.array([2.4, 2.5, 0.21, 2.5])
        n_qubits = 4
        n_layers = 5
        n_actions = 2
        learning_rates = {'uploading': 0.01, 'output': 0.1}
        gamma = 1.0
        
        # Initialize and train the quantum agent
        agent = QuantumAgent(n_qubits, n_layers, n_actions, beta=1.0)
        trained_agent, rewards = train_quantum(
            agent, env_name, n_episodes, batch_size, gamma, learning_rates, state_bounds,
            solved_threshold=solved_threshold, window=window
        )
        quantum_rewards_runs.append(rewards)
        quantum_agents.append(trained_agent)
    return quantum_rewards_runs, quantum_agents


def benchmark_simple_dqn(n_runs=10, n_episodes=1000, batch_size=64, gamma=0.99, solved_threshold=500.0, window=10):
    """
    Runs multiple training runs for the simple DQN agent.
    """
    simple_rewards_runs = []
    simple_agents = []
    env_name = "CartPole-v1"
    import gym
    test_env = gym.make(env_name)
    state_dim = test_env.observation_space.shape[0]
    n_actions = test_env.action_space.n
    test_env.close()
    
    for run in range(n_runs):
        print(f"Simple DQN Run {run+1}/{n_runs}")
        agent, rewards = train_dqn_simple(
            env_name, state_dim, n_actions, n_episodes=n_episodes, batch_size=batch_size, gamma=gamma,
            lr=1e-3, buffer_capacity=10000, epsilon_start=1.0, epsilon_final=0.01, epsilon_decay=500,
            solved_threshold=solved_threshold, window=window
        )
        simple_rewards_runs.append(rewards)
        simple_agents.append(agent)
    return simple_rewards_runs, simple_agents


def benchmark_advanced_dqn(n_runs=10, n_episodes=1000, batch_size=64, gamma=0.99, target_update_freq=10, solved_threshold=500.0, window=10):
    """
    Runs multiple training runs for the advanced DQN agent.
    """
    advanced_rewards_runs = []
    advanced_agents = []
    env_name = "CartPole-v1"
    import gym
    test_env = gym.make(env_name)
    state_dim = test_env.observation_space.shape[0]
    n_actions = test_env.action_space.n
    test_env.close()
    
    for run in range(n_runs):
        print(f"Advanced DQN Run {run+1}/{n_runs}")
        agent, rewards = train_dqn_advanced(
            env_name, state_dim, n_actions, n_episodes=n_episodes, batch_size=batch_size, gamma=gamma,
            lr=1e-3, buffer_capacity=10000, epsilon_start=1.0, epsilon_final=0.01, epsilon_decay=500,
            target_update_freq=target_update_freq, solved_threshold=solved_threshold, window=window
        )
        advanced_rewards_runs.append(rewards)
        advanced_agents.append(agent)
    return advanced_rewards_runs, advanced_agents


def plot_benchmark(quantum_rewards, simple_rewards, advanced_rewards, total_episodes=1000):
    """
    Generates and saves a benchmark plot comparing reward curves.
    The curves are truncated to the episode where they solved the environment.
    A marker indicating the average solved episode is added.
    """
    # Compute average solved episode for each method (using run lengths)
    quantum_lengths = np.array([len(run) for run in quantum_rewards])
    simple_lengths = np.array([len(run) for run in simple_rewards])
    advanced_lengths = np.array([len(run) for run in advanced_rewards])
    
    quantum_solved = int(np.mean(quantum_lengths))
    simple_solved = int(np.mean(simple_lengths))
    advanced_solved = int(np.mean(advanced_lengths))
    
    def truncate_and_aggregate(reward_runs, solved_ep):
        truncated = []
        for run in reward_runs:
            if len(run) >= solved_ep:
                truncated.append(np.array(run[:solved_ep]))
            else:
                # Pad with the final value if solved early
                pad = np.full(solved_ep - len(run), run[-1])
                truncated.append(np.concatenate([np.array(run), pad]))
        truncated = np.array(truncated)
        mean_curve = np.mean(truncated, axis=0)
        std_curve = np.std(truncated, axis=0)
        return mean_curve, std_curve
    
    quantum_mean, quantum_std = truncate_and_aggregate(quantum_rewards, quantum_solved)
    simple_mean, simple_std = truncate_and_aggregate(simple_rewards, simple_solved)
    advanced_mean, advanced_std = truncate_and_aggregate(advanced_rewards, advanced_solved)
    
    episodes_q = np.arange(1, quantum_solved+1)
    episodes_s = np.arange(1, simple_solved+1)
    episodes_a = np.arange(1, advanced_solved+1)
    
    plt.figure(figsize=(10, 6))
    
    plt.plot(episodes_q, quantum_mean, label=f"Quantum Agent (solved at ~{quantum_solved} eps)", color='blue')
    plt.fill_between(episodes_q, quantum_mean - quantum_std, quantum_mean + quantum_std, color='blue', alpha=0.2)
    
    plt.plot(episodes_s, simple_mean, label=f"Simple DQN (solved at ~{simple_solved} eps)", color='orange')
    plt.fill_between(episodes_s, simple_mean - simple_std, simple_mean + simple_std, color='orange', alpha=0.2)
    
    plt.plot(episodes_a, advanced_mean, label=f"Advanced DQN (solved at ~{advanced_solved} eps)", color='green')
    plt.fill_between(episodes_a, advanced_mean - advanced_std, advanced_mean + advanced_std, color='green', alpha=0.2)
    
    plt.xlabel("Episodes")
    plt.ylabel("Total Reward")
    plt.title("Benchmark: QPC vs DQN on CartPole")
    plt.legend()
    plt.grid(True)
    plt.savefig("outputs/benchmark_rewards.png")
    plt.show()


if __name__ == "__main__":
    ensure_output_dir()
    save_circuit_configuration()
    
    print("Starting Quantum Agent Benchmark...")
    quantum_rewards_runs, quantum_agents = benchmark_quantum(n_runs=10, n_episodes=1000, batch_size=10, solved_threshold=500.0, window=10)
    
    print("Starting Simple DQN Benchmark...")
    simple_rewards_runs, simple_agents = benchmark_simple_dqn(n_runs=10, n_episodes=1000, batch_size=64, gamma=0.99, solved_threshold=500.0, window=10)
    
    print("Starting Advanced DQN Benchmark...")
    advanced_rewards_runs, advanced_agents = benchmark_advanced_dqn(n_runs=10, n_episodes=1000, batch_size=64, gamma=0.99, target_update_freq=10, solved_threshold=500.0, window=10)
    
    print("Generating Benchmark Plot...")
    plot_benchmark(quantum_rewards_runs, simple_rewards_runs, advanced_rewards_runs, total_episodes=1000)
    
    # Select the best-performing quantum agent based on its final episode reward
    final_quantum_rewards = [run[-1] for run in quantum_rewards_runs]
    best_index = np.argmax(final_quantum_rewards)
    best_quantum_agent = quantum_agents[best_index]
    print(f"Best Quantum Agent: Run {best_index + 1} with final reward {final_quantum_rewards[best_index]}")
    
    # Generate animation using the best quantum agent
    total_reward, frames = evaluate_quantum(best_quantum_agent, "CartPole-v1", np.array([2.4, 2.5, 0.21, 2.5]), render=True)
    gif_path = "outputs/agent_animation.gif"
    imageio.mimsave(gif_path, frames, fps=30)
    print(f"Quantum Agent Animation saved to {gif_path} (Total Reward: {total_reward})")