# driver.py

import os
import numpy as np
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

# ====== Benchmarking Functions ====== #

def benchmark_quantum(n_runs=10, n_episodes=1000, batch_size=10):
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
            agent, env_name, n_episodes, batch_size, gamma, learning_rates, state_bounds
        )
        quantum_rewards_runs.append(rewards)
        quantum_agents.append(trained_agent)
    return quantum_rewards_runs, quantum_agents

def benchmark_simple_dqn(n_runs=10, n_episodes=1000, batch_size=64, gamma=0.99):
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
            lr=1e-3, buffer_capacity=10000, epsilon_start=1.0, epsilon_final=0.01, epsilon_decay=500
        )
        simple_rewards_runs.append(rewards)
        simple_agents.append(agent)
    return simple_rewards_runs, simple_agents

def benchmark_advanced_dqn(n_runs=10, n_episodes=1000, batch_size=64, gamma=0.99, target_update_freq=10):
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
            target_update_freq=target_update_freq
        )
        advanced_rewards_runs.append(rewards)
        advanced_agents.append(agent)
    return advanced_rewards_runs, advanced_agents

def plot_benchmark(quantum_rewards, simple_rewards, advanced_rewards):
    """
    Generates and saves a benchmark plot comparing reward curves with means and shaded standard deviations.
    """
    quantum_rewards = np.array(quantum_rewards)
    simple_rewards = np.array(simple_rewards)
    advanced_rewards = np.array(advanced_rewards)
    
    quantum_mean = np.mean(quantum_rewards, axis=0)
    quantum_std = np.std(quantum_rewards, axis=0)
    simple_mean = np.mean(simple_rewards, axis=0)
    simple_std = np.std(simple_rewards, axis=0)
    advanced_mean = np.mean(advanced_rewards, axis=0)
    advanced_std = np.std(advanced_rewards, axis=0)
    
    episodes = np.arange(1, len(quantum_mean) + 1)
    plt.figure(figsize=(10, 6))
    
    plt.plot(episodes, quantum_mean, label="Quantum Agent", color='blue')
    plt.fill_between(episodes, quantum_mean - quantum_std, quantum_mean + quantum_std, color='blue', alpha=0.2)
    
    plt.plot(episodes, simple_mean, label="Simple DQN", color='orange')
    plt.fill_between(episodes, simple_mean - simple_std, simple_mean + simple_std, color='orange', alpha=0.2)
    
    plt.plot(episodes, advanced_mean, label="Advanced DQN", color='green')
    plt.fill_between(episodes, advanced_mean - advanced_std, advanced_mean + advanced_std, color='green', alpha=0.2)
    
    plt.xlabel("Episodes")
    plt.ylabel("Total Reward")
    plt.title("Benchmark: Quantum vs Simple DQN vs Advanced DQN on CartPole")
    plt.legend()
    plt.grid(True)
    plt.savefig("outputs/benchmark_rewards.png")
    plt.show()


if __name__ == "__main__":
    ensure_output_dir()
    
    print("Starting Quantum Agent Benchmark...")
    quantum_rewards_runs, quantum_agents = benchmark_quantum(n_runs=10, n_episodes=1000, batch_size=10)
    
    print("Starting Simple DQN Benchmark...")
    simple_rewards_runs, simple_agents = benchmark_simple_dqn(n_runs=10, n_episodes=1000, batch_size=64, gamma=0.99)
    
    print("Starting Advanced DQN Benchmark...")
    advanced_rewards_runs, advanced_agents = benchmark_advanced_dqn(n_runs=10, n_episodes=1000, batch_size=64, gamma=0.99, target_update_freq=10)
    
    print("Generating Benchmark Plot...")
    plot_benchmark(quantum_rewards_runs, simple_rewards_runs, advanced_rewards_runs)
    
    # Select the best-performing quantum agent based on its final episode reward
    final_quantum_rewards = [rewards[-1] for rewards in quantum_rewards_runs]
    best_index = np.argmax(final_quantum_rewards)
    best_quantum_agent = quantum_agents[best_index]
    print(f"Best Quantum Agent: Run {best_index + 1} with final reward {final_quantum_rewards[best_index]}")

    # Generate animation using the best quantum agent
    total_reward, frames = evaluate_quantum(best_quantum_agent, "CartPole-v1", np.array([2.4, 2.5, 0.21, 2.5]), render=True)
    gif_path = "outputs/agent_animation.gif"
    import imageio
    imageio.mimsave(gif_path, frames, fps=30)
    print(f"Quantum Agent Animation saved to {gif_path} (Total Reward: {total_reward})")