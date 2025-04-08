# models/classical.py

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import gym
from collections import deque
import random
import copy


class DQN(nn.Module):
    """
    A Deep Q-Network with two hidden layers.
    """
    def __init__(self, state_dim, n_actions, hidden_dim=128):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions)
        )
    
    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    """
    Experience replay buffer for storing transitions.
    """
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states),
                np.array(actions),
                np.array(rewards),
                np.array(next_states),
                np.array(dones, dtype=np.float32))
    
    def __len__(self):
        return len(self.buffer)


def train_dqn_simple(env_name, state_dim, n_actions, n_episodes=500, batch_size=64, gamma=0.99,
                     lr=1e-3, buffer_capacity=10000, epsilon_start=1.0, epsilon_final=0.01, epsilon_decay=500):
    """
    Simple training loop for a DQN agent.
    Uses one network (no target network) and MSE loss.
    """
    env = gym.make(env_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    agent = DQN(state_dim, n_actions).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=lr)
    replay_buffer = ReplayBuffer(buffer_capacity)
    
    epsilon = epsilon_start
    episode_rewards = []
    
    for episode in range(n_episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            if np.random.random() < epsilon:
                action = np.random.randint(n_actions)
            else:
                with torch.no_grad():
                    q_values = agent(state_tensor)
                    action = q_values.argmax().item()
            
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            replay_buffer.push(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            
            if len(replay_buffer) >= batch_size:
                states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
                states_tensor = torch.FloatTensor(states).to(device)
                actions_tensor = torch.LongTensor(actions).unsqueeze(1).to(device)
                rewards_tensor = torch.FloatTensor(rewards).unsqueeze(1).to(device)
                next_states_tensor = torch.FloatTensor(next_states).to(device)
                dones_tensor = torch.FloatTensor(dones).unsqueeze(1).to(device)
                
                q_values = agent(states_tensor).gather(1, actions_tensor)
                with torch.no_grad():
                    next_q_values = agent(next_states_tensor).max(1, keepdim=True)[0]
                expected_q_values = rewards_tensor + gamma * next_q_values * (1 - dones_tensor)
                
                loss = F.mse_loss(q_values, expected_q_values)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
        epsilon = epsilon_final + (epsilon_start - epsilon_final) * np.exp(-episode / epsilon_decay)
        episode_rewards.append(total_reward)
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            print(f"Simple DQN: Episode {episode+1}/{n_episodes}, Avg Reward: {avg_reward:.2f}, Epsilon: {epsilon:.3f}")
    
    env.close()
    return agent, episode_rewards


def train_dqn_advanced(env_name, state_dim, n_actions, n_episodes=1000, batch_size=64, gamma=0.99,
                       lr=1e-3, buffer_capacity=10000, epsilon_start=1.0, epsilon_final=0.01, epsilon_decay=500,
                       target_update_freq=10):
    """
    Advanced training loop for a DQN agent.
    Uses a target network (updated periodically), Huber loss, and more robust replay.
    """
    env = gym.make(env_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    agent = DQN(state_dim, n_actions, hidden_dim=128).to(device)
    target_agent = copy.deepcopy(agent)
    target_agent.eval()
    
    optimizer = optim.Adam(agent.parameters(), lr=lr)
    replay_buffer = ReplayBuffer(buffer_capacity)
    
    epsilon = epsilon_start
    episode_rewards = []
    
    for episode in range(n_episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            if np.random.random() < epsilon:
                action = np.random.randint(n_actions)
            else:
                with torch.no_grad():
                    q_values = agent(state_tensor)
                    action = q_values.argmax().item()
            
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            replay_buffer.push(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            
            if len(replay_buffer) >= batch_size:
                states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
                states_tensor = torch.FloatTensor(states).to(device)
                actions_tensor = torch.LongTensor(actions).unsqueeze(1).to(device)
                rewards_tensor = torch.FloatTensor(rewards).unsqueeze(1).to(device)
                next_states_tensor = torch.FloatTensor(next_states).to(device)
                dones_tensor = torch.FloatTensor(dones).unsqueeze(1).to(device)
                
                q_values = agent(states_tensor).gather(1, actions_tensor)
                with torch.no_grad():
                    next_q_values = target_agent(next_states_tensor).max(1, keepdim=True)[0]
                expected_q_values = rewards_tensor + gamma * next_q_values * (1 - dones_tensor)
                
                loss = F.smooth_l1_loss(q_values, expected_q_values)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
        epsilon = epsilon_final + (epsilon_start - epsilon_final) * np.exp(-episode / epsilon_decay)
        episode_rewards.append(total_reward)
        
        if (episode + 1) % target_update_freq == 0:
            target_agent.load_state_dict(agent.state_dict())
        
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            print(f"Advanced DQN: Episode {episode+1}/{n_episodes}, Avg Reward: {avg_reward:.2f}, Epsilon: {epsilon:.3f}")
    
    env.close()
    return agent, episode_rewards


def evaluate_dqn(agent, env_name, state_dim, n_actions, render=False):
    """
    Evaluate a trained DQN agent.
    """
    env = gym.make(env_name, render_mode='rgb_array' if render else None)
    state, _ = env.reset()
    total_reward = 0
    done = False
    frames = []
    while not done:
        if render:
            frames.append(env.render())
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = agent(state_tensor)
            action = q_values.argmax().item()
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        state = next_state
        total_reward += reward
    env.close()
    return total_reward, frames
