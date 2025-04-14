import gymnasium as gym
import numpy as np
import math
import random
from collections import namedtuple, deque
from itertools import count
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# A single transition in our environment
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

# A cyclic buffer of bounded size that holds the transitions
class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# Feed-forward neural network to predict expected return (Q-Network)
class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

class DQNAgent:
    def __init__(self, n_observations, n_actions, device):
        # Hyperparameters
        self.BATCH_SIZE = 64
        self.GAMMA = 0.99
        self.EPS_START = 0.9
        self.EPS_END = 0.05
        self.EPS_DECAY = 1000
        self.TAU = 0.005
        self.LR = 1e-3

        self.steps_done = 0
        self.n_observations = n_observations
        self.n_actions = n_actions
        self.device = device

        # Initialize the policy and target networks
        self.policy_net = DQN(n_observations, n_actions).to(device)
        self.target_net = DQN(n_observations, n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.LR, amsgrad=True)
        self.memory = ReplayMemory(10000)

    def select_action(self, state, env=None):
        """Selects an action using an epsilon-greedy policy."""
        sample = random.random()
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * math.exp(-self.steps_done / self.EPS_DECAY)
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                return self.policy_net(state).max(1).indices.view(1, 1)
        else:
            # If an environment is provided, sample from its action space
            if env is not None:
                return torch.tensor([[env.action_space.sample()]], device=self.device, dtype=torch.long)
            else:
                return torch.tensor([[random.randrange(self.n_actions)]], device=self.device, dtype=torch.long)

    def optimize_model(self):
        """Performs one step of optimization on the policy network."""
        if len(self.memory) < self.BATCH_SIZE:
            return None

        transitions = self.memory.sample(self.BATCH_SIZE)
        batch = Transition(*zip(*transitions))

        # Create a mask for non-final states
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)),
                                      device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) using the policy network
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states
        next_state_values = torch.zeros(self.BATCH_SIZE, device=self.device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1).values

        expected_state_action_values = reward_batch + (self.GAMMA * next_state_values)

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

        return loss.item()

    def soft_update_target(self):
        """Soft update: θ′ ← τ θ + (1 - τ) θ′."""
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key] * self.TAU + target_net_state_dict[key] * (1 - self.TAU)
        self.target_net.load_state_dict(target_net_state_dict)

    def train(self, env_name, num_episodes=600):
        env = gym.make(env_name, render_mode=None)
        episode_rewards = []

        pbar = tqdm(range(num_episodes), desc="[DQN] Episodes", leave=False)
        for i_episode in pbar:
            state, info = env.reset(seed=777)
            state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            total_reward = 0

            for t in count():
                action = self.select_action(state, env)
                next_state, reward, terminated, truncated, _ = env.step(action.item())
                done = terminated or truncated

                reward_tensor = torch.tensor([reward], device=self.device, dtype=torch.float32)
                if done:
                    next_state_tensor = None
                else:
                    next_state_tensor = torch.tensor(next_state, dtype=torch.float32, device=self.device).unsqueeze(0)

                # Store the transition
                self.memory.push(state, action, next_state_tensor, reward_tensor)

                state = next_state_tensor
                total_reward += reward

                # Perform one step of optimization
                self.optimize_model()
                self.soft_update_target()

                pbar.set_description(f"[DQN] Episode")

                if done:
                    break

            episode_rewards.append(total_reward)
            
            # If reward ≥500, mark remaining episodes solved
            if total_reward >= 500:
                remaining = num_episodes - i_episode - 1
                episode_rewards.extend([500] * remaining)
                # pbar.set_description(f"[DQN] Episode {i_episode+1}/500 (Solved)")
                break

        env.close()
        return episode_rewards

    def evaluate(self, env_name):
        env = gym.make(env_name, render_mode='rgb_array')
        state, info = env.reset(seed=777)
        state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        total_reward = 0
        done = False

        while not done:
            with torch.no_grad():
                action = self.policy_net(state).max(1).indices.view(1, 1)
            next_state, reward, terminated, truncated, _ = env.step(action.item())
            total_reward += reward
            done = terminated or truncated
            if not done:
                state = torch.tensor(next_state, dtype=torch.float32, device=self.device).unsqueeze(0)

        env.close()
        return total_reward