import gymnasium as gym
import math
import random
import numpy as np
# Prevent deprecation warning
np.bool8 = np.bool_
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque, namedtuple
from itertools import count
import os
from tqdm import tqdm

# Set seed for reproducibility
seed = 777
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# Define a tuple for transitions
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

# ReplayMemory holds recent transitions for experience replay
class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# A simple feed-forward Q-network
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
    def __init__(self, n_observations, n_actions, memory_capacity=10000, batch_size=128,
                 gamma=0.99, eps_start=0.9, eps_end=0.05, eps_decay=1000, tau=0.005, lr=1e-4):
        self.n_observations = n_observations
        self.n_actions = n_actions
        self.batch_size = batch_size
        self.gamma = gamma
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.tau = tau
        self.steps_done = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy_net = DQN(n_observations, n_actions).to(self.device)
        self.target_net = DQN(n_observations, n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=lr, amsgrad=True)
        self.memory = ReplayMemory(memory_capacity)

    # Epsilon-greedy action selection
    def select_action(self, state):
        sample = random.random()
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * math.exp(-1. * self.steps_done / self.eps_decay)
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                return self.policy_net(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(self.n_actions)]], device=self.device, dtype=torch.long)

    # Optimize the Q-network using a batch of transitions
    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return None
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)),
                                      device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_action_values = self.policy_net(state_batch).gather(1, action_batch)
        next_state_values = torch.zeros(self.batch_size, device=self.device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0]
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

        # Soft-update target network
        for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(policy_param.data * self.tau + target_param.data * (1.0 - self.tau))
        return loss.item()

    # Train the agent for a specified number of episodes
    # If an episode reward reaches ≥500, the run is considered solved
    def train(self, env_name, num_episodes=500):
        env = gym.make(env_name, render_mode=None)
        episode_rewards = []
        pbar = tqdm(range(num_episodes), desc="[DQN] Episodes", leave=False)
        for i_episode in pbar:
            state, info = env.reset(seed=seed)
            state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            total_reward = 0
            while True:
                action = self.select_action(state)
                observation, reward, terminated, truncated, _ = env.step(action.item())
                done = terminated or truncated
                reward_tensor = torch.tensor([reward], device=self.device, dtype=torch.float32)
                next_state = None if done else torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
                self.memory.push(state, action, next_state, reward_tensor)
                state = next_state
                total_reward += reward
                self.optimize_model()
                if done:
                    break
            # Check if solved (>= 500)
            if total_reward >= 500:
                # Fill remainder with 500 and break early
                remaining = num_episodes - i_episode - 1
                episode_rewards.append(total_reward)
                episode_rewards.extend([500] * remaining)
                pbar.set_description(f"[DQN] Episode {i_episode+1}/{num_episodes} (Solved)")
                break
            else:
                episode_rewards.append(total_reward)
                pbar.set_description(f"[DQN] Episode {i_episode+1}/{num_episodes}")
        env.close()
        os.makedirs("outputs", exist_ok=True)
        np.savez("outputs/dqn_training_data.npz", episode_rewards=np.array(episode_rewards))
        # Only the reward history is saved for DQN
        return episode_rewards

    def evaluate(self, env_name):
        env = gym.make(env_name, render_mode='rgb_array')
        state, info = env.reset(seed=seed)
        total_reward = 0
        done = False
        while not done:
            state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            action = self.policy_net(state_tensor).max(1)[1].view(1, 1)
            state, reward, terminated, truncated, _ = env.step(action.item())
            total_reward += reward
            done = terminated or truncated
        env.close()
        return total_reward

# Uncomment the following to test training directly
# if __name__ == "__main__":
#     env_name = "CartPole-v1"
#     temp_env = gym.make(env_name)
#     n_observations = temp_env.observation_space.shape[0]
#     n_actions = temp_env.action_space.n
#     temp_env.close()
#     agent = DQNAgent(n_observations, n_actions)
#     rewards = agent.train(env_name, num_episodes=500)
#     print("DQN Training complete.")