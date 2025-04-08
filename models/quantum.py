# models/quantum.py

import pennylane as qml
import torch
import torch.nn as nn
import numpy as np
import gym
import matplotlib.pyplot as plt
from collections import defaultdict

class QuantumCircuit:
    """
    Generates a quantum circuit for the data re-uploading PQC approach.
    """
    def __init__(self, n_qubits, n_layers):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        # Define the quantum device
        self.dev = qml.device("default.qubit", wires=self.n_qubits)
        self.n_params_per_layer = self.n_qubits * 3  # Three rotation gates per qubit
        self.n_params = (self.n_layers + 1) * self.n_params_per_layer  # Additional final layer
        
        # Create a QNode with a torch interface
        self.circuit = qml.QNode(self._circuit_impl, self.dev, interface="torch")
        
    def _one_qubit_rotation(self, qubit, params):
        """
        Apply rotation operators around the three Cartesian axes to one qubit.
        """
        qml.RX(params[0], wires=qubit)
        qml.RY(params[1], wires=qubit)
        qml.RZ(params[2], wires=qubit)
        
    def _entangling_layer(self):
        """
        Apply a CZ gate between each qubit and its neighbor (circular connectivity).
        """
        for i in range(self.n_qubits):
            qml.CZ(wires=[i, (i + 1) % self.n_qubits])
    
    def _circuit_impl(self, params, inputs):
        """
        Defines the circuit: alternating variational and encoding layers.
        Returns the expectation value of the tensor-product observable.
        """
        # Reshape parameters and inputs for clarity
        params = params.reshape(self.n_layers + 1, self.n_qubits, 3)
        inputs = inputs.reshape(self.n_layers, self.n_qubits)
        
        for l in range(self.n_layers):
            # Variational layer: apply rotation gates
            for i in range(self.n_qubits):
                self._one_qubit_rotation(i, params[l, i])
            # Entangling layer
            self._entangling_layer()
            # Encoding layer: embed input data via RX rotations
            for i in range(self.n_qubits):
                qml.RX(inputs[l, i], wires=i)
        
        # Final variational layer
        for i in range(self.n_qubits):
            self._one_qubit_rotation(i, params[-1, i])
        
        # Return the expectation of the tensor product of PauliZ on all qubits
        # (CartPole has 4 state dimensions, so we have 4 qubit system.)
        return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1) @ qml.PauliZ(2) @ qml.PauliZ(3))

    def visualize_circuit(self):
        """
        Draws and saves a visualization of the quantum circuit.
        """
        dummy_params = np.zeros((self.n_layers + 1, self.n_qubits, 3))
        dummy_inputs = np.zeros((self.n_layers, self.n_qubits))
        params_flat = torch.tensor(dummy_params.flatten(), dtype=torch.float32)
        inputs_flat = torch.tensor(dummy_inputs.flatten(), dtype=torch.float32)
        fig, ax = qml.draw_mpl(self.circuit)(params_flat, inputs_flat)
        plt.savefig("outputs/quantum_circuit_structure.png")
        plt.close(fig)
        return fig, ax


class ReUploadingPQC(nn.Module):
    """
    Implements the data re-uploading PQC. This handles trainable variational parameters and 
    input scaling, and then feeds the processed inputs into the quantum circuit.
    """
    def __init__(self, n_qubits, n_layers, activation=nn.Identity()):
        super(ReUploadingPQC, self).__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.activation = activation
        
        # Create the quantum circuit
        self.quantum_circuit = QuantumCircuit(n_qubits, n_layers)
        # Initialize variational parameters (theta) and scaling factors (lambda)
        self.theta = nn.Parameter(torch.rand(1, self.quantum_circuit.n_params) * np.pi)
        self.lmbd = nn.Parameter(torch.ones(self.n_qubits * self.n_layers))
        
    def forward(self, x):
        batch_size = x.shape[0]
        # Repeat input data for each layer and scale using the trainable lambda parameters
        x_tiled = x.repeat(1, self.n_layers)
        scaled_inputs = torch.einsum("i,ji->ji", self.lmbd, x_tiled)
        squashed_inputs = self.activation(scaled_inputs)
        # Repeat theta parameters for each batch element
        thetas = self.theta.repeat(batch_size, 1)
        results = torch.zeros(batch_size, 1, device=x.device)
        for i in range(batch_size):
            results[i] = self.quantum_circuit.circuit(thetas[i], squashed_inputs[i])
        return results


class Alternating(nn.Module):
    """
    A simple linear layer with alternating-sign initialization.
    """
    def __init__(self, output_dim):
        super(Alternating, self).__init__()
        self.w = nn.Parameter(torch.Tensor([[(-1) ** i for i in range(output_dim)]]))
        
    def forward(self, x):
        return torch.matmul(x, self.w)


class QuantumAgent(nn.Module):
    """
    Full quantum RL agent that combines the PQC with an output layer.
    Agent outputs action probabilities through a softmax.
    """
    def __init__(self, n_qubits, n_layers, n_actions, beta=1.0):
        super(QuantumAgent, self).__init__()
        self.pqc = ReUploadingPQC(n_qubits, n_layers)
        self.alternating = Alternating(n_actions)
        self.beta = beta
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        x = self.pqc(x)
        x = self.alternating(x)
        x = x * self.beta
        return self.softmax(x)


def train_quantum(agent, env_name, n_episodes, batch_size, gamma, learning_rates, state_bounds):
    """
    Training loop for the quantum agent using a REINFORCE-style update.
    Returns the trained agent and the list of episode rewards.
    """
    optimizer_upload = torch.optim.Adam(agent.pqc.parameters(), lr=learning_rates['uploading'], amsgrad=True)
    optimizer_output = torch.optim.Adam(agent.alternating.parameters(), lr=learning_rates['output'], amsgrad=True)
    episode_rewards = []
    env = gym.make(env_name)
    
    for episode in range(n_episodes):
        state, _ = env.reset()
        done = False
        rewards_list = []
        states_list = []
        actions_list = []
        
        while not done:
            state_norm = np.asarray(state).flatten() / state_bounds
            state_tensor = torch.FloatTensor(state_norm).unsqueeze(0)
            with torch.no_grad():
                action_probs = agent(state_tensor).numpy()[0]
            action = np.random.choice(len(action_probs), p=action_probs)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            states_list.append(state_norm)
            actions_list.append(action)
            rewards_list.append(reward)
            state = next_state

        # Compute discounted returns
        R = 0
        returns = []
        for r in rewards_list[::-1]:
            R = r + gamma * R
            returns.insert(0, R)
        returns = np.array(returns)
        if len(returns) > 1:
            returns = (returns - np.mean(returns)) / (np.std(returns) + 1e-8)
        
        # Convert lists to torch tensors
        states_tensor = torch.FloatTensor(np.array(states_list))
        actions_tensor = torch.LongTensor(actions_list)
        returns_tensor = torch.FloatTensor(returns)
        
        # Zero gradients
        optimizer_upload.zero_grad()
        optimizer_output.zero_grad()
        
        # Forward pass
        logits = agent(states_tensor)
        action_masks = torch.zeros_like(logits)
        for i, a in enumerate(actions_tensor):
            action_masks[i, a] = 1
        p_actions = (logits * action_masks).sum(dim=1)
        log_probs = torch.log(p_actions)
        loss = -torch.sum(log_probs * returns_tensor) / states_tensor.shape[0]
        
        # Backward pass
        loss.backward()
        optimizer_upload.step()
        optimizer_output.step()
        
        episode_rewards.append(sum(rewards_list))
        if (episode + 1) % batch_size == 0:
            avg_reward = np.mean(episode_rewards[-batch_size:])
            print(f"Quantum: Episode {episode+1}/{n_episodes}, Avg Reward: {avg_reward:.2f}, Loss: {loss.item():.4f}")
    
    env.close()
    return agent, episode_rewards


def evaluate_quantum(agent, env_name, state_bounds, render=False):
    """
    Evaluate a trained quantum agent in the environment.
    Set render as True to return a list of frames to create an animation.
    """
    env = gym.make(env_name, render_mode='rgb_array' if render else None)
    state, _ = env.reset()
    total_reward = 0
    done = False
    frames = []
    while not done:
        if render:
            frames.append(env.render())
        state_norm = np.asarray(state).flatten() / state_bounds
        state_tensor = torch.FloatTensor(state_norm).unsqueeze(0)
        with torch.no_grad():
            action_probs = agent(state_tensor).numpy()[0]
        action = np.argmax(action_probs)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward
        state = next_state
    env.close()
    return total_reward, frames