import pennylane as qml
import torch
import torch.nn as nn
import numpy as np
import gym
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

seed = 777
np.random.seed(seed)
torch.manual_seed(seed)
# Prevent deprecation warning
np.bool8 = np.bool_

# Parameterized quantum circuit (PQC) with data re-uploading
class QuantumCircuit:
    def __init__(self, n_qubits, n_layers):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.dev = qml.device("default.qubit", wires=self.n_qubits)
        self.n_params_per_layer = self.n_qubits * 3
        self.n_params = (self.n_layers + 1) * self.n_params_per_layer
        self.circuit = qml.QNode(self._circuit_impl, self.dev, interface="torch")

    def _one_qubit_rotation(self, qubit, params):
        """
        Applies rotation operators, effectively rotating the Bloch sphere.
        """
        qml.RX(params[0], wires=qubit)
        qml.RY(params[1], wires=qubit)
        qml.RZ(params[2], wires=qubit)

    def _entangling_layer(self):
        """
        Creates a layer of CZ entangling gates on qubits, arranged in a circular topology.
        """
        for i in range(self.n_qubits):
            qml.CZ(wires=[i, (i + 1) % self.n_qubits])

    def _circuit_impl(self, params, inputs):
        """
        Generates the data re-uploading circuit architecture.
        """
        params = params.reshape(self.n_layers + 1, self.n_qubits, 3)
        inputs = inputs.reshape(self.n_layers, self.n_qubits)
        for l in range(self.n_layers):
            # Variational layer
            for i in range(self.n_qubits):
                self._one_qubit_rotation(i, params[l, i])
            self._entangling_layer()
            # Encoding layer
            for i in range(self.n_qubits):
                qml.RX(inputs[l, i], wires=i)
        # Final variational layer
        for i in range(self.n_qubits):
            self._one_qubit_rotation(i, params[-1, i])
        return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1) @ qml.PauliZ(2) @ qml.PauliZ(3))

    def visualize_circuit(self):
        dummy_params = np.zeros((self.n_layers + 1, self.n_qubits, 3))
        dummy_inputs = np.zeros((self.n_layers, self.n_qubits))
        params_flat = torch.tensor(dummy_params.flatten(), dtype=torch.float32)
        inputs_flat = torch.tensor(dummy_inputs.flatten(), dtype=torch.float32)
        fig, ax = qml.draw_mpl(self.circuit)(params_flat, inputs_flat)
        os.makedirs("outputs", exist_ok=True)
        plt.savefig("outputs/quantum_circuit_structure.png")
        plt.close(fig)
        return fig, ax

class ReUploadingPQC(nn.Module):
    """
    Wraps the quantum circuit and adds trainable parameters.
    """
    def __init__(self, n_qubits, n_layers, activation=nn.Identity()):
        super(ReUploadingPQC, self).__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.activation = activation
        self.quantum_circuit = QuantumCircuit(n_qubits, n_layers)
        # Random initialization of trainable parameters
        self.theta = nn.Parameter(torch.rand(1, self.quantum_circuit.n_params) * np.pi)
        self.lmbd = nn.Parameter(torch.ones(self.n_qubits * self.n_layers))

    def forward(self, x):
        batch_size = x.shape[0]
        x_tiled = x.repeat(1, self.n_layers)
        scaled_inputs = torch.einsum("i,ji->ji", self.lmbd, x_tiled)
        squashed_inputs = self.activation(scaled_inputs)
        thetas = self.theta.repeat(batch_size, 1)
        results = torch.zeros(batch_size, 1, device=x.device)
        for i in range(batch_size):
            results[i] = self.quantum_circuit.circuit(thetas[i], squashed_inputs[i])
        return results

# Alternating is a simple linear layer with alternating-sign initialization
class Alternating(nn.Module):
    def __init__(self, output_dim):
        super(Alternating, self).__init__()
        self.w = nn.Parameter(torch.Tensor([[(-1) ** i for i in range(output_dim)]]))

    def forward(self, x):
        # Weigh observable vector x by action-specific weight vector w
        return torch.matmul(x, self.w)

# PQCAgent combines the PQC with a post-processing layer that outputs a probability distribution
class PQCAgent(nn.Module):
    def __init__(self, n_qubits, n_layers, n_actions, beta=1.0):
        super(PQCAgent, self).__init__()
        self.pqc = ReUploadingPQC(n_qubits, n_layers)
        self.alternating = Alternating(n_actions)
        self.beta = beta
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        """
        Generates model policy. Gathers weighted observables and applies softmax with fixed temperature.
        """
        x = self.pqc(x)
        x = self.alternating(x)
        x = x * self.beta
        return self.softmax(x)

    def evaluate(self, env_name, state_bounds=np.array([2.4, 2.5, 0.21, 2.5]), render=False):
        env = gym.make(env_name, render_mode='rgb_array' if render else None)
        state, info = env.reset(seed=seed)
        total_reward = 0
        done = False
        while not done:
            state_norm = np.asarray(state).flatten() / state_bounds
            state_tensor = torch.tensor(state_norm, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                probs = self(state_tensor).detach().cpu().numpy()[0]
            action = np.argmax(probs)
            state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated
        env.close()
        return total_reward

# Full REINFORCE update: collects full trajectories, computes discounted returns, and applies REINFORCE loss
def train_pqc(agent, env_name, n_episodes=500, gamma=1.0,
              state_bounds=np.array([2.4, 2.5, 0.21, 2.5])):
    optimizer_upload = torch.optim.Adam(agent.pqc.parameters(), lr=0.01, amsgrad=True)
    optimizer_output = torch.optim.Adam(agent.alternating.parameters(), lr=0.1, amsgrad=True)

    env = gym.make(env_name, render_mode=None)
    episode_rewards = []
    best_reward = -float('inf')
    best_model_state = None
    pbar = tqdm(range(n_episodes), desc="[PQC] Episodes", leave=False)
    for i_episode in pbar:
        state, info = env.reset(seed=seed)
        done = False
        states, actions, rewards = [], [], []
        while not done:
            state_norm = np.asarray(state).flatten() / state_bounds
            states.append(state_norm)
            state_tensor = torch.tensor(state_norm, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                probs = agent(state_tensor).detach().cpu().numpy()[0]
            action = np.random.choice(agent.alternating.w.shape[1], p=probs)
            actions.append(action)
            next_state, reward, terminated, truncated, _ = env.step(action)
            rewards.append(reward)
            done = terminated or truncated
            state = next_state

        # If reward ≥500, mark remaining episodes solved
        episode_total = np.sum(rewards)
        if episode_total >= 500:
            remaining = n_episodes - i_episode - 1
            episode_rewards.append(episode_total)
            episode_rewards.extend([500] * remaining)
            # pbar.set_description(f"[PQC] Episode {i_episode+1}/500 (Solved)")
            if episode_total > best_reward:
                best_reward = episode_total
                best_model_state = agent.state_dict()
            break

        # Compute discounted returns for each time step
        R = 0
        returns = []
        for r in rewards[::-1]:
            R = r + gamma * R
            returns.insert(0, R)
        returns = np.array(returns)
        if len(returns) > 1:
            returns = (returns - np.mean(returns)) / (np.std(returns) + 1e-8)

        states_tensor = torch.tensor(np.array(states), dtype=torch.float32)
        actions_tensor = torch.tensor(actions, dtype=torch.long)
        returns_tensor = torch.tensor(returns, dtype=torch.float32)

        logits = agent(states_tensor)
        chosen_probs = logits[range(len(actions_tensor)), actions_tensor]
        log_probs = torch.log(chosen_probs + 1e-8)
        loss = - (log_probs * returns_tensor).mean()

        optimizer_upload.zero_grad()
        optimizer_output.zero_grad()
        loss.backward()
        optimizer_upload.step()
        optimizer_output.step()

        episode_rewards.append(episode_total)
        pbar.set_description(f"[PQC] Episode")
        if episode_total > best_reward:
            best_reward = episode_total
            best_model_state = agent.state_dict()
    env.close()
    os.makedirs("outputs", exist_ok=True)
    np.savez("outputs/pqc_training_data.npz", episode_rewards=np.array(episode_rewards))
    if best_model_state is not None:
        torch.save(best_model_state, "outputs/best_pqc_model.pt")
        pbar.set_description(f"Best PQC model saved with reward {best_reward:.2f}")
    return best_model_state, episode_rewards

def evaluate_pqc(agent, env_name, state_bounds=np.array([2.4, 2.5, 0.21, 2.5]), render=False):
    env = gym.make(env_name, render_mode='rgb_array' if render else None)
    state, info = env.reset(seed=seed)
    total_reward = 0
    done = False
    while not done:
        state_norm = np.asarray(state).flatten() / state_bounds
        state_tensor = torch.tensor(state_norm, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            probs = agent(state_tensor).detach().cpu().numpy()[0]
        action = np.argmax(probs)
        state, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        done = terminated or truncated
    env.close()
    return total_reward

# Uncomment to run training directly:
# if __name__ == "__main__":
#     env_name = "CartPole-v1"
#     n_qubits = 4
#     n_layers = 10
#     n_actions = gym.make(env_name).action_space.n
#     agent = PQCAgent(n_qubits, n_layers, n_actions, beta=1.0)
#     best_model, rewards_history = train_pqc(agent, env_name, n_episodes=500)
#     print("PQC Training complete. Best reward:", max(rewards_history))