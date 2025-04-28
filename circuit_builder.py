import pennylane as qml
import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym
import os
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.display import HTML
from IPython.display import display
from tqdm import tqdm

from configs import ENV_CONFIGS

seed = 777
np.random.seed(seed)
torch.manual_seed(seed)
np.bool8 = np.bool_  # Avoid deprecation warning

# Custom setattr to track quantum blocks
def _track_blocks_setattr(self, name, value):
    """
    Record VariationalBlock and EncodingBlock instances in order.
    """
    if isinstance(value, (VariationalBlock, EncodingBlock)):
        self._quantum_blocks.append(name)
    object.__setattr__(self, name, value)

# ====== Quantum Layer Blocks ====== #

class VariationalBlock(nn.Module):
    """
    Variational block that applies:
    1) Single-qubit rotation gates (Rx, Ry, Rz)
    2) Entangling gates (CZ) in a circular topology
    """
    def __init__(self, n_qubits: int):
        super().__init__()
        self.n_qubits = n_qubits
        self.params_per_block = 3 * n_qubits
        # Initialize trainable angles θ uniformly in [0, π]
        self.theta = nn.Parameter(torch.rand(1, self.params_per_block) * np.pi)

    def apply(self, angles: torch.Tensor):
        """
        Apply single-qubit rotations and entangling gates within a QNode.
        """
        # Apply Rx, Ry, Rz on each circuit wire
        for qubit_index in range(self.n_qubits):
            angle_x = angles[qubit_index, 0]
            angle_y = angles[qubit_index, 1]
            angle_z = angles[qubit_index, 2]
            
            qml.RX(angle_x, wires=qubit_index)
            qml.RY(angle_y, wires=qubit_index)
            qml.RZ(angle_z, wires=qubit_index)
            
        # Apply CZ entangling gates in a circular topology
        for qubit_index in range(self.n_qubits):
            next_qubit = (qubit_index + 1) % self.n_qubits
            qml.CZ(wires=[qubit_index, next_qubit])

class EncodingBlock(nn.Module):
    """
    Data-encoding block that maps classical data inputs to quantum qubits.
    """
    def __init__(self, n_qubits: int):
        super().__init__()
        self.n_qubits = n_qubits
        self.params_per_block = n_qubits

    def apply(self, inputs: torch.Tensor):
        """
        Apply Rx rotations within a QNode.
        """
        for qubit_index in range(self.n_qubits):
            angle = inputs[qubit_index]
            qml.RX(angle, wires=qubit_index)

class HardwareEfficientAnsatz:
    """
    Factory class to generate alternating variational and encoding blocks.
    HEA circumvents no-cloning theorem by copying classical data.
    """
    def __init__(self, n_layers: int):
        self.n_layers = n_layers

    def __call__(self, parent: nn.Module, prefix: str = 'ansatz_'):
        # Attach blocks in alternating order
        for i in range(self.n_layers):
            setattr(parent, f'{prefix}var{i}', VariationalBlock(parent.n_qubits))
            setattr(parent, f'{prefix}enc{i}', EncodingBlock(parent.n_qubits))
        # Final variational block after all encodings
        setattr(parent, f'{prefix}var{self.n_layers}', VariationalBlock(parent.n_qubits))

class Alternating(nn.Module):
    """
    Weight the Pauli product by learnable action-specific weight vector.
    """
    def __init__(self, n_actions: int):
        super().__init__()
        # Initialize weights to [+1, -1, +1, -1, ...]
        initial_weights = [(-1) ** i for i in range(n_actions)]
        self.w = nn.Parameter(torch.tensor(initial_weights, dtype=torch.float32).unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute logits from the observable values.
        """
        return x @ self.w

# ====== Block Sequence Factory ====== #

class BlockSequence:
    """
    Utility class to define custom block sequences.
    """
    def __init__(self):
        self.sequence = []
    
    def add_variational(self):
        """
        Add a variational block to the sequence.
        """
        self.sequence.append("variational")
        return self
    
    def add_encoding(self):
        """
        Add an encoding block to the sequence.
        """
        self.sequence.append("encoding")
        return self
    
    def __call__(self, parent: nn.Module, prefix: str = 'custom_'):
        """
        Apply the defined sequence to a parent module.
        """
        for i, block_type in enumerate(self.sequence):
            if block_type == "variational":
                setattr(parent, f'{prefix}var{i}', VariationalBlock(parent.n_qubits))
            elif block_type == "encoding":
                setattr(parent, f'{prefix}enc{i}', EncodingBlock(parent.n_qubits))

# ====== Parameterized Quantum Circuit ====== #

class ParameterizedQuantumCircuit(nn.Module):
    """
    PyTorch-style quantum policy network using the REINFORCE algorithm.
    """
    def __init__(self, env_name: str = 'CartPole-v1', n_qubits: int = None, beta: float = None, 
                 heuristic_ansatz: int = None, custom_blocks: BlockSequence = None):
        super().__init__()

        # Get environment configuration
        if env_name not in ENV_CONFIGS:
            raise ValueError(f"Unsupported environment: {env_name}. Supported environments: {list(ENV_CONFIGS.keys())}")
        env_config = ENV_CONFIGS[env_name]
        
        # Inspect environment to determine observation and action spaces
        env = gym.make(env_name)
        obs_shape = env.observation_space.shape
        assert len(obs_shape) == 1, "Only flat observation spaces supported."
        
        # Use provided n_qubits or auto-detect from environment
        self.n_qubits = n_qubits if n_qubits is not None else obs_shape[0]
        
        action_space = env.action_space
        assert isinstance(action_space, gym.spaces.Discrete), "Only discrete action spaces are supported."
        self.n_actions = action_space.n
        env.close()

        self.env_name = env_name
        # Use provided beta or get from environment config
        self.beta = beta if beta is not None else env_config['beta']

        # Track quantum blocks
        self._quantum_blocks = []
        self.__class__.__setattr__ = _track_blocks_setattr

        # Add quantum blocks based on specified pattern
        if custom_blocks is not None:
            custom_blocks(self)
        elif heuristic_ansatz is not None:
            HardwareEfficientAnsatz(heuristic_ansatz)(self)
        else:
            # Default to a simple pattern
            HardwareEfficientAnsatz(1)(self)

        # Classical post-processing layer
        self.alternating = Alternating(self.n_actions)
        self.softmax = nn.Softmax(dim=1)

        # Define PennyLane QNode
        self.dev = qml.device('default.qubit', wires=self.n_qubits)
        @qml.qnode(self.dev, interface='torch')
        def circuit(theta_flat: torch.Tensor, input_flat: torch.Tensor):
            """
            Executes the full PQC by iterating over all defined blocks.
            """
            var_idx = 0
            enc_idx = 0
            for name in self._quantum_blocks:
                block = getattr(self, name)
                param_count = block.params_per_block
                if isinstance(block, VariationalBlock):
                    angles = theta_flat[var_idx : var_idx + param_count]
                    angles = angles.reshape(self.n_qubits, 3)
                    block.apply(angles)
                    var_idx += param_count
                else:
                    inputs = input_flat[enc_idx : enc_idx + param_count]
                    block.apply(inputs)
                    enc_idx += param_count
            # Measure the product of Z on all wires (observable)
            observable = [qml.PauliZ(i) for i in range(self.n_qubits)]
            return qml.expval(qml.operation.Tensor(*observable))

        self.circuit = circuit

        # Check number of trainable parameters
        self.total_theta = sum(
            getattr(self, name).params_per_block
            for name in self._quantum_blocks
            if isinstance(getattr(self, name), VariationalBlock)
        )
        self.total_inputs = sum(
            getattr(self, name).params_per_block
            for name in self._quantum_blocks
            if isinstance(getattr(self, name), EncodingBlock)
        )

        # Initialize trainable parameters
        self._theta = nn.Parameter(torch.rand(self.total_theta) * np.pi)
        self._lambda = nn.Parameter(torch.ones(self.total_inputs))

    @classmethod
    def speedrun(cls, env_name: str = 'CartPole-v1', ansatz_layers: int = 20, beta: float = None, 
                 n_qubits: int = None) -> 'ParameterizedQuantumCircuit':
        """
        One-line constructor for a hardware-efficient ansatz with increased depth.
        """
        return cls(env_name=env_name, beta=beta, heuristic_ansatz=ansatz_layers, n_qubits=n_qubits)
    
    def animate(self, n_steps: int = None, save: bool = True, filename: str = 'performance.gif',
                fps: int = 20, state_bounds: np.ndarray = None):
        """
        Run the current policy in the environment, capture frames, and return an animation to be displayed inline in Jupyter.
        """
        env = gym.make(self.env_name, render_mode='rgb_array')
        if state_bounds is None:
            state_bounds = ENV_CONFIGS[self.env_name]['state_bounds']

        state, _ = env.reset(seed=seed)
        frames = []
        done = False
        steps = 0

        while not done and (n_steps is None or steps < n_steps):
            normalized = np.array(state) / state_bounds
            state_tensor = torch.tensor(normalized, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                probs = self(state_tensor).cpu().numpy()[0]
            action = int(np.argmax(probs))
            next_state, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            frame = env.render()
            frames.append(frame)
            state = next_state
            steps += 1

        env.close()

        fig = plt.figure(figsize=(6, 6))
        plt.axis('off')
        ims = []
        for img in frames:
            im = plt.imshow(img, animated=True)
            ims.append([im])

        ani = animation.ArtistAnimation(fig, ims, interval=1000/fps, blit=True, repeat_delay=1000)
        
        if save:
            os.makedirs('outputs', exist_ok=True)
            ani.save(os.path.join('outputs', filename), writer='pillow', fps=fps)
            print(f"Animation saved to outputs/{filename}")

        plt.close(fig)
        return HTML(ani.to_jshtml())  # Return only the HTML version for Jupyter

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: maps state inputs to action probabilities.
        """
        repeats = self.total_inputs // self.n_qubits
        x_tiled = x.repeat(1, repeats)
        scaled_inputs = x_tiled * self._lambda
        batch_size = x.shape[0]
        theta_batch = self._theta.unsqueeze(0).repeat(batch_size, 1)

        obs_values = torch.zeros(batch_size, 1)
        for idx in range(batch_size):
            obs_values[idx] = self.circuit(theta_batch[idx], scaled_inputs[idx])

        logits = self.alternating(obs_values) * self.beta
        action_probs = self.softmax(logits)
        return action_probs

    def train(self, gamma: float = 1.0, plot: bool = True, save: bool = True, 
              animate: bool = True, early_stopping: bool = True) -> tuple:
        """
        Train the policy using the REINFORCE algorithm.
        """
        # Use environment-specific configuration
        env_config = ENV_CONFIGS[self.env_name]
        n_episodes = env_config['n_episodes']
        state_bounds = env_config['state_bounds']
        learning_rate_theta = env_config['learning_rate_theta']
        learning_rate_lambda = env_config['learning_rate_lambda']
        learning_rate_weights = env_config['learning_rate_weights']
            
        env = gym.make(self.env_name)
        threshold = env_config.get('reward_threshold', 500)
        
        # Set up the optimizers with different learning rates
        optimizer_theta = torch.optim.Adam([self._theta], lr=learning_rate_theta, amsgrad=True)
        optimizer_lambda = torch.optim.Adam([self._lambda], lr=learning_rate_lambda, amsgrad=True)
        optimizer_weights = torch.optim.Adam(self.alternating.parameters(), lr=learning_rate_weights, amsgrad=True)

        self.loss_history = []
        self.reward_history = []
        best_reward = -float("inf")
        best_state = None

        for episode in range(n_episodes):
            state, _ = env.reset(seed=seed)
            done = False
            trajectory = []
            
            # Collect trajectory by rolling out the policy in the env
            while not done:
                obs_norm = np.array(state) / state_bounds
                state_t = torch.tensor(obs_norm, dtype=torch.float32).unsqueeze(0)
                # Predict action probabilities
                with torch.no_grad():
                    probs = self(state_t).cpu().numpy()[0]
                # Sample an action from the policy
                action = np.random.choice(self.n_actions, p=probs)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                trajectory.append((obs_norm, action, reward))
                state = next_state
            
            # Compute discounted returns from rewards
            returns = []
            R = 0.0
            for _, _, r in reversed(trajectory):
                R = r + gamma * R
                returns.insert(0, R)
            returns = np.array(returns, dtype=np.float32)
            if len(returns) > 1:
                returns = (returns - returns.mean()) / (returns.std() + 1e-8)

            # Prepare data for policy gradient
            states_np = np.stack([t[0] for t in trajectory])
            states = torch.from_numpy(states_np).float()
            actions = torch.tensor([t[1] for t in trajectory], dtype=torch.long)
            returns_t = torch.tensor(returns, dtype=torch.float32)

            # Calculate the policy loss
            logits = self(states)
            selected = logits[range(len(actions)), actions]
            log_probs = torch.log(selected + 1e-8)
            loss = -torch.mean(log_probs * returns_t)

            # Optimize the policy parameters
            optimizer_theta.zero_grad()
            optimizer_lambda.zero_grad()
            optimizer_weights.zero_grad()
            loss.backward()
            optimizer_theta.step()
            optimizer_lambda.step()
            optimizer_weights.step()

            total_reward = sum(r for _, _, r in trajectory)
            self.loss_history.append(loss.item())
            self.reward_history.append(total_reward)

            if total_reward > best_reward:
                best_reward = total_reward
                best_state = self.state_dict()

            print(f"Episode {episode+1}: Reward = {total_reward}", end="\r")

            if early_stopping and best_reward >= threshold:
                print(f"Environment solved in {episode+1} episodes! Reward={best_reward:.2f}")
                break

        env.close()
        print(f"\nTraining completed. Best reward: {best_reward:.2f}")

        if plot:
            self.plot_training()
        
        if save:
            os.makedirs('outputs', exist_ok=True)
            torch.save(best_state, 'outputs/pqc_best.pt')
            np.save('outputs/pqc_loss.npy', self.loss_history)
            np.save('outputs/pqc_reward.npy', self.reward_history)

        if animate:
            if save and best_state is not None:
                current_state = self.state_dict()
                self.load_state_dict(best_state)
                animation_html = self.animate(save=True, filename='best_performance.gif', state_bounds=state_bounds)
                display(animation_html)
                self.load_state_dict(current_state)
            else:
                animation_html = self.animate(save=save, filename='final_performance.gif', state_bounds=state_bounds)
                display(animation_html)

        return self.loss_history, self.reward_history

    def evaluate(self, render: bool = False, state_bounds: np.ndarray = None) -> float:
        """
        Evaluate the trained policy for one full episode.
        """
        env = gym.make(self.env_name, render_mode='rgb_array' if render else None)
        if state_bounds is None:
            state_bounds = ENV_CONFIGS[self.env_name]['state_bounds']

        state, _ = env.reset(seed=seed)
        total_reward = 0.0
        done = False

        while not done:
            normalized = np.array(state) / state_bounds
            state_tensor = torch.tensor(normalized, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                probs = self(state_tensor).cpu().numpy()[0]
            action = int(np.argmax(probs))
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            state = next_state

        env.close()
        return total_reward

    def plot_training(self):
        """
        Plot the episode rewards over training with improved aesthetics.
        """
        os.makedirs('outputs', exist_ok=True)
        
        plt.figure(figsize=(10, 6), dpi=300)
        
        # Plot episode rewards with transparency
        plt.plot(self.reward_history, color='dodgerblue', alpha=1.0, label='Episode Reward')
        
        # Add threshold line
        threshold = ENV_CONFIGS[self.env_name].get('reward_threshold')
        if threshold:
            plt.axhline(y=threshold, color='tab:red', linestyle='--', alpha=0.7, 
                        label=f'Solution Threshold ({threshold})')
        
        plt.xlabel('Episode', fontsize=12)
        plt.ylabel('Reward', fontsize=12)
        plt.title(f'Training Rewards on {self.env_name}', fontsize=14, fontweight='bold')
        plt.legend(loc='best', frameon=True, framealpha=0.9)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('outputs/pqc_training.png', dpi=300, bbox_inches='tight')
        plt.show()

# if __name__ == "__main__":
#     # Speedrun
#     pqc = ParameterizedQuantumCircuit.speedrun(env_name='LunarLander-v2', ansatz_layers=20)
#     pqc.train()