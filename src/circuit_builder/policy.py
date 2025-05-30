# src/circuit_builder/policy.py
import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym
import os
import locale
import matplotlib
import matplotlib.pyplot as plt
import qutip
from qutip import Bloch
from PIL import Image
from matplotlib import animation
from IPython.display import HTML
from IPython.display import Markdown, display
from tqdm import tqdm
import pennylane as qml
from pennylane import draw_mpl

from .configs import ENV_CONFIGS
from .blocks import (VariationalBlock, EncodingBlock, HardwareEfficientAnsatz, 
                     Alternating, BlockSequence, _track_blocks_setattr)


class ParameterizedQuantumCircuit(nn.Module):
    """
    PyTorch-style quantum policy network using the REINFORCE algorithm.
    """
    def __init__(self, env_name: str = 'CartPole-v1', n_qubits: int = None, beta: float = None, 
                 hardware_efficient_ansatz: int = None, custom_blocks: BlockSequence = None):
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

        # Add quantum blocks based on specified sequence
        if custom_blocks is not None:
            custom_blocks(self)
        elif hardware_efficient_ansatz is not None:
            HardwareEfficientAnsatz(hardware_efficient_ansatz)(self)
        else:
            # Default to a simple architecture
            HardwareEfficientAnsatz(1)(self)

        # Classical post-processing layer
        self.alternating = Alternating(self.n_actions)
        self.softmax = nn.Softmax(dim=1)

        # Define PennyLane QNode for the full circuit
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

        # Define PennyLane QNode for partial circuit execution (for Bloch sphere animation)
        @qml.qnode(self.dev, interface='torch')
        def partial_circuit(theta_flat: torch.Tensor, input_flat: torch.Tensor, max_block: int):
            """
            Executes the PQC up to max_block blocks for visualization purposes.
            """
            var_idx = 0
            enc_idx = 0
            for i in range(max_block):
                name = self._quantum_blocks[i]
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
            return [qml.density_matrix([i]) for i in range(self.n_qubits)]

        self.partial_circuit = partial_circuit

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
    def speedrun(cls, env_name: str = 'CartPole-v1', ansatz_layers: int = 5, beta: float = None, 
                 n_qubits: int = None) -> 'ParameterizedQuantumCircuit':
        """
        One-line constructor for a hardware-efficient ansatz with increased depth.
        """
        return cls(env_name=env_name, beta=beta, hardware_efficient_ansatz=ansatz_layers, n_qubits=n_qubits)

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

    def train(self, gamma: float = 1.0, plot: bool = False, save: bool = True, animate: bool = True, 
              early_stopping: bool = True, return_histories: bool = False) -> (None or 'tuple'): # type: ignore
        """
        Train the policy using the PQC REINFORCE algorithm.
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
        self.theta_history = []  # Record parameter evolution
        best_reward = -float("inf")
        best_state = None

        # Set fixed seed for reproducibility
        seed = 777
        
        # Training loop (tqdm better if there's no early stopping)
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
            self.theta_history.append(self._theta.clone().detach())

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
                gif_path = self.animate(save=True, filename='best_performance.gif', state_bounds=state_bounds)
                self.load_state_dict(current_state)
            else:
                gif_path = self.animate(save=save, filename='final_performance.gif', state_bounds=state_bounds)

        if return_histories:
            return self.loss_history, self.reward_history
        else:
            return None

    def evaluate(self, render: bool = False, state_bounds: np.ndarray = None) -> float:
        """
        Evaluate the trained policy for one full episode.
        """
        env = gym.make(self.env_name, render_mode='rgb_array' if render else None)
        if state_bounds is None:
            state_bounds = ENV_CONFIGS[self.env_name]['state_bounds']

        seed = 777
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

    def draw_circuit(self, style='pennylane_sketch', fontsize='small', dpi=300, title=None):
        """
        Draw the circuit diagram of the PQC with all trained parameters.
        """
        # Suppress a fontconfig warning
        os.environ['LC_ALL'] = 'en_US.UTF-8'
        os.environ['LANG']   = 'en_US.UTF-8'
        locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
                
        reuploading_input = [f"$x_{i % self.n_qubits}$" for i in range(self.total_inputs)]
        scaled_input = [f"{reuploading_input[i]} • {self._lambda.detach()[i].item():.2f}" for i in range(self.total_inputs)]
                
        drawer = draw_mpl(self.circuit, style=style, show_all=True, show_wires=True, decimals=2, fontsize=fontsize)
        fig, ax = drawer(theta_flat=self._theta.detach(), input_flat=scaled_input)
        if title is not None:
            fig.suptitle(title, fontweight="bold", fontsize=16) 
        plt.show()

    def plot_training(self):
        """
        Plot the episode rewards over training.
        """
        os.makedirs('outputs', exist_ok=True)
        
        plt.figure(figsize=(10, 6), dpi=300)
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
        plt.savefig('outputs/pqc_training.png', dpi=300, bbox_inches='tight')
        plt.show()

    def animate_bloch_spheres(self, x: torch.Tensor, save: bool = False, filename: str = 'circuit_animation.gif',
        fps: float = 20.0, dwell_time: float = 0.5, state_bounds: np.ndarray = None):
        """
        Animate the Bloch spheres layer-by-layer with smooth interpolation.
        """
        if state_bounds is None:
            state_bounds = ENV_CONFIGS[self.env_name]['state_bounds']

        # Prepare inputs & parameters
        x_norm     = torch.tensor(np.array(x) / state_bounds, dtype=torch.float32).unsqueeze(0)
        repeats    = self.total_inputs // self.n_qubits
        input_flat = (x_norm.repeat(1, repeats) * self._lambda).detach().squeeze(0)
        theta_flat = self._theta.detach()
        n_blocks   = len(self._quantum_blocks)

        # Compute raw Bloch vectors after each block
        raw_vectors = []
        for blk in range(n_blocks + 1):
            dm_list = self.partial_circuit(theta_flat, input_flat, blk)
            vecs = []
            for rho in dm_list:
                rho_np = rho.detach().numpy()
                vx = 2 * np.real(rho_np[0, 1])
                vy = 2 * np.imag(rho_np[0, 1])
                vz = np.real(rho_np[0, 0] - rho_np[1, 1])
                vecs.append(np.array([vx, vy, vz]))
            raw_vectors.append(np.stack(vecs))

        # Build the frame sequence
        dwell_frames = max(1, int(fps * dwell_time))
        interp_steps = dwell_frames
        all_frames   = []
        frame_layers = []

        for k in range(1, n_blocks + 1):
            # Interpolation: raw_vectors[k-1] -> raw_vectors[k]
            v0 = raw_vectors[k - 1]
            v1 = raw_vectors[k]
            for i in range(1, interp_steps + 1):
                a = i / float(interp_steps)
                all_frames.append((1 - a) * v0 + a * v1)
                frame_layers.append(k)
            # Dwell on raw_vectors[k]
            for _ in range(dwell_frames):
                all_frames.append(v1)
                frame_layers.append(k)

        # Set up Bloch spheres
        fig = plt.figure(figsize=(4 * self.n_qubits, 4), dpi=300)
        axes = [fig.add_subplot(1, self.n_qubits, i + 1, projection='3d') for i in range(self.n_qubits)]
        bloch_objs = []
        for ax in axes:
            b = Bloch(fig=fig, axes=ax)
            b.add_states(qutip.basis(2, 0))
            b.render()
            b.font_size = 12
            # b.view      = [30, 30]
            bloch_objs.append(b)

        def update(frame_idx):
            layer = frame_layers[frame_idx]
            vecs  = all_frames[frame_idx]
            for q, b in enumerate(bloch_objs):
                b.clear()
                b.add_vectors([vecs[q]])
                b.make_sphere()
                axes[q].set_title(f'Qubit {q+1}', pad=8)
            fig.suptitle(f'Applying layer {layer}/{n_blocks}', fontsize=14)
            return bloch_objs

        ani = animation.FuncAnimation(fig, update, frames=len(all_frames), interval=1000.0 / fps, blit=False, repeat=False)

        if save:
            os.makedirs('outputs', exist_ok=True)
            out_path = os.path.join('outputs', filename)
            ani.save(out_path, writer='pillow', fps=fps, dpi=300)
            print(f"Circuit animation saved to {out_path}")
            # Display Markdown to embed the GIF
            display(Markdown(f"![Animation]({out_path})"))

        plt.close(fig)
        return ani

    def animate(self, n_steps: int = None, save: bool = True, filename: str = 'performance.gif',
                fps: int = 20, state_bounds: np.ndarray = None):
        """
        Run the current policy in the environment, capture frames, and save as a GIF.
        """
        env = gym.make(self.env_name, render_mode='rgb_array')
        if state_bounds is None:
            state_bounds = ENV_CONFIGS[self.env_name]['state_bounds']

        seed = 777
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
            # Capture frame as PIL Image
            frame = env.render()
            im = Image.fromarray(frame)
            frames.append(im)
            state = next_state
            steps += 1

        env.close()

        # Save the GIF using PIL
        if save:
            os.makedirs('outputs', exist_ok=True)
            gif_path = os.path.join('outputs', filename)
            frames[0].save(gif_path, save_all=True, append_images=frames[1:], optimize=False, duration=int(1000/fps), loop=0)
            print(f"Animation saved to {gif_path}")
            # Display Markdown to embed the GIF
            display(Markdown(f"![Animation]({gif_path})"))

        return gif_path

    def animate_training(self, save: bool = False,
                        filename: str = 'training_animation.gif',
                        fps: int = 10):
        """
        Animate θ_i and reward vs episode.
        """
        if not hasattr(self, 'theta_history') or not hasattr(self, 'reward_history'):
            raise ValueError("Training history not recorded. Please run train first.")

        n_params = self.total_theta
        episodes = len(self.theta_history)

        # Shared x-axis for both panels
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), dpi=300, sharex=True, gridspec_kw={'height_ratios': [2, 1]})

        # Top panel: parameter evolution
        ax1.set_title('Rotation Gate Parameters', fontsize=12)
        ax1.set_ylabel('θ (rad)', fontsize=10)
        ax1.set_ylim(0, np.pi)
        ax1.set_yticks([0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi])
        ax1.set_yticklabels(['0', r'$\frac{\pi}{4}$', r'$\frac{\pi}{2}$', r'$\frac{3\pi}{4}$', r'$\pi$'], fontsize=9)

        cmap = plt.cm.gist_earth
        colors = cmap(np.linspace(0, 1, n_params))
        lines = [ax1.plot([], [], color=colors[i], linewidth=1)[0] for i in range(n_params)]

        # Bottom panel: reward history
        ax2.set_ylabel('Reward', fontsize=10)
        ax2.set_xlabel('Episode', fontsize=10)
        ax2.set_ylim(0, max(self.reward_history))
        reward_line, = ax2.plot([], [], color='crimson', linewidth=2)
        plt.tight_layout()

        def init():
            for ln in lines:
                ln.set_data([], [])
            reward_line.set_data([], [])
            return lines + [reward_line]

        def update(ep):
            x = np.arange(ep + 1)
            # Parameter update
            for i, ln in enumerate(lines):
                y = [self.theta_history[e][i].item() for e in x]
                ln.set_data(x, y)
            # Reward update
            reward_line.set_data(x, self.reward_history[:ep + 1])
            ax2.set_xlim(0, episodes)
            return lines + [reward_line]

        ani = animation.FuncAnimation(fig, update, frames=range(episodes), init_func=init, interval=1000/fps, blit=True, repeat=False)

        if save:
            os.makedirs('outputs', exist_ok=True)
            out_path = os.path.join('outputs', filename)
            ani.save(out_path, writer='pillow', fps=fps, dpi=300)
            display(Markdown(f"![Animation]({out_path})"))

        plt.close(fig)
        return ani