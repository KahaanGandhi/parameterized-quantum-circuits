# src/circuit_builder/policy.py
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
import pennylane as qml
from .blocks import VariationalBlock, EncodingBlock, HardwareEfficientAnsatz, BlockSequence, Alternating
from .configs import ENV_CONFIGS

# Reproducible seed
seed = 777
np.random.seed(seed)
torch.manual_seed(seed)
np.bool8 = np.bool_


class ParameterizedQuantumCircuit(nn.Module):
    """
    PyTorch-style quantum policy network using adapted REINFORCE.
    """
    def __setattr__(self, name, value):
        # Track block attachments in _quantum_blocks
        if hasattr(self, '_quantum_blocks') and isinstance(value, (VariationalBlock, EncodingBlock)):
            self._quantum_blocks.append(name)
        super().__setattr__(name, value)

    def __init__(self, env_name='CartPole-v1', n_qubits=None, beta=None,
                 hardware_efficient_ansatz=None, custom_blocks=None):
        super().__init__()
        config = ENV_CONFIGS[env_name]
        env = gym.make(env_name)
        obs = env.observation_space.shape
        self.n_qubits = n_qubits or obs[0]
        self.n_actions = env.action_space.n
        env.close()
        self.beta = beta or config['beta']

        # Keep track of block order
        self._quantum_blocks = []

        # Attach quantum layers
        if custom_blocks:
            custom_blocks(self)
        elif hardware_efficient_ansatz:
            HardwareEfficientAnsatz(hardware_efficient_ansatz)(self)
        else:
            HardwareEfficientAnsatz(1)(self)

        # Classical post-processing head
        self.alternating = Alternating(self.n_actions)
        self.softmax = nn.Softmax(dim=1)

        # Define the QNode circuit
        self.dev = qml.device('default.qubit', wires=self.n_qubits)
        @qml.qnode(self.dev, interface='torch')
        def circuit(theta, inputs):
            var_i = enc_i = 0
            # Apply each block in order
            for name in self._quantum_blocks:
                block = getattr(self, name)
                cnt = block.params_per_block
                if isinstance(block, VariationalBlock):
                    angles = theta[var_i:var_i+cnt].reshape(self.n_qubits, 3)
                    block.apply(angles)
                    var_i += cnt
                else:
                    block.apply(inputs[enc_i:enc_i+cnt])
                    enc_i += cnt
            # Measure Z on all wires
            obs_ops = [qml.PauliZ(i) for i in range(self.n_qubits)]
            return qml.expval(qml.operation.Tensor(*obs_ops))
        self.circuit = circuit

        # Count parameters for optimization
        self.total_theta = sum(getattr(self, n).params_per_block
                               for n in self._quantum_blocks
                               if isinstance(getattr(self, n), VariationalBlock))
        self.total_inputs = sum(getattr(self, n).params_per_block
                                for n in self._quantum_blocks
                                if isinstance(getattr(self, n), EncodingBlock))

        # Trainable parameters
        self._theta = nn.Parameter(torch.rand(self.total_theta) * np.pi)
        self._lambda = nn.Parameter(torch.ones(self.total_inputs))

    def forward(self, x):
        # Prepare inputs for QNode
        reps = self.total_inputs // self.n_qubits
        inp = x.repeat(1, reps) * self._lambda
        batch = inp.shape[0]
        thetas = self._theta.unsqueeze(0).repeat(batch, 1)
        vals = torch.zeros(batch, 1)
        for i in range(batch):
            vals[i] = self.circuit(thetas[i], inp[i])
        logits = self.alternating(vals) * self.beta
        return self.softmax(logits)

    def train(self, gamma=1.0, plot=False, save=True, animate=False,
              early_stopping=True, return_histories=False):
        config = ENV_CONFIGS[self.env_name]
        env = gym.make(self.env_name)
        thr = config['reward_threshold']
        # Optimizers for theta, lambda, and alternating weights
        opt_t = torch.optim.Adam([self._theta], lr=config['learning_rate_theta'])
        opt_l = torch.optim.Adam([self._lambda], lr=config['learning_rate_lambda'])
        opt_w = torch.optim.Adam(self.alternating.parameters(), lr=config['learning_rate_weights'])

        self.reward_history = []
        for ep in range(config['n_episodes']):
            state, _ = env.reset(seed=seed)
            traj, done = [], False
            
            # Collect trajectory
            while not done:
                norm = np.array(state) / config['state_bounds']
                s = torch.tensor(norm, dtype=torch.float32).unsqueeze(0)
                with torch.no_grad(): probs = self(s).cpu().numpy()[0]
                a = np.random.choice(self.n_actions, p=probs)
                ns, r, term, trunc, _ = env.step(a)
                done = term or trunc
                traj.append((norm, a, r))
                state = ns
                
            # Compute returns
            R, returns = 0, []
            for _, _, r in reversed(traj):
                R = r + gamma * R; returns.insert(0, R)
            rets = torch.tensor((returns - np.mean(returns)) / (np.std(returns) + 1e-8))
            states = torch.tensor([t[0] for t in traj], dtype=torch.float32)
            acts = torch.tensor([t[1] for t in traj])
            logps = torch.log(self(states)[range(len(acts)), acts] + 1e-8)
            loss = -torch.mean(logps * rets)
            
            # Optimize
            opt_t.zero_grad(); opt_l.zero_grad(); opt_w.zero_grad()
            loss.backward(); opt_t.step(); opt_l.step(); opt_w.step()
            total = sum(r for _,_,r in traj)
            self.reward_history.append(total)
            if early_stopping and len(self.reward_history) >= 100 and np.mean(self.reward_history[-100:]) >= thr:
                break
            
        env.close()
        
        if return_histories:
            return self.reward_history