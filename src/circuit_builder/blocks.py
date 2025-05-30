# src/circuit_builder/blocks.py
import torch
import torch.nn as nn
import numpy as np
import pennylane as qml


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


class BlockSequence:
    """
    Utility class to define custom block sequences.
    """
    def __init__(self):
        self.sequence = []
    
    def VariationalLayer(self):
        """
        Add a variational block to the sequence.
        """
        self.sequence.append("variational")
        return self
    
    def EncodingLayer(self):
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


# Custom setattr to track quantum blocks
def _track_blocks_setattr(self, name, value):
    """
    Record VariationalBlock and EncodingBlock instances in order.
    """
    if isinstance(value, (VariationalBlock, EncodingBlock)):
        self._quantum_blocks.append(name)
    object.__setattr__(self, name, value)