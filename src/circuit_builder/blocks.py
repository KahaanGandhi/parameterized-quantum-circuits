# src/circuit_builder/blocks.py
import pennylane as qml
import torch.nn as nn
import torch

# Custom setattr to track quantum blocks
def __track_blocks_setattr__(self, name, value):
    from .blocks import VariationalBlock, EncodingBlock
    if isinstance(value, (VariationalBlock, EncodingBlock)):
        self._quantum_blocks.append(name)
    super(self.__class__, self).__setattr__(name, value)


# ====== Quantum Layer Blocks ======

class VariationalBlock(nn.Module):
    """
    Variational block that applies:
      1) Single-qubit rotation gates (RX, RY, RZ)
      2) Entangling CZ gates in a circular topology
    """
    def __init__(self, n_qubits: int):
        super().__init__()
        self.n_qubits = n_qubits
        self.params_per_block = 3 * n_qubits
        # Initialize trainable angles θ uniformly in [0, π]
        self.theta = nn.Parameter(torch.rand(1, self.params_per_block) * qml.math.numpy.pi)

    def apply(self, angles):
        """
        Apply single-qubit rotations followed by entangling CZ gates.
        """
        # Per-qubit RX, RY, RZ
        for i in range(self.n_qubits):
            qml.RX(angles[i, 0], wires=i)
            qml.RY(angles[i, 1], wires=i)
            qml.RZ(angles[i, 2], wires=i)
        # Circular CZ entanglement
        for i in range(self.n_qubits):
            qml.CZ(wires=[i, (i + 1) % self.n_qubits])


class EncodingBlock(nn.Module):
    """
    Data-encoding block mapping classical inputs to RX rotations.
    """
    def __init__(self, n_qubits: int):
        super().__init__()
        self.n_qubits = n_qubits
        self.params_per_block = n_qubits

    def apply(self, inputs):
        """
        Apply RX rotations for each input feature.
        """
        for i in range(self.n_qubits):
            qml.RX(inputs[i], wires=i)


class HardwareEfficientAnsatz:
    """
    Factory to attach alternating variational and encoding blocks.
    """
    def __init__(self, n_layers: int):
        self.n_layers = n_layers

    def __call__(self, parent, prefix='ansatz_'):
        for i in range(self.n_layers):
            setattr(parent, f'{prefix}var{i}', VariationalBlock(parent.n_qubits))
            setattr(parent, f'{prefix}enc{i}', EncodingBlock(parent.n_qubits))
        # Final variational block
        setattr(parent, f'{prefix}var{self.n_layers}', VariationalBlock(parent.n_qubits))


class BlockSequence:
    """
    Define custom block sequences programmatically.
    """
    def __init__(self):
        self.sequence = []

    def VariationalLayer(self):
        self.sequence.append('variational')
        return self

    def EncodingLayer(self):
        self.sequence.append('encoding')
        return self

    def __call__(self, parent, prefix='custom_'):
        for i, block in enumerate(self.sequence):
            if block == 'variational':
                setattr(parent, f'{prefix}var{i}', VariationalBlock(parent.n_qubits))
            else:
                setattr(parent, f'{prefix}enc{i}', EncodingBlock(parent.n_qubits))


class Alternating(nn.Module):
    """
    Classical post-processing: linear weighting of observable outputs.
    """
    def __init__(self, n_actions: int):
        super().__init__()
        init = [(-1)**i for i in range(n_actions)]
        self.w = nn.Parameter(qml.math.numpy.array(init, requires_grad=True).reshape(1, -1))

    def forward(self, x):
        return x @ self.w

# Attach custom setattr to block classes
VariationalBlock.__setattr__ = __track_blocks_setattr__
EncodingBlock.__setattr__ = __track_blocks_setattr__