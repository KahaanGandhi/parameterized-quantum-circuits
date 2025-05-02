Parameterized quantum circuits (PQCs) enable quantum reinforcement learning by mapping classical states to action probabilities via trainable gates, which we optimize to solve control tasks like CartPole using PennyLane for simulation and PyTorch for gradient-based training.

Contents include:
* `circuit_builder.py`: Defines the full PQC architecture with variational and encoding layers, including methods for circuit construction, REINFORCE-based training, and model evaluation.
* `demo.ipynb`: Offers example workflows using PQCs, with a simple two-line end-to-end training example. Includes tools to draw circuit diagrams, animate Bloch spheres, and visualize model performance on reinforcement learning tasks.
* `report.pdf`: Presents a derivation of the parameter-shift rule and shows how its exact gradient estimates drive the learning algorithm.

_Final project for Phys 302 (Advanced Quantum Mechanics) at Haverford College._