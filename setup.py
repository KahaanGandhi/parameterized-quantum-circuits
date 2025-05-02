from setuptools import setup, find_packages

setup(
    name="parameterized_quantum_circuits",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pennylane",
        "torch",
        "numpy",
        "gymnasium[classic_control]",
        "matplotlib",
        "tqdm",
        "pillow",
        "qutip",
    ],
    python_requires=">=3.8",
)