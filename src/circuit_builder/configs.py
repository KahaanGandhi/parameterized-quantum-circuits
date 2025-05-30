# src/circuit_builder/configs.py
import numpy as np

# Environment-specific hyperparameters and training settings
ENV_CONFIGS = {
    'CartPole-v1': {
        'state_bounds': np.array([2.4, 2.5, 0.21, 2.5]),
        'learning_rate_theta': 0.01,
        'learning_rate_lambda': 0.1,
        'learning_rate_weights': 0.1,
        'beta': 1.0,
        'reward_threshold': 500,
        'n_episodes': 500
    },
    'LunarLander-v2': {
        'state_bounds': np.array([1.5,1.5,5.0,5.0,3.14,5.0,1.0,1.0]),
        'learning_rate_theta': 0.001,
        'learning_rate_lambda': 0.01,
        'learning_rate_weights': 0.01,
        'beta': 1.0,
        'reward_threshold': 200,
        'n_episodes': 4000
    }
}