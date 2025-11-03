# configs/training_configs.py
TRAINING_CONFIGS = {
    'fast': {
        'batch_size': 256,
        'epochs': 5,
        'lr': 1e-3,
        'num_steps': 10
    },
    'accurate': {
        'batch_size': 128, 
        'epochs': 20,
        'lr': 5e-4,
        'num_steps': 25
    }
}

# configs/network_configs.py
NETWORK_CONFIGS = {
    'small': {'hidden_size': 256, 'beta': 0.9},
    'medium': {'hidden_size': 512, 'beta': 0.95},
    'large': {'hidden_size': 1024, 'beta': 0.99}
}