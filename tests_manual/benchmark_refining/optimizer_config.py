"""
Configuration file for optimizer benchmarking
Adjust these parameters to customize your tests
"""

# Data files
MTZ_FILE = '/das/work/p17/p17490/Peter/Library/multicopy_refinement/scientific_testing/data/1A0F/1A0F.mtz'
PDB_FILE = '/das/work/p17/p17490/Peter/Library/multicopy_refinement/scientific_testing/data/1A0F/1A0F_shaken.pdb'

# Target weights - KEEP CONSISTENT ACROSS ALL TESTS
TARGET_WEIGHTS = {
    'xray': 1.0,
    'restraints': 1.0,
    'adp': 0.3
}

# Refinement targets to test
TARGETS = ['xyz', 'b']

# Number of optimization steps
NSTEPS = 10

# Optimizer configurations: {name: learning_rate}
# Learning rates are tuned for each optimizer type
OPTIMIZER_LR = {
    'LBFGS': 1.0,       # LBFGS typically uses lr=1.0
    'Adam': 0.01,       # Good default for Adam
    'AdamW': 0.01,      # Same as Adam
    'SGD': 0.01,        # With momentum
    'RMSprop': 0.01,    # Standard lr
    'Adagrad': 0.01,    # Adaptive lr
}

# Optional: Optimizer-specific hyperparameters
OPTIMIZER_KWARGS = {
    'LBFGS': {
        'max_iter': 20,
        'history_size': 10,
        'line_search_fn': None  # Can be 'strong_wolfe'
    },
    'Adam': {
        'betas': (0.9, 0.999),
        'eps': 1e-8
    },
    'AdamW': {
        'betas': (0.9, 0.999),
        'eps': 1e-8,
        'weight_decay': 0.01
    },
    'SGD': {
        'momentum': 0.9,
        'dampening': 0,
        'nesterov': False
    },
    'RMSprop': {
        'alpha': 0.99,
        'eps': 1e-8,
        'momentum': 0
    },
    'Adagrad': {
        'lr_decay': 0,
        'eps': 1e-10
    }
}

# Which optimizers to test (comment out to skip)
OPTIMIZERS_TO_TEST = [
    'LBFGS',
    'Adam',
    'AdamW',
    'SGD',
    'RMSprop',
    'Adagrad',
]
