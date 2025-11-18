#!/usr/bin/env python
"""
Check what happens with t<0.
"""

import torch
import numpy as np

# Rate matrix for A->B,B->C,C->D
K = torch.tensor([
    [-15.0, 0.0, 0.0, 0.0],
    [15.0, -7.5, 0.0, 0.0],
    [0.0, 7.5, -3.75, 0.0],
    [0.0, 0.0, 3.75, 0.0]
])

P0 = torch.tensor([1.0, 0.0, 0.0, 0.0])

print("Testing matrix exponential for different times:")
print()

for t in [-0.5, -0.1, 0.0, 0.1, 0.5, 1.0]:
    exp_Kt = torch.matrix_exp(K * t)
    P_t = exp_Kt @ P0
    print(f"t={t:5.1f}: P = {P_t.numpy()}, sum = {P_t.sum():.6f}")
