#!/usr/bin/env python
"""
Check matrix exponential with large rates and times.
"""

import torch

# Simple 2-state system: A -> B
K_simple = torch.tensor([
    [-1e10, 0],
    [1e10, 0]
], dtype=torch.float32)

print("Simple A->B system with k=1e10")
print("Rate matrix K:")
print(K_simple)

test_times = [1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2]

for t in test_times:
    try:
        exp_Kt = torch.matrix_exp(K_simple * t)
        P0 = torch.tensor([1.0, 0.0])
        P_t = exp_Kt @ P0
        print(f"\nt = {t:.2e}: P = {P_t.numpy()}, sum = {P_t.sum():.10f}")
        
        # Check if matrix exp has crazy values
        if exp_Kt.abs().max() > 1e10:
            print(f"  WARNING: matrix_exp max = {exp_Kt.abs().max():.2e}")
    except Exception as e:
        print(f"\nt = {t:.2e}: ERROR - {e}")

print("\n" + "="*70)
print("The problem: exp(k*t) overflows for large k*t")
print("Solution: Clamp populations or detect equilibrium")
print("="*70)
