"""
Example demonstrating how to use MixedTensor's make_refinable and make_fixed methods.

This example shows how to dynamically change which parts of a tensor are refinable
during optimization, which is useful for multi-stage refinement protocols.
"""

import torch
import torch.nn as nn
from multicopy_refinement.model import MixedTensor


def example_1_basic_selection():
    """Example 1: Basic usage of make_refinable and make_fixed with slices."""
    print("=" * 70)
    print("Example 1: Basic Selection with Slices")
    print("=" * 70)
    
    # Create initial tensor where nothing is refinable
    initial_values = torch.arange(100, dtype=torch.float64)
    mask = torch.zeros(100, dtype=torch.bool)
    
    mixed = MixedTensor(initial_values, refinable_mask=mask, requires_grad=True)
    print(f"\nInitial: {mixed}")
    print(f"Initial values (first 10): {mixed()[:10]}")
    
    # Make elements 20-30 refinable
    mixed.make_refinable(slice(20, 30))
    print(f"\nAfter making [20:30] refinable: {mixed}")
    
    # Optimize those elements
    optimizer = torch.optim.Adam([mixed.refinable_params], lr=0.1)
    target = torch.ones(100) * 50.0
    
    for i in range(10):
        optimizer.zero_grad()
        prediction = mixed()
        loss = ((prediction - target) ** 2).sum()
        loss.backward()
        optimizer.step()
        if i % 5 == 0:
            print(f"  Iter {i}: loss={loss.item():.2f}")
    
    result = mixed().detach()
    print(f"\nAfter optimization:")
    print(f"  Elements [20:30]: {result[20:30]}")
    print(f"  Elements [0:10] (should be unchanged): {result[0:10]}")
    
    # Now make those elements fixed again and make a different region refinable
    mixed.make_fixed(slice(20, 30), freeze_at_current=True)
    mixed.make_refinable(slice(40, 50))
    print(f"\nAfter fixing [20:30] and making [40:50] refinable: {mixed}")
    
    # Optimize the new region
    optimizer = torch.optim.Adam([mixed.refinable_params], lr=0.1)
    for i in range(10):
        optimizer.zero_grad()
        prediction = mixed()
        loss = ((prediction - target) ** 2).sum()
        loss.backward()
        optimizer.step()
    
    result = mixed().detach()
    print(f"\nAfter second optimization:")
    print(f"  Elements [40:50]: {result[40:50]}")
    print(f"  Elements [20:30] (should still be ~50): {result[20:30]}")
    print(f"  Elements [0:10] (should still be original): {result[0:10]}")


def example_2_boolean_mask():
    """Example 2: Using boolean masks for complex selections."""
    print("\n" + "=" * 70)
    print("Example 2: Boolean Mask Selection")
    print("=" * 70)
    
    # Create a 10x10 tensor
    initial_values = torch.randn(10, 10, dtype=torch.float64)
    
    # Start with nothing refinable
    mask = torch.zeros(10, 10, dtype=torch.bool)
    mixed = MixedTensor(initial_values, refinable_mask=mask, requires_grad=True)
    
    print(f"\nInitial: {mixed}")
    
    # Make diagonal elements refinable
    diagonal_mask = torch.eye(10, dtype=torch.bool)
    mixed.make_refinable(diagonal_mask)
    print(f"After making diagonal refinable: {mixed}")
    
    # Make upper triangle refinable (in addition to diagonal)
    upper_triangle = torch.triu(torch.ones(10, 10, dtype=torch.bool), diagonal=1)
    mixed.make_refinable(upper_triangle)
    print(f"After making upper triangle refinable: {mixed}")
    
    # Make lower triangle fixed (if any were refinable)
    lower_triangle = torch.tril(torch.ones(10, 10, dtype=torch.bool), diagonal=-1)
    mixed.make_fixed(lower_triangle)
    print(f"After fixing lower triangle: {mixed}")


def example_3_multidimensional_indexing():
    """Example 3: Using advanced indexing for multidimensional tensors."""
    print("\n" + "=" * 70)
    print("Example 3: Multidimensional Indexing")
    print("=" * 70)
    
    # Create a 3D tensor (e.g., for coordinates: N_atoms x 3)
    n_atoms = 50
    initial_coords = torch.randn(n_atoms, 3, dtype=torch.float64)
    
    # Initially, make atoms 10-20 refinable (all coordinates)
    mask = torch.zeros(n_atoms, 3, dtype=torch.bool)
    mask[10:20, :] = True
    
    mixed = MixedTensor(initial_coords, refinable_mask=mask, requires_grad=True)
    print(f"\nInitial: {mixed}")
    print(f"Refinable atoms: 10-20 (all coordinates)")
    
    # Make only X and Y coordinates of atoms 30-40 refinable
    xy_mask = torch.zeros(n_atoms, 3, dtype=torch.bool)
    xy_mask[30:40, 0:2] = True  # Only X and Y
    mixed.make_refinable(xy_mask)
    print(f"\nAfter adding atoms 30-40 (X,Y only): {mixed}")
    
    # Fix Z coordinates of atoms 10-20
    z_mask = torch.zeros(n_atoms, 3, dtype=torch.bool)
    z_mask[10:20, 2] = True  # Only Z
    mixed.make_fixed(z_mask, freeze_at_current=True)
    print(f"After fixing Z of atoms 10-20: {mixed}")


def example_4_staged_refinement():
    """Example 4: Realistic staged refinement protocol."""
    print("\n" + "=" * 70)
    print("Example 4: Staged Refinement Protocol")
    print("=" * 70)
    
    # Simulate atomic B-factors for a protein
    n_atoms = 200
    initial_b_factors = torch.ones(n_atoms, dtype=torch.float64) * 20.0
    target_b_factors = torch.randn(n_atoms, dtype=torch.float64) * 10 + 30.0
    
    # Start with nothing refinable
    mask = torch.zeros(n_atoms, dtype=torch.bool)
    mixed = MixedTensor(initial_b_factors, refinable_mask=mask, requires_grad=True)
    
    print(f"\nInitial state: All B-factors fixed at 20.0")
    print(f"Target: Random B-factors around 30.0")
    print(f"{mixed}")
    
    # Stage 1: Refine high-confidence atoms (first 50)
    print("\n--- Stage 1: Refine atoms 0-50 ---")
    mixed.make_refinable(slice(0, 50))
    print(f"{mixed}")
    
    optimizer = torch.optim.Adam([mixed.refinable_params], lr=0.5)
    for i in range(20):
        optimizer.zero_grad()
        prediction = mixed()
        loss = ((prediction - target_b_factors) ** 2).sum()
        loss.backward()
        optimizer.step()
    
    result = mixed().detach()
    print(f"RMSE atoms 0-50: {((result[0:50] - target_b_factors[0:50])**2).mean().sqrt():.3f}")
    
    # Stage 2: Fix stage 1, refine atoms 50-100
    print("\n--- Stage 2: Fix 0-50, refine 50-100 ---")
    mixed.make_fixed(slice(0, 50), freeze_at_current=True)
    mixed.make_refinable(slice(50, 100))
    print(f"{mixed}")
    
    optimizer = torch.optim.Adam([mixed.refinable_params], lr=0.5)
    for i in range(20):
        optimizer.zero_grad()
        prediction = mixed()
        loss = ((prediction - target_b_factors) ** 2).sum()
        loss.backward()
        optimizer.step()
    
    result = mixed().detach()
    print(f"RMSE atoms 50-100: {((result[50:100] - target_b_factors[50:100])**2).mean().sqrt():.3f}")
    
    # Stage 3: Refine all previously refined atoms together
    print("\n--- Stage 3: Refine all 0-100 together ---")
    mixed.make_refinable(slice(0, 100))
    print(f"{mixed}")
    
    optimizer = torch.optim.Adam([mixed.refinable_params], lr=0.2)
    for i in range(30):
        optimizer.zero_grad()
        prediction = mixed()
        loss = ((prediction - target_b_factors) ** 2).sum()
        loss.backward()
        optimizer.step()
    
    result = mixed().detach()
    print(f"Final RMSE atoms 0-100: {((result[0:100] - target_b_factors[0:100])**2).mean().sqrt():.3f}")
    print(f"RMSE atoms 100-200 (never refined): {((result[100:200] - target_b_factors[100:200])**2).mean().sqrt():.3f}")


def example_5_conditional_refinement():
    """Example 5: Conditional refinement based on tensor values."""
    print("\n" + "=" * 70)
    print("Example 5: Conditional Refinement")
    print("=" * 70)
    
    # Create tensor with some outliers
    n = 100
    initial_values = torch.randn(n, dtype=torch.float64) * 2 + 10.0
    # Add some outliers
    initial_values[torch.randperm(n)[:10]] += 20.0
    
    target = torch.ones(n, dtype=torch.float64) * 10.0
    
    # Start with all refinable
    mixed = MixedTensor(initial_values, refinable_mask=None, requires_grad=True)
    print(f"\nInitial: {mixed}")
    print(f"Initial RMSE: {((mixed().detach() - target)**2).mean().sqrt():.3f}")
    
    # Optimize
    optimizer = torch.optim.Adam([mixed.refinable_params], lr=0.1)
    for i in range(30):
        optimizer.zero_grad()
        prediction = mixed()
        loss = ((prediction - target) ** 2).sum()
        loss.backward()
        optimizer.step()
    
    result = mixed().detach()
    print(f"After optimization RMSE: {((result - target)**2).mean().sqrt():.3f}")
    
    # Identify outliers (values far from target)
    residuals = torch.abs(result - target)
    outlier_threshold = residuals.median() + 2 * residuals.std()
    outlier_mask = residuals > outlier_threshold
    
    print(f"\nIdentified {outlier_mask.sum()} outliers")
    
    # Fix outliers and continue refinement
    mixed.make_fixed(outlier_mask, freeze_at_current=False)  # Revert to original
    print(f"After fixing outliers: {mixed}")
    
    # Optimize again without outliers
    optimizer = torch.optim.Adam([mixed.refinable_params], lr=0.1)
    for i in range(30):
        optimizer.zero_grad()
        prediction = mixed()
        loss = ((prediction - target) ** 2).sum()
        loss.backward()
        optimizer.step()
    
    result = mixed().detach()
    non_outlier_rmse = ((result[~outlier_mask] - target[~outlier_mask])**2).mean().sqrt()
    print(f"Final RMSE (non-outliers): {non_outlier_rmse:.3f}")


def example_6_integration_with_nn_module():
    """Example 6: Using MixedTensor within a neural network module."""
    print("\n" + "=" * 70)
    print("Example 6: Integration with nn.Module")
    print("=" * 70)
    
    class ProteinRefinementModel(nn.Module):
        def __init__(self, n_atoms):
            super().__init__()
            # Coordinates (3 per atom)
            coords = torch.randn(n_atoms, 3, dtype=torch.float64)
            self.coords = MixedTensor(coords, requires_grad=True)
            
            # B-factors (1 per atom)
            b_factors = torch.ones(n_atoms, dtype=torch.float64) * 20.0
            self.b_factors = MixedTensor(b_factors, requires_grad=True)
            
        def forward(self):
            return {
                'coords': self.coords(),
                'b_factors': self.b_factors()
            }
        
        def parameters(self):
            """Return all refinable parameters."""
            for param in self.coords.parameters():
                yield param
            for param in self.b_factors.parameters():
                yield param
    
    # Create model
    model = ProteinRefinementModel(n_atoms=50)
    print(f"Created model with {50} atoms")
    print(f"  Coordinates: {model.coords}")
    print(f"  B-factors: {model.b_factors}")
    
    # Stage 1: Refine only backbone atoms (assume first 30 are backbone)
    print("\n--- Refining backbone coordinates ---")
    model.coords.make_refinable(slice(0, 30))  # All coords of atoms 0-30
    print(f"  Coordinates: {model.coords}")
    
    # Create optimizer with all model parameters
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    # Dummy optimization
    target_coords = torch.randn(50, 3, dtype=torch.float64)
    for i in range(10):
        optimizer.zero_grad()
        output = model()
        loss = ((output['coords'] - target_coords) ** 2).sum()
        loss.backward()
        optimizer.step()
    
    print(f"  Optimization complete")
    
    # Stage 2: Add side chain refinement
    print("\n--- Adding side chain refinement ---")
    model.coords.make_refinable(slice(30, 50))
    print(f"  Coordinates: {model.coords}")
    
    # Stage 3: Refine B-factors for all atoms
    print("\n--- Adding B-factor refinement ---")
    model.b_factors.make_all_refinable()
    print(f"  B-factors: {model.b_factors}")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    print(f"  Number of parameters: {sum(p.numel() for p in model.parameters())}")


if __name__ == "__main__":
    example_1_basic_selection()
    example_2_boolean_mask()
    example_3_multidimensional_indexing()
    example_4_staged_refinement()
    example_5_conditional_refinement()
    example_6_integration_with_nn_module()
    
    print("\n" + "=" * 70)
    print("All examples completed successfully!")
    print("=" * 70)
