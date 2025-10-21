"""
Example demonstrating how to use named MixedTensor parameters.

The MixedTensor class now supports naming parameters, which makes it much easier
to debug, log, and understand which parameters are being refined.
"""

import torch
from multicopy_refinement.model import MixedTensor, model

print("="*80)
print("1. Creating Named MixedTensors")
print("="*80)

# Create a named MixedTensor
xyz_coords = torch.randn(100, 3)
mask = torch.zeros(100, dtype=torch.bool)
mask[20:30] = True  # Only refine atoms 20-30

xyz = MixedTensor(xyz_coords, refinable_mask=mask, name='coordinates')

print(f"\nShort representation: {repr(xyz)}")
print(f"\nDetailed representation:\n{xyz}")

print("\n" + "="*80)
print("2. Using Named Parameters in a Model")
print("="*80)

# Load a model with named parameters
test_model = model()
test_model.load_pdb_from_file('/das/work/p17/p17490/Peter/Library/multicopy_refinement/test_data/dark.pdb')

# Print summary of all parameters
test_model.print_parameters_info()

print("\n" + "="*80)
print("3. Iterating Over Named Parameters for Optimization")
print("="*80)

# Get all refinable parameters with their names
params_to_optimize = []
for name, mixed_tensor in test_model.named_mixed_tensors():
    if mixed_tensor.get_refinable_count() > 0:
        print(f"Adding {name} to optimizer ({mixed_tensor.get_refinable_count()} parameters)")
        params_to_optimize.append(mixed_tensor.refinable_params)

# Create optimizer with named parameters
optimizer = torch.optim.Adam(params_to_optimize, lr=0.01)

print(f"\nOptimizer has {len(params_to_optimize)} parameter groups")

print("\n" + "="*80)
print("4. Accessing Individual Named Parameters")
print("="*80)

# Access by attribute
print(f"XYZ: {test_model.xyz.name}")
print(f"B-factor: {test_model.b.name}")
print(f"Occupancy: {test_model.occupancy.name}")

# Change the name dynamically
test_model.xyz.name = 'atomic_coordinates'
print(f"\nRenamed: {test_model.xyz.name}")

print("\n" + "="*80)
print("5. Using Names in Optimization Loop")
print("="*80)

# Simulate a refinement loop with logging
for iteration in range(3):
    optimizer.zero_grad()
    
    # Simulate some loss
    loss = 0
    for name, mixed_tensor in test_model.named_mixed_tensors():
        if mixed_tensor.get_refinable_count() > 0:
            values = mixed_tensor()
            loss += (values ** 2).sum()
    
    loss.backward()
    
    # Log gradients with names
    print(f"\nIteration {iteration}:")
    for name, mixed_tensor in test_model.named_mixed_tensors():
        if mixed_tensor.refinable_params.grad is not None:
            grad_norm = mixed_tensor.refinable_params.grad.norm().item()
            print(f"  {mixed_tensor.name} gradient norm: {grad_norm:.4e}")
    
    optimizer.step()

print("\n" + "="*80)
print("6. Cloning Preserves Names")
print("="*80)

cloned_xyz = test_model.xyz.clone()
print(f"Original name: {test_model.xyz.name}")
print(f"Cloned name: {cloned_xyz.name}")

print("\n" + "="*80)
print("Summary:")
print("="*80)
print("""
Key benefits of named MixedTensor parameters:

1. **Debugging**: Easily identify which parameter has issues
2. **Logging**: Track gradient norms and values by parameter name
3. **Monitoring**: Print parameter summaries during refinement
4. **Organization**: Clear structure when iterating over parameters
5. **Documentation**: Self-documenting code with meaningful names

Usage tips:
- Use descriptive names: 'atomic_xyz', 'b_factors', 'occupancies'
- Access names via the .name property
- Names are preserved when cloning
- Use model.print_parameters_info() for quick summary
- Use model.named_mixed_tensors() to iterate with names
""")
