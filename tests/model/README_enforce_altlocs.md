# Enforce Alternative Conformation Occupancy Constraints

## Overview

The `enforce_alternative_conformations()` method enforces occupancy constraints on alternative conformations in protein structures. It ensures that:

1. **Uniform occupancy within each conformation**: All atoms in the same conformation have the same occupancy value
2. **Sum to unity**: The occupancies across all conformations of a residue sum to 1.0

This is essential for maintaining physically meaningful occupancy values during structure refinement.

## Implementation

### Location
- **File**: `multicopy_refinement/model_new.py`
- **Class**: `model`
- **Method**: `enforce_alternative_conformations()`

### Algorithm

The method applies a two-step normalization process:

```
For each residue with alternative conformations:
    Step 1: Compute mean occupancy for each conformation
    Step 2: Normalize means to sum to 1.0
    Step 3: Set all atoms in each conformation to their normalized mean
```

### Mathematical Formulation

For a residue with conformations A, B, C, ...:

1. **Mean occupancy per conformation**:
   ```
   mean_A = (occ_A1 + occ_A2 + ... + occ_An) / n_atoms
   mean_B = (occ_B1 + occ_B2 + ... + occ_Bn) / n_atoms
   ...
   ```

2. **Normalize to sum to 1.0**:
   ```
   total = mean_A + mean_B + mean_C + ...
   norm_A = mean_A / total
   norm_B = mean_B / total
   ...
   ```

3. **Apply uniform occupancy**:
   ```
   occ_A1 = occ_A2 = ... = occ_An = norm_A
   occ_B1 = occ_B2 = ... = occ_Bn = norm_B
   ...
   ```

## Usage

### Basic Usage

```python
from multicopy_refinement.model_new import model

# Load a PDB file with alternative conformations
m = model()
m.load_pdb_from_file('structure.pdb')

# Enforce occupancy constraints
m.enforce_alternative_conformations()

# Occupancies now satisfy:
# 1. All atoms in same conformation have same occupancy
# 2. Occupancies across conformations sum to 1.0
```

### Integration with Refinement

The typical use case is within a refinement loop:

```python
# Set up optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(num_epochs):
    # Forward pass
    loss = compute_loss(model)
    
    # Backward pass
    loss.backward()
    
    # Update parameters
    optimizer.step()
    optimizer.zero_grad()
    
    # Enforce occupancy constraints
    model.enforce_alternative_conformations()
```

### Checking Constraints

You can verify that constraints are satisfied:

```python
occ = model.occupancy().detach()

# Check 1: Uniform occupancy within conformations
for group in model.altloc_pairs:
    for conf_indices in group:
        conf_occ = occ[conf_indices]
        std = conf_occ.std().item()
        print(f"Std deviation: {std:.10f}")  # Should be ~0

# Check 2: Sum to 1.0
for group in model.altloc_pairs:
    mean_occupancies = [occ[conf].mean() for conf in group]
    total = sum(mean_occupancies).item()
    print(f"Sum: {total:.6f}")  # Should be 1.0
```

## Examples

### Example 1: Simple Pair (A, B)

```python
# Before enforcement:
# Conf A: [0.60, 0.65, 0.62]  mean = 0.623
# Conf B: [0.40, 0.35, 0.38]  mean = 0.377
# Sum = 1.000 (already normalized)

m.enforce_alternative_conformations()

# After enforcement:
# Conf A: [0.623, 0.623, 0.623]  mean = 0.623
# Conf B: [0.377, 0.377, 0.377]  mean = 0.377
# Sum = 1.000
```

### Example 2: Triplet (A, B, C) Requiring Normalization

```python
# Before enforcement:
# Conf A: [0.50, 0.55]  mean = 0.525
# Conf B: [0.30, 0.35]  mean = 0.325
# Conf C: [0.15, 0.20]  mean = 0.175
# Sum = 1.025 (needs normalization)

m.enforce_alternative_conformations()

# After enforcement:
# Conf A: [0.512, 0.512]  mean = 0.512  (0.525 / 1.025)
# Conf B: [0.317, 0.317]  mean = 0.317  (0.325 / 1.025)
# Conf C: [0.171, 0.171]  mean = 0.171  (0.175 / 1.025)
# Sum = 1.000
```

### Example 3: Accessing Modified Occupancies

```python
# Get occupancies after enforcement
occ = model.occupancy().detach()

# For a specific residue group
group = model.altloc_pairs[0]

for i, conf_indices in enumerate(group):
    # Get atoms for this conformation
    atoms = model.pdb.loc[conf_indices.tolist()]
    altloc = atoms['altloc'].iloc[0]
    
    # Get occupancy (all atoms have same value)
    occupancy = occ[conf_indices[0]].item()
    
    print(f"Conformation {altloc}: occupancy = {occupancy:.4f}")
```

## Properties and Guarantees

### 1. Idempotent Operation
Calling `enforce_alternative_conformations()` multiple times produces the same result:

```python
m.enforce_alternative_conformations()
occ1 = m.occupancy().detach().clone()

m.enforce_alternative_conformations()
occ2 = m.occupancy().detach()

assert torch.allclose(occ1, occ2)  # True
```

### 2. Preserves Non-Altloc Atoms
Atoms without alternative conformations are not modified:

```python
# Get non-altloc atoms
non_altloc_mask = model.pdb['altloc'] == ''
non_altloc_indices = model.pdb[non_altloc_mask].index

occ_before = model.occupancy()[non_altloc_indices].clone()
model.enforce_alternative_conformations()
occ_after = model.occupancy()[non_altloc_indices]

assert torch.allclose(occ_before, occ_after)  # True
```

### 3. Handles Edge Cases
- **Zero occupancies**: If all occupancies are zero, they are distributed equally
- **Single conformation**: Residues with only one altloc are not processed
- **No altlocs**: Method returns immediately if no alternative conformations exist

## Implementation Details

### Data Flow

1. **Get current occupancies**: Extract from MixedTensor
2. **Process each group**: Apply mean and normalization
3. **Update MixedTensor**: Modify both fixed and refinable parameters

### Code Structure

```python
def enforce_alternative_conformations(self):
    # Early return if no altlocs
    if not hasattr(self, 'altloc_pairs') or len(self.altloc_pairs) == 0:
        return
    
    # Get current values
    current_occ = self.occupancy().detach().clone()
    
    # Process each residue group
    for group in self.altloc_pairs:
        # Step 1: Compute means
        mean_occupancies = [current_occ[conf].mean() for conf in group]
        
        # Step 2: Normalize
        total = sum(mean_occupancies)
        if total > 0:
            normalized_means = [m / total for m in mean_occupancies]
        else:
            normalized_means = [1.0 / len(group)] * len(group)
        
        # Step 3: Apply uniform values
        for conf_indices, norm_mean in zip(group, normalized_means):
            current_occ[conf_indices] = norm_mean
    
    # Update MixedTensor
    self.occupancy.fixed_values = current_occ.clone()
    if self.occupancy.refinable_mask.any():
        self.occupancy.refinable_params.data = current_occ[self.occupancy.refinable_mask].clone()
```

## Performance

- **Time Complexity**: O(n) where n is the number of atoms with alternative conformations
- **Memory**: Minimal - operates on existing tensors
- **In-place**: Modifies occupancy values in-place within the MixedTensor

## Testing

Comprehensive tests are available in:
- `tests/model/test_enforce_altlocs.py` - Full test suite (8 tests, all passing)
- `tests/model/example_enforce_usage.py` - Usage examples

Run tests:
```bash
cd tests/model
python test_enforce_altlocs.py
python example_enforce_usage.py
```

### Test Coverage

1. ✅ Basic enforcement on real data
2. ✅ Uniform occupancy within conformations
3. ✅ Occupancies sum to 1.0
4. ✅ Non-altloc atoms unchanged
5. ✅ Synthetic test cases
6. ✅ Triplet conformations (A, B, C)
7. ✅ Multiple enforcements (idempotency)
8. ✅ Statistics and edge cases

## Common Use Cases

### Use Case 1: During Refinement

Apply after each gradient update to maintain constraints:

```python
for epoch in range(num_epochs):
    loss.backward()
    optimizer.step()
    model.enforce_alternative_conformations()  # Apply constraints
```

### Use Case 2: Before Writing PDB

Ensure occupancies are correct before saving:

```python
# Refine structure
refine(model)

# Enforce constraints one final time
model.enforce_alternative_conformations()

# Write PDB
model.write_pdb('refined_structure.pdb')
```

### Use Case 3: Analyzing Conformations

Check occupancy distribution after enforcement:

```python
model.enforce_alternative_conformations()

occ = model.occupancy().detach()

for group in model.altloc_pairs:
    occupancies = [occ[conf].mean().item() for conf in group]
    print(f"Occupancies: {occupancies}")  # e.g., [0.6, 0.4] or [0.5, 0.3, 0.2]
```

## Notes

- **Automatic in refinement**: Should be called after each parameter update
- **Works with MixedTensor**: Properly updates both refinable and fixed parameters
- **Requires altloc_pairs**: Depends on `register_alternative_conformations()` being called first
- **No gradients**: Operates on detached tensors to avoid interfering with autograd

## Related Methods

- `register_alternative_conformations()`: Must be called first to identify altloc groups
- `update_pdb()`: Updates the internal PDB DataFrame with current parameter values
- `write_pdb()`: Writes the structure to a file with enforced occupancies

## References

Alternative conformations in protein crystallography typically represent:
- Different rotamer states of side chains
- Multiple binding modes of ligands
- Conformational flexibility of loops
- Crystal packing disorder

Proper occupancy refinement ensures that the model accurately represents the electron density observed in the experiment.
