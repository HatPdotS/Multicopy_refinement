# OccupancyTensor Quick Reference

## Installation

```python
from multicopy_refinement.model import OccupancyTensor
import torch
```

## Quick Start

### Basic Creation

```python
# No sharing (each atom independent)
occ = OccupancyTensor(initial_values=torch.rand(100))

# With sharing groups
occ = OccupancyTensor(
    initial_values=torch.ones(10),
    sharing_groups=[[0,1,2], [3,4,5]]  # atoms 0,1,2 share; atoms 3,4,5 share
)

# From PDB (residue-level sharing)
occ = OccupancyTensor.from_residue_groups(
    initial_values=torch.tensor(pdb['occupancy'].values),
    pdb_dataframe=pdb
)
```

### Usage

```python
# Get occupancies (automatically expands to full size)
occupancies = occ()  # Returns torch.Tensor in [0, 1]

# In optimization
optimizer = torch.optim.Adam([occ.refinable_params], lr=0.01)
for epoch in range(100):
    optimizer.zero_grad()
    occ_values = occ()
    loss = my_loss_function(occ_values)
    loss.backward()
    optimizer.step()
```

## Key Features

### Collapsed Storage

```python
# Internal storage is compressed
print(f"Atoms: {occ.shape[0]}")              # 100
print(f"Parameters: {occ.collapsed_shape[0]}")  # 20 (if 20 residues)
# Saves 80% memory!
```

### Sharing Groups

```python
# All atoms in group automatically have same value
groups = [[0,1,2], [3,4]]
occ = OccupancyTensor(torch.ones(5), sharing_groups=groups)

values = occ()
assert values[0] == values[1] == values[2]  # Group 0
assert values[3] == values[4]                # Group 1
```

### Refinable Mask

```python
# Only refine specific atoms (mask in full space)
refinable = torch.tensor([False, True, True, False, True])
occ = OccupancyTensor(
    initial_values=torch.ones(5),
    refinable_mask=refinable
)

# Only 3 parameters are optimizable
print(occ.get_refinable_count())  # 3
```

## Common Patterns

### Alternative Conformations

```python
# Residue with conformations A and B
conf_a = [0, 1, 2, 3]  # 4 atoms
conf_b = [4, 5, 6, 7]  # 4 atoms

occ = OccupancyTensor(
    initial_values=torch.tensor([0.6]*4 + [0.4]*4),
    sharing_groups=[conf_a, conf_b]
)

# Each conformation has one parameter
# Occupancies sum to 1.0 can be enforced in loss
```

### Partial Occupancies

```python
# Only refine atoms with occupancy < 1.0
mask = initial_occ < 1.0
occ = OccupancyTensor(
    initial_values=initial_occ,
    refinable_mask=mask
)
```

### Group Operations

```python
# Set all atoms in group to same value
occ.set_group_occupancy(group_idx=0, value=0.8)

# Get group occupancy
val = occ.get_group_occupancy(group_idx=0)

# Clamp values
occ = occ.clamp(min_value=0.1, max_value=0.9)
```

## Properties & Methods

| Property/Method | Description |
|----------------|-------------|
| `occ()` | Get full occupancy tensor (forward pass) |
| `occ.shape` | Full tensor shape `(n_atoms,)` |
| `occ.collapsed_shape` | Internal storage shape `(n_params,)` |
| `occ.expansion_mask` | Maps atoms to collapsed indices |
| `occ.refinable_params` | Optimizable parameters (collapsed) |
| `occ.get_refinable_count()` | Number of refinable parameters |
| `occ.set_group_occupancy(idx, val)` | Set group occupancy |
| `occ.get_group_occupancy(idx)` | Get group occupancy |
| `occ.clamp(min, max)` | Return clamped copy |

## Debugging

```python
# Check storage efficiency
print(f"Compression ratio: {occ.shape[0] / occ.collapsed_shape[0]:.1f}x")

# Verify sharing
for i, group in enumerate(occ.sharing_groups):
    vals = occ()[group]
    print(f"Group {i}: {vals}")  # Should all be equal

# Check expansion mask
print(occ.expansion_mask)  # [0,0,0, 1,1, 2, ...]

# Detailed info
print(occ)  # Shows shapes, refinable counts, etc.
```

## Common Mistakes

### ❌ Wrong: Providing collapsed values

```python
# DON'T do this:
initial = torch.ones(5)  # Only 5 values for 10 atoms
occ = OccupancyTensor(initial, sharing_groups=...)  # ERROR!
```

### ✅ Correct: Always provide full values

```python
# DO this:
initial = torch.ones(10)  # One value per atom
occ = OccupancyTensor(initial, sharing_groups=...)  # Automatically collapses
```

### ❌ Wrong: Mask in collapsed space

```python
# DON'T do this:
mask = torch.tensor([True, False, True])  # 3 groups
occ = OccupancyTensor(..., refinable_mask=mask)  # Wrong size!
```

### ✅ Correct: Mask in full space

```python
# DO this:
mask = torch.tensor([True, True, False, False, ...])  # One per atom
occ = OccupancyTensor(..., refinable_mask=mask)  # Automatically collapsed
```

## Performance Tips

1. **Use residue-level sharing** for typical proteins (10x memory savings)
2. **Batch operations** when possible (forward pass is vectorized)
3. **Keep sharing groups static** (don't recreate OccupancyTensor frequently)
4. **Use GPU** for large structures (`occ.to('cuda')` works)

## Examples

### Minimal Example
```python
occ = OccupancyTensor(torch.ones(100))
optimizer = torch.optim.Adam([occ.refinable_params], lr=0.01)

for _ in range(100):
    optimizer.zero_grad()
    loss = (occ() - target).pow(2).sum()
    loss.backward()
    optimizer.step()
```

### With Model Class
```python
from multicopy_refinement.model import Model

model = Model()
model.load_pdb_from_file('structure.pdb')

# Replace default occupancy with OccupancyTensor
model.occupancy = OccupancyTensor.from_residue_groups(
    initial_values=torch.tensor(model.pdb['occupancy'].values),
    pdb_dataframe=model.pdb,
    dtype=model.dtype_float,
    device=model.device
)

# Now use model.occupancy() in refinement
```

## Testing

Run tests to verify installation:
```bash
python tests/occupancy_manager/test_occupancy_tensor_collapsed.py
```

Should see: `ALL TESTS PASSED! ✓✓✓`

## Need Help?

- Full documentation: `README_COLLAPSED_STORAGE.md`
- Test examples: `test_occupancy_tensor_collapsed.py`
- Implementation: `multicopy_refinement/model.py` (class OccupancyTensor)
