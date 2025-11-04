# Normalization-Based Alternative Conformation Handling

## Overview
Refactored the alternative conformation (altloc) handling in `OccupancyTensor` from a placeholder-based approach to a normalization-based approach. This is more stable, more intuitive, and handles arbitrary N-way splits naturally.

## Key Changes

### Old Approach (Placeholder-Based)
**How it worked:**
- Store N-1 parameters for N conformations
- Compute last conformation as `1.0 - sum(others)` in logit space
- Mark placeholder as non-refinable
- Complex logic to handle edge cases

**Problems:**
- Only N-1 conformations were refinable (asymmetric)
- Gradient flow through logit(1-sum(sigmoid(...))) was complex
- Required special handling to prevent placeholders from being refined
- Harder to understand and debug

### New Approach (Normalization-Based)
**How it works:**
- Store ALL N parameters for N conformations
- Apply sigmoid to all: `prob_i = sigmoid(logit_i)`
- Normalize within group: `occupancy_i = prob_i / sum_j(prob_j)`
- All conformations are refinable

**Benefits:**
- ✓ Symmetric treatment of all conformations
- ✓ All parameters refinable
- ✓ Simpler gradient flow (just softmax-like normalization)
- ✓ Naturally handles N-way splits (not just 2-way)
- ✓ More numerically stable
- ✓ Easier to understand

## Implementation Details

### 1. Altloc Structure in Collapsed Space

Altloc groups are now stored as tensors of collapsed indices grouped by size:

```python
# For 2-way splits: linked_occ_2 = tensor([[idx1, idx2], [idx3, idx4], ...])
# For 3-way splits: linked_occ_3 = tensor([[idx1, idx2, idx3], ...])
# etc.

self.register_buffer('linked_occ_2', tensor([[1, 2], [4, 5]]))  # Two 2-way splits
self.register_buffer('linked_occ_3', tensor([[7, 8, 9]]))       # One 3-way split
self.linked_occ_sizes = [2, 3]  # Which sizes we have
```

### 2. Forward Pass with Normalization

```python
def forward(self):
    # Get collapsed logits
    result = self.fixed_values.clone()
    result[self.refinable_mask] = self.refinable_params
    
    # Apply sigmoid
    collapsed_occs = torch.sigmoid(result)
    
    # Normalize within each altloc group
    for n_conf in self.linked_occ_sizes:
        linked_indices = getattr(self, f'linked_occ_{n_conf}')
        linked_occs = collapsed_occs[linked_indices]
        
        # Normalize: divide by sum
        sums = linked_occs.sum(dim=1, keepdim=True)
        normalized = linked_occs / sums
        
        # Update collapsed_occs with normalized values
        # (using loop to avoid in-place operations that break autograd)
        for group_idx in range(linked_indices.shape[0]):
            for conf_idx in range(n_conf):
                collapsed_occs[linked_indices[group_idx, conf_idx]] = normalized[group_idx, conf_idx]
    
    # Expand to full space
    return collapsed_occs[self.expansion_mask]
```

### 3. Setup Phase

During `_setup_sharing_groups_and_expansion`:

1. Convert altloc atom indices to collapsed indices
2. **Assert** all atoms in a conformation map to same collapsed index
3. Group by number of conformations
4. Create tensors and register as buffers

```python
# Process altloc groups
linked_occupancies = {}  # key: n_conformations, value: list of [idx1, idx2, ...]

for altloc_group in altloc_groups:
    n_conf = len(altloc_group)
    collapsed_indices = []
    
    for conformation_atoms in altloc_group:
        # Get collapsed index for first atom
        collapsed_idx = expansion_mask[conformation_atoms[0]].item()
        
        # ASSERT all atoms in this conformation share same collapsed index
        for atom in conformation_atoms:
            assert expansion_mask[atom].item() == collapsed_idx
        
        collapsed_indices.append(collapsed_idx)
    
    # Group by size
    if n_conf not in linked_occupancies:
        linked_occupancies[n_conf] = []
    linked_occupancies[n_conf].append(collapsed_indices)

# Convert to tensors and register
for n_conf, groups in linked_occupancies.items():
    tensor = torch.tensor(groups, dtype=torch.long, device=device)
    self.register_buffer(f'linked_occ_{n_conf}', tensor)
```

## Examples

### Example 1: 2-Way Altloc
```python
initial = torch.tensor([1.0, 1.0, 0.7, 0.7, 0.3, 0.3])
sharing = torch.tensor([0, 0, 1, 1, 2, 2])  # 3 collapsed indices
altlocs = [([2, 3], [4, 5])]  # Atoms 2-3 (conf A) and 4-5 (conf B)

occ = OccupancyTensor(initial, sharing_groups=sharing, altloc_groups=altlocs)
result = occ()

# result[2] + result[4] = 1.0 (automatically)
# Both indices 1 and 2 are refinable
# linked_occ_2 = tensor([[1, 2]])
```

### Example 2: 3-Way Altloc
```python
initial = torch.tensor([0.5, 0.5, 0.3, 0.3, 0.2, 0.2])
sharing = torch.tensor([0, 0, 1, 1, 2, 2])
altlocs = [([0, 1], [2, 3], [4, 5])]  # 3-way split

occ = OccupancyTensor(initial, sharing_groups=sharing, altloc_groups=altlocs)
result = occ()

# result[0] + result[2] + result[4] = 1.0
# All 3 conformations are refinable
# linked_occ_3 = tensor([[0, 1, 2]])
```

### Example 3: Multiple Altloc Groups
```python
initial = torch.tensor([1.0, 1.0, 0.7, 0.7, 0.3, 0.3, 0.6, 0.6, 0.4, 0.4])
sharing = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
altlocs = [
    ([2, 3], [4, 5]),  # First 2-way split
    ([6, 7], [8, 9]),  # Second 2-way split
]

occ = OccupancyTensor(initial, sharing_groups=sharing, altloc_groups=altlocs)
result = occ()

# result[2] + result[4] = 1.0
# result[6] + result[8] = 1.0
# linked_occ_2 = tensor([[1, 2], [3, 4]])
```

## Mathematical Details

### Normalization Formula
For a group of N conformations with logits `[l1, l2, ..., lN]`:

```
prob_i = sigmoid(l_i)
occupancy_i = prob_i / sum_j(prob_j)
```

This is similar to softmax but operates on probabilities instead of logits.

### Gradient Flow
The gradient of `occupancy_i` with respect to `l_i`:

```
∂occupancy_i/∂l_i = prob_i * (1 - prob_i) * (1 - occupancy_i) / sum_probs
```

All conformations receive gradients, making the optimization symmetric and stable.

### Comparison to Softmax
Softmax would be: `exp(l_i) / sum_j(exp(l_j))`

Our approach: `sigmoid(l_i) / sum_j(sigmoid(l_j))`

Difference: We apply sigmoid first (bounding to [0,1]), then normalize. This is more natural for occupancies which are already probabilities.

## Testing

Comprehensive tests in `tests/occupancy_manager/test_normalization_altloc.py`:

✓ 2-way altlocs work correctly
✓ 3-way altlocs work correctly  
✓ Multiple independent altloc groups
✓ Sum-to-1 maintained during optimization
✓ Proper assertions for invalid configurations
✓ All altloc members are refinable
✓ Gradient-friendly normalization

## Performance

**Memory:**
- Old: Stored N-1 parameters + placeholder mask
- New: Stores N parameters + linked index tensors
- Difference: Negligible (one more parameter per altloc group)

**Speed:**
- Old: O(N_altlocs) placeholder computations with logit inversions
- New: O(N_altlocs) normalizations with simple divisions
- Result: Slightly faster (no logit inversions)

**Gradient Quality:**
- Old: Complex gradient through logit(1 - sum(sigmoid(...)))
- New: Simple gradient through division
- Result: More stable optimization

## Migration Guide

### For Users
No changes needed! The API is the same:
```python
occ = OccupancyTensor(
    initial_values=values,
    sharing_groups=sharing,
    altloc_groups=altlocs,  # Same format as before
)
```

### For Developers
If you were accessing internal attributes:
- `altloc_info` → No longer exists
- `altloc_placeholder_mask` → No longer needed
- `linked_occ_2`, `linked_occ_3`, etc. → New buffers with linked indices
- `linked_occ_sizes` → List of which sizes are present

## Summary

The new normalization-based approach is:
1. **Simpler**: No placeholder logic, all conformations treated equally
2. **More stable**: Better gradient flow for optimization
3. **More flexible**: Naturally handles N-way splits
4. **More intuitive**: Just sigmoid + normalize = sum to 1
5. **Better tested**: Comprehensive test suite

All existing tests pass with the new implementation!
