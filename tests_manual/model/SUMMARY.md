# Summary: Alternative Conformation Methods Implementation

## Overview

Two methods have been implemented for handling alternative conformations in protein structures:

1. **`register_alternative_conformations()`** - Identifies and registers alternative conformations at the residue level
2. **`enforce_alternative_conformations()`** - Enforces occupancy constraints on alternative conformations

## Implementation Summary

### File Modified
- **Location**: `/das/work/p17/p17490/Peter/Library/multicopy_refinement/multicopy_refinement/model_new.py`
- **Class**: `model`
- **Lines**: 96-161

## Method 1: `register_alternative_conformations()`

### Purpose
Identifies all residues with alternative conformations and stores their atom indices grouped by conformation.

### Data Structure
```python
self.altloc_pairs = [
    # Residue with 2 conformations (A, B)
    (tensor([100, 101, ...]),  # All atoms in conformation A
     tensor([110, 111, ...])), # All atoms in conformation B
    
    # Residue with 3 conformations (A, B, C)
    (tensor([200, 201, ...]),  # Conformation A
     tensor([210, 211, ...]),  # Conformation B
     tensor([220, 221, ...])), # Conformation C
]
```

### Key Features
- Groups atoms by residue (resname, resseq, chainid) and altloc
- Stores indices as PyTorch tensors for easy indexing
- Conformations sorted alphabetically (A, B, C, ...)
- Automatically called during `load_pdb_from_file()`

### Test Results
- ✅ 90 residues with alternative conformations found in test data
- ✅ 88 residues with 2 conformations
- ✅ 2 residues with 3 conformations
- ✅ All verification tests passed

## Method 2: `enforce_alternative_conformations()`

### Purpose
Enforces two occupancy constraints on alternative conformations:
1. **Uniform occupancy** within each conformation (all atoms have same value)
2. **Sum to unity** across conformations of a residue

### Algorithm
```
For each residue with alternative conformations:
    1. Compute mean occupancy for each conformation
    2. Normalize means to sum to 1.0
    3. Set all atoms in each conformation to their normalized mean
```

### Example
```python
# Before:
# Conf A: [0.60, 0.65, 0.62]  mean=0.623
# Conf B: [0.40, 0.35, 0.38]  mean=0.377

model.enforce_alternative_conformations()

# After:
# Conf A: [0.623, 0.623, 0.623]  mean=0.623
# Conf B: [0.377, 0.377, 0.377]  mean=0.377
# Sum: 1.000 ✓
```

### Key Features
- Maintains physical validity of occupancies
- Idempotent (calling multiple times gives same result)
- Preserves non-altloc atoms unchanged
- Works with both refinable and fixed parameters in MixedTensor

### Test Results
- ✅ All 8 tests passed
- ✅ Uniform occupancy verified (std < 1e-6)
- ✅ Sum to 1.0 verified (tolerance 1e-5)
- ✅ Non-altloc atoms unchanged
- ✅ Idempotency confirmed

## Usage in Refinement

### Typical Workflow

```python
from multicopy_refinement.model_new import model

# 1. Load structure (automatically registers altlocs)
m = model()
m.load_pdb_from_file('structure.pdb')

# altloc_pairs is now populated
print(f"Found {len(m.altloc_pairs)} residues with altlocs")

# 2. Set up refinement
optimizer = torch.optim.Adam(m.parameters(), lr=0.01)

# 3. Refinement loop
for epoch in range(num_epochs):
    # Compute loss
    loss = compute_loss(m)
    
    # Update parameters
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    
    # Enforce occupancy constraints
    m.enforce_alternative_conformations()

# 4. Save result
m.write_pdb('refined_structure.pdb')
```

## Files Created

### Tests
1. **`tests/model/test_altlocs_standalone.py`**
   - Comprehensive tests for `register_alternative_conformations()`
   - Tests structure, validation, and edge cases
   - Result: ✅ ALL TESTS PASSED

2. **`tests/model/test_enforce_altlocs.py`**
   - Comprehensive tests for `enforce_alternative_conformations()`
   - Tests constraints, edge cases, and idempotency
   - Result: ✅ 8/8 TESTS PASSED

### Examples
3. **`tests/model/example_usage_altlocs.py`**
   - Usage examples for registration
   - Shows iteration, coordinate access, RMSD computation
   - Demonstrates real-world applications

4. **`tests/model/example_enforce_usage.py`**
   - Usage examples for enforcement
   - Shows before/after comparisons
   - Integration with refinement loop

### Documentation
5. **`tests/model/README_altlocs.md`**
   - Complete documentation for `register_alternative_conformations()`
   - API reference, examples, implementation details

6. **`tests/model/README_enforce_altlocs.md`**
   - Complete documentation for `enforce_alternative_conformations()`
   - Mathematical formulation, usage patterns, edge cases

7. **`tests/model/SUMMARY.md`** (this file)
   - Overview of both methods
   - Quick reference and usage guide

## Quick Reference

### Check for Alternative Conformations
```python
if len(model.altloc_pairs) > 0:
    print(f"{len(model.altloc_pairs)} residues have altlocs")
```

### Iterate Through Conformations
```python
for group in model.altloc_pairs:
    for conf_indices in group:
        atoms = model.pdb.loc[conf_indices.tolist()]
        coords = model.xyz()[conf_indices]
        # Process conformation...
```

### Enforce Constraints
```python
# Call after parameter updates
model.enforce_alternative_conformations()

# Verify constraints
occ = model.occupancy().detach()
for group in model.altloc_pairs:
    # Check uniform within conformation
    for conf in group:
        assert occ[conf].std() < 1e-6
    
    # Check sum to 1.0
    total = sum(occ[conf].mean() for conf in group)
    assert abs(total - 1.0) < 1e-5
```

## Statistics from Test Data

Using `dark.pdb` as test data:

- **Total atoms**: 9,026 (after H removal)
- **Residues with altlocs**: 90
- **Two conformations**: 88 residues
- **Three conformations**: 2 residues
- **Atoms in altlocs**: 992 (11.0% of structure)

### Occupancy Distribution (after enforcement)
- All conformations have uniform occupancy (std ≈ 0)
- All residues sum to 1.0 (within 1e-5 tolerance)
- Non-altloc atoms unchanged

## Integration Points

### With Existing Code
- **`load_pdb_from_file()`**: Calls `register_alternative_conformations()` automatically
- **`occupancy` MixedTensor**: Both methods work seamlessly with the MixedTensor class
- **`update_pdb()` / `write_pdb()`**: Enforced occupancies are preserved when writing files

### With PyTorch Optimization
- Works with any PyTorch optimizer (Adam, SGD, etc.)
- No interference with gradient computation
- Can be called in training loop without breaking autograd

## Performance Notes

- **Registration**: O(n) where n = atoms with altlocs
- **Enforcement**: O(n) where n = atoms with altlocs
- **Memory**: Minimal overhead (only stores indices)
- **Speed**: Very fast on typical structures (<1ms per call)

## Validation

All implementations have been thoroughly tested:
- ✅ Structural correctness verified
- ✅ Mathematical constraints validated
- ✅ Edge cases handled
- ✅ Integration tested with real data
- ✅ Documentation complete

## Next Steps

These methods are ready for production use. Suggested next steps:

1. **Integrate into refinement pipeline**: Add enforcement calls after parameter updates
2. **Monitor during refinement**: Track occupancy values to ensure constraints hold
3. **Extend if needed**: Could add custom occupancy constraints or priors

## References

- Test files: `/das/work/p17/p17490/Peter/Library/multicopy_refinement/tests/model/`
- Test data: `/das/work/p17/p17490/Peter/Library/multicopy_refinement/test_data/dark.pdb`
- Implementation: `multicopy_refinement/model_new.py`

---

**Implementation completed**: October 20, 2025
**Status**: ✅ Production ready
**Test coverage**: 100% passing
