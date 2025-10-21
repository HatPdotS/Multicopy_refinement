# Implementation Summary: register_alternative_conformations

## Overview
Successfully implemented the `register_alternative_conformations()` method for the `model` class to identify and store alternative conformation pairs/groups in protein structures.

## Files Created/Modified

### Modified Files
1. **`/das/work/p17/p17490/Peter/Library/multicopy_refinement/multicopy_refinement/model_new.py`**
   - Implemented `register_alternative_conformations()` method (lines 91-137)
   - Method is automatically called during `load_pdb_from_file()`

### Test Files Created (in `/das/work/p17/p17490/Peter/Library/multicopy_refinement/tests/model/`)

1. **`test_alternative_conformations.py`** - Comprehensive test suite with 9 tests
2. **`example_altloc_usage.py`** - Demonstration of usage with real examples
3. **`verify_implementation.py`** - Quick verification script
4. **`README_ALTERNATIVE_CONFORMATIONS.md`** - Complete documentation

### Analysis/Debug Scripts Created
5. **`analyze_altlocs.py`** - Initial analysis of altloc patterns
6. **`check_index_vs_iloc.py`** - DataFrame index verification
7. **`check_groupby.py`** - GroupBy logic verification
8. **`debug_altlocs.py`** - Debugging helper
9. **`test_temp_pdb.py`** - Temporary PDB file testing

## Implementation Details

### Function Behavior
- **Input**: Called automatically during PDB loading
- **Output**: Populates `self.altloc_pairs` list with tuples of indices
- **Data Structure**: `[(idx1, idx2), (idx3, idx4, idx5), ...]`
  - Each tuple contains DataFrame indices of atoms representing alternative conformations
  - Sorted alphabetically by altloc identifier (A, B, C, ...)
  - Supports 2-way, 3-way, and N-way conformations

### Algorithm
```python
1. Filter atoms where altloc != ''
2. Group by (resname, resseq, chainid, name)
3. For each group with ≥2 atoms:
   a. Extract indices and altloc values
   b. Sort by altloc identifier
   c. Store as tuple in altloc_pairs
```

## Test Results

All 9 tests pass successfully:

```
================================================================================
Testing register_alternative_conformations
================================================================================

Real PDB file... ✓ PASSED
Indices match atoms... ✓ PASSED
Altloc ordering... ✓ PASSED
No duplicate indices... ✓ PASSED
Count conformations... ✓ PASSED
PDB without altlocs... ✓ PASSED
Specific residue... ✓ PASSED
Attribute exists... ✓ PASSED
Manual PDB... ✓ PASSED

================================================================================
Results: 9 passed, 0 failed out of 9 tests
================================================================================
```

### Test Coverage
- ✓ Real PDB files with altlocs
- ✓ PDB files without altlocs
- ✓ Manually created PDB files
- ✓ 2-way conformations
- ✓ 3-way conformations
- ✓ Index correctness
- ✓ Sorting consistency
- ✓ No duplicate indices
- ✓ Specific residue queries

## Example Results from dark.pdb

- **Total groups found**: 489
- **2-way conformations**: 475 groups
- **3-way conformations**: 14 groups

Example group (ARG-86 Chain B, CA atom):
- Indices: (8756, 8775)
- Altlocs: ['A', 'B']
- Coordinates:
  - Altloc A: (7.727, -32.220, -9.351), occupancy 0.57
  - Altloc B: (7.719, -32.211, -9.356), occupancy 0.43

## Usage Example

```python
from multicopy_refinement.model_new import model

# Load PDB file
m = model()
m.load_pdb_from_file('structure.pdb')

# Access alternative conformations
print(f"Found {len(m.altloc_pairs)} groups")

# Iterate over groups
for group in m.altloc_pairs:
    atoms = m.pdb.loc[list(group)]
    print(f"{atoms['name'].iloc[0]} - {atoms['resname'].iloc[0]}-{atoms['resseq'].iloc[0]}")
    print(f"  Altlocs: {atoms['altloc'].tolist()}")
```

## Key Features

✓ **Automatic detection** during PDB loading
✓ **Handles multiple conformations** (2-way, 3-way, N-way)
✓ **Consistent ordering** (sorted by altloc)
✓ **No duplicates** (each atom in at most one group)
✓ **Comprehensive testing** (9 tests covering various scenarios)
✓ **Well documented** (README, examples, inline comments)
✓ **Backward compatible** (doesn't break existing functionality)

## Performance

- **Time Complexity**: O(N log N) where N = atoms with altlocs
- **Space Complexity**: O(M) where M = number of groups
- **Tested on**: dark.pdb with 17,224 atoms, 2,114 atoms with altlocs
- **Result**: 489 groups identified instantly

## Integration

The method integrates seamlessly with the existing model class:
- Called automatically in `load_pdb_from_file()`
- Stores results in `self.altloc_pairs` attribute
- Compatible with existing MixedTensor refinement system
- No changes required to existing code

## Verification

✓ All new tests pass
✓ Original tests still pass
✓ Example scripts run successfully
✓ Documentation complete
