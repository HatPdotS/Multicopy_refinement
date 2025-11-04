# Model Class Cleanup Summary

**Date**: March 2025  
**Change**: Removed redundant occupancy handling methods

---

## What Was Removed

### `sanitize_occupancies()` method ❌ REMOVED

**Previous Implementation**:
```python
def sanitize_occupancies(self):
    """
    Sanitize occupancy values to ensure they are within valid bounds [0.0, 1.0].
    
    This method clamps all occupancy values to be within the range [0.0, 1.0].
    It updates both the fixed and refinable parameters in the occupancy MixedTensor.
    """
    self.occupancy = self.occupancy.clamp(0.0, 1.0)
    self.enforce_occ_alternative_conformations()
```

**Why Removed**:
- ✅ **Redundant**: `OccupancyTensor` with sigmoid parameterization **automatically** ensures all values are in [0,1]
- ✅ **No manual clamping needed**: The sigmoid function mathematically guarantees bounds
- ✅ **Cleaner API**: One less method users need to call
- ✅ **Performance**: No extra clamping operation needed

---

## What Was Kept (and Why)

### `enforce_occ_alternative_conformations()` method ✓ KEPT

**Updated Implementation**:
```python
def enforce_occ_alternative_conformations(self):
    """
    Enforce sum-to-one constraint for alternative conformations.
    
    This method ensures that alternative conformations (altlocs) of the same residue
    sum to 1.0. It should be called periodically during optimization to maintain
    this crystallographic constraint.
    
    The OccupancyTensor class already handles:
    - Residue-level sharing (atoms in same altloc have same occupancy)
    - Bounds [0,1] via sigmoid parameterization
    
    This method additionally enforces:
    - Sum to one: The occupancies across all altlocs of a residue sum to 1.0
    ...
    """
```

**Why Kept**:
- ✅ **Still needed**: Enforces sum-to-1 constraint for alternative conformations
- ✅ **Cannot be automated**: This is a **cross-group** constraint that requires knowing which groups are alternative conformations
- ✅ **Works with OccupancyTensor**: Updated to work correctly with collapsed storage
- ✅ **Tested**: Integration test verifies it works correctly
- ✅ **Called during optimization**: Should be called periodically (e.g., every N steps)

**Key Point**: `OccupancyTensor` handles **per-atom** or **per-group** constraints (bounds, sharing), but **altloc sum-to-1** is a **cross-group constraint** that requires knowledge of which groups are alternative conformations.

---

## Division of Responsibilities

### OccupancyTensor Class Handles:
1. ✅ **Sigmoid bounds [0,1]**: Automatic via sigmoid transformation
2. ✅ **Residue-level sharing**: Atoms in same residue share one parameter
3. ✅ **Collapsed storage**: Memory-efficient storage with expansion mask
4. ✅ **Gradient flow**: Proper backpropagation through sharing groups

### Model Class Handles:
1. ✅ **Alternative conformation tracking**: Identifies which atoms belong to which altloc
2. ✅ **Sum-to-1 normalization**: Enforces cross-group constraint for altlocs
3. ✅ **Residue grouping logic**: Decides which atoms should share occupancy

---

## Testing Results

Both integration tests still pass after cleanup:

### Test 1: Model Integration
```
✓ OccupancyTensor correctly integrated
✓ Collapsed storage working (19 → 4)
✓ Residue-level sharing enforced
✓ Only partial occupancies refined
✓ Fixed parameters stay fixed
✓ Optimization works correctly
```

### Test 2: Alternative Conformation Handling
```
✓ Alternative conformations properly normalized
✓ Altloc sum-to-1 constraint enforced
✓ Relative ratios preserved
✓ Sharing maintained within conformations
✓ Works correctly with collapsed storage
```

---

## Usage Pattern

### Before Cleanup (Old Way)
```python
model = Model()
model.load_pdb_from_file("structure.pdb")

# Manual sanitization needed
model.sanitize_occupancies()  # ❌ No longer needed!

# Optimize
for step in range(100):
    optimizer.zero_grad()
    loss = compute_loss(model)
    loss.backward()
    optimizer.step()
    
    if step % 10 == 0:
        model.sanitize_occupancies()  # ❌ Redundant!
```

### After Cleanup (New Way)
```python
model = Model()
model.load_pdb_from_file("structure.pdb")

# No manual sanitization - OccupancyTensor handles bounds automatically!

# Optimize
for step in range(100):
    optimizer.zero_grad()
    loss = compute_loss(model)
    loss.backward()
    optimizer.step()
    
    # Only enforce altloc constraints if structure has alternative conformations
    if step % 10 == 0 and len(model.altloc_pairs) > 0:
        model.enforce_occ_alternative_conformations()  # ✓ Still needed!
```

---

## Benefits of Cleanup

1. **Simpler API**: One less method to remember
2. **Less error-prone**: Can't forget to call `sanitize_occupancies()`
3. **Automatic guarantees**: Sigmoid ensures [0,1] mathematically
4. **Clearer separation**: Model class only handles cross-group constraints
5. **Better performance**: No unnecessary clamping operations

---

## Documentation Updates

Updated docstring for `enforce_occ_alternative_conformations()` to clarify:
- What OccupancyTensor already handles (bounds, sharing)
- What this method additionally provides (sum-to-1 for altlocs)
- When it should be called (periodically during optimization)
- How it works with collapsed storage

---

## Conclusion

The cleanup removes **redundant** occupancy sanitization while keeping **essential** alternative conformation constraint enforcement. This results in:

✅ Cleaner, simpler API  
✅ Automatic bounds guarantees  
✅ Proper separation of concerns  
✅ All tests still passing  
✅ Production-ready code

The `OccupancyTensor` class now handles everything it can handle automatically, while the `Model` class focuses on high-level structural constraints (altlocs) that require domain knowledge.
