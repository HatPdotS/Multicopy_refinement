# French-Wilson Bug Fix Summary

## Date: November 13, 2025

## Problem

The French-Wilson intensity-to-amplitude conversion was producing incorrect results compared to Phenix:
- **R-factor**: 0.3743 (extremely poor agreement)
- **All reflections** showed systematically incorrect F values
- Strong reflections were ~4-5× too small
- Weak reflections were ~10-20× too small

## Root Cause

**Critical bug in `interpolate_table()` function** (line 148 of `french_wilson.py`):

```python
# WRONG:
point = 10.0 * (h + h_min)

# CORRECT:
point = 10.0 * (h - h_min)
```

Since `h_min = -4.0`, the bug caused:
- Wrong: `point = 10.0 * (h + (-4.0)) = 10.0 * (h - 4.0)`
- Correct: `point = 10.0 * (h - (-4.0)) = 10.0 * (h + 4.0)`

This **completely scrambled the lookup table indexing**, causing all reflections to retrieve wrong zf values from the French & Wilson (1978) tables.

## Impact

The lookup tables map the normalized parameter `h` (ranging from -4 to +3 for acentrics, -4 to +4 for centrics) to corrected structure factor amplitudes. The bug meant:
- For `h = -0.102`, we were looking up index `~-41` instead of `~39`
- This caused all lookups to return values near the table boundaries (0.269 for centrics)
- The formula `F = zf × √σ_I` was correct, but zf values were completely wrong

## Fix

Changed line 148 in `multicopy_refinement/french_wilson.py`:

```python
def interpolate_table(h: torch.Tensor, table: torch.Tensor, h_min: float = -4.0) -> torch.Tensor:
    """
    Interpolate values from French-Wilson lookup table.
    
    Args:
        h: Normalized parameter (tensor of any shape)
        table: Lookup table tensor (1D)
        h_min: Minimum h value (default -4.0)
    
    Returns:
        Interpolated values (same shape as h)
    """
    # Map h to table index: point = 10.0 * (h - h_min)
    # For h_min = -4.0, this gives point = 10.0 * (h + 4.0)
    point = 10.0 * (h - h_min)  # ← FIXED: was (h + h_min)
    point = torch.clamp(point, 0.0, len(table) - 1.001)
    
    # ... rest of interpolation code ...
```

## Validation

After the fix:
- **R-factor**: 0.0005 (essentially perfect agreement with Phenix!)
- **All 3 test suites pass**:
  - `test_centric_determination.py` ✓
  - `test_core_functions.py` ✓
  - `test_module_integration.py` ✓
- Comparison against Phenix `F-obs-filtered` shows excellent agreement

## Lessons Learned

1. **Sign errors are insidious**: The bug was a simple sign error (`+` vs `-`) that completely broke the algorithm
2. **Validation is critical**: Without comparing against a reference implementation (Phenix), this bug would have gone unnoticed
3. **Diagnostic tools are essential**: The `diagnose_differences.py` script helped identify that zf values were constant (0.269), leading directly to the bug
4. **Test coverage matters**: Unit tests for the interpolation function itself would have caught this immediately

## Files Modified

1. `multicopy_refinement/french_wilson.py` - Fixed `interpolate_table()` function
2. `tests_manual/french_wilson/test_module_integration.py` - Fixed import name (`FrenchWilsonModule` → `FrenchWilson`) and converted unit_cell lists to tensors

## Files Created

1. `tests_manual/french_wilson/diagnose_differences.py` - Diagnostic script that helped identify the bug

## Performance

The fix has no performance impact - it's simply correcting an indexing error. The algorithm complexity remains O(n) for n reflections.
