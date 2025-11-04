# Scaling Fix Summary

## Problem Identified

After fixing the French-Wilson conversion to properly handle negative intensities, the scaling was not working correctly:

1. **Log-ratio distribution not centered at 0**: Mean was -0.262 (should be ~0)
2. **R-factors were worse** than before the French-Wilson fix
3. **Systematic bias** between negative and positive intensity reflections

## Root Cause

The `Scaler.calc_initial_scale()` method was calculating scale factors using **all reflections**, including those with negative intensities. 

### Why this was a problem:

1. **Negative intensity reflections** (8.42% of data) get systematically different F values from French-Wilson conversion
   - Mean F for negative I: 12.42
   - Mean F for positive I: 31.03
   
2. Including these in the scale calculation **biased the scale factors downward**, causing:
   - Under-scaling of positive intensity reflections (log-ratio = -0.301)
   - Over-scaling of negative intensity reflections (log-ratio = +0.165)
   - Overall mean log-ratio of -0.262

## Solution

Modified `Scaler.calc_initial_scale()` to **exclude reflections with negative intensities** from the scale factor calculation:

```python
# Exclude reflections with negative intensities from scale calculation
if hasattr(self.data, 'I') and self.data.I is not None:
    positive_mask = self.data.I > 0
    ...
else:
    positive_mask = torch.ones_like(fobs, dtype=torch.bool)

# Calculate ratios only for positive intensity reflections
ratios_masked = torch.where(positive_mask, ratios, torch.zeros_like(ratios))
counts_masked = torch.where(positive_mask, torch.ones_like(ratios), torch.zeros_like(ratios))
```

## Results

### Before Fix:
- Log-ratio mean: **-0.262**
- Log-ratio std: 0.614
- R-factor (before outlier removal): **0.3299**
- R-factor (after outlier removal): **0.2589**

### After Fix:
- Log-ratio mean: **-0.090** (71% improvement)
- Log-ratio std: 0.594
- R-factor (before outlier removal): **0.2787** (15.5% improvement)
- R-factor (after outlier removal): **0.2572** (0.7% improvement)

## Remaining Issues

The log-ratio distribution is still slightly off-center (-0.090 instead of 0), with different means for:
- Negative intensity reflections: +0.486 (over-scaled)
- Positive intensity reflections: -0.143 (under-scaled)

This suggests there may still be room for improvement in the French-Wilson conversion for negative intensities, but the current fix addresses the major scaling problem.

## Files Modified

- `/das/work/p17/p17490/Peter/Library/multicopy_refinement/multicopy_refinement/scaler.py`
  - Function: `calc_initial_scale()`
  - Change: Added filtering to exclude negative intensity reflections from scale calculation
