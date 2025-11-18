# R-Factor Explosion Bug - Fixed

## Problem

In round 3 of refinement, the B-factor optimization catastrophically failed:
- xyz refinement: Rfree = 0.2255 ✓ (excellent)
- **b refinement: Rfree = 0.5865** ❌ (disaster - 2.6x worse!)

## Root Cause

**The automatic weight scaling was computed once with all parameters active, but then applied during optimization where parameters were frozen separately.**

### What Happened:

1. At the start of `refine()`, weights were computed:
   ```python
   # Gradient norms computed with ALL parameters active (xyz + b)
   gx = grad_norm(loss_xray, all_params)
   ga = grad_norm(loss_adp, all_params)
   
   weight_adp = (gx / ga) / 0.2  # Results in ~5.0
   ```

2. By round 3, xyz geometry was excellent, so:
   - X-ray gradient was small (good fit)
   - ADP gradient was relatively larger
   - Ratio gx/ga was small → **weight_adp became ~5.0**

3. During B-factor refinement (with xyz frozen):
   ```python
   loss = 1.0 * xray_loss + 5.0 * adp_loss
   ```

4. The ADP loss (weight 5.0) dominated over X-ray loss (weight 1.0)

5. ADP loss definition:
   ```python
   def adp_loss(self):
       b_mean = torch.mean(b_current)
       loss = torch.mean((b_current - b_mean) ** 2)  # Minimize variance
   ```

6. **Optimizer pushed all B-factors toward their mean**, destroying information and worsening the fit by 6x!

## The Fix

**Recompute weights separately for each refinement target** (xyz vs b), with the appropriate parameters frozen:

```python
for target in ['xyz', 'b']:
    refinement.model.freeze_all()
    refinement.model.unfreeze(target)
    
    # Compute gradients with CURRENT freeze state
    loss_xray = refinement.xray_loss()
    loss_geom = refinement.restraints_loss()
    loss_adp = refinement.adp_loss()
    
    gx = grad_norm(loss_xray, refinement.model.parameters())
    gg = grad_norm(loss_geom, refinement.model.parameters())
    ga = grad_norm(loss_adp, refinement.model.parameters())
    
    # Weights are now appropriate for current target
    weight_adp = (gx / ga) / weights['adp']
    weight_restraints = (gx / gg) / weights['restraints']
    
    # Optimize with these target-specific weights
    ...
```

This ensures:
- During xyz refinement: weights reflect xyz-only gradients
- During b refinement: weights reflect b-only gradients
- The gradient ratios are correct for the active parameters

## Additional Fix

Added gradient zeroing to prevent accumulation:

```python
def grad_norm(loss, params):
    # Zero gradients before backward
    for p in params:
        if p.grad is not None:
            p.grad.zero_()
    loss.backward(retain_graph=True)
    ...
```

Without this, gradients accumulated across multiple `grad_norm()` calls, giving incorrect weight calculations.

## Files

- **Bug analysis:** `WEIGHT_SCALING_BUG_ANALYSIS.md`
- **Fixed script:** `test_LBFGS_FIXED.py`
- **Original (buggy):** `test_LBFGS.py`

## Expected Behavior After Fix

With the fix, B-factor refinement should:
1. Use appropriate weights for B-factor-only optimization
2. Balance X-ray fit with B-factor regularization correctly
3. Maintain or improve Rfree instead of exploding
4. Produce Rfree values in the 0.20-0.30 range throughout all rounds

## Key Lesson

**When using automatic weight scaling with frozen parameters, always recompute the weights with the same freeze/unfreeze state that will be used during optimization.**

The gradient magnitudes change dramatically when parameters are frozen, so weights computed with all parameters active are incorrect for parameter-specific refinement.
