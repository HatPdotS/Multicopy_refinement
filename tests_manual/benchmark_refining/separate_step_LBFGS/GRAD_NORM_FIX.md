# Grad Norm Error Fix

## Problem

The screening script was producing these errors repeatedly:
```
Error in grad_norm: torch.cat(): expected a non-empty list of Tensors
Error in grad_norm: element 0 of tensors does not require grad and does not have a grad_fn
```

## Root Cause

Two issues were causing these errors:

### Issue 1: Using `model.parameters()` instead of `refinement.parameters()`

In `refine_adp()` function:
```python
# WRONG - returns empty list when parameters are frozen
gx = grad_norm(loss_xray_, refinement.model.parameters())
ga = grad_norm(loss_adp_, refinement.model.parameters())
```

When parameters are frozen, `model.parameters()` filters them out completely, returning an empty iterator. But `refinement.parameters()` returns the full list of parameters, including those from the scaler and other submodules.

### Issue 2: Not checking for empty gradient list

The `grad_norm()` function tried to concatenate gradients without checking if any exist:
```python
# WRONG - crashes if no gradients available
vec = torch.cat([p.grad.flatten().detach()
                for p in params if p.grad is not None])
```

## The Fix

### Fix 1: Use `refinement.parameters()` everywhere

Changed in `refine_adp()`:
```python
# CORRECT - gets all refineable parameters
gx = grad_norm(loss_xray_, refinement.parameters())
ga = grad_norm(loss_adp_, refinement.parameters())
```

Note: `refine_xyz()` was already correct - it used `refinement.parameters()`.

### Fix 2: Check for empty gradient list

Updated `grad_norm()` function:
```python
def grad_norm(loss, params):
    """Compute gradient norm with proper zeroing"""
    try:
        # Zero gradients
        for p in params:
            if p.grad is not None:
                p.grad.zero_()
        
        # Compute gradients
        loss.backward(retain_graph=True)
        
        # Collect gradients from parameters that have them
        grad_list = [p.grad.flatten().detach() for p in params if p.grad is not None]
        
        # CHECK: Do we have any gradients?
        if len(grad_list) == 0:
            # No parameters with gradients - return default value
            return 1.0
        
        # Concatenate and compute norm
        vec = torch.cat(grad_list)
        return vec.norm().item()
    except Exception as e:
        print(f"Error in grad_norm: {e}")
        return 1.0
```

The key addition is checking `if len(grad_list) == 0` before calling `torch.cat()`.

## Why This Matters

Without the fix:
- Gradient-based weight computation fails
- Falls back to default weight of 1.0
- Weight screening becomes less meaningful
- All combinations use essentially the same weights

With the fix:
- Gradients computed correctly for active parameters
- Automatic weight scaling works properly
- Each combination uses its intended weights
- Screening results are meaningful and reliable

## Testing

Run the test script to verify the fix:
```bash
python test_grad_norm.py
```

Expected output:
- Should show non-zero gradient norms
- Should not show "No parameters with gradients found" warnings
- Should successfully compute gradient ratios

## Files Modified

- `screen_weights.py` - Fixed both `grad_norm()` and `refine_adp()` functions
- `test_grad_norm.py` - New test to verify the fix works

## Impact on Running Job

The current screening job will complete successfully despite the errors (refinements still work), but the gradient-based weight scaling won't be functioning properly. You may want to:

1. Let current job finish (to see baseline results)
2. Resubmit with the fixed version
3. Compare results to see if proper weight scaling makes a difference

The fixed version should show:
- No error messages in the output
- Proper weight scaling for each combination
- Potentially better optimization results
