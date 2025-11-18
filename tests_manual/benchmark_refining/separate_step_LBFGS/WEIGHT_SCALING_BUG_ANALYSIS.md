# Weight Scaling Bug Analysis

## Problem Summary

The B-factor refinement in round 3 produces catastrophically bad results:
- **Round 3 xyz:** Rwork=0.2028, Rfree=0.2255 ✓ (excellent!)
- **Round 3 b:** Rwork=0.5653, Rfree=0.5865 ❌ (disaster!)

The X-ray loss jumps from 45.29 to 287.77 (~6x increase), indicating B-factors are being set to terrible values.

## Root Cause

The issue is in how **automatic weight scaling** is being applied:

### Current Implementation (BUGGY):

```python
def refine(refinement, weights={'xray':1.0, 'restraints':10.0, 'adp':0.3}):
    refinement.get_scales()
    
    # Compute gradient norms with ALL parameters active
    loss_xray = refinement.xray_loss()
    loss_geom = refinement.restraints_loss()
    loss_adp = refinement.adp_loss()
    
    gx = grad_norm(loss_xray, refinement.model.parameters())
    gg = grad_norm(loss_geom, refinement.model.parameters())
    ga = grad_norm(loss_adp, refinement.model.parameters())
    
    # Compute weights based on gradient ratios
    weight_adp = (gx / (ga + 1e-12)) / weights['adp']
    weight_restraints = (gx / (gg + 1e-12)) / weights['restraints']
    
    # Set these weights ONCE for the entire refinement
    refinement.target_weights['restraints'] = weight_restraints
    refinement.target_weights['adp'] = weight_adp
    
    # Then refine xyz and b separately with the SAME weights
    for target in ['xyz', 'b']:
        refinement.model.freeze_all()
        refinement.model.unfreeze(target)
        # ... optimize ...
```

### Why This Fails:

1. **Weights are computed with all parameters active** (xyz + b together)
2. **But optimization happens with parameters frozen separately** (xyz alone, then b alone)
3. **The gradient magnitudes are completely different** when parameters are frozen!

Specifically:
- When computing `ga = grad_norm(loss_adp, all_params)`, the gradient includes contributions from both xyz and b
- But during B-factor refinement, only b is active, so the actual gradient magnitude is different
- The pre-computed weight `weight_adp` is therefore incorrect for B-factor-only refinement

4. **By round 3**, after two rounds of xyz refinement:
   - The xyz geometry is excellent (Rfree=0.2255)
   - The X-ray loss is low
   - The gradient ratio `gx/ga` becomes small
   - So `weight_adp` becomes **very large** (~5.0)
   
5. **During B-factor refinement**, the optimizer sees:
   ```
   total_loss = 1.0 * xray_loss + 5.0 * adp_loss
   ```
   
6. **The ADP loss dominates**:
   ```python
   def adp_loss(self):
       b_current = self.b()
       b_mean = torch.mean(b_current)
       loss = torch.mean((b_current - b_mean) ** 2)
       return loss
   ```
   This loss tries to make all B-factors equal to their mean!
   
7. **The optimizer prioritizes minimizing B-factor variance** (weight 5.0) over fitting the X-ray data (weight 1.0)
8. **Result:** All B-factors get pushed toward their mean value, destroying information and making the fit terrible

## Evidence from Log File

```
Round 3 effective weights: {'xray': 1.0, 'restraints': 1.988, 'adp': 4.956}
```

The ADP weight is ~5x the X-ray weight! This is way too high.

Compare with earlier rounds:
- Round 1: adp=4.996 (but xyz geometry was poor, so B-factor regularization helped)
- Round 2: adp=4.517 (still reasonable because xyz wasn't perfect yet)
- Round 3: adp=4.956 (NOW IT'S A PROBLEM because xyz is nearly perfect)

## Solutions

### Solution 1: Recompute weights for each target (RECOMMENDED)

```python
def refine(refinement, weights={'xray':1.0, 'restraints':10.0, 'adp':0.3}):
    refinement.get_scales()
    refinement.effective_weights = weights
    
    for target in ['xyz', 'b']:
        refinement.model.freeze_all()
        refinement.model.unfreeze(target)
        
        # Recompute weights with current frozen state
        loss_xray = refinement.xray_loss()
        loss_geom = refinement.restraints_loss()
        loss_adp = refinement.adp_loss()
        
        gx = grad_norm(loss_xray, refinement.model.parameters())
        gg = grad_norm(loss_geom, refinement.model.parameters())
        ga = grad_norm(loss_adp, refinement.model.parameters())
        
        weight_adp = (gx / (ga + 1e-12)) / weights['adp']
        weight_restraints = (gx / (gg + 1e-12)) / weights['restraints']
        
        refinement.target_weights['restraints'] = weight_restraints
        refinement.target_weights['adp'] = weight_adp
        
        print(f'Target {target} weights:', refinement.target_weights)
        
        # ... optimize with these weights ...
```

### Solution 2: Don't include ADP loss during xyz refinement

```python
def loss_function(refinement, target):
    loss = 0.0
    if 'xray' in refinement.target_weights:
        loss += refinement.target_weights['xray'] * refinement.xray_loss()
    if 'restraints' in refinement.target_weights:
        loss += refinement.target_weights['restraints'] * refinement.restraints_loss()
    
    # Only include ADP loss during B-factor refinement
    if target == 'b' and 'adp' in refinement.target_weights:
        loss += refinement.target_weights['adp'] * refinement.adp_loss()
    
    return loss
```

### Solution 3: Use fixed reasonable weights instead of automatic scaling

```python
def refine(refinement):
    refinement.get_scales()
    
    # Use fixed, reasonable weights
    for target in ['xyz', 'b']:
        refinement.model.freeze_all()
        refinement.model.unfreeze(target)
        
        if target == 'xyz':
            refinement.target_weights = {
                'xray': 1.0,
                'restraints': 2.0,  # Geometry restraints important for xyz
                'adp': 0.0           # Don't regularize B-factors during xyz refinement
            }
        else:  # target == 'b'
            refinement.target_weights = {
                'xray': 1.0,
                'restraints': 0.0,  # Geometry doesn't affect B-factors
                'adp': 0.5           # Moderate B-factor regularization
            }
        
        # ... optimize ...
```

## Recommended Fix

I recommend **Solution 1** because:
1. It preserves the automatic weight scaling concept
2. It computes appropriate weights for each refinement target
3. It's minimally invasive to the existing code

The key insight is: **weights must be computed with the same parameter freeze/unfreeze state that will be used during optimization**.

## Additional Bug: Missing Gradient Zeroing

The current `grad_norm` function doesn't zero gradients before backward:

```python
def grad_norm(loss, params):
    loss.backward(retain_graph=True)  # ❌ Accumulates on top of existing gradients!
    vec = torch.cat([p.grad.flatten().detach()
                        for p in params if p.grad is not None])
    return vec.norm().item()
```

Should be:

```python
def grad_norm(loss, params):
    for p in params:
        if p.grad is not None:
            p.grad.zero_()  # ✓ Zero gradients first
    loss.backward(retain_graph=True)
    vec = torch.cat([p.grad.flatten().detach()
                        for p in params if p.grad is not None])
    return vec.norm().item()
```

Without this, gradient norms accumulate across multiple calls, giving incorrect weight calculations.
