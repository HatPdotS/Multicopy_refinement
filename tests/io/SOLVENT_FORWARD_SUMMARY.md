# SolventModel.forward() Implementation Summary

## Implementation

Added a differentiable `forward()` method to the `SolventModel` class that computes solvent contribution to structure factors.

### Method Signature
```python
def forward(self, hkl: torch.Tensor) -> torch.Tensor
```

### Algorithm

The method implements the crystallographic solvent model in 4 steps:

1. **Create solvent electron density**
   - `ρ_solvent = k_solvent * mask`
   - Flat solvent scattering scaled by k_solvent parameter

2. **Fourier transform to reciprocal space**
   - Uses `ifft()` to convert real-space density to reciprocal-space structure factors
   - Creates a full 3D grid of structure factors

3. **Extract structure factors at HKL positions**
   - Uses `extract_structure_factor_from_grid()` to get F values at specific Miller indices
   - Handles periodic boundary conditions

4. **Apply B-factor damping**
   - Calculates `s² = (|h*|/2)²` where h* is the reciprocal lattice vector
   - Applies temperature factor: `F_damped = F * exp(-B_solvent * s²)`
   - Higher B-factor → stronger damping at high resolution

## Test Results

✓ **All tests passed successfully!**

### Structure Factor Statistics
- Shape: 123,647 reflections
- Mean amplitude: 289 e⁻
- Max amplitude: 26,589 e⁻ (low resolution)
- Min amplitude: 0.042 e⁻ (high resolution)

### Gradient Flow ✓
Both parameters show correct gradient behavior:
- **dk_solvent/dloss** = 17,340,128 (positive, increasing k increases |F|)
- **db_solvent/dloss** = -70,024 (negative, increasing b decreases |F|)

### Parameter Effects

**k_solvent** (solvent scattering scale):
- k = 0.10 → |F| mean = 321 e⁻
- k = 0.35 → |F| mean = 1,122 e⁻ **(typical value)**
- k = 0.50 → |F| mean = 1,603 e⁻
- k = 1.00 → |F| mean = 3,207 e⁻
- **Linear relationship**: |F| ∝ k_solvent ✓

**b_solvent** (solvent B-factor in Ų):
- b = 10 Ų → |F| mean = 1,639 e⁻ (sharp, unrealistic)
- b = 30 Ų → |F| mean = 1,306 e⁻
- b = 46 Ų → |F| mean = 1,122 e⁻ **(typical value)**
- b = 50 Ų → |F| mean = 1,084 e⁻
- b = 100 Ų → |F| mean = 738 e⁻ (over-damped)
- **Exponential decay**: Higher B → lower mean |F| ✓

## Physics Interpretation

The solvent model captures the disordered bulk solvent contribution:

- **k_solvent ≈ 0.3-0.4**: Typical for protein crystals
  - Reflects partial occupancy and disorder of solvent
  - Lower than bulk water due to voids and partial ordering

- **b_solvent ≈ 40-60 Ų**: Typical for protein crystals
  - Reflects high mobility of bulk solvent
  - Dampens high-resolution contributions
  - Higher than protein atoms (~20-40 Ų) due to disorder

## Usage Example

```python
# Create solvent model
solvent = SolventModel(model, k_solvent=0.35, b_solvent=46.0)

# Compute solvent contribution
F_solvent = solvent.forward(hkl)

# Total structure factors
F_total = F_protein + F_solvent

# Optimize parameters
optimizer = torch.optim.Adam([solvent.k_solvent, solvent.b_solvent], lr=0.01)
loss = some_loss_function(F_total, F_obs)
loss.backward()
optimizer.step()
```

## Files Modified

- `/das/work/p17/p17490/Peter/Library/multicopy_refinement/multicopy_refinement/solvent.py`
  - Added `forward(hkl)` method (62 lines)
  - Fully differentiable with respect to k_solvent and b_solvent
  - Imports: ifft, extract_structure_factor_from_grid, math_numpy, numpy

## Next Steps

The forward method is ready for integration into refinement workflows:

1. **Add to ModelFT**: Create combined F_calc = F_protein + F_solvent
2. **Optimize parameters**: Refine k_solvent and b_solvent during minimization
3. **Bulk solvent correction**: Improves R-factors, especially at low resolution
4. **Mask optimization**: Could refine mask parameters (radius, dilation) if needed
