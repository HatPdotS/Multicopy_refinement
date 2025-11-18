# SolventModel.forward() - Smoothed Mask Implementation

## Overview

Modified the `forward()` method of `SolventModel` to return a smoothed solvent mask that can be added to the protein electron density map before Fourier transform.

## Implementation Details

### Method Signature
```python
def forward(self) -> torch.Tensor
```
**Note:** No longer takes `hkl` parameter - returns real-space density map instead of structure factors.

### Algorithm

1. **Convert binary mask to float**
   - Takes the boolean `solvent_mask` and converts to float

2. **Apply 3D Gaussian smoothing**
   - Creates a 3D Gaussian kernel with σ = 1.5 voxels
   - Uses `F.conv3d` with circular padding for periodic boundaries
   - Smoothing creates soft edges at protein-solvent boundary
   - Prevents sharp discontinuities that cause artifacts in reciprocal space

3. **Scale by k_solvent**
   - Multiplies smoothed mask by the learnable `k_solvent` parameter
   - Returns: `ρ_solvent = k_solvent × smooth(mask)`

### Key Features

- **Fully differentiable**: Gradients flow through k_solvent
- **Periodic boundary conditions**: Uses circular padding for FFT compatibility  
- **Volume conservation**: Smoothing preserves total solvent volume (ratio ≈ 1.0000)
- **Soft edges**: Creates smooth transition between protein and solvent regions

## Test Results

✓ **All tests passed successfully!**

### Smoothing Verification
- **Original mask**: Binary (0.0, 1.0), only 2 unique values
- **Smoothed mask**: Continuous (0.0 to 0.35), 1,870,000 unique values
- **Conclusion**: ✓ Mask has been properly smoothed with soft edges

### Gradient Flow ✓
- **dk_solvent/dloss** = 6,328,025
- **Conclusion**: ✓ Fully differentiable with respect to k_solvent

### Linear Scaling ✓
Effect of k_solvent on smoothed density:
```
k_solvent = 0.10 → mean = 0.0408, max = 0.10
k_solvent = 0.35 → mean = 0.1428, max = 0.35
k_solvent = 0.50 → mean = 0.2041, max = 0.50
k_solvent = 1.00 → mean = 0.4081, max = 1.00
```
**Conclusion**: ✓ Perfect linear relationship between k_solvent and density

### Volume Conservation ✓
- Original mask sum: 6,328,020 voxels
- Smoothed mask sum: 6,328,025 voxels
- Ratio: 1.0000 (100.0% conserved)
- **Conclusion**: ✓ Gaussian smoothing preserves total solvent volume

## Usage Example

```python
# Create solvent model
solvent = SolventModel(model, k_solvent=0.35, b_solvent=46.0)

# Build protein density
protein_density = model.build_density_map()

# Get smoothed solvent contribution
solvent_density = solvent.forward()

# Combine protein + solvent in real space
total_density = protein_density + solvent_density

# Fourier transform combined map
from multicopy_refinement.math_torch import ifft, extract_structure_factor_from_grid

reciprocal_grid = ifft(total_density)
F_total = extract_structure_factor_from_grid(reciprocal_grid, hkl)

# Apply B-factor to solvent contribution if needed
# (usually applied in reciprocal space separately to F_solvent)
```

## Physics Interpretation

### Why Smooth the Mask?

1. **Sharp edges → Fourier artifacts**: A binary mask creates Gibbs phenomenon (ringing) in Fourier space
2. **Smooth edges → Clean FFT**: Gaussian smoothing with σ ≈ 1-2 voxels eliminates artifacts
3. **Physical reality**: Protein-solvent boundary is not infinitely sharp anyway

### Gaussian Width (σ = 1.5 voxels)

- At typical grid spacing (~0.4-0.5 Å/voxel), σ = 1.5 voxels ≈ 0.6-0.75 Å
- This creates a smooth transition zone at the protein surface
- Not too broad (would blur features) or too narrow (would leave artifacts)

### Volume Conservation

- The Gaussian kernel is normalized: ∫∫∫ G(x,y,z) dx dy dz = 1
- Therefore: ∫∫∫ smooth(mask) dx dy dz = ∫∫∫ mask dx dy dz
- Smoothing redistributes density but preserves total amount

## Benefits

1. **Cleaner structure factors**: No Fourier ringing from sharp edges
2. **Better refinement**: Smoother gradients, more stable optimization
3. **Physically motivated**: Represents realistic protein-solvent interface
4. **Efficient**: Single 3D convolution with periodic boundaries
5. **Differentiable**: Can optimize k_solvent during refinement

## Next Steps

The smoothed solvent mask is ready to be added to protein density:

1. **In ModelFT**: Add method to combine protein + solvent densities
2. **B-factor application**: May need to apply b_solvent separately in reciprocal space
3. **Refinement integration**: Include k_solvent (and optionally b_solvent) in parameters to optimize

## Files Modified

- `/das/work/p17/p17490/Peter/Library/multicopy_refinement/multicopy_refinement/solvent.py`
  - Modified `forward()` method: removed `hkl` parameter, now returns smoothed real-space density
  - Added 3D Gaussian smoothing with σ = 1.5 voxels
  - Uses circular padding for periodic boundaries
  - Returns: `k_solvent × smooth(mask)`
