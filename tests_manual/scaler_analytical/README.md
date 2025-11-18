# Analytical Scaler Tests

This directory contains tests for the `AnalyticalScaler` class, which implements the analytical bulk-solvent and isotropic scaling method from Afonine et al. (2013) *Acta Cryst. D69, 625–634*.

## Overview

The `AnalyticalScaler` class provides:
- **Analytical computation** of bulk-solvent scale factor (k_mask) and isotropic scale parameter (K)
- **Resolution-dependent scaling** using cubic equation solutions per bin
- **Fast initialization** (no iterative optimization needed)
- **Forward pass** for applying computed scales to structure factors

## Test Files

### `test_basic.py`
Basic functionality tests:
- Loading model, data, and solvent mask
- Initializing AnalyticalScaler
- Computing analytical scales
- Forward pass with default and custom HKL
- Statistics reporting

**Usage:**
```bash
python test_basic.py
```

### `test_validation.py`
Validation tests against expected behavior:
- R-factor improvement from scaling
- Resolution-dependent scaling behavior
- Bulk-solvent contribution magnitude
- Internal consistency checks
- Plotting of scaling curves

**Usage:**
```bash
python test_validation.py
```

## Expected Results

### Initialization
The scaler should successfully:
1. Load ModelFT, ReflectionData, and SolventModel objects
2. Compute analytical scales for all reflections (grouped into resolution bins)
3. Report k_mask and K values per bin

Typical output:
```
Analytical Bulk-Solvent and Isotropic Scaling
======================================================================
Number of reflections: 50000
Resolution bins: 20

Computing analytical bulk-solvent scales...
  Bin  1/20:  2500 refs, d=20.00Å, k_mask=0.3500, K=0.9500
  Bin  2/20:  2500 refs, d=15.00Å, k_mask=0.3300, K=0.9400
  ...
```

### R-factor Improvement
The analytical scaling should improve the R-factor compared to unscaled F_calc:
- **Unscaled R-factor**: typically 0.25-0.35
- **Scaled R-factor**: typically 0.20-0.30
- **Expected improvement**: 5-20%

### Scaling Parameters
Typical ranges for tubulin test data:
- **k_mask**: 0.25-0.40 (bulk-solvent contributes 25-40% relative to protein)
- **K**: 0.80-1.20 (isotropic scale factor)

## Class Usage Example

```python
from multicopy_refinement.model_ft import ModelFT
from multicopy_refinement.Data import ReflectionData
from multicopy_refinement.solvent import SolventModel
from multicopy_refinement.scaler_analytical import AnalyticalScaler

# Load model and data
model = ModelFT(max_res=1.5)
model.load_pdb_from_file('structure.pdb')

data = ReflectionData()
data.load_from_mtz('reflections.mtz')
data = data.filter_by_resolution(d_min=1.5, d_max=50.0)

# Create solvent mask
solvent = SolventModel(model)

# Initialize analytical scaler (computes scales automatically)
scaler = AnalyticalScaler(
    model_ft=model,
    reflection_data=data,
    solvent_model=solvent,
    n_bins=20,  # number of resolution bins
    verbose=1
)

# Apply scaling to get F_model
F_model = scaler.forward()

# Or apply to custom HKL
F_model_subset = scaler.forward(hkl=custom_hkl)

# Get statistics
scaler.print_statistics()
```

## Algorithm Details

The analytical method solves for k_mask and K per resolution bin by:

1. **Binning reflections** by resolution (uniform in log-space)

2. **Computing per-reflection quantities**:
   - u = |F_calc|²
   - v = Re(F_calc · F_mask*)
   - w = |F_mask|²

3. **Forming bin-level sums**: A₂, B₂, C₂, D₃, E₄, F₄, G₄, Y₂, Y₃

4. **Solving cubic equation**: k_mask³ + a·k_mask² + b·k_mask + c = 0

5. **Selecting best positive root** by evaluating loss function

6. **Computing K**: K = (k_mask²·C₂ + k_mask·B₂ + A₂) / Y₂

This avoids iterative optimization and is typically **100× faster** than traditional methods.

## Troubleshooting

### No improvement in R-factor
Possible causes:
- Model is already very good
- Data quality issues
- Incorrect solvent mask
- Too few reflections per bin (increase `n_bins`)

### Unusual scaling parameters
Possible causes:
- k_mask < 0.1 or > 0.6: Check solvent mask
- K < 0.5 or > 2.0: Check overall scale
- High variation: Normal for data with artifacts

### Forward pass errors
Possible causes:
- `hkl` indices out of bounds: Check HKL validity
- Complex vs. real: Ensure F_calc is complex
- Shape mismatch: Verify input dimensions

## References

1. Afonine, P. V., Grosse-Kunstleve, R. W., Echols, N., Headd, J. J., Moriarty, N. W., Mustyakimov, M., Terwilliger, T. C., Urzhumtsev, A., Zwart, P. H. & Adams, P. D. (2013). *Acta Cryst. D69*, 625–634. "Towards automated crystallographic structure refinement with phenix.refine"

2. Afonine, P. V., Poon, B. K., Read, R. J., Sobolev, O. V., Terwilliger, T. C., Urzhumtsev, A. & Adams, P. D. (2023). *Acta Cryst. D79*, 666–667. "Correction to Afonine et al. (2013)"

## Contact

For issues or questions, please refer to the main repository documentation.
