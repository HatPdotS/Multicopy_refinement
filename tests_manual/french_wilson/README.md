# French-Wilson Tests

This directory contains manual tests for the French-Wilson conversion implementation in `multicopy_refinement/french_wilson.py`.

## Overview

The French-Wilson method converts measured X-ray intensities (I) and their uncertainties (σ_I) into structure factor amplitudes (F) and their uncertainties (σ_F), using a Bayesian approach that properly handles weak and negative intensities.

## Verbosity Control

The `FrenchWilsonModule` supports verbosity control to manage console output:

**Verbosity Levels:**
- `verbose=0`: Silent mode (no output) - useful for production code or batch processing
- `verbose=1`: Basic info (DEFAULT) - shows reflections count, resolution range, space group, centric %
- `verbose=2`: Detailed info - adds binning parameters, rejection thresholds, device information

**Examples:**
```python
# Silent mode (production)
fw = FrenchWilsonModule(hkl, unit_cell, space_group, verbose=0)

# Default (interactive use)
fw = FrenchWilsonModule(hkl, unit_cell, space_group)  # verbose=1 by default

# Detailed (debugging)
fw = FrenchWilsonModule(hkl, unit_cell, space_group, verbose=2)
```

See `demo_verbosity.py` for a demonstration of all verbosity levels.

## Test Files

### 1. `test_centric_determination.py`
Tests the centric/acentric reflection determination using the Symmetry class.

**Tests:**
- P1 space group (all acentric)
- P21 space group (mixed)
- P212121 orthorhombic space group
- P-1 centrosymmetric (all centric)
- Batch processing with different array shapes
- Multiple common space groups

**Run:**
```bash
python test_centric_determination.py
```

### 2. `test_module_integration.py`
Tests the FrenchWilsonModule class with math_torch integration.

**Tests:**
- D-spacing calculation using `math_torch.get_scattering_vectors`
- Non-orthogonal unit cells (monoclinic, triclinic)
- Basic French-Wilson conversion
- Weak and negative intensities
- Functional API (`french_wilson_auto`)
- Different space groups
- Large datasets (1000+ reflections)

**Run:**
```bash
python test_module_integration.py
```

### 3. `test_core_functions.py`
Tests core French-Wilson conversion functions.

**Tests:**
- Acentric conversion (lookup tables and asymptotic formulas)
- Centric conversion (lookup tables and extended asymptotic formulas)
- Mixed centric/acentric reflections
- Resolution binning for mean intensity estimation
- Lookup table interpolation
- Asymptotic formulas for very strong reflections

**Run:**
```bash
python test_core_functions.py
```

## Key Features Tested

### 1. Improved Centric Determination
The new implementation uses the `Symmetry` class to determine centric vs acentric reflections based on actual space group symmetry operations, rather than simple heuristics. A reflection is centric if its Friedel mate (-h,-k,-l) is symmetry-equivalent to (h,k,l).

### 2. Integration with math_torch
The module now uses `math_torch.get_scattering_vectors()` for calculating reciprocal space vectors and d-spacings, ensuring consistency with other parts of the codebase.

### 3. Lookup Tables
The French-Wilson method uses lookup tables from the original 1978 paper for different ranges of the normalized parameter h:
- **Acentric**: Uses AC_ZF tables for h < 3.0, asymptotic formula for h ≥ 3.0
- **Centric**: Uses C_ZF tables for h < 4.0, extended asymptotic formula for h ≥ 4.0

### 4. Resolution Binning
Mean intensities are estimated by binning reflections by resolution (d-spacing) and using linear interpolation between bin centers.

## Expected Results

All tests should pass with the message "✓ [test name] test passed!" or "✓ [test name] test completed!".

### Key Assertions:
1. P1 space group should have 0 centric reflections
2. P-1 (centrosymmetric) should have 100% centric reflections
3. D-spacings should match theoretical values for simple cubic cells
4. Structure factors should be non-negative and finite
5. F² should be roughly comparable to I for strong reflections
6. Weak/negative intensities should produce reasonable F values (no NaN/Inf)

## Space Group Coverage

The tests cover various space group types:
- **P1**: Triclinic, no symmetry
- **P21**: Monoclinic
- **P212121**: Orthorhombic
- **P-1**: Triclinic, centrosymmetric
- **C2**: Monoclinic, C-centered
- **P222**: Orthorhombic

## Mathematical Background

### French-Wilson Conversion

For **acentric** reflections:
- Normalized parameter: h = (I/σ_I) - (σ_I/<I>)
- Uses Bayesian posterior with Wilson prior
- Rejects reflections with h < -4.0 or I/σ_I < -3.7

For **centric** reflections:
- Normalized parameter: h = (I/σ_I) - (σ_I/(2<I>))  [note factor of 2]
- Different lookup tables due to different probability distribution
- Same rejection criteria

### D-spacing Calculation

The relationship between Miller indices (h,k,l) and d-spacing:
```
1/d = |s| = |h·a* + k·b* + l·c*|
```
where a*, b*, c* are reciprocal lattice vectors.

## Troubleshooting

If tests fail:

1. **Import errors**: Ensure the path to the library is correct
2. **Symmetry errors**: Check that space group names are recognized
3. **Numerical differences**: Small floating-point differences are expected
4. **Device errors**: Tests run on CPU by default

## References

- French, S. & Wilson, K. (1978). "On the treatment of negative intensity observations". *Acta Cryst.* A34, 517-525.
- Phenix implementation: `cctbx/french_wilson.py`
