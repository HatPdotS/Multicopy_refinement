# CIF Readers - Legacy Format Compatibility

## Overview

The CIF readers (`ReflectionCIFReader` and `ModelCIFReader`) are fully compatible with the legacy format readers (`MTZ` and `PDB`). They return data in identical structures, allowing seamless drop-in replacement.

## Interface Compatibility

### Structure Factor Data

Both `ReflectionCIFReader` and `MTZ` return the same data structure:

```python
# Legacy MTZ reader
from multicopy_refinement.legacy_format_readers import MTZ
mtz_reader = MTZ(verbose=1).read('file.mtz')
data_dict, cell, spacegroup = mtz_reader()

# New CIF reader - IDENTICAL interface
from multicopy_refinement.cif_readers import ReflectionCIFReader
cif_reader = ReflectionCIFReader('file-sf.cif', verbose=1).read()
data_dict, cell, spacegroup = cif_reader()
```

**Return Structure:**
- `data_dict`: `dict` with numpy arrays
  - `'h'`, `'k'`, `'l'`: Miller indices (np.ndarray, int)
  - `'F'`, `'SIGF'`: Amplitudes and sigmas (np.ndarray, float, optional)
  - `'I'`, `'SIGI'`: Intensities and sigmas (np.ndarray, float, optional)
  - `'R-free-flags'`: Test set flags (np.ndarray, int32, optional)
  - `'F_col'`, `'I_col'`, `'R-free-source'`: Metadata strings
- `cell`: `np.ndarray` of 6 floats [a, b, c, α, β, γ]
- `spacegroup`: `str` with space group symbol

### Model/Structure Data

Both `ModelCIFReader` and `PDB` return the same data structure:

```python
# Legacy PDB reader
from multicopy_refinement.legacy_format_readers import PDB
pdb_reader = PDB(verbose=1).read('file.pdb')
dataframe, cell, spacegroup = pdb_reader()

# New CIF reader - IDENTICAL interface
from multicopy_refinement.cif_readers import ModelCIFReader
cif_reader = ModelCIFReader('file.cif', verbose=1).read()
dataframe, cell, spacegroup = cif_reader()
```

**Return Structure:**
- `dataframe`: `pd.DataFrame` with atom data
  - Standard columns: `'ATOM'`, `'serial'`, `'name'`, `'altloc'`, `'resname'`, `'chainid'`, `'resseq'`, `'icode'`
  - Coordinates: `'x'`, `'y'`, `'z'` (float64)
  - Properties: `'occupancy'`, `'tempfactor'` (float64)
  - Identification: `'element'`, `'charge'` (str, int)
  - Anisotropic: `'anisou_flag'`, `'u11'`, `'u22'`, `'u33'`, `'u12'`, `'u13'`, `'u23'`
  - Index: `'index'` (sequential integer)
  - Attributes: `df.attrs['cell']`, `df.attrs['spacegroup']`, `df.attrs['z']`
- `cell`: `list` of 6 floats [a, b, c, α, β, γ] or `None`
- `spacegroup`: `str` with space group symbol

## Usage Examples

### Drop-in Replacement for MTZ Files

```python
# Old code using MTZ
from multicopy_refinement.legacy_format_readers import MTZ
reader = MTZ(verbose=1).read('data.mtz')
data, cell, sg = reader()
h, k, l = data['h'], data['k'], data['l']
F_obs = data['F']

# New code using CIF - SAME interface!
from multicopy_refinement.cif_readers import ReflectionCIFReader
reader = ReflectionCIFReader('data-sf.cif', verbose=1).read()
data, cell, sg = reader()
h, k, l = data['h'], data['k'], data['l']
F_obs = data['F']
```

### Drop-in Replacement for PDB Files

```python
# Old code using PDB
from multicopy_refinement.legacy_format_readers import PDB
reader = PDB(verbose=1).read('structure.pdb')
df, cell, sg = reader()
coords = df[['x', 'y', 'z']].values
atoms = df['name']

# New code using CIF - SAME interface!
from multicopy_refinement.cif_readers import ModelCIFReader
reader = ModelCIFReader('structure.cif', verbose=1).read()
df, cell, sg = reader()
coords = df[['x', 'y', 'z']].values
atoms = df['name']
```

### Unified Interface

Both CIF and legacy readers support the same calling patterns:

```python
# Pattern 1: Initialize and call
reader = ReflectionCIFReader('file.cif')
data, cell, sg = reader()

# Pattern 2: Method chaining (like legacy readers)
data, cell, sg = ReflectionCIFReader('file.cif').read()()

# Pattern 3: Initialize first, read later
reader = ReflectionCIFReader('dummy.cif')  # placeholder
reader.read('actual_file.cif')
data, cell, sg = reader()
```

## Data Dictionary Keys Reference

### Reflection Data (ReflectionCIFReader / MTZ)

| Key | Type | Description | Always Present |
|-----|------|-------------|----------------|
| `'h'`, `'k'`, `'l'` | np.ndarray (int) | Miller indices | ✅ Yes |
| `'F'` | np.ndarray (float) | Structure factor amplitudes | ❌ Optional |
| `'SIGF'` | np.ndarray (float) | Amplitude uncertainties | ❌ Optional |
| `'I'` | np.ndarray (float) | Intensities | ❌ Optional |
| `'SIGI'` | np.ndarray (float) | Intensity uncertainties | ❌ Optional |
| `'R-free-flags'` | np.ndarray (int32) | Test set flags (0=free) | ❌ Optional |
| `'F_col'` | str | Source column name for F | ❌ Optional |
| `'SIGF_col'` | str | Source column name for SIGF | ❌ Optional |
| `'I_col'` | str | Source column name for I | ❌ Optional |
| `'SIGI_col'` | str | Source column name for SIGI | ❌ Optional |
| `'R-free-source'` | str | Source column for R-free | ❌ Optional |

### Model Data (ModelCIFReader / PDB)

All DataFrames contain these columns:

| Column | Type | Description |
|--------|------|-------------|
| `'ATOM'` | str | Record type (ATOM/HETATM) |
| `'serial'` | int64 | Atom serial number |
| `'name'` | str | Atom name |
| `'altloc'` | str | Alternate location indicator |
| `'resname'` | str | Residue name |
| `'chainid'` | str | Chain identifier |
| `'resseq'` | int64 | Residue sequence number |
| `'icode'` | str | Insertion code |
| `'x'`, `'y'`, `'z'` | float64 | Cartesian coordinates (Å) |
| `'occupancy'` | float64 | Occupancy factor |
| `'tempfactor'` | float64 | Temperature/B-factor (Å²) |
| `'element'` | str | Element symbol |
| `'charge'` | int | Formal charge |
| `'anisou_flag'` | bool | Has anisotropic data |
| `'u11'` through `'u23'` | float64 | Anisotropic displacement |
| `'index'` | int | Sequential row index |

## Advantages of CIF Readers

While maintaining full compatibility, CIF readers offer:

1. **Standard Format**: CIF is the official format for PDB deposits
2. **Richer Metadata**: Better preservation of experimental details
3. **Validation**: Automatic validation of file structure
4. **Column Flexibility**: Handles multiple naming conventions automatically
5. **Type Safety**: Strong typing with clear error messages

## Migration Guide

To migrate from legacy readers to CIF readers:

1. **No code changes needed** - interfaces are identical
2. **Just change the import and filename**:
   ```python
   # Change this:
   from multicopy_refinement.legacy_format_readers import MTZ
   data = MTZ().read('file.mtz')
   
   # To this:
   from multicopy_refinement.cif_readers import ReflectionCIFReader
   data = ReflectionCIFReader('file-sf.cif').read()
   ```

3. **The rest of your code remains unchanged**:
   ```python
   data_dict, cell, spacegroup = data()  # Same as before!
   h, k, l = data_dict['h'], data_dict['k'], data_dict['l']  # Same!
   ```

## Testing

Comprehensive compatibility tests are available:
- `tests_manual/cif_reader/test_comprehensive_compatibility.py` - Full test suite
- `tests_manual/cif_reader/test_legacy_compatibility.py` - Side-by-side comparison

Run tests:
```bash
cd /das/work/p17/p17490/Peter/Library/multicopy_refinement
python tests_manual/cif_reader/test_comprehensive_compatibility.py
```

## See Also

- **Edge Case Testing**: `tests_manual/cif_reader/EDGE_CASE_TESTING_SUMMARY.md`
- **CIF Reader API**: `multicopy_refinement/cif_readers.py`
- **Legacy Readers**: `multicopy_refinement/legacy_format_readers.py`
