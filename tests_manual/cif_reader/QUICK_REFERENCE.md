# CIF Readers - Quick Reference

## Import

```python
from multicopy_refinement.cif_readers import (
    ReflectionCIFReader,  # For structure factor data (.sf.cif)
    ModelCIFReader,       # For atomic coordinates (.cif)
    RestraintCIFReader    # For chemical restraints (monomer library)
)
```

## ReflectionCIFReader (Structure Factors)

```python
# Read structure factor CIF file
reader = ReflectionCIFReader('7JI4-sf.cif', verbose=1)
data, cell, spacegroup = reader()

# Access data (all are numpy arrays)
h, k, l = data['h'], data['k'], data['l']  # Miller indices
F = data['F']              # Amplitudes (if available)
SIGF = data['SIGF']        # Amplitude uncertainties
I = data['I']              # Intensities (if available)
SIGI = data['SIGI']        # Intensity uncertainties
rfree = data['R-free-flags']  # Test set flags (0=free)

# Cell and space group
print(f"Cell: {cell}")  # [a, b, c, alpha, beta, gamma]
print(f"Space group: {spacegroup}")
```

### Quick Checks
```python
reader = ReflectionCIFReader('file-sf.cif')

# Check what data is available
if reader.has_amplitudes():
    F_data = reader.get_amplitudes()
    
if reader.has_intensities():
    I_data = reader.get_intensities()
    
if reader.has_rfree_flags():
    print("Has R-free test set")
```

## ModelCIFReader (Atomic Coordinates)

```python
# Read model CIF file
reader = ModelCIFReader('7JI4.cif', verbose=1)
df, cell, spacegroup = reader()

# DataFrame has all atom data
coords = df[['x', 'y', 'z']].values  # Nx3 array
atoms = df['name']                    # Atom names
residues = df['resname']              # Residue names
chains = df['chainid']                # Chain IDs
occupancy = df['occupancy']           # Occupancy factors
bfactors = df['tempfactor']           # B-factors

# Cell and space group
print(f"Cell: {cell}")
print(f"Space group: {spacegroup}")
```

### Quick Checks
```python
reader = ModelCIFReader('file.cif')

if reader.has_coordinates():
    coords = reader.get_coordinates()  # Nx3 numpy array
    
if reader.has_anisotropic_data():
    # Has anisotropic U values
    u11 = df['u11']
    
if reader.has_occupancy():
    occ = df['occupancy']
```

## RestraintCIFReader (Chemical Restraints)

```python
# Read restraint dictionary
reader = RestraintCIFReader('external_monomer_library/a/ALA.cif')

# Get compound ID
comp_id = reader.get_compound_id()  # 'ALA'

# Get all restraints
restraints = reader.get_all_restraints()
# Returns: {'ALA': {'bonds': DataFrame, 'angles': DataFrame, ...}}

# Get specific restraint type
bonds = reader.get_bond_restraints(comp_id)
# DataFrame with: atom_id_1, atom_id_2, value_dist, value_dist_esd

angles = reader.get_compound_restraints(comp_id)['angles']
torsions = reader.get_compound_restraints(comp_id)['torsions']
```

### Quick Checks
```python
reader = RestraintCIFReader('ALA.cif')

if reader.has_bond_restraints():
    print("Has bond restraints")
    
if reader.has_angle_restraints():
    print("Has angle restraints")
    
if reader.has_torsion_restraints():
    print("Has torsion restraints")
```

## Common Patterns

### Legacy-Compatible Pattern
```python
# Works exactly like MTZ/PDB readers
reader = ReflectionCIFReader('file-sf.cif').read()
data, cell, sg = reader()
```

### Error Handling
```python
try:
    reader = ReflectionCIFReader('file-sf.cif')
    data, cell, sg = reader()
except ValueError as e:
    print(f"Invalid CIF file: {e}")
```

### Checking Data Availability
```python
reader = ReflectionCIFReader('file-sf.cif')
data, cell, sg = reader()

# Check what's present
has_F = 'F' in data
has_I = 'I' in data
has_rfree = 'R-free-flags' in data

print(f"Has amplitudes: {has_F}")
print(f"Has intensities: {has_I}")
print(f"Has R-free: {has_rfree}")
```

## Data Structure Reference

### ReflectionCIFReader Returns
```python
(data_dict, cell, spacegroup) = reader()

# data_dict: Dict[str, np.ndarray]
#   'h', 'k', 'l': int arrays (Miller indices)
#   'F', 'SIGF': float arrays (amplitudes, optional)
#   'I', 'SIGI': float arrays (intensities, optional)
#   'R-free-flags': int32 array (test flags, optional)
#   
# cell: np.ndarray of shape (6,)
#   [a, b, c, alpha, beta, gamma]
#   
# spacegroup: str
#   e.g., 'P 43 21 2'
```

### ModelCIFReader Returns
```python
(dataframe, cell, spacegroup) = reader()

# dataframe: pd.DataFrame with columns:
#   'ATOM', 'serial', 'name', 'altloc', 'resname',
#   'chainid', 'resseq', 'icode', 'x', 'y', 'z',
#   'occupancy', 'tempfactor', 'element', 'charge',
#   'u11', 'u22', 'u33', 'u12', 'u13', 'u23',
#   'anisou_flag', 'index'
#   
# cell: list of 6 floats or None
#   [a, b, c, alpha, beta, gamma]
#   
# spacegroup: str
#   e.g., 'P 1 21 1'
```

### RestraintCIFReader Returns
```python
restraints = reader.get_all_restraints()

# restraints: Dict[str, Dict[str, pd.DataFrame]]
# {
#   'ALA': {
#     'bonds': DataFrame(atom_id_1, atom_id_2, value_dist, value_dist_esd),
#     'angles': DataFrame(...),
#     'torsions': DataFrame(...),
#     'planes': DataFrame(...),
#     'chirals': DataFrame(...)
#   }
# }
```

## Tips

1. **Verbosity**: Use `verbose=1` for info, `verbose=2` for debug
2. **Check availability**: Use `has_*()` methods before accessing data
3. **Handle missing data**: Not all files have F, I, phases, or R-free
4. **Cell parameters**: May be None for some model files
5. **Column names**: Readers handle multiple naming conventions automatically

## Common Errors

```python
# ❌ File doesn't contain reflection data
ValueError: File does not contain reflection data (_refln loop)

# ❌ File doesn't contain atom coordinates
ValueError: File does not contain atomic coordinate data (_atom_site loop)

# ❌ Restraint file missing required parameters
ValueError: File does not contain bond restraint data (_chem_comp_bond)

# ❌ Required column not found
ValueError: Required column not found. Tried: [...]
```

## See Full Documentation

- `COMPLETE_SUMMARY.md` - Complete overview
- `LEGACY_COMPATIBILITY.md` - Migration from MTZ/PDB readers
- `EDGE_CASE_TESTING_SUMMARY.md` - Testing results

---

*Version 1.0 | November 2024*
