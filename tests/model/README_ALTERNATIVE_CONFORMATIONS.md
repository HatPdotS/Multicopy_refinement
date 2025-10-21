# Alternative Conformations Registration

## Overview

The `register_alternative_conformations()` method in the `model` class identifies and registers all alternative conformation groups in a protein structure loaded from a PDB file.

## What are Alternative Conformations?

In crystallographic structures, some atoms can occupy multiple conformations (positions) simultaneously. These are marked with altloc identifiers (typically 'A', 'B', 'C', etc.) in PDB files. For example:

```
ATOM     64   CA ACYS A   4      10.123  20.456  30.789  0.60 15.00           C
ATOM     65   CA BCYS A   4      10.234  20.567  30.890  0.40 15.00           C
```

Here, the CA atom of CYS-4 has two alternative conformations (A and B) with occupancies 0.60 and 0.40.

## Implementation

### Function Signature

```python
def register_alternative_conformations(self):
    """
    Identify and register all alternative conformation groups in the structure.
    
    The result is stored in self.altloc_pairs as a list of tuples, where each
    tuple contains the indices of atoms representing alternative conformations
    of the same atom.
    """
```

### What it Does

1. **Filters atoms** with alternative conformations (altloc != '')
2. **Groups atoms** by residue name, residue sequence number, chain ID, and atom name
3. **Identifies pairs/groups** where the same atom has multiple conformations
4. **Sorts** conformations alphabetically by altloc identifier
5. **Stores** the indices as tuples in `self.altloc_pairs`

### Output Format

The method populates `self.altloc_pairs` with a list of tuples:

- **2-way conformations**: `[(idx_A, idx_B), ...]`
- **3-way conformations**: `[(idx_A, idx_B, idx_C), ...]`
- **N-way conformations**: `[(idx_1, idx_2, ..., idx_N), ...]`

Each tuple contains the DataFrame indices of atoms that represent alternative conformations of the same atom.

## Usage Example

```python
from multicopy_refinement.model_new import model

# Load a PDB file
m = model()
m.load_pdb_from_file('structure.pdb')

# Alternative conformations are automatically registered
print(f"Found {len(m.altloc_pairs)} alternative conformation groups")

# Access the first group
first_group = m.altloc_pairs[0]
atoms = m.pdb.loc[list(first_group)]

# All atoms in this group represent the same atom in different conformations
print(f"Atom: {atoms['name'].iloc[0]}")
print(f"Residue: {atoms['resname'].iloc[0]}-{atoms['resseq'].iloc[0]}")
print(f"Conformations: {atoms['altloc'].tolist()}")
```

## Test Suite

Comprehensive tests are provided in `tests/model/test_alternative_conformations.py`:

### Test Coverage

1. **test_with_real_pdb_file**: Verifies the function works with real PDB files
2. **test_pair_indices_match_atoms**: Ensures indices correctly identify alternative conformations
3. **test_altloc_ordering**: Confirms conformations are sorted alphabetically
4. **test_no_duplicate_indices**: Verifies each atom appears in only one group
5. **test_count_conformations**: Tests handling of 2-way, 3-way, and N-way conformations
6. **test_pdb_without_altlocs**: Ensures graceful handling when no altlocs exist
7. **test_specific_residue_altlocs**: Tests specific known residues with altlocs
8. **test_altloc_pairs_attribute_exists**: Verifies the attribute is created
9. **test_manual_pdb_with_altlocs**: Tests with manually created PDB files

### Running the Tests

```bash
cd tests/model
python test_alternative_conformations.py
```

Expected output:
```
================================================================================
Testing register_alternative_conformations
================================================================================

Real PDB file... ✓ PASSED
Indices match atoms... ✓ PASSED
Altloc ordering... ✓ PASSED
No duplicate indices... ✓ PASSED
Count conformations... ✓ PASSED
PDB without altlocs... ✓ PASSED
Specific residue... ✓ PASSED
Attribute exists... ✓ PASSED
Manual PDB... ✓ PASSED

================================================================================
Results: 9 passed, 0 failed out of 9 tests
================================================================================
```

## Example Use Cases

### 1. Count Alternative Conformations

```python
# Count distribution by conformation size
group_sizes = {}
for pair in model.altloc_pairs:
    size = len(pair)
    group_sizes[size] = group_sizes.get(size, 0) + 1

print(f"2-way conformations: {group_sizes.get(2, 0)}")
print(f"3-way conformations: {group_sizes.get(3, 0)}")
```

### 2. Find Alternative Conformations for a Specific Residue

```python
# Find all altlocs for ARG-123 in chain A
arg123_groups = []
for pair in model.altloc_pairs:
    atoms = model.pdb.loc[list(pair)]
    if (atoms['resname'].iloc[0] == 'ARG' and 
        atoms['resseq'].iloc[0] == 123 and 
        atoms['chainid'].iloc[0] == 'A'):
        arg123_groups.append(pair)
```

### 3. Access Coordinates and Occupancies

```python
# Get coordinates for each conformation
for group in model.altloc_pairs[:5]:
    atoms = model.pdb.loc[list(group)]
    print(f"Atom: {atoms['name'].iloc[0]}")
    for idx, row in atoms.iterrows():
        print(f"  Altloc {row['altloc']}: "
              f"({row['x']:.3f}, {row['y']:.3f}, {row['z']:.3f}), "
              f"occ={row['occupancy']:.2f}")
```

### 4. Refine Specific Alternative Conformations

```python
# Make only altloc A refinable for a specific group
group = model.altloc_pairs[0]
atoms = model.pdb.loc[list(group)]

# Get the index for altloc A
altloc_a_idx = atoms[atoms['altloc'] == 'A'].index[0]

# Create a mask for refinement
refine_mask = torch.zeros(len(model.xyz()), dtype=torch.bool)
refine_mask[altloc_a_idx] = True
model.xyz.refine(refine_mask)
```

## Implementation Details

### Algorithm

1. Filter atoms where `altloc != ''`
2. Group by `['resname', 'resseq', 'chainid', 'name']`
3. For each group with ≥2 atoms:
   - Extract indices and altloc identifiers
   - Sort by altloc identifier (A, B, C, ...)
   - Store sorted indices as a tuple

### Time Complexity

- **O(N)** where N is the number of atoms with alternative conformations
- Grouping operation is O(N log N)
- Dominated by pandas groupby operation

### Space Complexity

- **O(M)** where M is the number of alternative conformation groups
- Stores one tuple per group

## Notes

- The function is automatically called during `load_pdb_from_file()`
- If no alternative conformations exist, `altloc_pairs` will be an empty list
- Indices are sorted alphabetically by altloc identifier for consistency
- Each atom index appears in at most one group (no duplicates)
- Compatible with structures containing 2, 3, or more alternative conformations

## See Also

- `example_altloc_usage.py`: Detailed usage examples
- `test_alternative_conformations.py`: Comprehensive test suite
- `MixedTensor` class: For selectively refining alternative conformations
