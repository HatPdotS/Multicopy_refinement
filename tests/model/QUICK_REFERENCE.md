# Quick Reference: Alternative Conformations

## Quick Start

```python
from multicopy_refinement.model_new import model

# Load PDB - altloc_pairs is automatically populated
m = model()
m.load_pdb_from_file('structure.pdb')

# Access all alternative conformation groups
print(len(m.altloc_pairs))  # Number of groups
```

## Data Structure

```python
# altloc_pairs is a list of tuples
# Each tuple contains DataFrame indices of alternative conformations
m.altloc_pairs = [
    (idx1, idx2),           # 2-way: A, B
    (idx3, idx4, idx5),     # 3-way: A, B, C
    (idx6, idx7),           # 2-way: A, B
    ...
]
```

## Common Operations

### 1. Count Groups
```python
print(f"Total groups: {len(m.altloc_pairs)}")
```

### 2. Iterate Over Groups
```python
for group in m.altloc_pairs:
    atoms = m.pdb.loc[list(group)]
    # Process atoms...
```

### 3. Get Atom Details
```python
group = m.altloc_pairs[0]
atoms = m.pdb.loc[list(group)]

# Same for all atoms in group
atom_name = atoms['name'].iloc[0]
residue = atoms['resname'].iloc[0]
resseq = atoms['resseq'].iloc[0]
chain = atoms['chainid'].iloc[0]

# Different for each conformation
altlocs = atoms['altloc'].tolist()      # ['A', 'B']
coords = atoms[['x', 'y', 'z']].values  # [[x1,y1,z1], [x2,y2,z2]]
occupancies = atoms['occupancy'].tolist()  # [0.6, 0.4]
```

### 4. Filter by Residue
```python
# Find all altlocs for specific residue
groups = []
for group in m.altloc_pairs:
    atoms = m.pdb.loc[list(group)]
    if (atoms['resname'].iloc[0] == 'ARG' and 
        atoms['resseq'].iloc[0] == 123):
        groups.append(group)
```

### 5. Count by Size
```python
from collections import Counter
sizes = Counter(len(g) for g in m.altloc_pairs)
print(f"2-way: {sizes[2]}, 3-way: {sizes[3]}")
```

## Test Files Location

All test files are in: `/das/work/p17/p17490/Peter/Library/multicopy_refinement/tests/model/`

### Run Tests
```bash
cd /das/work/p17/p17490/Peter/Library/multicopy_refinement/tests/model
python test_alternative_conformations.py
```

### Run Examples
```bash
python example_altloc_usage.py
python verify_implementation.py
```

## Key Properties

- **Automatic**: Populated during `load_pdb_from_file()`
- **Sorted**: Conformations ordered by altloc (A, B, C, ...)
- **Unique**: No duplicate indices
- **Comprehensive**: Handles 2-way, 3-way, N-way conformations
- **Empty list**: If no alternative conformations exist

## Documentation Files

1. `README_ALTERNATIVE_CONFORMATIONS.md` - Full documentation
2. `IMPLEMENTATION_SUMMARY.md` - Implementation details
3. `example_altloc_usage.py` - Usage examples
4. `test_alternative_conformations.py` - Test suite
