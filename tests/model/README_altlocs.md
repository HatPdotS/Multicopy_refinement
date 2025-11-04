# Alternative Conformations Registration

## Overview

The `register_alternative_conformations()` method identifies and registers all alternative conformations in a protein structure at the **residue level**. Alternative conformations (altlocs) represent different structural states of the same residue, commonly observed in X-ray crystallography data.

## Implementation

### Location
- **File**: `multicopy_refinement/model_new.py`
- **Class**: `model`
- **Method**: `register_alternative_conformations()`

### Data Structure

The method stores results in `self.altloc_pairs`, which is a list of tuples where:
- Each tuple represents one residue with alternative conformations
- Each element in the tuple is a `torch.Tensor` containing atom indices for that conformation
- Conformations are sorted alphabetically by altloc identifier (A, B, C, ...)

```python
# Structure:
self.altloc_pairs = [
    # Residue 1 with 2 conformations (A and B)
    (tensor([100, 101, 102, ...]),  # Conformation A indices
     tensor([110, 111, 112, ...])), # Conformation B indices
    
    # Residue 2 with 3 conformations (A, B, and C)
    (tensor([200, 201, ...]),       # Conformation A indices
     tensor([210, 211, ...]),       # Conformation B indices
     tensor([220, 221, ...])),      # Conformation C indices
    ...
]
```

## Usage

### Basic Usage

```python
from multicopy_refinement.model_new import model

# Load a PDB file
m = model()
m.load_pdb_from_file('structure.pdb')

# Alternative conformations are automatically registered
print(f"Found {len(m.altloc_pairs)} residues with alternative conformations")

# Iterate through groups
for group in m.altloc_pairs:
    n_conformations = len(group)
    n_atoms_per_conf = len(group[0])
    print(f"Residue with {n_conformations} conformations, {n_atoms_per_conf} atoms each")
```

### Accessing Conformation Data

```python
# Get the first residue with alternative conformations
first_group = m.altloc_pairs[0]

for i, atom_indices in enumerate(first_group):
    # Get atom information
    atoms = m.pdb.loc[atom_indices.tolist()]
    
    # Get residue identifier
    resname = atoms['resname'].iloc[0]  # e.g., 'ARG'
    resseq = atoms['resseq'].iloc[0]    # e.g., 123
    chainid = atoms['chainid'].iloc[0]  # e.g., 'A'
    altloc = atoms['altloc'].iloc[0]    # e.g., 'A', 'B', 'C'
    
    print(f"Conformation {altloc}: {resname}-{resseq} Chain {chainid}")
    print(f"  Atom indices: {atom_indices.tolist()}")
    
    # Get coordinates
    coords = m.xyz()[atom_indices]
    print(f"  Coordinates shape: {coords.shape}")
```

### Computing RMSD Between Conformations

```python
import torch

def compute_rmsd(coords1, coords2):
    """Compute RMSD between two sets of coordinates."""
    diff = coords1 - coords2
    squared_diff = (diff ** 2).sum(dim=1)
    rmsd = torch.sqrt(squared_diff.mean())
    return rmsd

# For residues with exactly 2 conformations
pairs = [group for group in m.altloc_pairs if len(group) == 2]

for indices_a, indices_b in pairs:
    coords_a = m.xyz()[indices_a]
    coords_b = m.xyz()[indices_b]
    rmsd = compute_rmsd(coords_a, coords_b)
    print(f"RMSD: {rmsd:.3f} Å")
```

## Properties and Guarantees

### 1. Residue-Level Grouping
All atoms belonging to the same residue and altloc identifier are grouped together.

### 2. Consistent Ordering
- Conformations within a group are sorted alphabetically by altloc (A, B, C, ...)
- Atom order within each conformation matches the original PDB file

### 3. Equal Atom Counts
All conformations within a group have the same number of atoms.

```python
for group in m.altloc_pairs:
    lengths = [len(conf) for conf in group]
    assert all(l == lengths[0] for l in lengths), "All conformations have same atom count"
```

### 4. Matching Atom Names
Corresponding atoms in different conformations have the same atom names.

```python
group = m.altloc_pairs[0]
atoms_a = m.pdb.loc[group[0].tolist()]
atoms_b = m.pdb.loc[group[1].tolist()]

# Atom names match (though order may vary)
assert set(atoms_a['name']) == set(atoms_b['name'])
```

### 5. No Overlapping Indices
Each atom index appears in at most one conformation within a group.

## Examples

### Example 1: Count Alternative Conformations

```python
m = model()
m.load_pdb_from_file('structure.pdb')

# Count by number of conformations
conf_counts = {}
for group in m.altloc_pairs:
    n_conf = len(group)
    conf_counts[n_conf] = conf_counts.get(n_conf, 0) + 1

print(f"Total residues with altlocs: {len(m.altloc_pairs)}")
for n, count in sorted(conf_counts.items()):
    print(f"  {n} conformations: {count} residues")
```

### Example 2: Extract Specific Conformation

```python
# Extract all atoms from conformation 'A'
for group in m.altloc_pairs:
    atoms = m.pdb.loc[group[0].tolist()]
    if atoms['altloc'].iloc[0] == 'A':
        # Get coordinates for conformation A
        coords_a = m.xyz()[group[0]]
        # Process coordinates...
```

### Example 3: Analyze Conformational Differences

```python
# For residues with 3 conformations
triplets = [g for g in m.altloc_pairs if len(g) == 3]

for group in triplets:
    # Compute all pairwise RMSDs
    for i in range(len(group)):
        for j in range(i+1, len(group)):
            coords_i = m.xyz()[group[i]]
            coords_j = m.xyz()[group[j]]
            rmsd = compute_rmsd(coords_i, coords_j)
            
            altloc_i = m.pdb.loc[group[i][0].item(), 'altloc']
            altloc_j = m.pdb.loc[group[j][0].item(), 'altloc']
            print(f"  {altloc_i} ↔ {altloc_j}: RMSD = {rmsd:.3f} Å")
```

## Testing

Comprehensive tests are available in:
- `tests/model/test_register_altlocs.py` - Full test suite
- `tests/model/test_altlocs_standalone.py` - Standalone verification
- `tests/model/example_usage_altlocs.py` - Usage examples

Run tests:
```bash
cd tests/model
python test_altlocs_standalone.py
python example_usage_altlocs.py
```

## Implementation Details

### Algorithm

1. **Filter atoms with altlocs**: Select all atoms where `altloc != ''`
2. **Group by residue**: Group by `(resname, resseq, chainid)`
3. **For each residue**:
   - Find all unique altloc identifiers
   - For each altloc, collect all atom indices
   - Convert to tensors and store as tuple
4. **Only register if multiple conformations exist**

### Code

```python
def register_alternative_conformations(self):
    self.altloc_pairs = []
    pdb_with_altlocs = self.pdb[self.pdb['altloc'] != '']
    
    if len(pdb_with_altlocs) == 0:
        return
    
    grouped = pdb_with_altlocs.groupby(['resname', 'resseq', 'chainid'])
    
    for (resname, resseq, chainid), group in grouped:
        unique_altlocs = sorted(group['altloc'].unique())
        
        if len(unique_altlocs) > 1:
            conformation_tensors = []
            for altloc in unique_altlocs:
                altloc_atoms = group[group['altloc'] == altloc]
                indices = torch.tensor(altloc_atoms['index'].tolist(), dtype=torch.long)
                conformation_tensors.append(indices)
            
            self.altloc_pairs.append(tuple(conformation_tensors))
```

## Notes

- **Automatic Registration**: The method is called automatically when loading a PDB file via `load_pdb_from_file()`
- **Hydrogen Stripping**: By default, hydrogens are stripped before registration
- **Empty Structures**: If no alternative conformations exist, `altloc_pairs` will be an empty list
- **Tensor Type**: All indices are stored as `torch.long` tensors for compatibility with PyTorch operations

## Related Attributes

- `self.pdb`: Pandas DataFrame containing all atom information
- `self.xyz`: MixedTensor containing atomic coordinates
- `self.occupancy`: MixedTensor containing occupancy values
- `self.b`: MixedTensor containing B-factors

## Performance

- Registration is performed once during PDB loading
- Time complexity: O(n) where n is the number of atoms with altlocs
- Memory: Stores only atom indices, not full atomic data
