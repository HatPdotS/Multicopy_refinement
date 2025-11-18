# Restraints Module Documentation

This directory contains the implementation and tests for the crystallographic restraints system used in the multicopy refinement package.

## Overview

The `Restraints` class provides a comprehensive system for handling geometric restraints in crystallographic model refinement. It parses CIF (Crystallographic Information File) restraints dictionaries and builds efficient PyTorch tensor representations for:

- **Bond length restraints**: Maintain ideal distances between bonded atoms
- **Angle restraints**: Maintain ideal angles between three bonded atoms
- **Torsion angle restraints**: Maintain ideal dihedral angles between four bonded atoms

## Class: `Restraints`

### Location
`multicopy_refinement/restraints_new.py`

### Initialization

```python
from multicopy_refinement.model import Model
from multicopy_refinement.restraints_new import Restraints

# Load model
model = Model()
model.load_pdb_from_file('structure.pdb')

# Create restraints
restraints = Restraints(model, 'restraints.cif')
```

### Data Structure

All restraints are stored as PyTorch tensors with the following structure:

#### Bond Length Restraints
- **`bond_indices`**: `torch.Tensor` of shape `(N_bonds, 2)`
  - Each row contains indices of two atoms forming a bond
  - Type: `torch.long`
- **`bond_references`**: `torch.Tensor` of shape `(N_bonds,)`
  - Expected bond lengths in Ångströms
  - Type: `torch.float32`
- **`bond_sigmas`**: `torch.Tensor` of shape `(N_bonds,)`
  - Uncertainty (σ) values for bond lengths in Ångströms
  - Type: `torch.float32`

#### Angle Restraints
- **`angle_indices`**: `torch.Tensor` of shape `(N_angles, 3)`
  - Each row contains indices of three atoms forming an angle
  - The middle atom (index 1) is the vertex
  - Type: `torch.long`
- **`angle_references`**: `torch.Tensor` of shape `(N_angles,)`
  - Expected angles in degrees
  - Type: `torch.float32`
- **`angle_sigmas`**: `torch.Tensor` of shape `(N_angles,)`
  - Uncertainty (σ) values for angles in degrees
  - Type: `torch.float32`

#### Torsion Angle Restraints
- **`torsion_indices`**: `torch.Tensor` of shape `(N_torsions, 4)`
  - Each row contains indices of four atoms forming a torsion angle
  - Type: `torch.long`
- **`torsion_references`**: `torch.Tensor` of shape `(N_torsions,)`
  - Expected torsion angles in degrees
  - Type: `torch.float32`
- **`torsion_sigmas`**: `torch.Tensor` of shape `(N_torsions,)`
  - Uncertainty (σ) values for torsion angles in degrees
  - Type: `torch.float32`

### Methods

#### `summary()`
Print a detailed summary of all restraints including counts, ranges, and statistics.

```python
restraints.summary()
```

Output:
```
================================================================================
Restraints Summary
================================================================================
CIF file: /path/to/restraints.cif
Residue types in dictionary: 20

Bond Length Restraints: 1234
  Shape: torch.Size([1234, 2])
  Reference distances: min=1.200, max=1.900, mean=1.450
  Sigmas: min=0.0001, max=0.0200, mean=0.0150

Angle Restraints: 2345
  Shape: torch.Size([2345, 3])
  Reference angles: min=90.00°, max=180.00°, mean=120.50°
  Sigmas: min=0.5000°, max=3.0000°, mean=1.5000°

Torsion Angle Restraints: 456
  Shape: torch.Size([456, 4])
  Reference torsions: min=-180.00°, max=180.00°, mean=0.00°
  Sigmas: min=5.0000°, max=30.0000°, mean=15.0000°
================================================================================
```

#### `cuda(device=None)`
Move all restraint tensors to CUDA device.

```python
restraints.cuda()  # Move to default CUDA device
restraints.cuda(0)  # Move to CUDA device 0
```

#### `cpu()`
Move all restraint tensors to CPU.

```python
restraints.cpu()
```

## Usage Examples

### Example 1: Basic Usage

```python
from multicopy_refinement.model import Model
from multicopy_refinement.restraints_new import Restraints

# Load model
model = Model()
model.load_pdb_from_file('structure.pdb')

# Create restraints
restraints = Restraints(model, 'restraints.cif')

# Print summary
restraints.summary()

# Access bond length information
print(f"Total bonds: {restraints.bond_indices.shape[0]}")
print(f"Bond indices:\n{restraints.bond_indices[:5]}")  # First 5 bonds
print(f"Expected lengths:\n{restraints.bond_references[:5]}")
print(f"Uncertainties:\n{restraints.bond_sigmas[:5]}")
```

### Example 2: Computing Bond Length Deviations

```python
import torch

# Get model coordinates
xyz = model.xyz()

# Extract atom positions for each bond
xyz1 = xyz[restraints.bond_indices[:, 0]]
xyz2 = xyz[restraints.bond_indices[:, 1]]

# Compute actual bond lengths
bond_lengths = torch.sqrt(torch.sum((xyz1 - xyz2) ** 2, dim=1))

# Compute deviations from ideal values
deviations = bond_lengths - restraints.bond_references

# Compute normalized deviations (in units of σ)
normalized_deviations = deviations / restraints.bond_sigmas

print(f"RMS deviation: {torch.sqrt((deviations**2).mean()):.3f} Å")
print(f"RMS normalized deviation: {torch.sqrt((normalized_deviations**2).mean()):.2f} σ")
```

### Example 3: Computing Angle Deviations

```python
import torch
import numpy as np

# Get model coordinates
xyz = model.xyz()

# Extract atom positions for each angle
xyz1 = xyz[restraints.angle_indices[:, 0]]
xyz2 = xyz[restraints.angle_indices[:, 1]]  # Vertex
xyz3 = xyz[restraints.angle_indices[:, 2]]

# Compute vectors from vertex
v1 = xyz1 - xyz2
v2 = xyz3 - xyz2

# Normalize vectors
v1_norm = v1 / torch.sqrt(torch.sum(v1**2, dim=1, keepdim=True))
v2_norm = v2 / torch.sqrt(torch.sum(v2**2, dim=1, keepdim=True))

# Compute angles
dot_product = torch.clamp(torch.sum(v1_norm * v2_norm, dim=1), -1.0, 1.0)
angles_rad = torch.arccos(dot_product)
angles_deg = angles_rad * 180.0 / np.pi

# Compute deviations
angle_deviations = angles_deg - restraints.angle_references
normalized_deviations = angle_deviations / restraints.angle_sigmas

print(f"RMS angle deviation: {torch.sqrt((angle_deviations**2).mean()):.2f}°")
print(f"RMS normalized deviation: {torch.sqrt((normalized_deviations**2).mean()):.2f} σ")
```

### Example 4: Using in Optimization

```python
import torch
import torch.optim as optim

# Enable gradients for coordinates
model.xyz.refine_all()  # Make all coordinates refinable

# Define optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(100):
    optimizer.zero_grad()
    
    # Get current coordinates
    xyz = model.xyz()
    
    # Compute bond length loss
    xyz1 = xyz[restraints.bond_indices[:, 0]]
    xyz2 = xyz[restraints.bond_indices[:, 1]]
    bond_lengths = torch.sqrt(torch.sum((xyz1 - xyz2) ** 2, dim=1))
    bond_deviations = (bond_lengths - restraints.bond_references) / restraints.bond_sigmas
    bond_loss = (bond_deviations ** 2).mean()
    
    # Compute angle loss (similar to bond loss)
    # ... angle computation code ...
    
    # Total loss
    loss = bond_loss  # + angle_loss + torsion_loss + other_terms
    
    # Backpropagate and update
    loss.backward()
    optimizer.step()
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item():.4f}")
```

### Example 5: Device Management

```python
# Create restraints on CPU
restraints = Restraints(model, 'restraints.cif')

# Move model and restraints to GPU
if torch.cuda.is_available():
    model.cuda()
    restraints.cuda()
    
    # Now all computations will be on GPU
    xyz = model.xyz()  # On GPU
    bond_lengths = compute_bond_lengths(xyz, restraints.bond_indices)  # On GPU
    
    # Move back to CPU when needed
    model.cpu()
    restraints.cpu()
```

## CIF File Format

The restraints are parsed from standard CIF restraints dictionary files. Each residue type should have:

### Bond Length Section (`_chem_comp_bond`)
```
loop_
_chem_comp_bond.comp_id
_chem_comp_bond.atom_id_1
_chem_comp_bond.atom_id_2
_chem_comp_bond.type
_chem_comp_bond.value_dist
_chem_comp_bond.value_dist_esd
ALA  N   CA   single  1.450  0.020
ALA  CA  C    single  1.520  0.020
...
```

### Angle Section (`_chem_comp_angle`)
```
loop_
_chem_comp_angle.comp_id
_chem_comp_angle.atom_id_1
_chem_comp_angle.atom_id_2
_chem_comp_angle.atom_id_3
_chem_comp_angle.value_angle
_chem_comp_angle.value_angle_esd
ALA  N   CA  C    110.5  1.5
ALA  CA  C   O    120.0  2.0
...
```

### Torsion Angle Section (`_chem_comp_tor`)
```
loop_
_chem_comp_tor.comp_id
_chem_comp_tor.atom_id_1
_chem_comp_tor.atom_id_2
_chem_comp_tor.atom_id_3
_chem_comp_tor.atom_id_4
_chem_comp_tor.value_angle
_chem_comp_tor.value_angle_esd
ALA  N   CA  C   O    180.0  5.0
...
```

## Testing

Comprehensive tests are provided in this directory:

### Test Files

1. **`test_restraints_creation.py`**
   - Tests basic creation and initialization
   - Verifies data structure correctness
   - Tests device management (CPU/CUDA)

2. **`test_bond_lengths.py`**
   - Tests bond length computation
   - Verifies bond indices validity
   - Tests gradient computation

3. **`test_angles.py`**
   - Tests angle computation
   - Verifies angle indices validity
   - Tests gradient computation

4. **`test_torsions.py`**
   - Tests torsion angle computation
   - Verifies torsion indices validity
   - Tests gradient computation

### Running Tests

Run all tests:
```bash
cd /das/work/p17/p17490/Peter/Library/multicopy_refinement/tests/restraints
python test_restraints_creation.py
python test_bond_lengths.py
python test_angles.py
python test_torsions.py
```

Run a specific test:
```bash
python test_bond_lengths.py
```

## Performance Considerations

### Memory Usage
- All restraints are stored as tensors, which is memory-efficient
- For a typical protein with 1000 atoms:
  - ~2000 bonds: ~50 KB
  - ~3000 angles: ~100 KB
  - ~500 torsions: ~30 KB
  - Total: ~200 KB

### Computational Efficiency
- All operations are vectorized using PyTorch
- Can be moved to GPU for faster computation
- Gradients computed efficiently for optimization

### Scalability
- Linear scaling with number of atoms
- Efficient for structures with 10,000+ atoms
- Can handle multiple chains and heterogeneous systems

## Known Limitations

1. **HETATM handling**: Currently skips HETATM entries. Future versions may include support for ligands and non-standard residues.

2. **Plane restraints**: Not yet implemented. Planned for future versions.

3. **Inter-residue restraints**: Currently only handles intra-residue restraints. Peptide bond restraints between residues may be added in the future.

4. **Alternative conformations**: Restraints are built for all atoms, but special handling of alternative conformations may be needed.

## Future Enhancements

- [ ] Add support for plane restraints
- [ ] Add support for inter-residue (peptide bond) restraints
- [ ] Add support for HETATM and ligand restraints
- [ ] Add custom restraint generation utilities
- [ ] Add restraint visualization tools
- [ ] Add restraint statistics and quality metrics

## References

- CIF format specification: [IUCr CIF](https://www.iucr.org/resources/cif)
- Engh & Huber (1991) protein geometry parameters
- Geometry restraints in crystallographic refinement

## Contact

For questions or issues, please contact the maintainer or open an issue in the repository.
