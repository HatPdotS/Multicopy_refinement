"""
Example usage of the register_alternative_conformations function.

This demonstrates how to use the alternative conformation registration
feature to identify and work with alternative conformations at the residue level.
"""
import sys
sys.path.insert(0, '/das/work/p17/p17490/Peter/Library/multicopy_refinement/multicopy_refinement')

import importlib.util
import torch

# Load model_new directly
spec = importlib.util.spec_from_file_location(
    "model_new", 
    "/das/work/p17/p17490/Peter/Library/multicopy_refinement/multicopy_refinement/model_new.py"
)
model_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model_module)
model = model_module.model

# Create a model and load a PDB file
print("Loading PDB file with alternative conformations...")
m = model()
m.load_pdb_from_file('/das/work/p17/p17490/Peter/Library/multicopy_refinement/test_data/dark.pdb')

print(f"\nFound {len(m.altloc_pairs)} residues with alternative conformations")

# Example 1: Iterate through all alternative conformation groups
print("\n" + "="*80)
print("Example 1: Iterating through alternative conformation groups")
print("="*80)

for i, conformation_group in enumerate(m.altloc_pairs[:3]):  # Show first 3
    print(f"\nResidue group {i+1}:")
    print(f"  Number of alternative conformations: {len(conformation_group)}")
    
    for j, atom_indices in enumerate(conformation_group):
        # Get the atoms for this conformation
        atoms = m.pdb.loc[atom_indices.tolist()]
        
        # Extract residue information
        resname = atoms['resname'].iloc[0]
        resseq = atoms['resseq'].iloc[0]
        chainid = atoms['chainid'].iloc[0]
        altloc = atoms['altloc'].iloc[0]
        
        print(f"  Conformation {altloc}:")
        print(f"    Residue: {resname}-{resseq} Chain {chainid}")
        print(f"    Number of atoms: {len(atom_indices)}")
        print(f"    Atom indices: {atom_indices.tolist()}")

# Example 2: Accessing coordinates for alternative conformations
print("\n" + "="*80)
print("Example 2: Accessing coordinates for alternative conformations")
print("="*80)

# Take the first residue with alternative conformations
first_group = m.altloc_pairs[0]
print(f"\nFirst residue has {len(first_group)} conformations")

for i, atom_indices in enumerate(first_group):
    # Get coordinates using the model's xyz tensor
    coords = m.xyz()[atom_indices]
    
    # Get atom information
    atoms = m.pdb.loc[atom_indices.tolist()]
    altloc = atoms['altloc'].iloc[0]
    
    print(f"\nConformation {altloc}:")
    print(f"  Shape of coordinates: {coords.shape}")
    print(f"  First atom coordinates: {coords[0].tolist()}")
    print(f"  Center of mass: {coords.mean(dim=0).tolist()}")

# Example 3: Computing RMSD between alternative conformations
print("\n" + "="*80)
print("Example 3: Computing RMSD between alternative conformations")
print("="*80)

def compute_rmsd(coords1, coords2):
    """Compute RMSD between two sets of coordinates."""
    diff = coords1 - coords2
    squared_diff = (diff ** 2).sum(dim=1)
    rmsd = torch.sqrt(squared_diff.mean())
    return rmsd

# Look at residues with exactly 2 conformations
pairs = [group for group in m.altloc_pairs if len(group) == 2]
print(f"\nFound {len(pairs)} residues with exactly 2 conformations")
print("\nRMSD between conformations for first 5 residues:")

for i, (indices_a, indices_b) in enumerate(pairs[:5]):
    coords_a = m.xyz()[indices_a]
    coords_b = m.xyz()[indices_b]
    
    rmsd = compute_rmsd(coords_a, coords_b)
    
    atoms_a = m.pdb.loc[indices_a.tolist()]
    resname = atoms_a['resname'].iloc[0]
    resseq = atoms_a['resseq'].iloc[0]
    chainid = atoms_a['chainid'].iloc[0]
    
    print(f"{i+1}. {resname}-{resseq} Chain {chainid}: RMSD = {rmsd:.3f} Angstrom")

# Example 4: Working with triplet conformations
print("\n" + "="*80)
print("Example 4: Residues with 3 or more conformations")
print("="*80)

triplets = [group for group in m.altloc_pairs if len(group) >= 3]
print(f"\nFound {len(triplets)} residues with 3+ conformations")

for i, group in enumerate(triplets):
    print(f"\nResidue {i+1}:")
    print(f"  Number of conformations: {len(group)}")
    
    # Get info from first conformation
    atoms = m.pdb.loc[group[0].tolist()]
    resname = atoms['resname'].iloc[0]
    resseq = atoms['resseq'].iloc[0]
    chainid = atoms['chainid'].iloc[0]
    print(f"  Residue: {resname}-{resseq} Chain {chainid}")
    
    # Compute pairwise RMSDs
    print("  Pairwise RMSDs:")
    for j in range(len(group)):
        for k in range(j+1, len(group)):
            coords_j = m.xyz()[group[j]]
            coords_k = m.xyz()[group[k]]
            rmsd = compute_rmsd(coords_j, coords_k)
            
            altloc_j = m.pdb.loc[group[j][0].item(), 'altloc']
            altloc_k = m.pdb.loc[group[k][0].item(), 'altloc']
            print(f"    {altloc_j} <-> {altloc_k}: {rmsd:.3f} Angstrom")

# Example 5: Summary statistics
print("\n" + "="*80)
print("Example 5: Summary statistics")
print("="*80)

conf_counts = {}
for group in m.altloc_pairs:
    n_conf = len(group)
    conf_counts[n_conf] = conf_counts.get(n_conf, 0) + 1

print(f"\nTotal residues with alternative conformations: {len(m.altloc_pairs)}")
for n_conf, count in sorted(conf_counts.items()):
    print(f"  Residues with {n_conf} conformations: {count}")

# Count total atoms involved in alternative conformations
total_alt_atoms = sum(len(conf) * len(group) for group in m.altloc_pairs for conf in [group[0]])
print(f"\nTotal atoms involved in alternative conformations: {total_alt_atoms}")
print(f"Percentage of structure: {100.0 * total_alt_atoms / len(m.pdb):.1f}%")

print("\n" + "="*80)
print("Examples complete!")
print("="*80)
