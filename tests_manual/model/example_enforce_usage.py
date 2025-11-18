"""
Example usage of the enforce_alternative_conformations method.

This demonstrates how to enforce occupancy constraints on alternative conformations.
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

print("="*80)
print("EXAMPLE: Enforcing Alternative Conformation Occupancy Constraints")
print("="*80)

# Load a PDB file
m = model()
m.load_pdb_from_file('/das/work/p17/p17490/Peter/Library/multicopy_refinement/test_data/dark.pdb')

print(f"\nLoaded structure with {len(m.altloc_pairs)} residues having alternative conformations")

# Example 1: Show occupancies before enforcement
print("\n" + "="*80)
print("Example 1: Occupancies before enforcement")
print("="*80)

occ_before = m.occupancy().detach().clone()

print("\nFirst 5 residues:")
for i, group in enumerate(m.altloc_pairs[:5]):
    atoms_first = m.pdb.loc[group[0].tolist()]
    resname = atoms_first['resname'].iloc[0]
    resseq = atoms_first['resseq'].iloc[0]
    chainid = atoms_first['chainid'].iloc[0]
    
    print(f"\n{i+1}. {resname}-{resseq} Chain {chainid}:")
    
    mean_occupancies = []
    for j, conf_indices in enumerate(group):
        atoms = m.pdb.loc[conf_indices.tolist()]
        altloc = atoms['altloc'].iloc[0]
        conf_occ = occ_before[conf_indices]
        mean_occ = conf_occ.mean().item()
        std_occ = conf_occ.std().item()
        mean_occupancies.append(mean_occ)
        
        print(f"   Altloc {altloc}: mean={mean_occ:.4f}, std={std_occ:.6f}, values={conf_occ[:3].tolist()}...")
    
    total = sum(mean_occupancies)
    print(f"   Sum of means: {total:.6f}")

# Example 2: Enforce constraints
print("\n" + "="*80)
print("Example 2: Enforcing occupancy constraints")
print("="*80)

print("\nCalling enforce_alternative_conformations()...")
m.enforce_alternative_conformations()
print("✓ Constraints enforced")

# Example 3: Show occupancies after enforcement
print("\n" + "="*80)
print("Example 3: Occupancies after enforcement")
print("="*80)

occ_after = m.occupancy().detach()

print("\nSame 5 residues after enforcement:")
for i, group in enumerate(m.altloc_pairs[:5]):
    atoms_first = m.pdb.loc[group[0].tolist()]
    resname = atoms_first['resname'].iloc[0]
    resseq = atoms_first['resseq'].iloc[0]
    chainid = atoms_first['chainid'].iloc[0]
    
    print(f"\n{i+1}. {resname}-{resseq} Chain {chainid}:")
    
    mean_occupancies = []
    for j, conf_indices in enumerate(group):
        atoms = m.pdb.loc[conf_indices.tolist()]
        altloc = atoms['altloc'].iloc[0]
        conf_occ = occ_after[conf_indices]
        mean_occ = conf_occ.mean().item()
        std_occ = conf_occ.std().item()
        mean_occupancies.append(mean_occ)
        
        print(f"   Altloc {altloc}: mean={mean_occ:.4f}, std={std_occ:.10f}, values={conf_occ[:3].tolist()}...")
    
    total = sum(mean_occupancies)
    print(f"   Sum of means: {total:.6f} {'✓' if abs(total - 1.0) < 1e-5 else '✗'}")

# Example 4: Verify constraints are satisfied
print("\n" + "="*80)
print("Example 4: Verifying constraints")
print("="*80)

all_uniform = True
all_sum_to_one = True

for group in m.altloc_pairs:
    # Check uniform occupancy within each conformation
    for conf_indices in group:
        std = occ_after[conf_indices].std().item()
        if std > 1e-6:
            all_uniform = False
            break
    
    # Check sum to 1.0
    mean_occupancies = [occ_after[conf].mean() for conf in group]
    total = sum(mean_occupancies).item()
    if abs(total - 1.0) > 1e-5:
        all_sum_to_one = False

print(f"✓ All conformations have uniform occupancy: {all_uniform}")
print(f"✓ All groups sum to 1.0: {all_sum_to_one}")

# Example 5: Use in refinement loop
print("\n" + "="*80)
print("Example 5: Integration with refinement")
print("="*80)

print("""
Typical usage in a refinement loop:

    # Set up optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    for epoch in range(num_epochs):
        # Forward pass
        loss = compute_loss(model)
        
        # Backward pass
        loss.backward()
        
        # Update parameters
        optimizer.step()
        optimizer.zero_grad()
        
        # Enforce occupancy constraints
        model.enforce_alternative_conformations()
        
        # The occupancies now satisfy:
        # 1. Uniform within each conformation
        # 2. Sum to 1.0 across conformations
""")

# Example 6: Statistics
print("\n" + "="*80)
print("Example 6: Summary statistics")
print("="*80)

conf_counts = {}
for group in m.altloc_pairs:
    n = len(group)
    conf_counts[n] = conf_counts.get(n, 0) + 1

print(f"\nTotal residues with alternative conformations: {len(m.altloc_pairs)}")
for n, count in sorted(conf_counts.items()):
    print(f"  {n} conformations: {count} residues")

# Count atoms involved
total_alt_atoms = sum(len(conf) for group in m.altloc_pairs for conf in group)
print(f"\nTotal atoms in alternative conformations: {total_alt_atoms}")
print(f"Percentage of structure: {100.0 * total_alt_atoms / len(m.pdb):.1f}%")

# Show occupancy distribution
all_alt_occupancies = []
for group in m.altloc_pairs:
    for conf in group:
        all_alt_occupancies.append(occ_after[conf].mean().item())

print(f"\nOccupancy distribution:")
print(f"  Min: {min(all_alt_occupancies):.4f}")
print(f"  Max: {max(all_alt_occupancies):.4f}")
print(f"  Mean: {sum(all_alt_occupancies)/len(all_alt_occupancies):.4f}")

print("\n" + "="*80)
print("Examples complete!")
print("="*80)
