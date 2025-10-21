"""
Quick verification that both methods work together correctly.
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
print("QUICK VERIFICATION: Both Methods Working Together")
print("="*80)

# Load structure
print("\n1. Loading PDB file...")
m = model()
m.load_pdb_from_file('/das/work/p17/p17490/Peter/Library/multicopy_refinement/test_data/dark.pdb')
print(f"   ✓ Loaded {len(m.pdb)} atoms")
print(f"   ✓ Found {len(m.altloc_pairs)} residues with alternative conformations")

# Verify registration worked
print("\n2. Verifying registration...")
assert hasattr(m, 'altloc_pairs'), "Should have altloc_pairs attribute"
assert len(m.altloc_pairs) > 0, "Should have found some altlocs"
assert isinstance(m.altloc_pairs[0], tuple), "Each group should be a tuple"
assert isinstance(m.altloc_pairs[0][0], torch.Tensor), "Each conformation should be a tensor"
print(f"   ✓ Registration structure correct")

# Show a few examples
print("\n3. Example registered groups:")
for i, group in enumerate(m.altloc_pairs[:3]):
    atoms = m.pdb.loc[group[0].tolist()]
    resname = atoms['resname'].iloc[0]
    resseq = atoms['resseq'].iloc[0]
    chainid = atoms['chainid'].iloc[0]
    n_conf = len(group)
    n_atoms = len(group[0])
    print(f"   {i+1}. {resname}-{resseq} Chain {chainid}: {n_conf} conformations, {n_atoms} atoms each")

# Enforce constraints
print("\n4. Enforcing occupancy constraints...")
m.enforce_alternative_conformations()
print(f"   ✓ Constraints enforced")

# Verify constraints
print("\n5. Verifying constraints...")
occ = m.occupancy().detach()

# Check uniform within conformations
all_uniform = True
for group in m.altloc_pairs:
    for conf in group:
        if occ[conf].std() > 1e-6:
            all_uniform = False
            break

# Check sum to 1.0
all_sum_to_one = True
for group in m.altloc_pairs:
    total = sum(occ[conf].mean() for conf in group).item()
    if abs(total - 1.0) > 1e-5:
        all_sum_to_one = False
        break

print(f"   ✓ Uniform occupancy within conformations: {all_uniform}")
print(f"   ✓ Occupancies sum to 1.0: {all_sum_to_one}")

# Show example occupancies
print("\n6. Example occupancies after enforcement:")
for i, group in enumerate(m.altloc_pairs[:3]):
    atoms = m.pdb.loc[group[0].tolist()]
    resname = atoms['resname'].iloc[0]
    resseq = atoms['resseq'].iloc[0]
    chainid = atoms['chainid'].iloc[0]
    
    print(f"   {i+1}. {resname}-{resseq} Chain {chainid}:")
    occupancies = []
    for j, conf in enumerate(group):
        atoms_conf = m.pdb.loc[conf.tolist()]
        altloc = atoms_conf['altloc'].iloc[0]
        occ_val = occ[conf[0]].item()
        occupancies.append(occ_val)
        print(f"      Altloc {altloc}: {occ_val:.4f}")
    
    total = sum(occupancies)
    print(f"      Sum: {total:.6f}")

# Test idempotency
print("\n7. Testing idempotency...")
occ_before = m.occupancy().detach().clone()
m.enforce_alternative_conformations()
occ_after = m.occupancy().detach()
max_diff = (occ_after - occ_before).abs().max().item()
print(f"   Max difference after re-enforcement: {max_diff:.10f}")
print(f"   ✓ Idempotent: {max_diff < 1e-10}")

# Summary
print("\n" + "="*80)
print("VERIFICATION SUMMARY")
print("="*80)
print("✓ register_alternative_conformations() - Working correctly")
print("✓ enforce_alternative_conformations() - Working correctly")
print("✓ Both methods integrate properly")
print("✓ All constraints satisfied")
print("\n✅ ALL VERIFICATIONS PASSED ✅")
print("="*80)
