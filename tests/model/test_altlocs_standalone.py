"""
Standalone test for register_alternative_conformations.
This avoids import issues by loading model_new directly.
"""
import sys
import os
import torch

# Set up path before any imports
sys.path.insert(0, '/das/work/p17/p17490/Peter/Library/multicopy_refinement/multicopy_refinement')

# Prevent __init__.py from being loaded
import importlib
import importlib.util

# Load model_new directly without going through package __init__
spec = importlib.util.spec_from_file_location(
    "model_new", 
    "/das/work/p17/p17490/Peter/Library/multicopy_refinement/multicopy_refinement/model_new.py"
)
model_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model_module)
model = model_module.model

print("Testing register_alternative_conformations with real data...")
print("="*80)

# Test 1: Load real PDB and check structure
m = model()
m.load_pdb_from_file('/das/work/p17/p17490/Peter/Library/multicopy_refinement/test_data/dark.pdb')

print(f"\nTotal alternative conformation groups found: {len(m.altloc_pairs)}")
print(f"Type of altloc_pairs: {type(m.altloc_pairs)}")

# Show details of first few groups
for i, pair in enumerate(m.altloc_pairs[:5]):
    print(f"\n--- Group {i+1} ---")
    print(f"Type: {type(pair)}")
    print(f"Number of conformations: {len(pair)}")
    
    for j, conf_tensor in enumerate(pair):
        print(f"  Conformation {j+1}:")
        print(f"    Type: {type(conf_tensor)}")
        print(f"    Shape: {conf_tensor.shape}")
        print(f"    Dtype: {conf_tensor.dtype}")
        print(f"    Number of atoms: {len(conf_tensor)}")
        print(f"    First 5 indices: {conf_tensor[:5].tolist()}")
        
        # Get actual atoms from PDB
        indices = conf_tensor.tolist()
        atoms = m.pdb.loc[indices]
        altloc = atoms['altloc'].iloc[0]
        resname = atoms['resname'].iloc[0]
        resseq = atoms['resseq'].iloc[0]
        chainid = atoms['chainid'].iloc[0]
        atom_names = atoms['name'].tolist()
        
        print(f"    Residue: {resname}-{resseq} Chain {chainid}")
        print(f"    Altloc: {altloc}")
        print(f"    Atom names: {atom_names[:5]}...")

# Test 2: Verify all groups have correct structure
print("\n" + "="*80)
print("Verification Tests")
print("="*80)

all_pass = True

# Test: All items are tuples
for i, pair in enumerate(m.altloc_pairs):
    if not isinstance(pair, tuple):
        print(f"✗ Group {i} is not a tuple!")
        all_pass = False

# Test: All conformations are tensors
for i, pair in enumerate(m.altloc_pairs):
    for j, conf in enumerate(pair):
        if not isinstance(conf, torch.Tensor):
            print(f"✗ Group {i}, conf {j} is not a tensor!")
            all_pass = False

# Test: All conformations in a group have same length
for i, pair in enumerate(m.altloc_pairs):
    lengths = [len(conf) for conf in pair]
    if not all(l == lengths[0] for l in lengths):
        print(f"✗ Group {i} has mismatched lengths: {lengths}")
        all_pass = False

# Test: No overlapping indices within a group
for i, pair in enumerate(m.altloc_pairs):
    all_indices = []
    for conf in pair:
        all_indices.extend(conf.tolist())
    if len(all_indices) != len(set(all_indices)):
        print(f"✗ Group {i} has overlapping indices!")
        all_pass = False

# Test: All atoms in a conformation have the same altloc
for i, pair in enumerate(m.altloc_pairs[:10]):  # Check first 10
    for j, conf in enumerate(pair):
        atoms = m.pdb.loc[conf.tolist()]
        unique_altlocs = atoms['altloc'].unique()
        if len(unique_altlocs) != 1:
            print(f"✗ Group {i}, conf {j} has multiple altlocs: {unique_altlocs}")
            all_pass = False

if all_pass:
    print("\n✓ ALL VERIFICATION TESTS PASSED!")
else:
    print("\n✗ SOME TESTS FAILED!")

# Summary statistics
print("\n" + "="*80)
print("Summary Statistics")
print("="*80)
print(f"Total residues with alternative conformations: {len(m.altloc_pairs)}")

conf_counts = {}
for pair in m.altloc_pairs:
    n_conf = len(pair)
    conf_counts[n_conf] = conf_counts.get(n_conf, 0) + 1

for n_conf, count in sorted(conf_counts.items()):
    print(f"  Residues with {n_conf} conformations: {count}")

print("\n✓ Test complete!")
