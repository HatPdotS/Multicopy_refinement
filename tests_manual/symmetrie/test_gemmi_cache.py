#!/usr/bin/env python3
"""
Test script to verify the cached gemmi symmetry operations.
"""

import sys
import os
sys.path.insert(0, '/das/work/p17/p17490/Peter/Library/multicopy_refinement')

import torch
import gemmi

# Load cached operations
cache_file = '/das/work/p17/p17490/Peter/Library/multicopy_refinement/caching/files/gemmi_symmetry_operations.pt'
cached_ops = torch.load(cache_file)

print("="*60)
print("GEMMI CACHE VALIDATION")
print("="*60)
print(f"\n✅ Loaded {len(cached_ops)} space groups from cache")
print(f"📁 Cache file: gemmi_symmetry_operations.pt")
print(f"💾 File size: {os.path.getsize(cache_file) / 1024:.1f} KB\n")

# Test a few representative space groups
test_space_groups = [
    'P1', 'P-1', 'P21', 'C2', 'P222', 'P212121',
    'P4', 'P41', 'I4', 'P3', 'P31', 'P6', 'P61',
    'P23', 'F23', 'I23', 'Pm-3', 'Fm-3m', 'Ia-3d'
]

print("Testing cached operations against gemmi...")
print("-"*60)

all_match = True
for sg_name in test_space_groups:
    if sg_name not in cached_ops:
        print(f"❌ {sg_name}: Not in cache!")
        all_match = False
        continue
    
    # Get cached operations
    cached_rot, cached_tran = cached_ops[sg_name]
    
    # Get fresh gemmi operations
    try:
        gemmi_sg = gemmi.SpaceGroup(sg_name)
        gemmi_ops = [(torch.tensor(i.rot, dtype=torch.float64)/24, 
                      torch.tensor(i.tran, dtype=torch.float64)/24) 
                     for i in gemmi_sg.operations()]
        gemmi_rot = torch.stack([op[0] for op in gemmi_ops])
        gemmi_tran = torch.stack([op[1] for op in gemmi_ops])
    except Exception as e:
        print(f"⚠️  {sg_name}: Gemmi error: {e}")
        continue
    
    # Check if they match
    rot_match = torch.allclose(cached_rot.double(), gemmi_rot, atol=1e-10)
    tran_match = torch.allclose(cached_tran.double(), gemmi_tran, atol=1e-10)
    
    if rot_match and tran_match:
        print(f"✅ {sg_name:12s}: {len(cached_rot):3d} operations - PERFECT MATCH")
    else:
        print(f"❌ {sg_name:12s}: {len(cached_rot):3d} operations - MISMATCH!")
        all_match = False

print("\n" + "="*60)
print("SUMMARY")
print("="*60)

if all_match:
    print("✅ All tested space groups match gemmi perfectly!")
    print(f"✅ Cache is valid and ready to use")
else:
    print("❌ Some space groups don't match - cache may need regeneration")

print("\n" + "="*60)
print("USAGE EXAMPLE")
print("="*60)
print("""
# Load cached operations in your code:
import torch
cache_file = 'caching/files/gemmi_symmetry_operations.pt'
cached_ops = torch.load(cache_file)

# Get operations for a space group:
rot_matrices, translations = cached_ops['P212121']

# Use the operations:
print(f"P212121 has {len(rot_matrices)} symmetry operations")
for i, (rot, trans) in enumerate(zip(rot_matrices, translations)):
    print(f"Operation {i+1}:")
    print(f"  Rotation:\\n{rot}")
    print(f"  Translation: {trans}")
""")
print("="*60)
