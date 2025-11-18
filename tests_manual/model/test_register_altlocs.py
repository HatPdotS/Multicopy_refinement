"""
Test suite for the register_alternative_conformations method.

This tests the identification and registration of alternative conformations
at the residue level.
"""
import sys
import os
import torch
import pandas as pd
import tempfile

# Add parent directory to path for imports
sys.path.insert(0, '/das/work/p17/p17490/Peter/Library/multicopy_refinement/multicopy_refinement')
sys.path.insert(0, '/das/work/p17/p17490/Peter/Library/multicopy_refinement')

from model_new import model
import pdb_tools


def test_load_real_pdb_with_altlocs():
    """Test loading a real PDB file with alternative conformations."""
    print("\n" + "="*80)
    print("TEST: Load real PDB file with alternative conformations")
    print("="*80)
    
    m = model()
    m.load_pdb_from_file('/das/work/p17/p17490/Peter/Library/multicopy_refinement/test_data/dark.pdb')
    
    print(f"Total alternative conformation groups: {len(m.altloc_pairs)}")
    assert len(m.altloc_pairs) > 0, "Should have found alternative conformations"
    
    # Check structure of first few groups
    for i, pair in enumerate(m.altloc_pairs[:3]):
        print(f"\nGroup {i+1}:")
        print(f"  Number of conformations: {len(pair)}")
        for j, conf_tensor in enumerate(pair):
            print(f"    Conformation {j+1}: {len(conf_tensor)} atoms, indices={conf_tensor[:5].tolist()}...")
            assert isinstance(conf_tensor, torch.Tensor), "Each conformation should be a tensor"
            assert conf_tensor.dtype == torch.long, "Indices should be long integers"
            assert len(conf_tensor) > 0, "Each conformation should have atoms"
    
    print("\n✓ Test passed: Real PDB loaded successfully")
    return m


def test_pair_structure():
    """Test that alternative conformation pairs have correct structure."""
    print("\n" + "="*80)
    print("TEST: Verify structure of altloc_pairs")
    print("="*80)
    
    m = model()
    m.load_pdb_from_file('/das/work/p17/p17490/Peter/Library/multicopy_refinement/test_data/dark.pdb')
    
    for i, pair in enumerate(m.altloc_pairs):
        # Check it's a tuple
        assert isinstance(pair, tuple), f"Group {i} should be a tuple"
        
        # Check we have at least 2 conformations
        assert len(pair) >= 2, f"Group {i} should have at least 2 conformations"
        
        # Check all elements are tensors
        for j, conf in enumerate(pair):
            assert isinstance(conf, torch.Tensor), f"Group {i}, conf {j} should be a tensor"
            assert conf.dtype == torch.long, f"Group {i}, conf {j} should be long tensor"
            assert conf.dim() == 1, f"Group {i}, conf {j} should be 1D tensor"
    
    print(f"✓ All {len(m.altloc_pairs)} groups have correct structure")


def test_conformations_same_length():
    """Test that all conformations in a group have the same number of atoms."""
    print("\n" + "="*80)
    print("TEST: Conformations in same group should have same number of atoms")
    print("="*80)
    
    m = model()
    m.load_pdb_from_file('/das/work/p17/p17490/Peter/Library/multicopy_refinement/test_data/dark.pdb')
    
    for i, pair in enumerate(m.altloc_pairs):
        lengths = [len(conf) for conf in pair]
        assert all(l == lengths[0] for l in lengths), \
            f"Group {i}: All conformations should have same number of atoms, got {lengths}"
        print(f"Group {i+1}: {len(pair)} conformations, each with {lengths[0]} atoms ✓")
    
    print(f"\n✓ All groups have matching atom counts")


def test_indices_are_valid():
    """Test that all indices are valid and within bounds."""
    print("\n" + "="*80)
    print("TEST: All indices should be valid")
    print("="*80)
    
    m = model()
    m.load_pdb_from_file('/das/work/p17/p17490/Peter/Library/multicopy_refinement/test_data/dark.pdb')
    
    max_index = len(m.pdb) - 1
    
    for i, pair in enumerate(m.altloc_pairs):
        for j, conf in enumerate(pair):
            # Check all indices are within bounds
            assert conf.min() >= 0, f"Group {i}, conf {j}: indices should be >= 0"
            assert conf.max() <= max_index, f"Group {i}, conf {j}: indices should be <= {max_index}"
            
            # Check indices are unique within conformation
            unique_indices = torch.unique(conf)
            assert len(unique_indices) == len(conf), \
                f"Group {i}, conf {j}: indices should be unique"
    
    print(f"✓ All indices are valid and within bounds [0, {max_index}]")


def test_atoms_have_correct_properties():
    """Test that atoms in alternative conformations have correct altloc values."""
    print("\n" + "="*80)
    print("TEST: Atoms should have correct altloc values")
    print("="*80)
    
    m = model()
    m.load_pdb_from_file('/das/work/p17/p17490/Peter/Library/multicopy_refinement/test_data/dark.pdb')
    
    for i, pair in enumerate(m.altloc_pairs[:3]):  # Check first 3 groups
        print(f"\nGroup {i+1}:")
        for j, conf in enumerate(pair):
            # Get atoms for this conformation
            indices = conf.tolist()
            atoms = m.pdb.loc[indices]
            
            # Check all have same altloc
            altlocs = atoms['altloc'].unique()
            assert len(altlocs) == 1, f"Group {i}, conf {j}: should have single altloc"
            
            # Check all have same residue identifier
            resnames = atoms['resname'].unique()
            resseqs = atoms['resseq'].unique()
            chainids = atoms['chainid'].unique()
            
            assert len(resnames) == 1, f"Group {i}, conf {j}: should have single resname"
            assert len(resseqs) == 1, f"Group {i}, conf {j}: should have single resseq"
            assert len(chainids) == 1, f"Group {i}, conf {j}: should have single chainid"
            
            print(f"  Conf {j+1}: {resnames[0]}-{resseqs[0]} Chain {chainids[0]}, "
                  f"altloc={altlocs[0]}, {len(atoms)} atoms ✓")
    
    print("\n✓ All atoms have correct properties")


def test_no_overlap_between_conformations():
    """Test that different conformations don't share any atom indices."""
    print("\n" + "="*80)
    print("TEST: Conformations should not share atom indices")
    print("="*80)
    
    m = model()
    m.load_pdb_from_file('/das/work/p17/p17490/Peter/Library/multicopy_refinement/test_data/dark.pdb')
    
    for i, pair in enumerate(m.altloc_pairs):
        # Combine all indices from all conformations
        all_indices = []
        for conf in pair:
            all_indices.extend(conf.tolist())
        
        # Check no duplicates
        assert len(all_indices) == len(set(all_indices)), \
            f"Group {i}: Conformations should not share atom indices"
    
    print(f"✓ No overlapping indices in {len(m.altloc_pairs)} groups")


def test_matching_atom_names():
    """Test that corresponding atoms in different conformations have matching names."""
    print("\n" + "="*80)
    print("TEST: Corresponding atoms should have matching names")
    print("="*80)
    
    m = model()
    m.load_pdb_from_file('/das/work/p17/p17490/Peter/Library/multicopy_refinement/test_data/dark.pdb')
    
    for i, pair in enumerate(m.altloc_pairs[:5]):  # Check first 5 groups
        # Get atom names for each conformation
        atom_names_list = []
        for conf in pair:
            indices = conf.tolist()
            atoms = m.pdb.loc[indices]
            # Sort by atom name to ensure consistent ordering
            atom_names = sorted(atoms['name'].tolist())
            atom_names_list.append(atom_names)
        
        # Check all conformations have same atom names
        for j in range(1, len(atom_names_list)):
            assert atom_names_list[0] == atom_names_list[j], \
                f"Group {i}: All conformations should have matching atom names"
        
        print(f"Group {i+1}: {len(pair)} conformations with matching atoms ({len(atom_names_list[0])} atoms) ✓")
    
    print("\n✓ All conformations have matching atom names")


def test_empty_pdb():
    """Test handling of PDB with no alternative conformations."""
    print("\n" + "="*80)
    print("TEST: Handle PDB with no alternative conformations")
    print("="*80)
    
    # Create a simple PDB with no altlocs
    pdb_content = """CRYST1   50.000   50.000   50.000  90.00  90.00  90.00 P 1
ATOM      1  N   ALA A   1      10.000  10.000  10.000  1.00 20.00           N
ATOM      2  CA  ALA A   1      11.000  11.000  11.000  1.00 20.00           C
ATOM      3  C   ALA A   1      12.000  12.000  12.000  1.00 20.00           C
ATOM      4  O   ALA A   1      13.000  13.000  13.000  1.00 20.00           O
END
"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as f:
        f.write(pdb_content)
        temp_file = f.name
    
    try:
        m = model()
        m.load_pdb_from_file(temp_file)
        
        assert len(m.altloc_pairs) == 0, "Should have no alternative conformations"
        print("✓ Empty altloc_pairs for PDB with no alternative conformations")
    finally:
        os.unlink(temp_file)


def test_create_synthetic_altloc_pdb():
    """Test with a synthetic PDB containing known alternative conformations."""
    print("\n" + "="*80)
    print("TEST: Synthetic PDB with known alternative conformations")
    print("="*80)
    
    # Create a PDB with 1 residue having 2 conformations (A and B)
    pdb_content = """CRYST1   50.000   50.000   50.000  90.00  90.00  90.00 P 1
ATOM      1  N  AALA A   1      10.000  10.000  10.000  0.50 20.00           N
ATOM      2  CA AALA A   1      11.000  11.000  11.000  0.50 20.00           C
ATOM      3  C  AALA A   1      12.000  12.000  12.000  0.50 20.00           C
ATOM      4  N  BALA A   1      10.100  10.100  10.100  0.50 20.00           N
ATOM      5  CA BALA A   1      11.100  11.100  11.100  0.50 20.00           C
ATOM      6  C  BALA A   1      12.100  12.100  12.100  0.50 20.00           C
END
"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as f:
        f.write(pdb_content)
        temp_file = f.name
    
    try:
        m = model()
        m.load_pdb_from_file(temp_file)
        
        print(f"Number of altloc groups: {len(m.altloc_pairs)}")
        assert len(m.altloc_pairs) == 1, "Should have exactly 1 residue with altlocs"
        
        pair = m.altloc_pairs[0]
        assert len(pair) == 2, "Should have 2 conformations (A and B)"
        
        # Check conformation A
        conf_a = pair[0]
        assert len(conf_a) == 3, "Conformation A should have 3 atoms"
        atoms_a = m.pdb.loc[conf_a.tolist()]
        assert all(atoms_a['altloc'] == 'A'), "All atoms should have altloc A"
        print(f"Conformation A: indices={conf_a.tolist()}, atoms={atoms_a['name'].tolist()}")
        
        # Check conformation B
        conf_b = pair[1]
        assert len(conf_b) == 3, "Conformation B should have 3 atoms"
        atoms_b = m.pdb.loc[conf_b.tolist()]
        assert all(atoms_b['altloc'] == 'B'), "All atoms should have altloc B"
        print(f"Conformation B: indices={conf_b.tolist()}, atoms={atoms_b['name'].tolist()}")
        
        print("✓ Synthetic PDB test passed")
    finally:
        os.unlink(temp_file)


def test_triplet_conformations():
    """Test residues with 3 alternative conformations."""
    print("\n" + "="*80)
    print("TEST: Residues with 3 conformations (A, B, C)")
    print("="*80)
    
    m = model()
    m.load_pdb_from_file('/das/work/p17/p17490/Peter/Library/multicopy_refinement/test_data/dark.pdb')
    
    # Find groups with 3 conformations
    triplets = [pair for pair in m.altloc_pairs if len(pair) == 3]
    
    print(f"Found {len(triplets)} residues with 3 conformations")
    
    if len(triplets) > 0:
        for i, triplet in enumerate(triplets[:2]):  # Show first 2
            print(f"\nTriplet {i+1}:")
            for j, conf in enumerate(triplet):
                indices = conf.tolist()
                atoms = m.pdb.loc[indices]
                altloc = atoms['altloc'].iloc[0]
                resname = atoms['resname'].iloc[0]
                resseq = atoms['resseq'].iloc[0]
                chainid = atoms['chainid'].iloc[0]
                print(f"  Conf {altloc}: {resname}-{resseq} Chain {chainid}, "
                      f"{len(atoms)} atoms, indices={indices[:3]}...")
        
        print(f"\n✓ Found and verified {len(triplets)} triplet conformations")
    else:
        print("Note: No triplet conformations found in this PDB")


def run_all_tests():
    """Run all tests."""
    print("\n" + "="*80)
    print("RUNNING ALL TESTS FOR register_alternative_conformations")
    print("="*80)
    
    tests = [
        test_load_real_pdb_with_altlocs,
        test_pair_structure,
        test_conformations_same_length,
        test_indices_are_valid,
        test_atoms_have_correct_properties,
        test_no_overlap_between_conformations,
        test_matching_atom_names,
        test_empty_pdb,
        test_create_synthetic_altloc_pdb,
        test_triplet_conformations,
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            test_func()
            passed += 1
        except AssertionError as e:
            print(f"\n✗ TEST FAILED: {test_func.__name__}")
            print(f"  Error: {e}")
            failed += 1
        except Exception as e:
            print(f"\n✗ TEST ERROR: {test_func.__name__}")
            print(f"  Error: {e}")
            failed += 1
    
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    print(f"Passed: {passed}/{len(tests)}")
    print(f"Failed: {failed}/{len(tests)}")
    
    if failed == 0:
        print("\n✓ ALL TESTS PASSED! ✓")
    else:
        print(f"\n✗ {failed} TEST(S) FAILED")
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
