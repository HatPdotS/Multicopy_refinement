"""
Test suite for the enforce_alternative_conformations method.

This tests the enforcement of occupancy constraints:
1. All atoms in a conformation have the same occupancy (mean occupancy)
2. Occupancies across conformations sum to 1.0
"""
import sys
import os
import torch
import pandas as pd
import tempfile

sys.path.insert(0, '/das/work/p17/p17490/Peter/Library/multicopy_refinement/multicopy_refinement')

import importlib.util

# Load model_new directly
spec = importlib.util.spec_from_file_location(
    "model_new", 
    "/das/work/p17/p17490/Peter/Library/multicopy_refinement/multicopy_refinement/model_new.py"
)
model_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model_module)
model = model_module.model


def test_basic_enforcement():
    """Test basic occupancy enforcement on real data."""
    print("\n" + "="*80)
    print("TEST 1: Basic occupancy enforcement")
    print("="*80)
    
    m = model()
    m.load_pdb_from_file('/das/work/p17/p17490/Peter/Library/multicopy_refinement/test_data/dark.pdb')
    
    print(f"Loaded {len(m.altloc_pairs)} residues with alternative conformations")
    
    # Get occupancies before enforcement
    occ_before = m.occupancy().detach().clone()
    
    # Show some examples before
    print("\nBefore enforcement (first 3 groups):")
    for i, group in enumerate(m.altloc_pairs[:3]):
        print(f"\nGroup {i+1}:")
        for j, conf_indices in enumerate(group):
            conf_occ = occ_before[conf_indices]
            print(f"  Conf {j+1}: occupancies = {conf_occ.tolist()}")
            print(f"          mean = {conf_occ.mean().item():.4f}, std = {conf_occ.std().item():.4f}")
    
    # Enforce constraints
    m.enforce_alternative_conformations()
    
    # Get occupancies after enforcement
    occ_after = m.occupancy().detach()
    
    # Show same examples after
    print("\nAfter enforcement (first 3 groups):")
    for i, group in enumerate(m.altloc_pairs[:3]):
        print(f"\nGroup {i+1}:")
        for j, conf_indices in enumerate(group):
            conf_occ = occ_after[conf_indices]
            print(f"  Conf {j+1}: occupancies = {conf_occ.tolist()}")
            print(f"          mean = {conf_occ.mean().item():.4f}, std = {conf_occ.std().item():.4f}")
        
        # Check sum equals 1
        sum_occupancies = sum(occ_after[conf].mean() for conf in group)
        print(f"  Sum of mean occupancies: {sum_occupancies.item():.6f}")
    
    print("\n✓ Test completed")


def test_uniform_occupancy_within_conformation():
    """Test that all atoms in a conformation have the same occupancy."""
    print("\n" + "="*80)
    print("TEST 2: Uniform occupancy within each conformation")
    print("="*80)
    
    m = model()
    m.load_pdb_from_file('/das/work/p17/p17490/Peter/Library/multicopy_refinement/test_data/dark.pdb')
    
    # Enforce constraints
    m.enforce_alternative_conformations()
    
    occ = m.occupancy().detach()
    
    all_uniform = True
    for i, group in enumerate(m.altloc_pairs):
        for j, conf_indices in enumerate(group):
            conf_occ = occ[conf_indices]
            # Check if all occupancies are the same (within tolerance)
            std = conf_occ.std().item()
            if std > 1e-6:
                print(f"✗ Group {i}, Conf {j}: occupancies not uniform (std={std:.6f})")
                all_uniform = False
    
    if all_uniform:
        print(f"✓ All {len(m.altloc_pairs)} groups have uniform occupancies within each conformation")
    
    assert all_uniform, "All conformations should have uniform occupancies"


def test_occupancies_sum_to_one():
    """Test that occupancies across conformations sum to 1.0."""
    print("\n" + "="*80)
    print("TEST 3: Occupancies sum to 1.0 across conformations")
    print("="*80)
    
    m = model()
    m.load_pdb_from_file('/das/work/p17/p17490/Peter/Library/multicopy_refinement/test_data/dark.pdb')
    
    # Enforce constraints
    m.enforce_alternative_conformations()
    
    occ = m.occupancy().detach()
    
    all_sum_to_one = True
    tolerance = 1e-5
    
    for i, group in enumerate(m.altloc_pairs):
        # Get mean occupancy for each conformation
        mean_occupancies = [occ[conf].mean() for conf in group]
        total = sum(mean_occupancies).item()
        
        if abs(total - 1.0) > tolerance:
            print(f"✗ Group {i}: sum = {total:.6f} (not 1.0)")
            all_sum_to_one = False
    
    if all_sum_to_one:
        print(f"✓ All {len(m.altloc_pairs)} groups sum to 1.0 (within tolerance {tolerance})")
    
    assert all_sum_to_one, "All groups should sum to 1.0"


def test_no_effect_on_non_altloc_atoms():
    """Test that atoms without altlocs are not affected."""
    print("\n" + "="*80)
    print("TEST 4: Non-altloc atoms unchanged")
    print("="*80)
    
    m = model()
    m.load_pdb_from_file('/das/work/p17/p17490/Peter/Library/multicopy_refinement/test_data/dark.pdb')
    
    # Get all altloc atom indices
    altloc_indices = set()
    for group in m.altloc_pairs:
        for conf in group:
            altloc_indices.update(conf.tolist())
    
    # Get non-altloc indices
    all_indices = set(range(len(m.pdb)))
    non_altloc_indices = all_indices - altloc_indices
    non_altloc_tensor = torch.tensor(sorted(non_altloc_indices), dtype=torch.long)
    
    print(f"Non-altloc atoms: {len(non_altloc_indices)}")
    print(f"Altloc atoms: {len(altloc_indices)}")
    
    # Store occupancies before
    occ_before = m.occupancy().detach().clone()
    
    # Enforce constraints
    m.enforce_alternative_conformations()
    
    # Get occupancies after
    occ_after = m.occupancy().detach()
    
    # Check non-altloc atoms are unchanged
    if len(non_altloc_indices) > 0:
        diff = (occ_after[non_altloc_tensor] - occ_before[non_altloc_tensor]).abs().max().item()
        print(f"Max difference in non-altloc atoms: {diff:.10f}")
        
        assert diff < 1e-6, "Non-altloc atoms should be unchanged"
        print("✓ Non-altloc atoms unchanged")
    else:
        print("Note: All atoms have altlocs in this structure")


def test_synthetic_case():
    """Test with synthetic data with known values."""
    print("\n" + "="*80)
    print("TEST 5: Synthetic test case")
    print("="*80)
    
    # Create a synthetic PDB with controlled occupancies
    pdb_content = """CRYST1   50.000   50.000   50.000  90.00  90.00  90.00 P 1
ATOM      1  N  AALA A   1      10.000  10.000  10.000  0.60 20.00           N
ATOM      2  CA AALA A   1      11.000  11.000  11.000  0.65 20.00           C
ATOM      3  C  AALA A   1      12.000  12.000  12.000  0.55 20.00           C
ATOM      4  N  BALA A   1      10.100  10.100  10.100  0.40 20.00           N
ATOM      5  CA BALA A   1      11.100  11.100  11.100  0.35 20.00           C
ATOM      6  C  BALA A   1      12.100  12.100  12.100  0.45 20.00           C
END
"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as f:
        f.write(pdb_content)
        temp_file = f.name
    
    try:
        m = model()
        m.load_pdb_from_file(temp_file)
        
        print(f"Loaded {len(m.altloc_pairs)} residue with alternative conformations")
        
        # Before enforcement
        occ_before = m.occupancy().detach().clone()
        print("\nBefore enforcement:")
        group = m.altloc_pairs[0]
        for j, conf_indices in enumerate(group):
            conf_occ = occ_before[conf_indices]
            print(f"  Conf {j+1}: {conf_occ.tolist()}, mean={conf_occ.mean().item():.4f}")
        
        # Expected: Conf A mean = (0.6+0.65+0.55)/3 = 0.6
        #           Conf B mean = (0.4+0.35+0.45)/3 = 0.4
        #           After normalization: A=0.6, B=0.4 (already sum to 1.0)
        
        # Enforce
        m.enforce_alternative_conformations()
        
        occ_after = m.occupancy().detach()
        print("\nAfter enforcement:")
        for j, conf_indices in enumerate(group):
            conf_occ = occ_after[conf_indices]
            print(f"  Conf {j+1}: {conf_occ.tolist()}, mean={conf_occ.mean().item():.4f}")
        
        # Verify
        conf_a_occ = occ_after[group[0]]
        conf_b_occ = occ_after[group[1]]
        
        # All should be the same within conformation
        assert conf_a_occ.std() < 1e-6, "Conf A should have uniform occupancy"
        assert conf_b_occ.std() < 1e-6, "Conf B should have uniform occupancy"
        
        # Should sum to 1.0
        total = conf_a_occ.mean() + conf_b_occ.mean()
        assert abs(total.item() - 1.0) < 1e-5, f"Should sum to 1.0, got {total.item()}"
        
        print(f"\n✓ Conf A: {conf_a_occ[0].item():.4f}, Conf B: {conf_b_occ[0].item():.4f}, Sum: {total.item():.4f}")
        print("✓ Synthetic test passed")
        
    finally:
        os.unlink(temp_file)


def test_triplet_conformations():
    """Test enforcement with 3 conformations."""
    print("\n" + "="*80)
    print("TEST 6: Triplet conformations (A, B, C)")
    print("="*80)
    
    m = model()
    m.load_pdb_from_file('/das/work/p17/p17490/Peter/Library/multicopy_refinement/test_data/dark.pdb')
    
    # Find triplets
    triplets = [g for g in m.altloc_pairs if len(g) == 3]
    print(f"Found {len(triplets)} residues with 3 conformations")
    
    if len(triplets) == 0:
        print("Skipping test - no triplets found")
        return
    
    # Enforce
    m.enforce_alternative_conformations()
    
    occ = m.occupancy().detach()
    
    for i, group in enumerate(triplets):
        print(f"\nTriplet {i+1}:")
        mean_occupancies = []
        for j, conf_indices in enumerate(group):
            conf_occ = occ[conf_indices]
            mean_occ = conf_occ.mean().item()
            mean_occupancies.append(mean_occ)
            print(f"  Conf {j+1}: mean={mean_occ:.4f}, std={conf_occ.std().item():.6f}")
        
        total = sum(mean_occupancies)
        print(f"  Sum: {total:.6f}")
        
        # Check uniform within conformation
        for j, conf_indices in enumerate(group):
            assert occ[conf_indices].std() < 1e-6, f"Conf {j} should be uniform"
        
        # Check sum to 1.0
        assert abs(total - 1.0) < 1e-5, f"Should sum to 1.0, got {total}"
    
    print("\n✓ All triplets pass constraints")


def test_multiple_enforcements():
    """Test that multiple enforcements are idempotent."""
    print("\n" + "="*80)
    print("TEST 7: Multiple enforcements (idempotency)")
    print("="*80)
    
    m = model()
    m.load_pdb_from_file('/das/work/p17/p17490/Peter/Library/multicopy_refinement/test_data/dark.pdb')
    
    # Enforce once
    m.enforce_alternative_conformations()
    occ_after_first = m.occupancy().detach().clone()
    
    # Enforce again
    m.enforce_alternative_conformations()
    occ_after_second = m.occupancy().detach()
    
    # Should be identical
    max_diff = (occ_after_second - occ_after_first).abs().max().item()
    print(f"Max difference after second enforcement: {max_diff:.10f}")
    
    assert max_diff < 1e-6, "Multiple enforcements should be idempotent"
    print("✓ Multiple enforcements are idempotent")


def test_statistics():
    """Show statistics about the enforcement."""
    print("\n" + "="*80)
    print("TEST 8: Enforcement statistics")
    print("="*80)
    
    m = model()
    m.load_pdb_from_file('/das/work/p17/p17490/Peter/Library/multicopy_refinement/test_data/dark.pdb')
    
    occ_before = m.occupancy().detach().clone()
    
    # Calculate statistics before
    print("\nBefore enforcement:")
    within_conf_stds = []
    sum_deviations = []
    
    for group in m.altloc_pairs:
        for conf in group:
            within_conf_stds.append(occ_before[conf].std().item())
        
        mean_occs = [occ_before[conf].mean() for conf in group]
        total = sum(mean_occs).item()
        sum_deviations.append(abs(total - 1.0))
    
    print(f"  Mean std within conformations: {sum(within_conf_stds)/len(within_conf_stds):.6f}")
    print(f"  Max std within conformations: {max(within_conf_stds):.6f}")
    print(f"  Mean deviation from sum=1.0: {sum(sum_deviations)/len(sum_deviations):.6f}")
    print(f"  Max deviation from sum=1.0: {max(sum_deviations):.6f}")
    
    # Enforce
    m.enforce_alternative_conformations()
    occ_after = m.occupancy().detach()
    
    # Calculate statistics after
    print("\nAfter enforcement:")
    within_conf_stds = []
    sum_deviations = []
    
    for group in m.altloc_pairs:
        for conf in group:
            within_conf_stds.append(occ_after[conf].std().item())
        
        mean_occs = [occ_after[conf].mean() for conf in group]
        total = sum(mean_occs).item()
        sum_deviations.append(abs(total - 1.0))
    
    print(f"  Mean std within conformations: {sum(within_conf_stds)/len(within_conf_stds):.10f}")
    print(f"  Max std within conformations: {max(within_conf_stds):.10f}")
    print(f"  Mean deviation from sum=1.0: {sum(sum_deviations)/len(sum_deviations):.10f}")
    print(f"  Max deviation from sum=1.0: {max(sum_deviations):.10f}")
    
    print("\n✓ Statistics calculated")


def run_all_tests():
    """Run all tests."""
    print("\n" + "="*80)
    print("RUNNING ALL TESTS FOR enforce_alternative_conformations")
    print("="*80)
    
    tests = [
        test_basic_enforcement,
        test_uniform_occupancy_within_conformation,
        test_occupancies_sum_to_one,
        test_no_effect_on_non_altloc_atoms,
        test_synthetic_case,
        test_triplet_conformations,
        test_multiple_enforcements,
        test_statistics,
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
            import traceback
            traceback.print_exc()
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
