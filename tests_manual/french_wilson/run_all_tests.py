#!/das/work/p17/p17490/CONDA/muticopy_refinement/bin/python
"""
Master test runner for French-Wilson module tests.

Runs all test scripts and provides a summary of results.
"""

import sys
import subprocess
import os

# Get the directory containing this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

TEST_SCRIPTS = [
    "test_centric_determination.py",
    "test_core_functions.py",
    "test_module_integration.py",
]

def run_test(test_script):
    """Run a single test script and return success status"""
    print(f"\n{'='*70}")
    print(f"Running: {test_script}")
    print(f"{'='*70}")
    
    test_path = os.path.join(SCRIPT_DIR, test_script)
    python_path = "/das/work/p17/p17490/CONDA/muticopy_refinement/bin/python"
    
    try:
        result = subprocess.run(
            [python_path, test_path],
            capture_output=False,
            text=True,
            cwd=SCRIPT_DIR
        )
        return result.returncode == 0
    except Exception as e:
        print(f"Error running {test_script}: {e}")
        return False

def main():
    """Run all tests and report results"""
    print("="*70)
    print("French-Wilson Module Test Suite")
    print("="*70)
    print(f"Running {len(TEST_SCRIPTS)} test scripts...")
    
    results = {}
    
    for script in TEST_SCRIPTS:
        success = run_test(script)
        results[script] = success
    
    # Print summary
    print("\n" + "="*70)
    print("Test Summary")
    print("="*70)
    
    for script, success in results.items():
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"{script:40s} {status}")
    
    # Overall result
    all_passed = all(results.values())
    n_passed = sum(results.values())
    n_total = len(results)
    
    print("="*70)
    if all_passed:
        print(f"All {n_total} test scripts passed! ✓")
        return 0
    else:
        print(f"{n_passed}/{n_total} test scripts passed")
        print(f"{n_total - n_passed} test script(s) failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
