#!/das/work/p17/p17490/CONDA/muticopy_refinement/bin/python
"""
Master script to run all edge case tests for CIF readers
"""

import sys
import subprocess
from pathlib import Path

def run_test(script_name, description):
    """Run a single test script"""
    print("=" * 80)
    print(f"RUNNING: {description}")
    print("=" * 80)
    print()
    
    script_path = Path(__file__).parent / script_name
    
    try:
        result = subprocess.run(
            ['/das/work/p17/p17490/CONDA/muticopy_refinement/bin/python', str(script_path)],
            cwd=script_path.parent,
            capture_output=False,
            text=True
        )
        return result.returncode == 0
    except Exception as e:
        print(f"Error running {script_name}: {e}")
        return False

def main():
    print()
    print("#" * 80)
    print("#" + " " * 78 + "#")
    print("#" + " " * 20 + "CIF READER EDGE CASE TESTING" + " " * 30 + "#")
    print("#" + " " * 78 + "#")
    print("#" * 80)
    print()
    
    tests = [
        ('test_reflection_reader_edgecases.py', 'Reflection CIF Reader (Structure Factors)'),
        ('test_model_reader_edgecases.py', 'Model CIF Reader (Atomic Structures)'),
        ('test_restraint_reader_edgecases.py', 'Restraint CIF Reader (Monomer Library)'),
    ]
    
    results = {}
    
    for script, description in tests:
        success = run_test(script, description)
        results[description] = success
        print()
        print()
    
    # Final summary
    print("#" * 80)
    print("#" + " " * 78 + "#")
    print("#" + " " * 30 + "FINAL SUMMARY" + " " * 35 + "#")
    print("#" + " " * 78 + "#")
    print("#" * 80)
    print()
    
    for description, success in results.items():
        status = "✓ COMPLETED" if success else "✗ FAILED"
        print(f"{status}: {description}")
    
    print()
    print("Results saved in:")
    output_dir = Path(__file__).parent
    print(f"  - {output_dir / 'reflection_reader_edgecase_results.txt'}")
    print(f"  - {output_dir / 'model_reader_edgecase_results.txt'}")
    print(f"  - {output_dir / 'restraint_reader_edgecase_results.txt'}")
    print()

if __name__ == "__main__":
    main()
