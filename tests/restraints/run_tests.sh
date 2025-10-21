#!/bin/bash
#
# Test runner script for restraints module
#
# This script runs all tests for the restraints module.
# Make sure you have the appropriate Python environment activated
# with torch, numpy, pandas, and other dependencies installed.
#

echo "================================================================================"
echo "Running Restraints Module Tests"
echo "================================================================================"
echo ""

# Set the base directory
TEST_DIR="/das/work/p17/p17490/Peter/Library/multicopy_refinement/tests/restraints"

# Function to run a test
run_test() {
    local test_file=$1
    echo "--------------------------------------------------------------------------------"
    echo "Running: $test_file"
    echo "--------------------------------------------------------------------------------"
    python3 "$TEST_DIR/$test_file"
    local exit_code=$?
    if [ $exit_code -eq 0 ]; then
        echo "✓ Test passed"
    else
        echo "✗ Test failed with exit code $exit_code"
        return $exit_code
    fi
    echo ""
}

# Change to the test directory
cd "$TEST_DIR" || exit 1

# Run demo first
echo "Running demo..."
run_test "demo_restraints.py" || exit 1

# Run all tests
echo "Running all tests..."
run_test "test_restraints_creation.py" || exit 1
run_test "test_bond_lengths.py" || exit 1
run_test "test_angles.py" || exit 1
run_test "test_torsions.py" || exit 1

echo "================================================================================"
echo "All tests completed successfully! ✓"
echo "================================================================================"
