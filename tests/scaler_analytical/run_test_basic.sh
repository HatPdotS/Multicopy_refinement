#!/bin/bash
#SBATCH --job-name=test_scaler
#SBATCH --output=/das/work/p17/p17490/Peter/Library/multicopy_refinement/tests/scaler_analytical/test_basic.out
#SBATCH --error=/das/work/p17/p17490/Peter/Library/multicopy_refinement/tests/scaler_analytical/test_basic.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G

# Load environment
source /das/work/p17/p17490/CONDA/muticopy_refinement/bin/activate

# Run the test
cd /das/work/p17/p17490/Peter/Library/multicopy_refinement/tests/scaler_analytical
/das/work/p17/p17490/CONDA/muticopy_refinement/bin/python test_basic.py

echo ""
echo "Test completed at $(date)"
