#!/das/work/p17/p17490/CONDA/muticopy_refinement/bin/python -u
#SBATCH -c 16
#SBATCH -o /das/work/p17/p17490/Peter/Library/multicopy_refinement/tests/validation_on_different_samples/Br/shake_coords/refine_shaken_cuda_all.log

#SBATCH --gres=gpu:1
#SBATCH -p gpu-day

from multicopy_refinement.base_refinement import Refinement



mtz = '/das/work/p17/p17490/Peter/Library/multicopy_refinement/tests/validation_on_different_samples/Br/BR_LCLS_refine_8.mtz'
pdb = '/das/work/p17/p17490/Peter/Library/multicopy_refinement/tests/validation_on_different_samples/Br/shake_coords/shaken_BR_LCLS_refine_8.pdb'
cif = '/das/work/p17/p17490/Peter/Library/multicopy_refinement/tests/validation_on_different_samples/Br/LYR_open.cif'

refinement = Refinement(mtz, pdb, cif, max_res=1.5,verbose=1)

print(50*'=')

print("Moving refinement to GPU...")

print(50*'=')

refinement.cuda()

print(50*'=')

print("Starting refinement on shaken structure...")

print(50*'=')

refinement.run_refinement()


refinement.model.write_pdb('/das/work/p17/p17490/Peter/Library/multicopy_refinement/tests/validation_on_different_samples/Br/shake_coords/refined_shaken_BR_LCLS_refine_8.pdb')