#!/das/work/p17/p17490/CONDA/muticopy_refinement/bin/python

#SBATCH -c 16
#SBATCH -o /das/work/p17/p17490/Peter/Library/multicopy_refinement/tests/bulk_solvent/solvent_mask_new.out

from multicopy_refinement.solvent_new import SolventModel
from multicopy_refinement.model_ft import ModelFT



pdb = '/das/work/p17/p17490/Peter/Library/multicopy_refinement/tests/refinement/dark.pdb'

model = ModelFT().load_pdb_from_file(pdb)


solvent_model = SolventModel(
    model=model)