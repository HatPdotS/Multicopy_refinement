from multicopy_refinement import restraints_handler
import multicopy_refinement.Model as Model
from time import time

cif_path = '/das/work/p17/p17490/Peter/Library/multicopy_refinement/test_data/Merged_restraints_all_opened.cif'

restraints = restraints_handler.restraints(cif_path)


M = Model.model()
M.load_pdb_from_file('/das/work/p17/p17490/Peter/Library/multicopy_refinement/test_data/test_all.pdb')



residues = list(M.residues.values())

losses = []

t = time()

for residue in residues:
    losses.append(restraints.get_deviations(residue))

print(time()-t)
