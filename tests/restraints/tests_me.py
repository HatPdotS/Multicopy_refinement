import torch
import numpy as np
from multicopy_refinement.model import Model
from multicopy_refinement.restraints_new import Restraints


model = Model()
test_pdb = '/das/work/p17/p17490/Peter/Library/multicopy_refinement/test_data/dark.pdb'
model.load_pdb_from_file(test_pdb)

cif_path = '/das/work/p17/p17490/Peter/Library/multicopy_refinement/test_data/elbow.AZO.dark_pdb.001.cif'
restraints = Restraints(model, cif_path)

bond_indices = restraints.bond_indices

pdb = model.pdb

xyz = model.xyz()

bondlengths = torch.linalg.norm(
    xyz[bond_indices[:, 0], :] - xyz[bond_indices[:, 1], :],
    axis=1
)

deviations = (bondlengths - restraints.bond_references) / restraints.bond_sigmas