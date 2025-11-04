#!/das/work/p17/p17490/CONDA/muticopy_refinement/bin/python

#SBATCH -c 32
#SBATCH -o /das/work/p17/p17490/Peter/Library/multicopy_refinement/test_data/test_P21.out

from multicopy_refinement.Model import model
import multicopy_refinement.math_numpy as mnp
import torch
import numpy as np
from multicopy_refinement.math_torch import add_atom_to_map_isotropic_periodic
import gemmi
import multicopy_refinement.get_scattering_factor_torch as gsf
import multicopy_refinement.Data as Data
import multicopy_refinement.refinement as refinement

M = model()
M.load_pdb_from_file('/das/work/p17/p17490/Peter/Library/multicopy_refinement/test_data/tubulin/dark.pdb')
M.use_structure_factor_fast = False

hkl = Data.read_mtz('/das/work/p17/p17490/Peter/Library/multicopy_refinement/test_data/tubulin/dark.mtz')

ref = refinement.Refinement(hkl,model=M,use_parametrization=True)
with torch.no_grad():   
    f = ref.get_structure_factor()


print(f.shape)
