import torch
import multicopy_refinement.get_scattering_factor as gsf
import multicopy_refinement.math_numpy as math_np
from torch import tensor

class Refinement:
    def __init__(self, hkl_df, model,Fobs_key='Fobs'):
        self.hkl_df = hkl_df
        self.hkl = torch.tensor(self.hkl_df.reset_index().loc[:, ['h', 'k', 'l']].values)
        self.fobs = torch.tensor(self.hkl_df.reset_index().loc[:, [Fobs_key]].values)
        self.model = model
        self.cell = model.cell
        self.space_group = model.space_group
        s = math_np.get_s(self.hkl, self.cell)
        self.unique_structure_factors = {key: tensor(value) for key,value in gsf.get_scattering_factors_unique(model.pdb,s).items()}
        self.s = tensor(s)

    
    def get_structure_factor_for_residue(self, residue):
        xyz = residue.get_xyz()
        occupancy = residue.get_occupancy()
        if residue.aniso_flag:
            b = residue.get_b()



