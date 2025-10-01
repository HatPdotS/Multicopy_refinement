from multicopy_refinement.Model import model
import multicopy_refinement.math_numpy as mnp
import torch
import numpy as np
from multicopy_refinement.math_torch import add_atom_to_map_isotropic_periodic
import gemmi
import multicopy_refinement.get_scattering_factor_torch as gsf

class ModelFT(model):
    """
    ModelFT is a subclass of Model that implements the Fourier Transform (FT) based structure factor calculation method for structure refinement.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def load_pdb_from_file(self, filename):
        """
        Load a PDB file and initialize the model.
        """
        super().load_pdb_from_file(filename)
        self.setup_grids()
        self.parametrization = gsf.get_parameterization(self.pdb)
    
    def setup_grids(self,max_res=0.5):
        self.max_res = max_res
        self.real_space_grid = mnp.get_real_grid(self.cell,self.max_res)
        self.real_space_grid = torch.tensor(self.real_space_grid.astype(np.float64))
        self.map = torch.tensor(np.zeros(self.real_space_grid.shape[:-1],dtype=np.float64))
        self.voxel_size = self.real_space_grid[1,1,1] - self.real_space_grid[0,0,0]
        self.inv_frac_matrix = torch.tensor(mnp.get_inv_fractional_matrix(self.cell))
        self.frac_matrix = torch.tensor(mnp.get_fractional_matrix(self.cell))

    
    def build_atom_full(self, residue):
        xyz = residue.get_xyz()
        b = residue.get_b()
        elements = residue.get_atoms()
        A = torch.stack([self.parametrization[element][0] for element in elements], dim=0)
        B = torch.stack([self.parametrization[element][1] for element in elements], dim=0)
        C = torch.stack([self.parametrization[element][2] for element in elements], dim=0)
        self.map = add_atom_to_map_isotropic_periodic(self.map, xyz, b, self.real_space_grid, self.voxel_size,self.inv_frac_matrix, self.frac_matrix,A,B,C)

    def save_map(self, filename):
        """
        Save the map to a file.
        """
        np_map = self.map.detach().cpu().numpy().astype(np.float32)
        # np_map = np.asfortranarray(np_map)  # Ensure Fortran order for gemmi
        print("Map shape before saving:", self.map.shape, "sum:", self.map.sum())
        map_ccp = gemmi.Ccp4Map()
        
        map_ccp.grid = gemmi.FloatGrid(np_map,gemmi.UnitCell(*self.cell),gemmi.SpaceGroup(self.spacegroup))
        map_ccp.setup(0.0)
        map_ccp.update_ccp4_header()
        map_ccp.write_ccp4_map(filename)