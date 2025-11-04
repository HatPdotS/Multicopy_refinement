from multicopy_refinement.Model import model
import multicopy_refinement.math_numpy as mnp
import torch
import numpy as np
from multicopy_refinement.math_torch import find_relevant_voxels, vectorized_add_to_map, vectorized_add_to_map_aniso
import gemmi
import multicopy_refinement.get_scattering_factor_torch as gsf
from multicopy_refinement.Model import residue, projected_residue
from multicopy_refinement.map_symmetry import MapSymmetry
from torch import nn
from tqdm import tqdm


class ModelDS(model.Model):
    """
    Subclass of Model for direct summation refinement.
    Includes methods for building density maps from atomic models.
    """
    def __init__(self):
        super().__init__()

    def setup_extinction(self,e_start=1):
        self.extinction_parameter = nn.Parameter(torch.tensor([e_start], dtype=torch.float64, requires_grad=True))
        self.global_corrections.append(self.extinction_correction)
        self.correction_parameters.append(self.extinction_parameter)
    
    def setup_absorption(self,abs_coeffs=None):
        if abs_coeffs is None:
            abs_coeffs = torch.ones(6, dtype=torch.float64, requires_grad=True)
        if abs_coeffs is not None and not isinstance(abs_coeffs, torch.Tensor):
            abs_coeffs = torch.tensor(abs_coeffs, dtype=torch.float64, requires_grad=True)
        self.abs_coeffs = nn.Parameter(abs_coeffs)
        self.global_corrections.append(self.absorption_correction)
        self.correction_parameters.append(self.abs_coeffs)
    
    def extinction_correction(self):
        f_calc_abs = torch.abs(self.f_calc)
        extinction_factor = 1.0 / (1.0 + self.extinction_parameter * 1e-10 * f_calc_abs**2 / (self.s + 1e-10))
        return extinction_factor
    
    def apply_corrections(self):
        """Apply all global corrections to the calculated structure factors"""
        f_calc = self.f_calc
        for correction in self.global_corrections:
            f_calc = correction() * f_calc
        # Apply scale factor last
        f_calc = self.scale * f_calc
        return f_calc

    def get_structure_factor(self, hkl, scattering_vectors, s):
        """Calculate structure factors for all residues in the model and apply corrections"""
        self.s = s
        self.hkl = hkl
        self.scattering_vectors = scattering_vectors
        if self.use_structure_factor_fast:
            self.get_structur_factor_not_corrected_fast(hkl, scattering_vectors, s)
        else:
            self.get_structure_factor_not_corrected(hkl, scattering_vectors, s)
        self.f_calc = self.apply_corrections()
        return self.f_calc
    
    def get_structur_factor_not_corrected_fast(self, hkl, scattering_vectors, s):
        residues_isotropic = []
        residues_anisotropic = []
        residues_special = []
        for res in self.residues.values():
            if res.corrections or isinstance(res, projected_residue):
                residues_special.append(res)
            if res.anisou_flag:
                residues_anisotropic.append(res)
            else:
                residues_isotropic.append(res)
        f_start = [res.get_structure_factor(hkl, scattering_vectors, s) for res in residues_special]
        if residues_isotropic:
            if self._cached_iso_xyz is None:
                self.cache_isotropic_residues()
            fractional_iso = self._cached_iso_xyz @ self.inv_fractional_matrix.T
            occs_iso = torch.cat([res.get_occupancy().expand(res.n_atoms) for res in residues_isotropic])
            scattering_factors_iso = self._cached_iso_scattering
            Bs = self._cached_iso_b
            scattering_iso = math_torch.iso_structure_factor_torched(hkl,s,fractional_iso,occs_iso,scattering_factors_iso,Bs,self.spacegroup_function)
            f_start.append(scattering_iso)
        if residues_anisotropic:
            if self._cached_aniso_xyz is None:
                self.cache_anisotropic_residues()
            fractional_aniso = self._cached_aniso_xyz @ self.inv_fractional_matrix.T    
            occs_aniso = torch.cat([res.get_occupancy().expand(res.n_atoms) for res in residues_anisotropic])
            scattering_factors_aniso = self._cached_aniso_scattering
            Us = self._cached_aniso_U
            scattering_aniso = math_torch.aniso_structure_factor_torched(hkl,scattering_vectors,fractional_aniso,
                                                                    occs_aniso,scattering_factors_aniso,Us,
                                                                    self.spacegroup_function)
            f_start.append(scattering_aniso)
        fs = torch.stack(f_start, dim=0)
        self.f_calc = torch.sum(fs, dim=0)
        return self.f_calc
    
    def get_f_ds(self,hkl):
        return ds.direct_summation(self.pdb,hkl,self.cell,space_group=self.spacegroup)
