import torch
import multicopy_refinement.get_scattering_factor_torch as gsf
import multicopy_refinement.math_numpy as math_np
from torch import tensor
from multicopy_refinement import math_torch
from multicopy_refinement import Model as mod
import torch.optim as optim
from tqdm import tqdm   
import pandas as pd

class Refinement:
    def __init__(self, hkl_df: pd.DataFrame, model: mod.model ,Fobs_key=None,I_obs_key=None,Fobs_keys_possible=['F','Fobs','F-obs-filtered'],Iobs_keys_possible=['I','Iobs']
                 ,restraints=None,weigth_xray=1,weight_restraints=0.1,structure_factors_to_refine=[],use_parametrization=False):
        hkl_df = hkl_df.dropna()
        self.hkl_df = hkl_df
        self.use_parametrization = use_parametrization
        self.scale = torch.tensor(1,dtype=torch.float64)
        self.hkl = torch.tensor(self.hkl_df.reset_index().loc[:, ['h', 'k', 'l']].values.astype(int))
        if Fobs_key is not None:
            self.Fobs = tensor(self.hkl_df[Fobs_key].values)
        elif I_obs_key is not None:
            self.Iobs = tensor(self.hkl_df[I_obs_key].values)
        elif any([key in Fobs_keys_possible for key in self.hkl_df ]):
            Fobs_key = [key for key in self.hkl_df if key in Fobs_keys_possible][0]
            self.Fobs = tensor(self.hkl_df[Fobs_key].values)
        elif any([key in Iobs_keys_possible for key in self.hkl_df ]):
            Iobs_key = [key for key in self.hkl_df if key in Iobs_keys_possible][0]
            self.Iobs = tensor(self.hkl_df[Iobs_key].values)
        else:
            raise ValueError("No Fobs or Iobs key found in hkl_df. Please provide a valid key.")
        self.model = model
        self.cell = model.cell
        self.spacegroup = model.spacegroup
        s = math_np.get_s(self.hkl, self.cell)
        self.parametrization = gsf.get_parameterization(model.pdb)
        self.s = tensor(s)
        scattering_vector = math_np.get_scattering_vectors(self.hkl, self.cell)
        self.scattering_vectors = torch.tensor(scattering_vector)
        B_inv = math_np.get_inv_fractional_matrix(self.cell)
        self.B_inv = torch.tensor(B_inv)
        self.nominal_weight_xray = weigth_xray
        self.scattering_factors_unique = gsf.get_scattering_factors_unique(self.model.pdb,self.s)
        self.nominal_weight_restraints = weight_restraints
        self.structure_factors_to_refine = structure_factors_to_refine
        if restraints is not None:
            self.restraints = restraints

    def cartesian_to_fractional(self,coords):
        coords = torch.matmul(coords,self.B_inv.T)
        return coords
    
    def get_structure_factor_for_residue(self, residue):
        xyz = residue.get_xyz()
        xyz_fractional = self.cartesian_to_fractional(xyz)
        occupancy = residue.get_occupancy()
        atoms = residue.get_atoms()
        if self.use_parametrization:
            scattering_factors = gsf.calc_scattering_factors_paramtetrization(self.parametrization, self.s, atoms)
        else:
            scattering_factors = gsf.get_scattering_factors(self.scattering_factors_unique,atoms)
        if residue.anisou_flag:
            U = residue.get_U()
            f = math_torch.aniso_structure_factor_torched(self.hkl, self.scattering_vectors, xyz_fractional, scattering_factors, occupancy, U, self.spacegroup)
        else:
            b = residue.get_b()
            f = math_torch.iso_structure_factor_torched(self.hkl, self.s, xyz_fractional, scattering_factors, occupancy, b, self.spacegroup)
        if residue.use_anharmonic:
            f = f * torch.exp(math_torch.anharmonic_correction(self.hkl, residue.anharmonic))
        return f

    def get_structure_factor(self):
        return torch.sum(torch.vstack([self.get_structure_factor_for_residue(residue) for residue in self.model.residues.values()]), axis=0)

    def get_restraint_loss_log(self):
        restraint_loss = [self.restraints.get_deviations(residue) for residue in self.model.residues.values()]
        restraint_loss = [loss for loss in restraint_loss if loss is not None]
        restraint_loss = torch.cat(restraint_loss, dim=0)
        return torch.sum(torch.log(1 + restraint_loss**2))

    def refine(self, selection_to_refine=[(None,None,None,None)],lr=0.01,n_iter=100):
        self.optimizer = self.setup_optimizer(selection_to_refine=selection_to_refine,lr=lr)
        self.norm_weigths()
        for i in tqdm(range(n_iter)):
            self.optimizer.zero_grad()
            loss = self.standard_loss()
            print(loss.item())
            loss.backward()
            self.optimizer.step()

    def calc_rfactor(self):
        f_calc = self.get_structure_factor()
        F = torch.abs(f_calc) * self.scale
        diff = torch.sum(torch.abs(F - self.Fobs))
        r_factor = diff / torch.sum(torch.abs(self.Fobs))
        return r_factor

    def norm_weigths(self):
        with torch.no_grad():
            nominal_loss_xray = self.xray_loss_linear()
            nominal_loss_restraints = self.get_restraint_loss_log()
            self.weight_xray = self.nominal_weight_xray / nominal_loss_xray
            self.weight_restraints = self.nominal_weight_restraints / nominal_loss_restraints
    
    def update_scale(self):
        with torch.no_grad():
            f_calc = self.get_structure_factor()
            F = torch.abs(f_calc)
            self.scale = torch.sum(self.Fobs) / torch.sum(F)

    def xray_loss_linear(self):
        f_calc = self.get_structure_factor()
        absorption = self.get_absorption()
        F = torch.abs(f_calc) * absorption
        return torch.sum(torch.abs(F - self.Fobs))

    def standard_loss(self):
        f_calc = self.get_structure_factor()
        absorption = self.get_absorption()
        F = torch.abs(f_calc) * absorption
        diff = torch.sum(torch.abs(F - self.Fobs))
        if self.restraints is not None:
            restraint_loss = self.get_restraint_loss_log()
            loss = diff * self.weight_xray + self.weight_restraints * restraint_loss
        else:
            loss = diff * self.weight_xray
        return loss
    
    def get_absorption(self):
        s_norm = self.scattering_vectors / torch.norm(self.scattering_vectors**2)
        Y00 = torch.ones_like(self.scattering_vectors[:,0])
        Y10 = s_norm[:,2]  # z
        Y11 = s_norm[:,0]  # x
        Y12 = s_norm[:,1]  # y
        Y20 = 1.5 * s_norm[:,2]**2 - 0.5
        Y21 = s_norm[:,0] * s_norm[:,2]
        # Apply correction - using exponential to ensure positive values
        harmonic_sum = (self.model.abs_coeffs[0] * Y00 + 
                        self.model.abs_coeffs[1] * Y10 + 
                        self.model.abs_coeffs[2] * Y11 + 
                        self.model.abs_coeffs[3] * Y12 + 
                        self.model.abs_coeffs[4] * Y20 + 
                        self.model.abs_coeffs[5] * Y21)
        absorption = torch.exp(-harmonic_sum)  
        return absorption
    
    def loss_absorption(self):
        return torch.sum(torch.abs(self.Fobs - self.get_absorption() * torch.abs(self.fcalc)))
    
    def get_rfactor_absorption(self):
        with torch.no_grad():
            f_calc = self.get_structure_factor()
            F = torch.abs(f_calc) * self.get_absorption()
            diff = torch.sum(torch.abs(F - self.Fobs))
            r_factor = diff / torch.sum(torch.abs(self.Fobs))
            return r_factor

    def optimize_absorption(self):
        self.optimizer_absorption = optim.Adam([self.model.abs_coeffs,self.scale], lr=0.01)
        with torch.no_grad():
            self.fcalc = self.get_structure_factor()
        for i in tqdm(range(100000)):
            self.optimizer_absorption.zero_grad()
            loss = self.loss_absorption()
            loss.backward()
            self.optimizer_absorption.step()

    def check_key_in_selection(self,key,selection):
        in_selection = []
        for i in range(len(selection)):
            not_in_selection = True
            for j in range(len(selection[i])):
                if selection[i][j] is not None:
                    if key[j] != selection[i][j]:
                        not_in_selection = False
                        break
            if not_in_selection:
                in_selection.append(True)
            else:
                in_selection.append(False)
                break
        return any(in_selection)

    def setup_optimizer(self, selection_to_refine=[(None,None,None,None)],lr=0.01):
        standard_tensors = [self.scale,self.model.abs_coeffs]
        if self.use_parametrization:
            for atom in self.structure_factors_to_refine:
                standard_tensors.extend(self.parametrization[atom])
        tensors_to_optimize = list(set([t for key in self.model.residues.keys() if self.check_key_in_selection(key,selection_to_refine) for t in self.model.residues[key].get_tensors_to_refine()]))
        tensors_to_optimize += standard_tensors
        self.optimizer = optim.Adam(tensors_to_optimize, lr=lr)
        return self.optimizer
    
    def loss_F_quadratic(self):
        f_calc = self.get_structure_factor()
        F = torch.abs(f_calc)
        diff = F - self.Fobs
        return torch.sum(diff**2)

    def loss_F_log(self):
        f_calc = self.get_structure_factor()
        F = torch.abs(f_calc)
        diff = F / self.Fobs
        diff.clamp_(min=1e-6,max=1e6)
        diff = torch.log(diff)
        return torch.sum(diff**2)
    
    def write_mtz(self,filename):
        import reciprocalspaceship as rs
        f_calc = self.get_structure_factor()
        F = torch.abs(f_calc) * self.scale
        Fobs = self.Fobs
        phases = torch.angle(f_calc) / torch.pi * 180
        hkl = self.hkl.detach().cpu().numpy()
        phases = phases.detach().cpu().numpy()
        self.absorption = self.get_absorption()
        F = (F * self.absorption).detach().cpu().numpy()
        Fobs = Fobs.detach().cpu().numpy()
        twoFOFC = 2 * Fobs - F
        F_diff = Fobs - F
        dataset = rs.DataSet({'H':hkl[:,0],'K':hkl[:,1],'L':hkl[:,2],'F-model':F,'F-obs':Fobs,'2FO-FC':twoFOFC,'Fobs-Fcalc':F_diff,'PHIF-model':phases}).set_index(['H','K','L'])
        dataset.cell = self.cell
        dataset.spacegroup = self.spacegroup
        dataset.infer_mtz_dtypes(inplace=True)
        dataset.write_mtz(filename)

    def _get_structure_factor_all_one_operation(self):
        anisou = self.model.pdb.loc[self.model.pdb['anisou_flag']]
        iso = self.model.pdb.loc[~self.model.pdb['anisou_flag']]
        fcalc = torch.zeros(self.hkl.shape[0], dtype=torch.complex128)
        if anisou.shape[0] > 0:
            structure_factors = torch.concat([self.unique_structure_factors[atom] for atom in anisou['element'].values], dim=1)
            xyz = self.cartesian_to_fractional(tensor(anisou[['x','y','z']].values))
            U = tensor(anisou[['u11','u22','u33','u12','u13','u23']].values)
            occupancy = tensor(anisou['occupancy'].values)
            fcalc += math_torch.aniso_structure_factor_torched(self.hkl, self.scattering_vectors, xyz,occupancy, structure_factors , U, self.spacegroup)
        if iso.shape[0] > 0:
            structure_factors = torch.concat([self.unique_structure_factors[atom] for atom in iso['element'].values],dim=1)
            xyz = self.cartesian_to_fractional(tensor(iso[['x','y','z']].values))
            b = tensor(iso['tempfactor'].values)
            occupancy = tensor(iso['occupancy'].values)
            fcalc += math_torch.iso_structure_factor_torched(self.hkl, self.s, xyz, occupancy, structure_factors, b, self.spacegroup)
        return fcalc