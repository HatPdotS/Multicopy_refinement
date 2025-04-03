import torch
import multicopy_refinement.get_scattering_factor as gsf
import multicopy_refinement.math_numpy as math_np
from torch import tensor
from multicopy_refinement import math_torch
from multicopy_refinement import Model as mod
import torch.optim as optim
from tqdm import tqdm   

class Refinement:
    def __init__(self, hkl_df, model: mod.model ,Fobs_key=None,I_obs_key=None,Fobs_keys_possible=['F','Fobs'],Iobs_keys_possible=['I','Iobs']
                 ,restraints=None,weigth_xray=1,weight_restraints=0.1):
        self.hkl_df = hkl_df
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
        self.unique_structure_factors = {key: tensor(value) for key,value in gsf.get_scattering_factors_unique(model.pdb,s).items()}
        self.s = tensor(s)
        scattering_vector = math_np.get_scattering_vectors(self.hkl, self.cell)
        self.scattering_vectors = torch.tensor(scattering_vector)
        B_inv = math_np.get_inv_fractional_matrix(self.cell)
        self.B_inv = torch.tensor(B_inv)
        self.weight_xray = weigth_xray
        self.weight_restraints = weight_restraints
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
        scattering_factors = torch.concat([self.unique_structure_factors[atom] for atom in atoms], dim=1)
        if residue.anisou_flag:
            U = residue.get_U()
            F = math_torch.aniso_structure_factor_torched(self.hkl, self.scattering_vectors, xyz_fractional, scattering_factors, occupancy, U, self.spacegroup)
        else:
            b = residue.get_b()
            F = math_torch.iso_structure_factor_torched(self.hkl, self.s, xyz_fractional, scattering_factors, occupancy, b, self.spacegroup)
        return F

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
            self.model.sanitize_occ()

    def calc_rfactor(self):
        f_calc = self.get_structure_factor()
        F = torch.abs(f_calc)
        diff = torch.sum(torch.abs(F - self.Fobs))
        r_factor = diff / torch.sum(torch.abs(self.Fobs))
        return r_factor

    def norm_weigths(self):
        with torch.no_grad():
            nominal_loss_xray = self.xray_loss_linear()
            nominal_loss_restraints = self.get_restraint_loss_log()
            self.weight_xray = self.weight_xray / nominal_loss_xray
            self.weight_restraints = self.weight_restraints / nominal_loss_restraints

    def xray_loss_linear(self):
        f_calc = self.get_structure_factor()
        F = torch.abs(f_calc)
        return torch.sum(torch.abs(F - self.Fobs))

    def standard_loss(self):
        f_calc = self.get_structure_factor()
        F = torch.abs(f_calc)
        diff = torch.sum(torch.abs(F - self.Fobs))
        if self.restraints is not None:
            restraint_loss = self.get_restraint_loss_log()
            loss = diff * self.weight_xray + self.weight_restraints * restraint_loss
        else:
            loss = diff * self.weight_xray
        return loss

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
        tensors_to_optimize = [t for key in self.model.residues.keys() if self.check_key_in_selection(key,selection_to_refine) for t in self.model.residues[key].get_tensors_to_refine()]
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