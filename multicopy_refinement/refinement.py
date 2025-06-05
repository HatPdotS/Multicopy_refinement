import torch
import multicopy_refinement.get_scattering_factor_torch as gsf
import multicopy_refinement.math_numpy as math_np
from torch import tensor
from multicopy_refinement import math_torch
from multicopy_refinement import Model as mod
import torch.optim as optim
from tqdm import tqdm   
import pandas as pd
import multicopy_refinement.io as io
from torch import nn
import multicopy_refinement.symmetrie as sym
import torch

class Refinement(nn.Module):
    def __init__(self, hkl_df: pd.DataFrame, model: mod.model ,Fobs_key=None,I_obs_key=None,Fobs_keys_possible=['F','Fobs','F-obs-filtered'],Iobs_keys_possible=['I','Iobs']
                 ,restraints=None,weigth_xray=1,weight_restraints=0.1,structure_factors_to_refine=[],use_parametrization=False,min_res=15.0,max_res=0.9):
        nn.Module.__init__(self)
        hkl_df = hkl_df.dropna()
        hkl_df['resolution'] = io.get_resolution(hkl_df[['h','k','l']],model.cell)
        hkl_df = hkl_df.loc[(hkl_df['resolution'] < min_res) & (hkl_df['resolution'] > max_res)]
        self.hkl_df = hkl_df
        self.use_parametrization = use_parametrization
        self.scale = torch.tensor(1,dtype=torch.float64)
        self.hkl = torch.tensor(self.hkl_df.reset_index().loc[:, ['h', 'k', 'l']].values.astype(int))
        self.Fobs, self.Fobs_sigma = io.get_f(hkl_df)
        self.model = model
        self.cell = model.cell
        self.spacegroup = model.spacegroup
        self.spacegroup_function = sym.get_space_group_function(self.spacegroup)
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
        self.setup_scattering_factors(self.model)
    
    def setup_scattering_factors(self,model):
        for residue in model.residues.values():
            if self.use_parametrization:
                residue.get_scattering_factor_parametrizatian(self.parametrization, self.s)
            else:
                residue.get_scattering_factor(self.scattering_factors_unique)

    def get_restraint_loss_exp(self, model, scale=0.1):
        """Extremely aggressive restraint using exponential penalty.
        
        Args:
            model: The model containing residues
            scale: Controls steepness of exponential penalty
        """
        restraint_loss = [self.restraints.get_deviations(residue) for residue in model.residues.values()]
        restraint_loss = [loss for loss in restraint_loss if loss is not None]
        restraint_loss = torch.cat(restraint_loss, dim=0)
        # exp(x) - 1 ensures small values stay small but large values grow extremely fast
        return torch.sum(torch.exp(scale * torch.abs(restraint_loss)) - 1.0)
    
    def get_restraint_loss_smooth_huber(self, model, delta=1.0, power=3.0):
        """Smooth transition from quadratic to higher power penalty.
        
        Args:
            model: The model containing residues
            delta: Threshold for transition
            power: Higher power for large deviations (try 3.0 to 4.0)
        """
        restraint_loss = [self.restraints.get_deviations(residue) for residue in model.residues.values()]
        restraint_loss = [loss for loss in restraint_loss if loss is not None]
        restraint_loss = torch.cat(restraint_loss, dim=0)
        
        abs_loss = torch.abs(restraint_loss)
        quadratic = 0.5 * restraint_loss**2
        # Scale the higher power term to match quadratic at the transition
        higher_power = (delta**(2-power)/2) * abs_loss**power
        
        # Use quadratic loss for values <= delta, higher power for values > delta
        mask = abs_loss <= delta
        loss = torch.where(mask, quadratic, higher_power)
        
        return torch.sum(loss)
    
    def get_restraint_loss_clamped_exp(self, model, scale=0.2, max_value=10000.0):
        """Controlled exponential penalty with clamping to prevent explosion.
        Args:
            model: The model containing residues
            scale: Controls steepness (0.1-0.3 = moderate)
            max_value: Maximum value for any individual penalty to prevent explosion
        """
        restraint_loss = [self.restraints.get_deviations(residue) for residue in model.residues.values()]
        restraint_loss = [loss for loss in restraint_loss if loss is not None]
        restraint_loss = torch.cat(restraint_loss, dim=0)
        
        # Apply exponential but clamp to prevent explosion
        exp_penalty = torch.exp(scale * torch.abs(restraint_loss)) - 1.0
        clamped_penalty = torch.clamp(exp_penalty, max=max_value)
        
        return torch.sum(clamped_penalty)

    def loss_squared(self,deviations):
        return torch.sum(deviations**2)
    
    def get_restraint_loss_cubic(self, model):
        """Standard squared loss for restraints.
        
        Args:
            model: The model containing residues
        """
        restraint_loss = [self.restraints.get_deviations(residue) for residue in model.residues.values()]
        restraint_loss = [loss for loss in restraint_loss if loss is not None]
        restraint_loss = torch.cat(restraint_loss, dim=0)
        return torch.sum(restraint_loss**4)

    def cartesian_to_fractional(self,coords):
        coords = torch.matmul(coords,self.B_inv.T)
        return coords
    
    def cuda(self):
        self.hkl = self.hkl.cuda()
        self.s = self.s.cuda()
        self.scattering_vectors = self.scattering_vectors.cuda()
        self.B_inv = self.B_inv.cuda()
        self.Fobs = self.Fobs.cuda()
        if hasattr(self,'Iobs'):
            self.Iobs = self.Iobs.cuda()
        if hasattr(self,'scale'):
            self.scale = self.scale.cuda()
        self.model.cuda()
        self.B_inv.cuda()
        self.Fobs_sigma = self.Fobs_sigma.cuda()
        self.scattering_factors_unique = {i: self.scattering_factors_unique[i].cuda() for i in self.scattering_factors_unique}
        for key in self.parametrization.keys():
            for i,t in enumerate(self.parametrization[key]):
                self.parametrization[key][i] = t.cuda()

    def cpu(self):
        self.hkl = self.hkl.cpu()
        self.s = self.s.cpu()
        self.scattering_vectors = self.scattering_vectors.cpu()
        self.B_inv = self.B_inv.cpu()
        self.Fobs = self.Fobs.cpu()
        if hasattr(self,'Iobs'):
            self.Iobs = self.Iobs.cpu()
        if hasattr(self,'scale'):
            self.scale = self.scale.cpu()
        self.model.cpu()
        self.B_inv.cpu()
        self.Fobs_sigma = self.Fobs_sigma.cpu()
        for key in self.parametrization.keys():
            for i,t in enumerate(self.parametrization[key]):
                self.parametrization[key][i] = t.cpu()

    def get_structure_factor_for_residue(self, residue):
        xyz = residue.get_xyz()
        if torch.isnan(xyz).any():
            print(residue.resname)
            raise ValueError("xyz contains NaN values")
        xyz_fractional = self.cartesian_to_fractional(xyz)
        occupancy = residue.get_occupancy()
        atoms = residue.get_atoms()
        if self.use_parametrization and any ([atom in self.structure_factors_to_refine for atom in atoms]):
            scattering_factors = gsf.calc_scattering_factors_paramtetrization(self.parametrization, self.s, atoms)
        else:
            if not hasattr(residue,'scattering_factors'):
                if self.use_parametrization:
                    residue.scattering_factors = gsf.calc_scattering_factors_paramtetrization(self.parametrization, self.s, atoms)
                else:
                    residue.scattering_factors = gsf.get_scattering_factors(self.scattering_factors_unique, atoms)
            scattering_factors = residue.scattering_factors
        if residue.anisou_flag:
            U = residue.get_U()
            f = math_torch.aniso_structure_factor_torched(self.hkl, self.scattering_vectors, xyz_fractional,
                                                          occupancy, scattering_factors, U, 
                                                          self.spacegroup_function)
        else:
            b = residue.get_b()
            f = math_torch.iso_structure_factor_torched(self.hkl, self.s, xyz_fractional,occupancy, scattering_factors, b, self.spacegroup_function)
        if residue.use_anharmonic:
            f = f * math_torch.anharmonic_correction(self.hkl, residue.anharmonic)
        if residue.use_core_deformation:
            f = f * math_torch.core_deformation(residue.core_deformation,self.s)
        return f

    def get_structure_factor_for_residue_compilable(self, residue):
        structure_factor = residue.get_structure_factor(self.hkl,self.scattering_vectors,self.s)
        return structure_factor

    def get_structure_factor(self):
        # f_calc = self.get_structure_factor_no_corrections_compilable()
        # absorption = self.get_absorption(self.model)
        # f_calc = f_calc * absorption
        # extinction_factor = self.get_extinction_factor(f_calc,self.model)
        # f_calc = f_calc * extinction_factor * self.scale
        # return f_calc
        self.fcalc = self.model.get_structure_factor(self.hkl,self.scattering_vectors,self.s)
        fcalc_corrected = self.model.apply_corrections()
        return fcalc_corrected

    def get_structure_factor_no_corrections(self):
        f_calc = torch.sum(torch.vstack([self.get_structure_factor_for_residue(residue) for residue in self.model.residues.values()]), axis=0)
        return f_calc
    
    def get_structure_factor_no_corrections_compilable(self):
        vstacked = torch.stack([self.get_structure_factor_for_residue_compilable(residue) for residue in self.model.residues.values()])
        f_calc = torch.sum(vstacked, axis=0)
        return f_calc

    def get_restraint_loss_log(self,model):
        restraint_loss = self.get_deviations(model)
        return self.loss_log(restraint_loss)
    
    def get_restraint_loss_squared(self,model):
        restraint_loss = self.get_deviations(model)
        return self.loss_squared(restraint_loss)

    def loss_log(self,loss):
        return torch.sum(torch.log(1 + loss**2))
    
    def get_deviations(self,model):
        restraint_loss = [self.restraints.get_deviations(residue) for residue in model.residues.values()]
        restraint_loss = [loss for loss in restraint_loss if loss is not None]
        if restraint_loss:
            restraint_loss = torch.cat(restraint_loss, dim=0)
            return restraint_loss
        else:
            return torch.tensor([0])
    
    def get_restraint_loss_lin(self,model):
        restraint_loss = [self.restraints.get_deviations(residue) for residue in model.residues.values()]
        restraint_loss = [loss for loss in restraint_loss if loss is not None]
        restraint_loss = torch.cat(restraint_loss, dim=0)
        return torch.sum(torch.abs(restraint_loss))
    
    def get_restraint_loss_sigmoidal(self,model):
        restraint_loss = [self.restraints.get_deviations(residue) for residue in model.residues.values()]
        restraint_loss = [loss for loss in restraint_loss if loss is not None]
        restraint_loss = torch.cat(restraint_loss, dim=0)
        return torch.sum(torch.sigmoid(torch.abs(restraint_loss))/5)
    
    def loss_compilable(self):
        f_calc = self.get_structure_factor_compilable()
        F = torch.sum(f_calc**2,axis=0) ** 0.5
        diff = torch.sum(torch.abs(F - self.Fobs))
        restraints_loss = self.get_restraint_loss_log(self.model)
        loss = diff * self.weight_xray + self.weight_restraints * restraints_loss
        return loss
    
    def get_structure_factor_compilable(self):
        f_calc = self.get_structure_factor_no_corrections_compilable()
        absorption = self.get_absorption(self.model)
        f_calc = f_calc * absorption
        extinction_factor = self.get_extinction_factor(f_calc,self.model)
        f_calc = f_calc * extinction_factor * self.scale
        return f_calc


    def refine(self, selection_to_refine=[(None,None,None,None)],lr=0.01,n_iter=100,grad_threshold=1e-10):
        self.optimizer = self.setup_optimizer(selection_to_refine=selection_to_refine,lr=lr)
        self.norm_weights()
        for i in tqdm(range(n_iter)):
            self.optimizer.zero_grad()
            loss = self.standard_loss()

            # print(loss.item())
            loss.backward()
            for param_group in self.optimizer.param_groups:
                for param in param_group['params']:
                    if param.grad is not None:
                        # Count NaNs (optional for debugging)
                        nan_count = torch.isnan(param.grad).sum().item()
                        if nan_count > 0:
                            print(f"Replacing {nan_count} NaN gradients with zeros")
                         
                        # Replace all NaNs with zeros using torch.nan_to_num
                        param.grad = torch.nan_to_num(param.grad, nan=0.0)
            torch.nn.utils.clip_grad_value_(self.optimizer.param_groups[0]['params'], clip_value=1.0)
            self.optimizer.step()

    def refine_compile(self, selection_to_refine=[(None,None,None,None)],lr=0.01,n_iter=100,grad_threshold=1e-10):
        self.optimizer = self.setup_optimizer(selection_to_refine=selection_to_refine,lr=lr)
        self.norm_weights()
        loss_function = torch.compile(self.loss_compilable)
        # loss_function = self.loss_compilable
        for i in tqdm(range(n_iter)):
            self.optimizer.zero_grad()
            loss = loss_function()
            print(loss.item())
            loss.backward()
            for param_group in self.optimizer.param_groups:
                for param in param_group['params']:
                    if param.grad is not None:
                        # Count NaNs (optional for debugging)
                        nan_count = torch.isnan(param.grad).sum().item()
                        if nan_count > 0:
                            print(f"Replacing {nan_count} NaN gradients with zeros")
                        
                        # Replace all NaNs with zeros using torch.nan_to_num
                        param.grad = torch.nan_to_num(param.grad, nan=0.0)
            torch.nn.utils.clip_grad_value_(self.optimizer.param_groups[0]['params'], clip_value=1.0)
            self.optimizer.step()

    def norm_weights(self):
        with torch.no_grad():
            nominal_loss_xray = self.xray_loss_rfactor10()
            self.weight_xray = self.nominal_weight_xray / nominal_loss_xray
            if hasattr(self,'restraints'): 
                deviations = self.get_deviations(self.model)
                deviations[:] = 3
                nominal_loss_restraints = self.loss_squared(deviations)
                self.weight_restraints = self.nominal_weight_restraints / nominal_loss_restraints
    
    def xray_loss_linear(self):
        f_calc = self.get_structure_factor()
        F = torch.abs(f_calc)
        return torch.sum(torch.abs(F - self.Fobs))

    def xray_loss_rfactor10(self):
        f_calc = self.get_structure_factor()
        F = torch.abs(f_calc)
        return torch.sum(torch.abs(F*0.1))

    def get_rfactor(self):
        f_calc = self.get_structure_factor()
        F = torch.abs(f_calc)
        diff = torch.sum(torch.abs(F - self.Fobs))
        r_factor = diff / torch.sum(torch.abs(self.Fobs))
        return r_factor
    
    def get_rfactor_weigthed(self):
        f_calc = self.get_structure_factor()
        I_over_sigma = self.Fobs / self.Fobs_sigma
        I_over_sigma[torch.isnan(I_over_sigma)] = 1e-6
        I_over_sigma = torch.clamp(I_over_sigma, min=1e-6, max=3)
        I_over_sigma = I_over_sigma / torch.mean(I_over_sigma)
        F = torch.abs(f_calc)
        diff = torch.sum(torch.abs(F - self.Fobs)*I_over_sigma)
        r_factor = diff / torch.sum(torch.abs(self.Fobs)*I_over_sigma)
        return r_factor

    def standard_loss(self):
        f_calc = self.get_structure_factor()
        if torch.isnan(f_calc).any():
            print("Fcalc contains NaN values")
            raise ValueError("Fcalc contains NaN values")
        F = torch.abs(f_calc)
        diff = torch.sum(torch.abs(F - self.Fobs))
        if hasattr(self,'restraints'):
            restraint_loss = self.get_restraint_loss_squared(self.model)
            if torch.isnan(restraint_loss).any():
                print("Restraint loss contains NaN values")
                raise ValueError("Restraint loss contains NaN values")
            loss = diff * self.weight_xray + self.weight_restraints * restraint_loss
        else:
            loss = diff * self.weight_xray
        return loss
    
    def get_absorption(self,model):
        s_norm = self.scattering_vectors / torch.norm(self.scattering_vectors**2)
        Y00 = torch.ones_like(self.scattering_vectors[:,0])
        Y10 = s_norm[:,2]  # z
        Y11 = s_norm[:,0]  # x
        Y12 = s_norm[:,1]  # y
        Y20 = 1.5 * s_norm[:,2]**2 - 0.5
        Y21 = s_norm[:,0] * s_norm[:,2]
        # Apply correction - using exponential to ensure positive values
        harmonic_sum = (model.abs_coeffs[0] * Y00 + 
                        model.abs_coeffs[1] * Y10 + 
                        model.abs_coeffs[2] * Y11 + 
                        model.abs_coeffs[3] * Y12 + 
                        model.abs_coeffs[4] * Y20 + 
                        model.abs_coeffs[5] * Y21)
        absorption = torch.exp(-harmonic_sum)  
        return absorption
    
    def get_extinction_factor(self,fcalc,model):
        """Calculate extinction correction factors for each reflection"""
        # Get sin(theta)/lambda values (proportional to resolution)
        s_values = self.s
        
        # Calculate extinction parameter based on reflection strength
        # Stronger reflections (typically lower resolution) have more extinction
        f_calc_abs = torch.abs(fcalc)
        
        # Simple isotropic extinction model (Zachariasen/Becker-Coppens formalism)
        # extinction_factor = 1 / (1 + 2 * extinction_param * f_calc_abs**2 / s_values)
        
        # Single-parameter model where ext_param is refined
        ext_param = model.extinction_parameter  # This would be a learnable parameter
        extinction_factor = 1.0 / (1.0 + ext_param * 1e-10 * f_calc_abs**2 / (s_values + 1e-10))
        
        return extinction_factor

    def loss_absorption(self,model):
        return torch.sum(torch.abs(self.Fobs - self.get_absorption(model) * torch.abs(self.fcalc)* self.scale * self.get_extinction_factor(self.fcalc,model)))

    def optimize_scale(self):
        self.optimizer_absorption = optim.Adam(self.model.correction_parameters, lr=0.01)
        with torch.no_grad():
            self.fcalc = self.model.get_structure_factor(self.hkl,self.scattering_vectors,self.s)
        for i in tqdm(range(1000)):
            self.optimizer_absorption.zero_grad()
            fcalc_corrected = self.model.apply_corrections()
            loss = torch.sum((self.Fobs - torch.abs(fcalc_corrected)) ** 2)
            loss.backward()
            if i % 100 == 0:
                print(f"Iteration {i}, Loss: {loss.item()}")
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
        standard_tensors = [self.scale,self.model.abs_coeffs,self.model.extinction_parameter]
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
        F = torch.abs(f_calc)
        Fobs = self.Fobs
        phases = torch.angle(f_calc) / torch.pi * 180
        hkl = self.hkl.detach().cpu().numpy()
        phases = phases.detach().cpu().numpy()
        F = (F).detach().cpu().numpy()
        Fobs = Fobs.detach().cpu().numpy()
        twoFOFC = 2 * Fobs - F
        F_diff = Fobs - F
        dataset = rs.DataSet({'H':hkl[:,0],'K':hkl[:,1],'L':hkl[:,2],'F-model':F,'F-obs':Fobs,'2FOFCWT':twoFOFC,'Fobs-Fcalc':F_diff,'PHIF-model':phases}).set_index(['H','K','L'])
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
    
class make_mtz(Refinement):
    def __init__(self, pdb_path, dmin,path_out=None,noise=0.0):
        """
        Initializes the make_mtz class with a PDB file and resolution.
        Computes the structure factor for the given PDB file.
        """
        nn.Module.__init__(self)
        import reciprocalspaceship as rs
        self.use_parametrization = False
        self.model = mod.model()
        self.model.load_pdb_from_file(pdb_path)
        self.dmin = dmin
        self.spacegroup = self.model.spacegroup
        self.cell = self.model.cell
        self.hkl,self.scattering_vectors = self.get_indices()
        self.spacegroup_function = sym.get_space_group_function(self.spacegroup)

        s = math_np.get_s(self.hkl, self.cell)
        self.parametrization = gsf.get_parameterization(self.model.pdb)
        self.s = tensor(s)
        scattering_vector = math_np.get_scattering_vectors(self.hkl, self.cell)
        self.scattering_vectors = torch.tensor(scattering_vector)
        B_inv = math_np.get_inv_fractional_matrix(self.cell)
        self.B_inv = torch.tensor(B_inv)
        self.scattering_factors_unique = gsf.get_scattering_factors_unique(self.model.pdb,self.s)
        f = self.get_structure_factor_no_corrections()
        self.F = torch.abs(f)
        self.F += torch.randn_like(self.F) * noise * self.F # Add noise if specified
        self.phase = torch.angle(f) / torch.pi * 180  # Convert to degrees
        self.SIGF = self.F * 0.1  # Example sigma, can be adjusted
        dataset = rs.DataSet({'H':self.hkl[:,0].detach().cpu().numpy(),'K':self.hkl[:,1].detach().cpu().numpy(),
                      'L':self.hkl[:,2].detach().cpu().numpy(),'F-model':self.F.detach().cpu().numpy(),'SIGF':self.SIGF.detach().cpu().numpy(),
                      'PHI-model':self.phase.detach().cpu().numpy()}).set_index(['H','K','L'])
        dataset.cell = self.cell
        dataset.spacegroup = self.spacegroup
        dataset.infer_mtz_dtypes(inplace=True)
        path_out = path_out if path_out is not None else pdb_path.replace('.pdb','.mtz')
        dataset.write_mtz(path_out)

    def structure_factor_from_pdb(self):
        """
        Loads a PDB file, updates the model, and computes the structure factor.
        Returns the calculated structure factor as a torch tensor.
        """
        # Setup scattering factors for the new model
        self.setup_scattering_factors(self.model)
        # Compute structure factor
        f_calc = self.get_structure_factor()
        return f_calc
    
    def get_indices(self):

        """
        Computes the grid for the structure factor.
        Returns a DataFrame with H, K, L, F-model, and PHIF-model.
        """
        cell = self.model.cell
        a, b, c, alpha, beta, gamma = cell
        # Estimate max index for each axis
        # Conservative: up to 1/dmin for each axis, but can be more precise
        h = torch.arange(torch.ceil(torch.tensor(a / self.dmin)).int() + 1, dtype=torch.int32)
        k = torch.arange(-int(torch.ceil(torch.tensor(b / self.dmin))), int(torch.ceil(torch.tensor(b / self.dmin))) + 1, dtype=torch.int32)
        l = torch.arange(-int(torch.ceil(torch.tensor(c / self.dmin))), int(torch.ceil(torch.tensor(c / self.dmin))) + 1, dtype=torch.int32)
        hkl = torch.cartesian_prod(h, k, l)
        scattering_vectors = math_torch.get_scattering_vectors(hkl, cell)
        resolution = 1.0/ torch.norm(scattering_vectors, dim=1)
        mask = resolution > self.dmin
        hkl = hkl[mask]
        scattering_vectors = scattering_vectors[mask]
        return hkl, scattering_vectors