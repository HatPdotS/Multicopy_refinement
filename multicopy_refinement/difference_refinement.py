import multicopy_refinement.refinement as refinement
from torch import tensor
import pandas as pd
from multicopy_refinement import Model as mod
import torch
from multicopy_refinement import math_numpy as math_np
import multicopy_refinement.get_scattering_factor_torch as gsf
import multicopy_refinement.io as io
import numpy as np
from torch import optim
from tqdm import tqdm
from torch import nn
import multicopy_refinement.symmetrie as sym
import reciprocalspaceship as rs

class Difference_refinement(refinement.Refinement):

    def __init__(self, Fobs_dark: pd.DataFrame,Fobs_light: pd.DataFrame, model_dark: mod.model,model_light: mod.model ,Fobs_key=None,I_obs_key=None,Fobs_keys_possible=['F','Fobs','F-obs-filtered'],Iobs_keys_possible=['I','Iobs']
                 ,restraints=None,weigth_xray=1,weight_restraints=0.1,structure_factors_to_refine=[],use_parametrization=False,alpha_start=0.1,refine_alpha=True):
        nn.Module.__init__(self)
        Fobs_dark = Fobs_dark.dropna().set_index(['h','k','l'])
        Fobs_light = Fobs_light.dropna().set_index(['h','k','l'])
        F_all = Fobs_dark.join(Fobs_light, lsuffix='_dark', rsuffix='_light', how='inner')
        Dark_columns = [name for name in F_all.columns if '_dark' in name]
        Light_columns = [name for name in F_all.columns if '_light' in name]
        Fobs_dark = F_all[Dark_columns]
        Fobs_dark = Fobs_dark.rename(columns={name: name.replace('_dark', '') for name in Dark_columns})
        Fobs_light = F_all[Light_columns]
        Fobs_light = Fobs_light.rename(columns={name: name.replace('_light', '') for name in Light_columns})
        self.hkl = tensor(np.array(Fobs_dark.reset_index().loc[:,['h','k','l']]),dtype=torch.float64)
        self.use_parametrization = use_parametrization
        self.Fobs_dark, self.Fobs_dark_sigma = io.get_f(Fobs_dark)
        self.Fobs_light, self.Fobs_light_sigma = io.get_f(Fobs_light)
        self.model_dark = model_dark
        self.model_light = model_light
        self.cell = model_dark.cell
        self.spacegroup = model_dark.spacegroup
        self.spacegroup_function = sym.get_space_group_function(self.spacegroup)
        s = math_np.get_s(self.hkl, self.cell)
        self.parametrization = gsf.get_parameterization(model_dark.pdb)
        self.s = tensor(s)
        scattering_vector = math_np.get_scattering_vectors(self.hkl, self.cell)
        self.scattering_vectors = torch.tensor(scattering_vector)
        B_inv = math_np.get_inv_fractional_matrix(self.cell)
        self.B_inv = torch.tensor(B_inv)
        self.nominal_weight_xray = tensor(weigth_xray,dtype=torch.float64)
        self.scattering_factors_unique = gsf.get_scattering_factors_unique(self.model_dark.pdb,self.s)
        self.nominal_weight_restraints = tensor(weight_restraints,dtype=torch.float64)
        self.structure_factors_to_refine = structure_factors_to_refine
        self.alpha = nn.Parameter(tensor(alpha_start,dtype=torch.float64,requires_grad=True))
        self.scale_light = nn.Parameter(tensor(1.0,requires_grad=True))
        self.scale_xray = tensor(1.0,requires_grad=True)
        self.scale_refinement = tensor(1.0,requires_grad=True)
        self.setup_scattering_factors(self.model_dark)
        self.setup_scattering_factors(self.model_light)
        self.refine_alpha = refine_alpha
        
        if restraints is not None:
            self.restraints = restraints
        else:
            self.restraints = None

    def cuda(self):
        nn.Module.cuda(self)
        self.Fobs_dark = self.Fobs_dark.cuda()
        self.Fobs_light = self.Fobs_light.cuda()
        self.model_dark.cuda()
        self.model_light.cuda()
        self.hkl = self.hkl.cuda()
        self.s = self.s.cuda()
        self.scattering_vectors = self.scattering_vectors.cuda()
        self.B_inv = self.B_inv.cuda()
        self.scattering_factors_unique = {i: self.scattering_factors_unique[i].cuda() for i in self.scattering_factors_unique}
        for key in self.parametrization.keys():
            for i,t in enumerate(self.parametrization[key]):
                self.parametrization[key][i] = t.cuda()
        self.Fobs_dark_sigma = self.Fobs_dark_sigma.cuda()
        self.Fobs_light_sigma = self.Fobs_light_sigma.cuda()
        self.model_dark.cuda()
        self.model_light.cuda()
    
    def cpu(self):
        nn.Module.cpu(self)
        self.Fobs_dark = self.Fobs_dark.cpu()
        self.Fobs_light = self.Fobs_light.cpu()
        self.model_dark.cpu()
        self.model_light.cpu()
        self.hkl = self.hkl.cpu()
        self.s = self.s.cpu()
        self.scattering_vectors = self.scattering_vectors.cpu()
        self.B_inv = self.B_inv.cpu()
        for key in self.parametrization.keys():
            for i,t in enumerate(self.parametrization[key]):
                self.parametrization[key][i] = t.cpu()
        self.scattering_factors_unique = {i: self.scattering_factors_unique[i].cpu() for i in self.scattering_factors_unique}
        self.Fobs_dark_sigma = self.Fobs_dark_sigma.cpu()
        self.Fobs_light_sigma = self.Fobs_light_sigma.cpu()
        
    def get_structure_factor_for_model_no_corrections(self,model):
        f_calc = torch.sum(torch.vstack([self.get_structure_factor_for_residue(residue) for residue in model.residues.values()]), axis=0)
        return f_calc
    
    def get_structure_factor_for_model(self,model):
        f_calc = self.get_structure_factor_for_model_no_corrections(model)
        absorption = self.get_absorption(model)
        extinction = self.get_extinction_factor(f_calc,model)
        f_calc = f_calc * absorption * extinction * model.scale
        return f_calc
    
    def loss_diff(self):
        fcalc = self.get_structure_factor_for_model(self.model_light)
        diff_fobs = (self.Fobs_dark - self.Fobs_light * self.scale_light)
        diff_fcalc = self.Fcalc_dark - torch.abs(fcalc * self.alpha + self.fcalc_dark * (1-self.alpha))
        return torch.sum((diff_fobs - diff_fcalc)**2)
    
    def loss_simple(self):
        fcalc_light = self.get_structure_factor_for_model(self.model_light)
        fcalc = fcalc_light * self.alpha + self.fcalc_dark * (1-self.alpha)
        Fcalc = torch.abs(fcalc)
        return torch.sum(torch.abs(self.Fobs_light - Fcalc))

    def get_rfactor(self):
        f_calc = self.get_f_combined()
        F = torch.abs(f_calc)
        diff = torch.sum(torch.abs(F - self.Fobs_light))
        r_factor = diff / torch.sum(torch.abs(self.Fobs_light))
        return r_factor

    def get_f_combined(self):
        if not hasattr(self, 'fcalc_dark'):
            self.fcalc_dark = self.get_structure_factor_for_model(self.model_dark)
        fcalc_light = self.get_structure_factor_for_model(self.model_light)
        fcalc = fcalc_light * self.alpha + self.fcalc_dark * (1-self.alpha)
        return fcalc

    def optimize_scale(self,fcalc,Fobs,model):
        self.optimizer_absorption = optim.Adam([model.abs_coeffs,model.scale,model.extinction_parameter], lr=0.01)
        self.fcalc = fcalc
        self.Fobs = Fobs
        self.scale = model.scale
        for i in tqdm(range(1000)):
            self.optimizer_absorption.zero_grad()
            loss = self.loss_absorption(model)
            loss.backward()
            self.optimizer_absorption.step()

    def loss_scale(self):
        return torch.sum(torch.abs(self.Fobs_dark - self.Fobs_light * self.scale_light)**2)

    def find_scale_light(self):
        self.optimizer_scale = optim.Adam([self.scale_light], lr=0.01)
        for i in tqdm(range(100)):
            self.optimizer_scale.zero_grad()
            loss = self.loss_scale()
            loss.backward()
            self.optimizer_scale.step()
        print('Scale light:',self.scale_light.item())

    def write_mtz(self,filename):
        fcalc_light = self.get_structure_factor_for_model(self.model_light)
        fcalc_dark = self.get_structure_factor_for_model(self.model_dark)
        f_calc = fcalc_light * self.alpha + fcalc_dark * (1-self.alpha)

        phase = torch.angle(f_calc) 
        phase_dark = torch.angle(fcalc_dark)
        Fobs_grafted = torch.complex(self.Fobs_light * torch.cos(phase), self.Fobs_dark * torch.sin(phase))
        Fobs_dark_grafted = torch.complex(self.Fobs_dark * torch.cos(phase_dark), self.Fobs_dark * torch.sin(phase_dark))
        fdiff = Fobs_dark_grafted - Fobs_grafted
        Fdiff = torch.abs(fdiff).detach().cpu().numpy()
        Diff_phase = (torch.angle(fdiff) / torch.pi * 180).detach().cpu().numpy()
        f_extr = (Fobs_grafted - fcalc_dark * (1-self.alpha)) / self.alpha
        phi_light = (torch.angle(fcalc_light)/ torch.pi * 180).detach().cpu().numpy()
        F_extr = torch.abs(f_extr).detach().cpu().numpy()
        F = torch.abs(f_calc)
        Fobs = self.Fobs_light
        phases = torch.angle(f_calc) / torch.pi * 180
        hkl = self.hkl.detach().cpu().numpy()
        phases = phases.detach().cpu().numpy()
        F = (F).detach().cpu().numpy()
        Fobs = Fobs.detach().cpu().numpy()
        twoFOFC = 2 * Fobs - F
        F_diff = Fobs - F
        dataset = rs.DataSet({'H':hkl[:,0],'K':hkl[:,1],'L':hkl[:,2],'F-model':F,'F-obs':Fobs,'2FOFCWT':twoFOFC,
                              'Fobs-Fcalc':F_diff,'PHIF-model':phases,'Fext':F_extr,'PHF_light':phi_light,
                              'Fcorrected_diff': Fdiff,'PHF_diff_corrected':Diff_phase}).set_index(['H','K','L'])
        dataset.cell = self.cell
        dataset.spacegroup = self.spacegroup
        dataset.infer_mtz_dtypes(inplace=True)
        dataset.write_mtz(filename)

    def get_scales(self):
        with torch.no_grad():
            fcalc_dark = self.get_structure_factor_for_model(self.model_dark)
        self.optimize_scale(fcalc_dark,self.Fobs_dark,self.model_dark)
        with torch.no_grad():
            fcalc_light = self.get_structure_factor_for_model(self.model_light)
        self.optimize_scale(fcalc_light,self.Fobs_light,self.model_light)

    def setup_optimizer(self, selection_to_refine=[(None,None,None,None)],lr=0.01):
        standard_tensors_light = [self.model_light.scale,self.model_light.abs_coeffs,self.model_light.extinction_parameter]
        standard_tensors_dark = [self.model_dark.scale,self.model_dark.abs_coeffs,self.model_dark.extinction_parameter]
        if self.refine_alpha:
            standard_tensors = [self.alpha]
        else:
            standard_tensors = []
        if self.use_parametrization:
            for atom in self.structure_factors_to_refine:
                standard_tensors.extend(self.parametrization[atom])
        tensors_to_optimize = list(set([t for key in self.model_light.residues.keys() if self.check_key_in_selection(key,selection_to_refine) for t in self.model_light.residues[key].get_tensors_to_refine()]))
        tensors_to_optimize += standard_tensors
        self.optimizer = optim.Adam(tensors_to_optimize, lr=lr)
        return self.optimizer

    def norm_weights(self):
        with torch.no_grad():
            nominal_loss_xray = torch.sum(torch.abs(self.fcalc_dark*0.1))
            self.weight_xray = self.nominal_weight_xray / nominal_loss_xray
            if hasattr(self,'restraints'): 
                deviations = self.get_deviations(self.model)
                deviations[:] = 3
                nominal_loss_restraints = self.loss_squared(deviations)
                self.weight_restraints = self.nominal_weight_restraints / nominal_loss_restraints
    
    def loss(self):
        loss = self.loss_simple() * self.weight_xray   
        if self.restraints is not None:
            loss += self.get_restraint_loss_squared(self.model_light) * self.weight_restraints
        return loss


    def refine(self,n_iter=1000,lr=0.01,selection_to_refine=[(None,None,None,None)]):
        self.optimizer = self.setup_optimizer(lr=lr,selection_to_refine=selection_to_refine)
        with torch.no_grad():
            self.fcalc_dark = self.get_structure_factor_for_model(self.model_dark)
            self.Fcalc_dark = torch.abs(self.fcalc_dark)
            self.model = self.model_dark
            self.norm_weights()
            del self.model
        for i in tqdm(range(n_iter)):
            self.optimizer.zero_grad()
            loss = self.loss()
            if i % 10 == 0:
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

