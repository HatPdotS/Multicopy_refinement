from multicopy_refinement import restraints
from multicopy_refinement.Data import ReflectionData
from multicopy_refinement.model_ft import ModelFT
from torch.nn import Module as nnModule
from multicopy_refinement.kinetics import KineticModel
import torch
from  multicopy_refinement.restraints import Restraints
from multicopy_refinement.math_torch import nll_xray,log_loss
from multicopy_refinement.solvent_new import SolventModel
from multicopy_refinement.scaler import Scaler

class Refinement(nnModule):
    def __init__(self, mtz_file, pdb,  cif= None, verbose=1,max_res=None,device=torch.device('cpu')):
        """
        Refinement class to handle the overall refinement process.
        Args:
            mtz_file (str): Path to the MTZ file containing reflection data.cd 
            pdb (str): Path to the PDB file containing the initial model.
            cif (str, optional): Path to the CIF file for restraints. Defaults to None.
            verbose (int, optional): Verbosity level. Defaults to 1.
            max_res (float, optional): Maximum resolution for reflections. Defaults to None.
        """
        super().__init__()
        self.device = device
        self.to(self.device)
        self.verbose = verbose
        self.mtz_file = mtz_file
        self.pdb = pdb
        self.reflection_data = ReflectionData(verbose=self.verbose)
        self.reflection_data.load_from_mtz(mtz_file)
        if max_res:
            self.reflection_data = self.reflection_data.cut_res(max_res)
            self.max_res = max_res
        else:
            self.max_res = self.reflection_data.get_max_res()
        self.model = ModelFT(verbose=self.verbose,max_res=self.max_res,device=self.device)
        self.model.load_pdb_from_file(pdb)
        self.scaler = Scaler(self.model, self.reflection_data, verbose=self.verbose, device=self.device)
        self.target_weights = {'xray': 1.0, 'restraints': 0.5}
        self.lr = 1e-3
        self.restraints = Restraints(self.model, cif, self.verbose)
        self.target_ratio = 1.3  # Target ratio of XRay test to work NLL
        self.balance_weights()
    
    def get_scales(self):
        self.scaler.initialize()
        self.reflection_data.find_outliers(self.model, self.scaler, z_threshold=4.0)
        self.scaler.fit_all_scales()
        self.reflection_data.find_outliers(self.model, self.scaler, z_threshold=4.0)

    def setup_scaler(self):
        self.scaler = Scaler(self.model, self.reflection_data)

    def parameters(self):
        parameters = list(self.model.parameters())
        parameters += list(self.scaler.parameters())
        return parameters
    
    def get_fcalc(self,hkl=None):
        if hkl is None:
            hkl, _,_, _ = self.reflection_data()
        return self.model(hkl)
    
    def get_fcalc_scaled(self,hkl=None):
        fcalc = self.get_fcalc(hkl)
        fcalc_scaled = self.scaler(fcalc)
        return fcalc_scaled

    def get_Fcalc(self,hkl=None):
        return torch.abs(self.get_fcalc(hkl))

    def get_F_calc_scaled(self,hkl=None):
        return torch.abs(self.get_fcalc_scaled(hkl))

    def nll_xray(self):
        """
        Compute X-ray negative log-likelihood for Rfree test set only.
        
        Assumes Gaussian distribution: P(F_obs | F_calc, σ) ∝ exp(-0.5*(F_obs - |F_calc|)²/σ²)
        NLL = 0.5*(F_obs - |F_calc|)²/σ² + log(σ) + 0.5*log(2π)
        
        Returns:
            torch.Tensor: Total NLL summed over Rwork reflections (1 is work, 0 is test)
        """
        hkl, F_obs, sigma_F_obs, rfree_mask = self.reflection_data()
        Fcalc_all = self.get_F_calc_scaled(hkl)

        
        # Filter to only Rfree reflections (test set)
        F_obs_work = F_obs[rfree_mask]
        sigma_F_obs_work = sigma_F_obs[rfree_mask]
        F_calc = Fcalc_all[rfree_mask]

        F_obs_test = F_obs[~rfree_mask]
        sigma_F_obs_test = sigma_F_obs[~rfree_mask]
        F_calc_test = Fcalc_all[~rfree_mask]

        return nll_xray(F_obs_work, F_calc, sigma_F_obs_work), nll_xray(F_obs_test, F_calc_test, sigma_F_obs_test)

    def loss(self):
        xray_work, xray_test = self.nll_xray()
        restraints = self.restraints.loss()
        total_loss = self.effective_weights['xray'] * xray_work + self.effective_weights['restraints'] * restraints
        return total_loss, xray_work, restraints, xray_test

    def balance_weights(self):
        self.effective_weights = {}
        with torch.no_grad():
            xray_work, xray_test = self.nll_xray()
            restraints = self.restraints.loss()
            xray_base = 10 / torch.clamp(xray_work, min=5).item() * self.target_weights['xray']
            restraints_base = 10 / torch.clamp(restraints, min=5).item() * self.target_weights['restraints']
            self.effective_weights['xray'] = xray_base
            self.effective_weights['restraints'] = restraints_base

    def update_weights(self):
        target_ratio = self.target_ratio
        with torch.no_grad():
            xray_work, xray_test = self.nll_xray()
            effective_ratio = xray_test / xray_work 
            current_ratio = self.target_weights['xray'] / self.target_weights['restraints']
            if current_ratio < 0.1 or current_ratio > 10.0:
                if self.verbose > 0:
                    print("Warning: Extreme weight ratio detected, skipping weight update to maintain stability.")
                    print(f"  Current Ratio: {current_ratio:.4f}, Effective Ratio: {effective_ratio.item():.4f}, Target Ratio: {target_ratio:.4f}")
                return
            if effective_ratio > target_ratio:
                self.target_weights['restraints'] *= 1.1
            if effective_ratio < target_ratio:
                self.target_weights['restraints'] *= 0.9
            self.balance_weights()

    def setup_optimizer(self, **kwargs):
        from torch.optim import Adam
        self.optimizer = Adam(self.parameters(), **kwargs)

    def run_refinement(self, macro_cycles=5, n_steps=10, lr=[1e-2,5e-4,1e-3, 5e-4, 1e-4]):
        self.setup_optimizer(lr=lr[0])
        for cycle in range(macro_cycles):
            self.get_scales()
            self.balance_weights()
            if self.verbose > 0:
                print(f"Starting macro cycle {cycle+1}/{macro_cycles} with learning rate {self.lr if isinstance(self.lr, float) else self.lr[cycle]}")
            for _lr in lr:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = _lr
                for step in range(n_steps):
                    self.optimizer.zero_grad()
                    total_loss, xray_work, restraints, xray_test = self.loss()
                    total_loss.backward()
                    self.optimizer.step()
                    if self.verbose > 2:
                        print(f"  Step {step+1}/{n_steps}, Total Loss: {total_loss.item():.4f}, XRay Work NLL: {xray_work.item():.4f}, Restraints Loss: {restraints.item():.4f}, XRay Test NLL: {xray_test.item():.4f}")
                self.balance_weights()
                if self.verbose > 1:
                    print(f"  Ran for {_lr}, Total Loss: {total_loss.item():.4f}, XRay Work NLL: {xray_work.item():.4f}, Restraints Loss: {restraints.item():.4f}, XRay Test NLL: {xray_test.item():.4f}")
            if self.verbose > 0:
                rwork, rfree = self.get_rfactor()
                print(f"  R-work: {rwork:.4f}, R-free: {rfree:.4f}")
                print(f"Completed macro cycle {cycle+1}/{macro_cycles}. Updated weights: {self.target_weights}")

    def cuda(self):
        super().cuda()
        self.model.cuda()  # Explicitly call cuda on model to update its device attributes
        self.reflection_data.cuda()
        self.scaler.cuda() if hasattr(self.scaler, 'cuda') else None  # Also update scaler if it has cuda method
        self.restraints.cuda() if hasattr(self.restraints, 'cuda') else None  # Also update restraints if it has cuda method
        self.device = torch.device('cuda')
        return self
    
    def cpu(self):
        super().cpu()
        self.model.cpu()  # Explicitly call cpu on model to update its device attribute
        self.scaler.cpu() if hasattr(self.scaler, 'cpu') else None  # Also update scaler if it has cpu method
        self.restraints.cpu() if hasattr(self.restraints, 'cpu') else None  # Also update restraints if it has cpu method
        self.device = torch.device('cpu')
        return self

    def get_rfactor(self):
        return self.scaler.rfactor()

    def update_outliers(self, z_threshold=4.0):
        with torch.no_grad():
            self.reflection_data = self.reflection_data.update_outliers(self.model, self.scaler, z_threshold=z_threshold)
            self.register_buffer('hkl', self.reflection_data.get_hkl())
            self.setup_scaler()

    def plot_fcalc_vs_fobs(self,outpath='fcalc_vs_fobs.png'):
        import matplotlib.pyplot as plt
        with torch.no_grad():
            hkl, F_obs, sigma_F_obs, self.rfree_flags = self.reflection_data()
            self.get_Fcalc()
            F_calc = self.F_calc
            F_obs_amp = torch.abs(F_obs).cpu().numpy()
            F_calc_amp = torch.abs(F_calc).cpu().numpy()
            plt.figure(figsize=(8,8))
            plt.scatter(F_obs_amp, F_calc_amp, alpha=0.5)
            plt.plot([0, max(F_obs_amp)], [0, max(F_obs_amp)], color='red', linestyle='--')
            plt.xlabel('Observed |F|')
            plt.ylabel('Calculated |F|')
            plt.title('Fcalc vs Fobs')
            plt.grid()
            plt.savefig(outpath)
    
    def write_out_mtz(self, out_mtz_path='refined_output.mtz'):
        import reciprocalspaceship as rs
        with torch.no_grad():
            hkl, F_obs, sigma_F_obs, self.rfree_flags = self.reflection_data()
            Iobs = self.reflection_data.I
            Sigma_Iobs = self.reflection_data.I_sigma
            self.get_F_calc_scaled()
            F_calc = self.F_calc_scaled
            # Create a new ReflectionData object to hold the output
            diff = F_obs - F_calc
            twofofc = 2 * F_obs - F_calc
            phases = torch.angle(self.f_calc).rad2deg()
            data = rs.DataSet({'H': hkl[:,0].cpu().numpy(),
                         'K': hkl[:,1].cpu().numpy(),
                         'L': hkl[:,2].cpu().numpy(),
                        'I-obs': Iobs.cpu().numpy(),
                            'SIGI-obs': Sigma_Iobs.cpu().numpy() if Sigma_Iobs is not None else torch.ones_like(Iobs).cpu().numpy(),
                         'F-obs': F_obs.cpu().numpy(),
                         'SIGF-obs': sigma_F_obs.cpu().numpy(),
                         'F-model': F_calc.cpu().numpy(),
                         'PHIF-model': phases.cpu().numpy(),
                         'FOFC': diff.cpu().numpy(),
                         '2FOFC': twofofc.cpu().numpy(),
                         'PH2FOFC': phases.cpu().numpy(),
                         'PHFOFC': (phases).cpu().numpy(),
                            'R-free-flags': self.rfree_flags.cpu().numpy().astype(int)})
            data = data.set_index(['H','K','L'])
            data = data.infer_mtz_dtypes()
            data.cell = self.reflection_data.dataset.cell
            data.spacegroup = self.reflection_data.dataset.spacegroup
            data.write_mtz(out_mtz_path)
