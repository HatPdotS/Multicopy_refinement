"""

A class for scaling and post corrections of scattering factors.

Currently implements:
- Overall scale per resolution bin
- B-factor per resolution bin
- Anisotropy correction
- Solvent model correction

"""



import torch
import torch.nn as nn
from multicopy_refinement.Data import ReflectionData
from multicopy_refinement.math_torch import get_scattering_vectors, U_to_matrix, nll_xray, get_rfactors, bin_wise_rfactors, nll_xray_lognormal
from multicopy_refinement.solvent_new import SolventModel
from multicopy_refinement.debug_utils import DebugMixin
from multicopy_refinement.utils import ModuleReference

class Scaler(DebugMixin, nn.Module):
    def __init__(self, model, data: ReflectionData, nbins: int = 20, verbose: int = 1,device=torch.device('cpu')):
        """
        Scaler class to apply scaling and corrections to calculated structure factors.
        self.device = fcalc.device
        Args:
            fcalc (torch.Tensor): Calculated structure factors.
            fobs (torch.Tensor): Observed structure factors.
            hkl (torch.Tensor): Miller indices corresponding to the structure factors.
        """
        super(Scaler, self).__init__()
        self.device = device
        self.to(self.device)
        # Wrap model in ModuleReference to prevent registration as submodule
        self._model = ModuleReference(model)
        self._data = data

        self.nbins = nbins
        self.verbose = verbose
        self.cell = data.cell
        # Don't store hkl directly - always access it from data to avoid device mismatch
        # self.hkl will be a property that accesses data.hkl
        self.register_buffer('s',get_scattering_vectors(data.hkl, self.cell))
        bins, self.nbins = self._data.get_bins(self.nbins)
        self.register_buffer('bins', bins)
        if self.verbose > 0:
            print(f"Initialized Scaler with {self.nbins} bins.")
        self.frozen = False

    def initialize(self):
        self.calc_initial_scale()
        self.setup_solvent()
        self.setup_anisotropy_correction()

    @property
    def hkl(self):
        return self._data.hkl
    
    def freeze(self):
        self.frozen = True

    def unfreeze(self):
        self.frozen = False

    def calc_initial_scale(self):
        """
        Calculate the initial scale factor based on the ratio of observed to calculated structure factors.
        Excludes reflections with negative intensities to avoid bias from French-Wilson conversion.
        """
        hkl, fobs, sigma, rfree = self._data(mask=False)
        fcalc = self._model(hkl)
        if self.verbose > 0:
            print(f"Calculating initial scale factors using {self.nbins} bins.")
        assert torch.all(torch.isfinite(fcalc)), "Non-finite values found in fcalc during initial scale calculation."
        
        scales = torch.zeros(self.nbins, device=self.device, dtype=fobs.dtype)
        counts = torch.zeros(self.nbins, device=self.device, dtype=fobs.dtype)
        fcalc_amp = torch.abs(fcalc).to(fobs.dtype)
        # Exclude reflections with negative intensities from scale calculation
        # These have biased F values from French-Wilson conversion
        if hasattr(self._data, 'I') and self._data.I is not None:
            positive_mask = self._data.I > 0
            if self.verbose > 1:
                n_excluded = (~positive_mask).sum().item()
                print(f"Excluding {n_excluded} negative intensity reflections from scale calculation")
        else:
            positive_mask = torch.ones_like(fobs, dtype=torch.bool)
        
        # Calculate ratios only for positive intensity reflections

        mask = (self._data.masks() & rfree & positive_mask).to(torch.bool)
        bins = self.bins[mask].to(torch.int64)

        fobs = fobs.clamp(min=1e-3)[mask]
        fcalc_amp = fcalc_amp.clamp(min=1e-3)[mask]

        log_ratios = torch.log(fobs) - torch.log(fcalc_amp)
        assert torch.all(torch.isfinite(log_ratios)), f"Non-finite log ratios encountered in initial scale calculation {torch.sum(~torch.isfinite(log_ratios)).item()}"
    
        counts_vals = torch.ones_like(self.bins, device=self.device, dtype=fobs.dtype)
        sum_log_scales = torch.scatter_add(scales, 0, bins, log_ratios)
        counts = torch.scatter_add(counts, 0, bins, counts_vals)
        log_scale = sum_log_scales / (counts + 1e-6)
        initial_log_scale = log_scale
        if self.verbose > 1:
            print("Initial scale factors per bin:", initial_log_scale.detach().cpu().numpy())
        self.log_scale = nn.Parameter(initial_log_scale.detach())    
        return self.log_scale
    
    def setup_anisotropy_correction(self):
        self.U = nn.Parameter(torch.normal(0, 0.001, (6,), dtype=torch.float32, device=self.device))

    def anisotropy_correction(self):
        U = U_to_matrix(self.U)
        exp = -2 * torch.pi ** 2 * torch.einsum('ij,jk,ik->i', self.s, U, self.s)
        return torch.exp(exp)

    def fit_anisotropy(self,fcalc: torch.Tensor):
        if not hasattr(self, 'U'):
            self.U = nn.Parameter(torch.normal(0, 0.01, (6,), dtype=torch.float32, device=self.device))
        hkl, fobs, sigma, rfree = self._data()

        fobs = fobs.to(torch.float32).detach()

        fcalc = torch.abs(fcalc).to(torch.float32).detach()
        optimizer = torch.optim.Adam([self.U, self.log_scale], lr=1e-1)
        for i in range(100):
            optimizer.zero_grad()
            scaled_fcalc = self.forward(fcalc)
            loss = nll_xray(fobs[rfree], scaled_fcalc[rfree], sigma[rfree])

            loss.backward()
            optimizer.step()
            if self.verbose > 0 and (i % 10 == 0 or i == 99):
                print(f"Anisotropy fit iteration {i+1}/100, Loss: {loss.item():.4f}")

    def setup_solvent(self):
        self.solvent = SolventModel(self._model, device=self.device, radius=1.1, k_solvent=0.35, b_solvent=46.0, verbose=self.verbose)
        self.solvent.update_solvent()

    def fit_all_scales(self):
        hkl, fobs, sigma, rfree = self._data()
        fobs = fobs.to(torch.float32).detach()
        fcalc = self._model(hkl).detach()
        for lr in [1e-1, 5e-2, 1e-2]:
            optimizer = torch.optim.Adam(self.parameters(), lr=lr)
            for i in range(20):
                optimizer.zero_grad()
                scaled_fcalc = self.forward(fcalc)
                nll_loss = nll_xray(fobs[rfree], scaled_fcalc[rfree], sigma[rfree])
                if torch.isnan(nll_loss):
                    raise ValueError("NaN encountered in NLL loss during scale fitting.")
                nll_log_loss_xray = nll_xray_lognormal(fobs[rfree], scaled_fcalc[rfree], sigma[rfree])
                loss = nll_loss 
                loss.backward()
                optimizer.step()
            if self.verbose > 1: print(f"Solvent fit after step, Loss: {loss.item():.4f}, NLL: {nll_loss.item():.4f}, LogLoss: {nll_log_loss_xray.item():.4f}")
        
    def cuda(self, device=None):
        """
        Move the Scaler module to GPU.

        Args:
            device (torch.device, optional): The target device. If None, uses the default CUDA device.
        """
        super().cuda(device)
        if hasattr(self, 'solvent'):
            self.solvent.cuda(device)
        self.device = next(self.parameters()).device
        if self.verbose > 1:
            print(f"Scaler moved to device: {self.device}")

    def cpu(self):
        """
        Move the Scaler module to CPU.
        """
        super().cpu()
        if hasattr(self, 'solvent'):
            self.solvent.cpu()
        self.device = next(self.parameters()).device
        if self.verbose > 1:
            print("Scaler moved to CPU")

    def rfactor(self):
        """
        Calculate the R-factor between observed and calculated structure factors.

        Args:
            fcalc (torch.Tensor): Calculated structure factors.
        Returns:
            float: R-factor value.
        """
        hkl, fobs, _, rfree = self._data()
        fcalc = self._model(hkl)
        fcalc_scaled = self.forward(fcalc)
        return get_rfactors(torch.abs(fobs), torch.abs(fcalc_scaled), rfree)

    def bin_wise_rfactor(self, fcalc=None):
        """
        Calculate the bin-wise R-factor between observed and calculated structure factors.

        Args:
            fcalc (torch.Tensor): Calculated structure factors.

        Returns:
            tuple: Mean resolution per bin, R work per bin, R free per bin.
        """
        hkl, fobs, _, rfree = self._data()
        if fcalc is None:
            fcalc = self._model(hkl)
        fcalc_scaled = self.forward(fcalc)
        mean_res_per_bin = self._data.mean_res_per_bin()
        return mean_res_per_bin, *bin_wise_rfactors(torch.abs(fobs), torch.abs(fcalc_scaled), rfree, self.bins[self._data.masks()])

    def setup_bin_wise_bfactor(self):
        self.bin_wise_bfactor = nn.Parameter(torch.zeros(self.nbins, dtype=torch.float32, device=self.device))

    def bin_wise_bfactor_correction(self):
        b_expanded = self.bin_wise_bfactor[self.bins]
        s = torch.norm(self.s, dim=1) 
        s_squared = s ** 2  # Now s² is correct for B-factor formula
        exp = -b_expanded * s_squared / 4
        return torch.exp(exp)
    
    def get_binwise_mean_intensity(self):
        hkl, fobs, _, rfree = self._data()
        Fcalc = torch.abs(self(self._model(hkl)))
        intensities = torch.abs(fobs) ** 2
        calc_intensities = torch.abs(Fcalc) ** 2
        mean_obs_intensity = torch.zeros(self.nbins, device=self.device)
        mean_calc_intensity = torch.zeros(self.nbins, device=self.device)
        counts = torch.zeros(self.nbins, device=self.device)
        counts_vals = torch.ones_like(Fcalc, device=self.device, dtype=fobs.dtype)
        mask = self._data.get_mask()
        mean_obs_intensity = torch.scatter_add(mean_obs_intensity, 0, self.bins.to(torch.int64)[mask][rfree], intensities[rfree])
        mean_calc_intensity = torch.scatter_add(mean_calc_intensity, 0, self.bins.to(torch.int64)[mask][rfree], calc_intensities[rfree])
        counts = torch.scatter_add(counts, 0, self.bins.to(torch.int64)[mask][rfree], counts_vals[rfree])
        mean_obs_intensity = mean_obs_intensity / (counts + 1e-6)
        mean_calc_intensity = mean_calc_intensity / (counts + 1e-6)
        return mean_obs_intensity, mean_calc_intensity, self._data.mean_res_per_bin()

    def screen_solvent_params(self,steps=4):
        hkl, fobs, sigma, rfree = self._data()
        fobs = fobs.to(torch.float32).detach()
        fcalc = self._model(hkl).detach()
        best_log_k_solvent = self.solvent.log_k_solvent
        best_b_solvent = self.solvent.b_solvent
        best_loss = float('inf')
        ksol_start, ksol_end = torch.log(torch.tensor(0.1)), torch.log(torch.tensor(1.4))
        for log_k_solvent in torch.linspace(ksol_start, ksol_end, steps=steps):
            for b_solvent in torch.linspace(20.0, 120.0, steps=steps):
                self.solvent.log_k_solvent.data = log_k_solvent
                self.solvent.b_solvent.data = b_solvent
                scaled_fcalc = self.forward(fcalc)
                nll_loss = nll_xray(fobs[rfree], scaled_fcalc[rfree], sigma[rfree])
                if nll_loss.item() < best_loss:
                    best_loss = nll_loss.item()
                    best_log_k_solvent = log_k_solvent
                    best_b_solvent = b_solvent
        self.solvent.log_k_solvent.data = best_log_k_solvent
        self.solvent.b_solvent.data = best_b_solvent
        if self.verbose > 0:
            print(f"Optimal solvent parameters found: log_k_solvent={best_log_k_solvent.item()}, b_solvent={best_b_solvent.item()}, NLL Loss={best_loss:.4f}")

    def refine_lbfgs(self,
                     nsteps: int = 5,
                     lr: float = 1.0,
                     max_iter: int = 20,
                     history_size: int = 10,
                     verbose: bool = True):
        """
        Refine scale parameters using LBFGS optimizer.
        
        This method optimizes the anisotropic scaling and B-factor parameters
        that relate calculated structure factors to observed structure factors.
        Uses the L-BFGS quasi-Newton optimization method for fast convergence.
        
        Args:
            nsteps: Number of LBFGS steps
            lr: Learning rate (typically 1.0 for LBFGS)
            max_iter: Maximum iterations per line search
            history_size: Number of previous gradients to store for Hessian approximation
            verbose: Print progress information
            
        Returns:
            Dictionary with refinement metrics including steps, xray_work, xray_test, rwork, rfree
            
        Example:
            >>> scaler.unfreeze()
            >>> metrics = scaler.refine_lbfgs(nsteps=5, verbose=True)
            >>> scaler.freeze()
        """
        # Ensure scaler is unfrozen
        was_frozen = self.frozen
        if was_frozen:
            self.unfreeze()
        
        # Create LBFGS optimizer for scaler parameters only
        optimizer = torch.optim.LBFGS(
            self.parameters(),
            lr=lr,
            max_iter=max_iter,
            history_size=history_size
        )
        
        def closure():
            optimizer.zero_grad()
            fcalc_scaled = self.forward(fcalc)
            loss = nll_xray(fobs[rfree], fcalc_scaled[rfree], sigma[rfree])
            loss.backward(retain_graph=True)
            return loss
        
        # Track metrics
        metrics = {
            'target': 'scales',
            'steps': [],
            'xray_work': [],
            'xray_test': [],
            'rwork': [],
            'rfree': []
        }
        
        if verbose and self.verbose > 0:
            print("Refining scales with LBFGS...")

        hkl, fobs, sigma, rfree = self._data()
        fcalc = self._model(hkl).detach()
        
        # Run optimization
        for step in range(nsteps):
            optimizer.step(closure)
            
            # Evaluate metrics
            with torch.no_grad():
                hkl, fobs, sigma, rfree = self._data()
                fcalc_scaled = self.forward(fcalc)
                
                xray_work = nll_xray(fobs[rfree], fcalc_scaled[rfree], sigma[rfree])
                xray_test = nll_xray(fobs[~rfree], fcalc_scaled[~rfree], sigma[~rfree])
                rwork, rfree_val = get_rfactors(torch.abs(fobs), torch.abs(fcalc_scaled), rfree)
                
                metrics['steps'].append(step + 1)
                metrics['xray_work'].append(xray_work.item())
                metrics['xray_test'].append(xray_test.item())
                metrics['rwork'].append(rwork)
                metrics['rfree'].append(rfree_val)
                
                if verbose and self.verbose > 0:
                    print(f"  Step {step+1}/{nsteps}: "
                          f"Rwork={rwork:.4f}, Rfree={rfree_val:.4f}, "
                          f"NLL_work={xray_work.item():.2f}, NLL_test={xray_test.item():.2f}")
        
        # Restore frozen state
        if was_frozen:
            self.freeze()
        
        if verbose and self.verbose > 0:
            print("Scale refinement complete.\n")
        
        return metrics

    def parameters(self, recurse = True):
        if self.frozen:
            return []
        return super().parameters(recurse)

    def forward(self, fcalc, use_mask=True):
        """
        Forward pass for the Scaler module.

        Args:
            fcalc (torch.Tensor): Calculated structure factors.

        Returns:
            torch.Tensor: Scaled structure factors.
        """

        if use_mask:
            mask = self._data.masks().to(torch.bool)

        else:
            mask = torch.ones(fcalc.shape[0], dtype=torch.bool, device=self.device)

        if hasattr(self, 'U'):
            anisotropy_factors = self.anisotropy_correction()
            aniso_correction = anisotropy_factors[mask]
        else:
            aniso_correction = 1.0

        if hasattr(self, 'solvent'):
            f_sol = self.solvent(self.hkl)
            f_sol = f_sol[mask]
        else:
            f_sol = 0.0

        if hasattr(self, 'log_scale'):
            K_overall = torch.exp(self.log_scale[self.bins[mask]])
        else:
            K_overall = 1.0
        
        if hasattr(self, 'bin_wise_bfactor'):
            bfactor_factors = self.bin_wise_bfactor_correction()
            b_overall = bfactor_factors[mask]
        else:
            b_overall = 1.0
        
        fcalc = K_overall * b_overall * (aniso_correction * fcalc + f_sol)

        return fcalc
