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

class Scaler(nn.Module):
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
        self.model = model
        self.nbins = nbins
        self.data = data
        self.verbose = verbose
        self.cell = data.cell
        # Don't store hkl directly - always access it from data to avoid device mismatch
        # self.hkl will be a property that accesses data.hkl
        self.register_buffer('s',get_scattering_vectors(data.hkl, self.cell))
        bins, self.nbins = self.data.get_bins(self.nbins)
        self.register_buffer('bins', bins)
        if self.verbose > 0:
            print(f"Initialized Scaler with {self.nbins} bins.")
    
    @property
    def hkl(self):
        """Access hkl from data to ensure it's always on the correct device."""
        return self.data.hkl

    def initialize(self):
        self.setup_solvent()
        self.calc_initial_scale()
        self.setup_anisotropy_correction()

    def calc_initial_scale(self):
        """
        Calculate the initial scale factor based on the ratio of observed to calculated structure factors.
        Excludes reflections with negative intensities to avoid bias from French-Wilson conversion.
        """
        hkl, fobs, sigma, rfree = self.data(mask=False)
        fcalc = self.model(hkl)
        if self.verbose > 0:
            print(f"Calculating initial scale factors using {self.nbins} bins.")

        scales = torch.zeros(self.nbins, device=self.device, dtype=fobs.dtype)
        counts = torch.zeros(self.nbins, device=self.device, dtype=fobs.dtype)
        fcalc_amp = torch.abs(fcalc).to(fobs.dtype)
        # Exclude reflections with negative intensities from scale calculation
        # These have biased F values from French-Wilson conversion
        if hasattr(self.data, 'I') and self.data.I is not None:
            positive_mask = self.data.I > 0
            if self.verbose > 1:
                n_excluded = (~positive_mask).sum().item()
                print(f"Excluding {n_excluded} negative intensity reflections from scale calculation")
        else:
            positive_mask = torch.ones_like(fobs, dtype=torch.bool)
        
        # Calculate ratios only for positive intensity reflections
        log_ratios = torch.log(fobs + 1e-6) - torch.log(fcalc_amp + 1e-6)
        counts_vals = torch.ones_like(self.bins, device=self.device, dtype=fobs.dtype)
        mask = (self.data.get_mask() & rfree & positive_mask).to(torch.bool)
        sum_log_scales = torch.scatter_add(scales, 0, self.bins.to(torch.int64)[mask], log_ratios[mask])
        counts = torch.scatter_add(counts, 0, self.bins.to(torch.int64)[mask], counts_vals[mask])
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
        hkl, fobs, sigma, rfree = self.data()

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
        self.solvent = SolventModel(self.model, device=self.device, radius=1.1, k_solvent=0.35, b_solvent=46.0, verbose=self.verbose)
        self.solvent.update_solvent()

    def fit_all_scales(self):
        hkl, fobs, sigma, rfree = self.data()
        fobs = fobs.to(torch.float32).detach()
        fcalc = self.model(hkl).detach()
        for lr in [1e-1, 5e-2, 1e-2]:
            optimizer = torch.optim.Adam(self.parameters(), lr=lr)
            for i in range(20):
                optimizer.zero_grad()
                scaled_fcalc = self.forward(fcalc)
                nll_loss = nll_xray(fobs[rfree], scaled_fcalc[rfree], sigma[rfree])
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
        hkl, fobs, _, rfree = self.data()
        fcalc = self.model(hkl)
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
        hkl, fobs, _, rfree = self.data()
        if fcalc is None:
            fcalc = self.model(hkl)
        fcalc_scaled = self.forward(fcalc)
        mean_res_per_bin = self.data.mean_res_per_bin()
        return mean_res_per_bin, *bin_wise_rfactors(torch.abs(fobs), torch.abs(fcalc_scaled), rfree, self.bins[self.data.masks()])

    def setup_bin_wise_bfactor(self):
        self.bin_wise_bfactor = nn.Parameter(torch.zeros(self.nbins, dtype=torch.float32, device=self.device))

    def bin_wise_bfactor_correction(self):
        b_expanded = self.bin_wise_bfactor[self.bins]
        s = torch.norm(self.s, dim=1) / 2.0  # This is sin(θ)/λ
        s_squared = s ** 2  # Now s² is correct for B-factor formula
        exp = -b_expanded * s_squared
        return torch.exp(exp)
    
    def get_binwise_mean_intensity(self):
        hkl, fobs, _, rfree = self.data()
        Fcalc = torch.abs(self(self.model(hkl)))
        intensities = torch.abs(fobs) ** 2
        calc_intensities = torch.abs(Fcalc) ** 2
        mean_obs_intensity = torch.zeros(self.nbins, device=self.device)
        mean_calc_intensity = torch.zeros(self.nbins, device=self.device)
        counts = torch.zeros(self.nbins, device=self.device)
        counts_vals = torch.ones_like(Fcalc, device=self.device, dtype=fobs.dtype)
        mask = self.data.get_mask()
        mean_obs_intensity = torch.scatter_add(mean_obs_intensity, 0, self.bins.to(torch.int64)[mask][rfree], intensities[rfree])
        mean_calc_intensity = torch.scatter_add(mean_calc_intensity, 0, self.bins.to(torch.int64)[mask][rfree], calc_intensities[rfree])
        counts = torch.scatter_add(counts, 0, self.bins.to(torch.int64)[mask][rfree], counts_vals[rfree])
        mean_obs_intensity = mean_obs_intensity / (counts + 1e-6)
        mean_calc_intensity = mean_calc_intensity / (counts + 1e-6)
        return mean_obs_intensity, mean_calc_intensity, self.data.mean_res_per_bin()

    def screen_solvent_params(self):
        hkl, fobs, sigma, rfree = self.data()
        fobs = fobs.to(torch.float32).detach()
        fcalc = self.model(hkl).detach()
        best_log_k_solvent = self.solvent.log_k_solvent
        best_b_solvent = self.solvent.b_solvent
        best_loss = float('inf')
        ksol_start, ksol_end = torch.log(torch.tensor(0.1)), torch.log(torch.tensor(1.0))
        for log_k_solvent in torch.linspace(ksol_start, ksol_end, steps=4):
            for b_solvent in torch.linspace(20.0, 80.0, steps=4):
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

    def forward(self, fcalc, use_mask=True):
        """
        Forward pass for the Scaler module.

        Args:
            fcalc (torch.Tensor): Calculated structure factors.

        Returns:
            torch.Tensor: Scaled structure factors.
        """
        if use_mask:
            mask = self.data.masks().to(torch.bool)
        else:
            mask = torch.ones(fcalc.shape[0], dtype=torch.bool, device=self.device)
        if hasattr(self, 'solvent'):
            f_sol = self.solvent(self.hkl)
            fcalc = fcalc + f_sol[mask]
        corrections = torch.ones_like(fcalc, device=self.device)
        if hasattr(self, 'log_scale'):
            corrections = corrections * torch.exp(self.log_scale[self.bins[mask]])
        if hasattr(self, 'U'):
            anisotropy_factors = self.anisotropy_correction()
            corrections = corrections * anisotropy_factors[mask]
        if hasattr(self, 'bin_wise_bfactor'):
            bfactor_factors = self.bin_wise_bfactor_correction()
            corrections = corrections * bfactor_factors[mask]
        return fcalc * corrections