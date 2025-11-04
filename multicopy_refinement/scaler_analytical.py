"""
Analytical Bulk-Solvent and Isotropic Scaling

Implementation of the analytical bulk-solvent and isotropic scaling method from:
Afonine et al. (2013) Acta Cryst. D69, 625–634
(corrected by 2023 addendum: Afonine et al., Acta Cryst. D79, 666–667)

This provides closed-form solutions for bulk-solvent scale factor k_mask and
overall isotropic scale parameter K, avoiding iterative optimization.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple, Dict
from multicopy_refinement.model_ft import ModelFT
from multicopy_refinement.Data import ReflectionData
from multicopy_refinement.solvent import SolventModel
from multicopy_refinement.math_torch import get_scattering_vectors, U_to_matrix, nll_xray


class AnalyticalScaler(nn.Module):
    """
    Analytical bulk-solvent and isotropic scaling.
    
    Upon initialization, computes optimal k_mask and K parameters per resolution shell
    using analytical formulas (cubic equation solution). Then applies these scales
    in the forward pass.
    
    The model structure factor is:
        F_model(s) = k_total(s) * [F_calc(s) + k_mask(s) * F_mask(s)]
    
    where:
        k_total(s) = k_overall * k_anisotropic(s) * k_isotropic(s)
        k_isotropic(s) = 1 / sqrt(K(s))
    """
    
    def __init__(
        self,
        model_ft: ModelFT,
        reflection_data: ReflectionData,
        solvent_model: SolventModel,
        n_bins: int = 20,
        k_overall: float = 1.0,
        k_anisotropic: Optional[torch.Tensor] = None,
        verbose: int = 1
    ):
        """
        Initialize analytical scaler and compute scaling parameters.
        
        Args:
            model_ft: ModelFT object for computing F_calc (unscaled)
            reflection_data: ReflectionData object containing F_obs, hkl, cell, etc.
            solvent_model: SolventModel object for computing F_mask
            n_bins: Number of resolution bins for analytical optimization
            k_overall: Overall scale factor (default: 1.0)
            k_anisotropic: Per-reflection anisotropic scale (optional)
            verbose: Verbosity level (0=silent, 1=normal, 2=detailed)
        """
        super(AnalyticalScaler, self).__init__()
        
        self.verbose = verbose
        self.model_ft = model_ft
        self.reflection_data = reflection_data
        self.solvent_model = solvent_model
        self.n_bins = n_bins
        
        # Store overall and anisotropic scales
        self.k_overall = torch.tensor(k_overall, dtype=torch.float32)
        self.k_anisotropic = k_anisotropic
        
        # Validate inputs
        if reflection_data.F is None:
            raise ValueError("ReflectionData must contain F_obs (amplitude data)")
        if reflection_data.hkl is None:
            raise ValueError("ReflectionData must contain hkl indices")
        if reflection_data.cell is None:
            raise ValueError("ReflectionData must contain unit cell parameters")
        
        # Get data from reflection_data
        self.hkl = reflection_data.hkl
        self.F_obs = reflection_data.F
        self.cell = reflection_data.cell
        
        # Compute scattering vectors
        self.scattering_vectors = get_scattering_vectors(self.hkl, self.cell)
        
        # Storage for computed scales (will be filled during initialization)
        self.kmask_per_ref: Optional[torch.Tensor] = None
        self.K_per_ref: Optional[torch.Tensor] = None
        self.bin_info: Optional[Dict] = None
        
        if self.verbose > 0:
            print("\n" + "="*70)
            print("Analytical Bulk-Solvent and Isotropic Scaling")
            print("="*70)
            print(f"Number of reflections: {len(self.hkl)}")
            print(f"Resolution bins: {n_bins}")
        
        # Compute scaling parameters analytically
        self._compute_analytical_scales()
        
        if self.verbose > 0:
            print("="*70)
            print("Analytical scaling initialization complete")
            print("="*70 + "\n")
    
    def _compute_analytical_scales(self):
        """
        Compute k_mask and K analytically per resolution bin.
        
        This is the core implementation of the Afonine et al. (2013) method.
        """
        if self.verbose > 0:
            print("\nComputing analytical bulk-solvent scales...")
        
        # Get unscaled structure factors
        with torch.no_grad():
            fcalc = self.model_ft.get_structure_factor(self.hkl, recalc=False)
            fmask = self.solvent_model.get_rec_solvent(self.hkl)
        
        # Create resolution bins with equal number of reflections per bin
        # using the ReflectionData object's get_bins method
        bin_indices, actual_n_bins = self.reflection_data.get_bins(
            n_bins=self.n_bins, 
            min_per_bin=100
        )
        
        # Update n_bins to actual number created
        self.n_bins = actual_n_bins
        
        # Compute s = sin(theta)/lambda for bin statistics
        s = torch.sqrt(torch.sum(self.scattering_vectors**2, dim=1)) / 2.0
        
        # Initialize per-reflection arrays
        self.kmask_per_ref = torch.zeros_like(s)
        self.K_per_ref = torch.ones_like(s)
        
        # Store bin information
        self.bin_info = {
            'bin_centers': [],
            'kmask_values': [],
            'K_values': [],
            'n_reflections': [],
            'resolution_min': [],
            'resolution_max': []
        }
        
        # Process each bin
        for bin_idx in range(self.n_bins):
            mask = (bin_indices == bin_idx)
            n_ref_in_bin = mask.sum().item()
            
            if n_ref_in_bin < 2:
                # Skip bins with too few reflections
                if self.verbose > 1:
                    print(f"  Bin {bin_idx+1}/{self.n_bins}: skipped (only {n_ref_in_bin} reflections)")
                continue
            
            # Extract data for this bin
            fcalc_bin = fcalc[mask]
            fmask_bin = fmask[mask]
            Fobs_bin = self.F_obs[mask]
            s_bin = s[mask]
            
            # Apply k_overall and k_anisotropic if provided
            if self.k_anisotropic is not None:
                k_anis_bin = self.k_anisotropic[mask]
                I_s = (self.k_overall * k_anis_bin * Fobs_bin) ** 2
            else:
                I_s = (self.k_overall * Fobs_bin) ** 2
            
            # Compute u, v, w for each reflection in bin
            # u = |F_calc|^2
            # v = Re(F_calc * F_mask*)
            # w = |F_mask|^2
            u_s = torch.abs(fcalc_bin) ** 2
            v_s = torch.real(fcalc_bin * torch.conj(fmask_bin))
            w_s = torch.abs(fmask_bin) ** 2
            
            # Compute bin-level sums (Eqs. in paper)
            # Note: subscripts are just indices, not exponents!
            A_2 = torch.sum(u_s)
            B_2 = torch.sum(2 * v_s)
            C_2 = torch.sum(w_s)
            D_3 = torch.sum(v_s * w_s)
            Y_2 = torch.sum(I_s)
            Y_3 = torch.sum(I_s * v_s)
            
            # Additional sums for cubic equation
            E_4 = torch.sum(w_s ** 2)
            F_4 = torch.sum(2 * v_s * w_s)
            G_4 = torch.sum(v_s ** 2)
            
            # Form cubic equation: k_mask^3 + a*k_mask^2 + b*k_mask + c = 0
            # Following corrected formulas from 2023 addendum
            
            # Numerator and denominator for cubic coefficients
            denom = Y_2 * E_4 - Y_3 * F_4 / 2
            
            if torch.abs(denom) < 1e-10:
                # Degenerate case - fall back to k_mask = 0
                kmask_bin = 0.0
                K_bin = A_2 / Y_2 if Y_2 > 0 else 1.0
            else:
                # Cubic coefficients
                a = (Y_2 * F_4 - 2 * Y_3 * E_4) / (2 * denom)
                b = (Y_2 * G_4 + Y_2 * C_2 - Y_3 * B_2 - Y_3 * D_3) / denom
                c = (Y_2 * D_3 - Y_3 * C_2) / denom
                
                # Solve cubic equation
                kmask_bin, K_bin = self._solve_cubic_and_select_best(
                    a, b, c, A_2, B_2, C_2, Y_2,
                    u_s, v_s, w_s, I_s
                )
            
            # Store results for this bin
            self.kmask_per_ref[mask] = kmask_bin
            self.K_per_ref[mask] = K_bin
            
            # Record bin info
            s_mean = s_bin.mean().item()
            d_mean = 1.0 / (2.0 * s_mean) if s_mean > 0 else float('inf')
            d_min = 1.0 / (2.0 * s_bin.max().item()) if s_bin.max() > 0 else float('inf')
            d_max = 1.0 / (2.0 * s_bin.min().item()) if s_bin.min() > 0 else float('inf')
            
            self.bin_info['bin_centers'].append(s_mean)
            self.bin_info['kmask_values'].append(kmask_bin)
            self.bin_info['K_values'].append(K_bin)
            self.bin_info['n_reflections'].append(n_ref_in_bin)
            self.bin_info['resolution_min'].append(d_min)
            self.bin_info['resolution_max'].append(d_max)
            
            if self.verbose > 1:
                print(f"  Bin {bin_idx+1}/{self.n_bins}: "
                      f"{n_ref_in_bin:5d} refs, "
                      f"d={d_mean:.2f}Å, "
                      f"k_mask={kmask_bin:.4f}, "
                      f"K={K_bin:.4f}")
        
        if self.verbose > 0:
            valid_bins = len([k for k in self.bin_info['kmask_values'] if k != 0])
            print(f"\nAnalytical optimization complete:")
            print(f"  Valid bins: {valid_bins}/{self.n_bins}")
            print(f"  k_mask range: {min(self.bin_info['kmask_values']):.4f} - {max(self.bin_info['kmask_values']):.4f}")
            print(f"  K range: {min(self.bin_info['K_values']):.4f} - {max(self.bin_info['K_values']):.4f}")
    
    def _solve_cubic_and_select_best(
        self,
        a: float, b: float, c: float,
        A_2: float, B_2: float, C_2: float, Y_2: float,
        u_s: torch.Tensor, v_s: torch.Tensor, w_s: torch.Tensor, I_s: torch.Tensor
    ) -> Tuple[float, float]:
        """
        Solve cubic equation and select best positive root.
        
        Args:
            a, b, c: Cubic equation coefficients (k^3 + a*k^2 + b*k + c = 0)
            A_2, B_2, C_2, Y_2: Bin-level sums for computing K
            u_s, v_s, w_s, I_s: Per-reflection quantities for evaluating loss
            
        Returns:
            kmask_bin: Selected k_mask value
            K_bin: Corresponding K value
        """
        # Convert to numpy for root finding
        coeffs = np.array([1.0, a.item(), b.item(), c.item()])
        roots = np.roots(coeffs)
        
        # Filter for real, positive roots
        positive_real_roots = []
        for root in roots:
            if np.isreal(root) and np.real(root) > 0:
                positive_real_roots.append(np.real(root))
        
        if len(positive_real_roots) == 0:
            # No positive roots - fall back to k_mask = 0
            kmask_bin = 0.0
            K_bin = A_2 / Y_2 if Y_2 > 0 else 1.0
        else:
            # Evaluate loss for each candidate
            best_loss = float('inf')
            best_kmask = 0.0
            best_K = 1.0
            
            for kmask_candidate in positive_real_roots:
                # Skip unrealistic k_mask values (should be order 0-1, max ~5)
                if kmask_candidate > 5.0:
                    continue
                    
                # Compute K from k_mask
                K_candidate = (kmask_candidate**2 * C_2 + kmask_candidate * B_2 + A_2) / Y_2
                K_candidate = max(K_candidate, 1e-6)  # Ensure positive
                
                # Evaluate loss: sum[(|F_calc + k_mask*F_mask|^2 - K*I_s)^2]
                model_intensity = (u_s + 2 * kmask_candidate * v_s + kmask_candidate**2 * w_s)
                residual = model_intensity - K_candidate * I_s
                loss = torch.sum(residual ** 2).item()
                
                if loss < best_loss:
                    best_loss = loss
                    best_kmask = kmask_candidate
                    best_K = K_candidate
            
            # Also evaluate k_mask=0 as a candidate
            K_zero = A_2 / Y_2 if Y_2 > 0 else 1.0
            model_intensity_zero = u_s
            residual_zero = model_intensity_zero - K_zero * I_s
            loss_zero = torch.sum(residual_zero ** 2).item()
            
            if loss_zero < best_loss:
                best_loss = loss_zero
                best_kmask = 0.0
                best_K = K_zero
            
            kmask_bin = best_kmask
            K_bin = best_K
        
        return kmask_bin, K_bin
    
    def forward(self, fcalc: Optional[torch.Tensor] = None, hkl: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Apply analytical scaling to structure factors.
        
        Args:
            fcalc: Calculated structure factors (optional, will compute from model if None)
            hkl: Miller indices (optional, uses stored hkl if None)
            
        Returns:
            F_model: Scaled model structure factors
        """
        # Determine which hkl to use
        if hkl is None:
            hkl_to_use = self.hkl
            kmask = self.kmask_per_ref
            K = self.K_per_ref
            # Compute anisotropic correction fresh each time (allows gradients to flow during training)
            k_anis = self.compute_anisotropy_correction() if hasattr(self, 'U') else self.k_anisotropic
        else:
            # Need to compute scales for new hkl set
            # For simplicity, we'll use nearest-neighbor interpolation based on resolution
            s_query = torch.sqrt(torch.sum(
                get_scattering_vectors(hkl, self.cell)**2, dim=1
            )) / 2.0
            s_ref = torch.sqrt(torch.sum(self.scattering_vectors**2, dim=1)) / 2.0
            
            # Find nearest reference reflection for each query
            # This is a simple approach - could be improved with interpolation
            kmask = torch.zeros(len(hkl), dtype=torch.float32)
            K = torch.ones(len(hkl), dtype=torch.float32)
            
            for i, s_val in enumerate(s_query):
                nearest_idx = torch.argmin(torch.abs(s_ref - s_val))
                kmask[i] = self.kmask_per_ref[nearest_idx]
                K[i] = self.K_per_ref[nearest_idx]
            
            k_anis = None  # Don't have anisotropic for new hkl
            hkl_to_use = hkl
        
        # Get F_calc if not provided
        if fcalc is None:
            fcalc = self.model_ft.get_structure_factor(hkl_to_use, recalc=False)
        
        # Get F_mask
        fmask = self.solvent_model.get_rec_solvent(hkl_to_use)
        
        # Apply bulk-solvent correction: F_calc + k_mask * F_mask
        fcalc_with_solvent = fcalc + kmask.to(fcalc.device) * fmask
        
        # Apply isotropic scale
        # The analytical method minimizes: sum[(|F_calc+k_mask*F_mask|^2 - K*I_obs)^2]
        # This means: |F_model|^2 ≈ K * I_obs = K * F_obs^2
        # Therefore: |F_model| ≈ sqrt(K) * F_obs
        # But we want: F_model_scaled ≈ F_obs
        # So we need to scale down by 1/sqrt(K)
        k_isotropic = 1.0 / torch.sqrt(K.to(fcalc.device) + 1e-10)  # Add small epsilon for stability
        
        # Apply overall and anisotropic scales
        if k_anis is not None:
            k_total = self.k_overall * k_anis.to(fcalc.device) * k_isotropic
        else:
            k_total = self.k_overall * k_isotropic
        
        # Final scaled model structure factors
        F_model = k_total * fcalc_with_solvent
        
        return F_model
    
    def get_scaling_statistics(self) -> Dict:
        """
        Get statistics about the computed scaling parameters.
        
        Returns:
            Dictionary with scaling statistics
        """
        stats = {
            'n_bins': self.n_bins,
            'n_reflections': len(self.hkl),
            'kmask_mean': torch.mean(self.kmask_per_ref).item(),
            'kmask_std': torch.std(self.kmask_per_ref).item(),
            'kmask_min': torch.min(self.kmask_per_ref).item(),
            'kmask_max': torch.max(self.kmask_per_ref).item(),
            'K_mean': torch.mean(self.K_per_ref).item(),
            'K_std': torch.std(self.K_per_ref).item(),
            'K_min': torch.min(self.K_per_ref).item(),
            'K_max': torch.max(self.K_per_ref).item(),
            'bin_info': self.bin_info
        }
        return stats
    
    def print_statistics(self):
        """Print scaling statistics."""
        stats = self.get_scaling_statistics()
        
        print("\n" + "="*70)
        print("Analytical Scaling Statistics")
        print("="*70)
        print(f"Total reflections: {stats['n_reflections']}")
        print(f"Resolution bins: {stats['n_bins']}")
        print(f"\nBulk-solvent scale (k_mask):")
        print(f"  Mean: {stats['kmask_mean']:.4f}")
        print(f"  Std:  {stats['kmask_std']:.4f}")
        print(f"  Min:  {stats['kmask_min']:.4f}")
        print(f"  Max:  {stats['kmask_max']:.4f}")
        print(f"\nIsotropic scale parameter (K):")
        print(f"  Mean: {stats['K_mean']:.4f}")
        print(f"  Std:  {stats['K_std']:.4f}")
        print(f"  Min:  {stats['K_min']:.4f}")
        print(f"  Max:  {stats['K_max']:.4f}")
        print("="*70 + "\n")
    
    def setup_anisotropy(self):
        """Initialize anisotropic U matrix parameters."""
        if not hasattr(self, 'U'):
            device = self.kmask_per_ref.device
            self.U = nn.Parameter(torch.normal(0, 0.01, (6,), dtype=torch.float32, device=device))
    
    def compute_anisotropy_correction(self) -> torch.Tensor:
        """
        Compute anisotropic B-factor correction exp(-2π²·s^T·U·s).
        
        Returns:
            Anisotropic scale factors for each reflection
        """
        if not hasattr(self, 'U'):
            return torch.ones(len(self.hkl), dtype=torch.float32, device=self.kmask_per_ref.device)
        
        U_matrix = U_to_matrix(self.U)
        s = self.scattering_vectors
        
        # Compute anisotropic correction: exp(-2π²·s^T·U·s)
        exponent = -2 * torch.pi ** 2 * torch.einsum('ij,jk,ik->i', s, U_matrix, s)
        return torch.exp(exponent)
    
    def fit_anisotropy(self, n_iterations: int = 50, learning_rate: float = 1e-1):
        """
        Fit anisotropic B-factor correction using iterative optimization.
        
        This refines the 6-parameter anisotropic U matrix to minimize the
        difference between observed and calculated structure factors.
        
        Args:
            n_iterations: Number of optimization iterations
            learning_rate: Learning rate for Adam optimizer
            
        Returns:
            Tuple of (r_work_history, r_free_history) lists tracking R-factors
        """
        if self.verbose > 0:
            print("\n" + "="*70)
            print("Fitting Anisotropic B-factor Correction")
            print("="*70)
        
        # Setup anisotropic parameters
        self.setup_anisotropy()
        
        # Get observed data
        _, F_obs, sigma, rfree = self.reflection_data()
        F_obs = F_obs.to(torch.float32).detach()
        sigma = sigma.to(torch.float32).detach()
        rfree = rfree.to(torch.bool)
        
        # Compute F_calc with current bulk-solvent scales
        with torch.no_grad():
            F_calc = self.model_ft.get_structure_factor(self.hkl, recalc=False)
        
        # Setup optimizer
        optimizer = torch.optim.Adam([self.U], lr=learning_rate)
        
        # Track R-factors
        r_work_history = []
        r_free_history = []
        
        for i in range(n_iterations):
            optimizer.zero_grad()
            
            # Apply analytical scaling with current anisotropic correction
            F_model = self.forward(F_calc)
            F_model_amp = torch.abs(F_model).to(torch.float32)
            
            # Compute loss on work set only
            loss = nll_xray(F_obs[rfree], F_model_amp[rfree], sigma[rfree])
            
            loss.backward()
            optimizer.step()
            
            if self.verbose > 0 and (i % 10 == 0 or i == n_iterations - 1):
                # Compute R-factors for monitoring
                with torch.no_grad():
                    from multicopy_refinement.math_torch import rfactor
                    r_work = rfactor(F_obs[rfree], F_model_amp[rfree])
                    r_free = rfactor(F_obs[~rfree], F_model_amp[~rfree])
                    r_work_history.append(r_work)
                    r_free_history.append(r_free)
                print(f"  Iteration {i+1}/{n_iterations}: Loss={loss.item():.4f}, "
                      f"R-work={r_work:.4f}, R-free={r_free:.4f}")
        
        # Update k_anisotropic for use in forward pass
        with torch.no_grad():
            self.k_anisotropic = self.compute_anisotropy_correction()
        
        if self.verbose > 0:
            print("="*70)
            print("Anisotropic fitting complete")
            print("="*70 + "\n")
        
        return r_work_history, r_free_history
    
    def refine_with_anisotropy(self, n_macro_cycles: int = 2, 
                               n_aniso_iterations: int = 50,
                               learning_rate: float = 1e-1):
        """
        Iteratively refine analytical scales and anisotropic correction.
        
        This performs macro-cycles of:
        1. Re-compute analytical k_mask and K with current anisotropic correction
        2. Fit anisotropic U matrix with current k_mask and K
        
        Args:
            n_macro_cycles: Number of macro-cycles (analytical + anisotropic)
            n_aniso_iterations: Iterations per anisotropic fitting
            learning_rate: Learning rate for anisotropic fitting
        """
        if self.verbose > 0:
            print("\n" + "="*70)
            print("Iterative Refinement: Analytical Scales + Anisotropy")
            print("="*70)
            print(f"Macro-cycles: {n_macro_cycles}")
            print(f"Anisotropic iterations per cycle: {n_aniso_iterations}")
        
        for cycle in range(n_macro_cycles):
            if self.verbose > 0:
                print(f"\n{'='*70}")
                print(f"Macro-cycle {cycle + 1}/{n_macro_cycles}")
                print(f"{'='*70}")
            
            # Step 1: Re-compute analytical scales with current anisotropic correction
            if self.verbose > 0:
                print("\nRe-computing analytical bulk-solvent scales...")
            self._compute_analytical_scales()
            
            # Step 2: Fit anisotropic correction
            if self.verbose > 0:
                print(f"\nFitting anisotropic correction...")
            self.fit_anisotropy(n_iterations=n_aniso_iterations, learning_rate=learning_rate)
        
        if self.verbose > 0:
            print("="*70)
            print("Iterative refinement complete")
            print("="*70 + "\n")

