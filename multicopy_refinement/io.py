import pandas as pd
import numpy as np
from multicopy_refinement import math_numpy as math_np
from multicopy_refinement import math_torch
import gemmi
import torch
import pdb_tools
from torch import tensor
from typing import Optional, Tuple, Dict, List
import reciprocalspaceship as rs


class ReflectionData:
    """
    Class for handling crystallographic reflection data.
    
    Initialized empty and populated via loader methods.
    All data arrays are stored as PyTorch tensors.
    """
    
    # Priority list for amplitude/intensity columns (higher priority first)
    AMPLITUDE_PRIORITY = [
        'F-obs', 'FOBS', 'FP', 'F',  # Direct observations
        'F-obs-filtered', 'FOBS-filtered',  # Filtered observations
        'F(+)', 'FPLUS',  # Anomalous pairs
        'FMEAN', 'F-pk', 'F_pk',  # Mean or peak values
        'FO', 'FODD',  # Other amplitude variants
        'F-model', 'FC', 'FCALC',  # Calculated (lowest priority)
    ]
    
    INTENSITY_PRIORITY = [
        'I-obs', 'IOBS', 'I', 'IMEAN',  # Direct observations
        'I-obs-filtered', 'IOBS-filtered',  # Filtered observations
        'I(+)', 'IPLUS', 'IP',  # Anomalous pairs
        'I-pk', 'I_pk',  # Peak values
        'IHLI', 'I_full', 'IOBS_full', 'IO',  # Other intensity variants
    ]
    
    PHASE_PRIORITY = [
        'PHIB', 'PHI',  # Best phases (from density modification)
        'PHIF-model', 'PHFC', 'PHICALC',  # Calculated phases
        'PHIM', 'PHIW',  # Other phase variants
        'phase',  # Generic phase column
    ]
    
    RFREE_FLAG_NAMES = [
        'R-free-flags', 'RFREE', 'FreeR_flag', 'FREE',  # Common names
        'R-free', 'Rfree', 'FREER', 'FREE_FLAG',  # Variants
        'test', 'TEST', 'free', 'Free',  # Generic names
    ]
    
    def __init__(self,verbose: int = 1):
        """Initialize empty ReflectionData object."""
        # Miller indices
        self.hkl: Optional[torch.Tensor] = None  # Shape: (N, 3), dtype: int32
        self.verbose: int = verbose  # Verbosity level for logging
        # Amplitudes/Intensities
        self.F: Optional[torch.Tensor] = None  # Structure factor amplitudes, shape: (N,)
        self.F_sigma: Optional[torch.Tensor] = None  # Uncertainties, shape: (N,)
        self.I: Optional[torch.Tensor] = None  # Intensities, shape: (N,)
        self.I_sigma: Optional[torch.Tensor] = None  # Uncertainties, shape: (N,)
        
        # Phases
        self.phase: Optional[torch.Tensor] = None  # Phases in radians, shape: (N,)
        self.fom: Optional[torch.Tensor] = None  # Figure of merit, shape: (N,)
        
        # R-free flags
        self.rfree_flags: Optional[torch.Tensor] = None  # R-free test set flags, shape: (N,), dtype: int32
        self.rfree_source: Optional[str] = None  # Name of the R-free column
        
        # Metadata
        self.cell: Optional[torch.Tensor] = None  # Unit cell parameters [a,b,c,alpha,beta,gamma]
        self.spacegroup: Optional[str] = None
        self.resolution: Optional[torch.Tensor] = None  # Resolution per reflection, shape: (N,)
        
        # Data source tracking
        self.amplitude_source: Optional[str] = None
        self.intensity_source: Optional[str] = None
        self.phase_source: Optional[str] = None
        
        # Original dataframe for reference
        self._raw_data: Optional[rs.DataSet] = None
        self.source = None
        
    def load_from_mtz(self, path: str, expand_to_p1: bool = False, verbose=None) -> 'ReflectionData':
        """
        Load reflection data from MTZ file using reciprocalspaceship.
        
        Args:
            path: Path to MTZ file
            expand_to_p1: Whether to expand to P1 (all symmetry-related reflections)
            
        Returns:
            self (for method chaining)
        """
        print(f"Loading MTZ file: {path}")
        
        # Read MTZ file
        dataset = rs.read_mtz(path)
        
        if verbose is not None:
            self.verbose = verbose

        if expand_to_p1:
            if self.verbose > 0: print("Expanding to P1...")
            dataset = dataset.expand_to_p1()
        
        # Store raw data
        self._raw_data = dataset.copy()
        self.dataset = dataset
        
        # Extract unit cell and space group
        self.cell = torch.tensor([
            dataset.cell.a, dataset.cell.b, dataset.cell.c,
            dataset.cell.alpha, dataset.cell.beta, dataset.cell.gamma
        ], dtype=torch.float32)
        self.spacegroup = str(dataset.spacegroup)

        if self.verbose > 0: print(f"Unit cell: {self.cell.tolist()}")
        if self.verbose > 0: print(f"Space group: {self.spacegroup}")

        self._extract_amplitudes_and_intensities()

        # Extract Miller indices
        dataset_reset = self.dataset.reset_index()
        self.hkl = torch.tensor(
            dataset_reset[['H', 'K', 'L']].values.astype(np.int32),
            dtype=torch.int32
        )
        if self.verbose > 0: print(f"Loaded {len(self.hkl)} reflections")
        
        # Extract amplitude/intensity data
        
        # Extract phase data
        self._extract_phases(self.dataset)
        
        # Calculate resolution (needed for R-free flag generation)
        self._calculate_resolution()
        
        # Extract R-free flags (or generate if not present)
        self._extract_rfree_flags(self.dataset)
        

        if self.verbose > 1:
            print("\n" + "="*60)
            print("ReflectionData Summary:")
            print("="*60)
            print(f"Reflections: {len(self.hkl)}")
            if self.F is not None:
                print(f"Amplitudes: {self.amplitude_source} (σ: {'Yes' if self.F_sigma is not None else 'No'})")
            if self.I is not None:
                print(f"Intensities: {self.intensity_source} (σ: {'Yes' if self.I_sigma is not None else 'No'})")
            if self.phase is not None:
                print(f"Phases: {self.phase_source} (FOM: {'Yes' if self.fom is not None else 'No'})")
            if self.rfree_flags is not None:
                n_free = (self.rfree_flags == 0).sum().item()
                n_work = (self.rfree_flags != 0).sum().item()
                free_pct = 100.0 * n_free / len(self.rfree_flags) if len(self.rfree_flags) > 0 else 0
                print(f"R-free flags: {self.rfree_source} ({n_free} free / {n_work} work, {free_pct:.1f}% free)")
            else:
                self.generate_rfree_flags(free_fraction=0.02, n_bins=20, min_per_bin=100)
            if self.resolution is not None:
                print(f"Resolution range: {self.resolution.min():.2f} - {self.resolution.max():.2f} Å")
            print("="*60 + "\n")
        self = self.sanitize_F()
        return self
    
    def _extract_amplitudes_and_intensities(self) -> None:
        """
        Extract amplitude and intensity data with priority ordering.
        
        Prioritizes based on column priority lists, with intensities preferred over
        amplitudes when both are present at similar priority levels (observations).
        Automatically converts intensities to amplitudes using French-Wilson.
        """
        available_cols = set(self.dataset.columns)
        
        # Find the highest priority intensity column
        intensity_col = None
        intensity_priority_idx = None
        for idx, col in enumerate(self.INTENSITY_PRIORITY):
            if col in available_cols:
                dtype = str(self.dataset.dtypes[col])
                # Check if this is actually intensity data using reciprocalspaceship dtype
                if 'Intensity' in dtype or 'J' in dtype:
                    intensity_col = col
                    intensity_priority_idx = idx
                    break
        
        # Find the highest priority amplitude column
        amplitude_col = None
        amplitude_priority_idx = None
        for idx, col in enumerate(self.AMPLITUDE_PRIORITY):
            if col in available_cols:
                dtype = str(self.dataset.dtypes[col])
                # Check if this is amplitude data using reciprocalspaceship dtype
                if 'SFAmplitude' in dtype or 'F' in dtype:
                    amplitude_col = col
                    amplitude_priority_idx = idx
                    break
        
        # Decision logic: prefer intensities if they're observation-quality
        # Observation-quality means they appear in the first ~4 entries of priority list
        use_intensity = False
        if intensity_col is not None:
            # Always prefer intensities if found, unless only model amplitudes available
            if amplitude_col is None:
                use_intensity = True
            elif intensity_priority_idx <= 3:  # Observation-quality intensity
                use_intensity = True
            elif amplitude_priority_idx > 3:  # Both are low priority, prefer intensity
                use_intensity = True
        
        if use_intensity and intensity_col is not None:
            self.dataset.dropna(subset=[intensity_col], inplace=True)
            # Use intensity data
            self.I = torch.tensor(self.dataset[intensity_col].to_numpy(), dtype=torch.float32)
            self.intensity_source = intensity_col
            
            # Find corresponding sigma with comprehensive search
            sigma_col = self._find_sigma_column(self.dataset, intensity_col, is_intensity=True)
            if sigma_col is not None:
                self.I_sigma = torch.tensor(self.dataset[sigma_col].to_numpy(), dtype=torch.float32)
                if self.verbose > 0: print(f"Found intensity data: {intensity_col} (σ: {sigma_col})")
                
            else:
                self.I_sigma = None
                if self.verbose > 0: print(f"Found intensity data: {intensity_col} (no σ found)")

            # Convert to amplitudes using French-Wilson
            if self.verbose > 0: print(f"Converting intensities to amplitudes using French-Wilson...")
            self.F, self.F_sigma = math_torch.french_wilson_conversion(self.I, self.I_sigma)
            self.amplitude_source = f"{intensity_col} (French-Wilson)"

            if self.verbose > 0: print(f"  ✓ I({intensity_col}) → F (French-Wilson) with {'σ' if self.F_sigma is not None else 'no σ'}")

        elif amplitude_col is not None:
            self.dataset.dropna(subset=[amplitude_col], inplace=True)
            # Use amplitude data
            self.F = torch.tensor(self.dataset[amplitude_col].to_numpy(), dtype=torch.float32)
            self.amplitude_source = amplitude_col
            
            # Find corresponding sigma
            sigma_col = self._find_sigma_column(self.dataset, amplitude_col, is_intensity=False)
            if sigma_col is not None:
                self.F_sigma = torch.tensor(self.dataset[sigma_col].to_numpy(), dtype=torch.float32)
                if self.verbose > 0: print(f"Found amplitude data: {amplitude_col} (σ: {sigma_col})")
            else:
                self.F_sigma = None
                if self.verbose > 0: print(f"Found amplitude data: {amplitude_col} (no σ found)")
        
        else:
            # No suitable data found
            print("WARNING: No suitable amplitude or intensity data found!")
            print(f"Available columns: {sorted(available_cols)}")
            print(f"Column dtypes:")
            for col in sorted(available_cols):
                print(f"  {col}: {self.dataset.dtypes[col]}")
    
    def _find_sigma_column(self, dataset: rs.DataSet, data_col: str, is_intensity: bool) -> Optional[str]:
        """
        Find the sigma (uncertainty) column corresponding to a data column.
        
        Args:
            dataset: reciprocalspaceship dataset
            data_col: Name of the data column (e.g., 'IOBS', 'F-obs')
            is_intensity: Whether the data column is intensity (True) or amplitude (False)
            
        Returns:
            Name of sigma column if found, None otherwise
        """
        available_cols = set(dataset.columns)
        
        # Build list of possible sigma column names
        sigma_variants = []
        
        # Common patterns
        sigma_variants.extend([
            f'SIG{data_col}',           # SIGIOBS
            f'SIGM{data_col}',          # SIGMIOBS (gemmi style)
            f'{data_col}_sigma',        # IOBS_sigma
            f'{data_col}-sigma',        # IOBS-sigma
        ])
        
        # Pattern replacements
        if is_intensity:
            # For intensities: I-obs → SIGI-obs, IOBS → SIGIOBS, etc.
            sigma_variants.extend([
                data_col.replace('I', 'SIGI', 1),
                data_col.replace('I-', 'SIGI-'),
                data_col.replace('IOBS', 'SIGIOBS'),
                data_col.replace('IMEAN', 'SIGIMEAN'),
            ])
            # Generic intensity sigmas
            sigma_variants.extend(['SIGI', 'SIGIMEAN', 'SIGI-obs', 'SIGIOBS'])
        else:
            # For amplitudes: F-obs → SIGF-obs, FOBS → SIGFOBS, etc.
            sigma_variants.extend([
                data_col.replace('F', 'SIGF', 1),
                data_col.replace('F-', 'SIGF-'),
                data_col.replace('FOBS', 'SIGFOBS'),
                data_col.replace('FP', 'SIGFP'),
            ])
            # Generic amplitude sigmas
            sigma_variants.extend(['SIGF', 'SIGFOBS', 'SIGF-obs', 'SIGFP'])
        
        # Check each variant in order
        for sigma_col in sigma_variants:
            if sigma_col in available_cols:
                # Verify it's actually a standard deviation dtype
                dtype = str(dataset.dtypes[sigma_col])
                if 'Stddev' in dtype or 'Sigma' in dtype or 'SIG' in sigma_col.upper():
                    return sigma_col
        
        return None
    
    def _extract_phases(self, dataset: rs.DataSet) -> None:
        """Extract phase data with priority ordering."""
        available_cols = set(dataset.columns)
        
        for col in self.PHASE_PRIORITY:
            if col in available_cols:
                dtype = dataset.dtypes[col]
                # Check if this is phase data
                if 'Phase' in str(dtype) or 'PHI' in col or 'phase' in col.lower():
                    phase_deg = dataset[col].to_numpy()
                    self.phase = torch.tensor(np.deg2rad(phase_deg), dtype=torch.float32)
                    self.phase_source = col
                    
                    # Look for figure of merit
                    fom_variants = ['FOM', 'fom', 'FigureOfMerit', 'FOMB', 'FOMC']
                    for fom_col in fom_variants:
                        if fom_col in available_cols:
                            self.fom = torch.tensor(dataset[fom_col].to_numpy(), dtype=torch.float32)
                            break

                    if self.verbose > 0: print(f"Found phase data: {col} (FOM: {'Yes' if self.fom is not None else 'No'})")
                    return

        if self.verbose > 0: print("No phase data found (this is normal for experimental data)")

    def _extract_rfree_flags(self, dataset: rs.DataSet) -> None:
        """
        Extract R-free flags from the dataset.
        
        R-free flags typically use the convention:
        - 0 = test set (free reflections, not used in refinement)
        - 1+ = work set (used in refinement)
        
        Some programs may use different conventions, but we standardize to this.
        """
        available_cols = set(dataset.columns)
        
        for col in self.RFREE_FLAG_NAMES:
            if col in available_cols:
                # Get the data type from reciprocalspaceship
                dtype = str(dataset.dtypes[col])
                
                # Check if this looks like flag data (usually integer types)
                if 'int' in dtype.lower() or 'flag' in dtype.lower() or 'I' in dtype:
                    try:
                        flags = dataset[col].to_numpy()
                        
                        # Handle NaN values or object types
                        if flags.dtype == object or not np.issubdtype(flags.dtype, np.integer):
                            # Try to convert to integer, replacing NaN with -1
                            flags = pd.to_numeric(flags, errors='coerce')
                            flags = np.nan_to_num(flags, nan=-1).astype(np.int32)
                        else:
                            flags = flags.astype(np.int32)
                        
                        self.rfree_flags = torch.tensor(flags, dtype=torch.int32)
                        self.rfree_source = col
                        
                        # Get unique flag values
                        unique_flags = torch.unique(self.rfree_flags).tolist()
                        n_free = (self.rfree_flags == 0).sum().item()
                        n_work = (self.rfree_flags != 0).sum().item()
                        free_pct = 100.0 * n_free / len(self.rfree_flags) if len(self.rfree_flags) > 0 else 0
                        
                        # Check if convention is flipped (more "free" than "work")
                        # Standard convention: 0=free (test set, ~5-10%), other=work (~90-95%)
                        # If free > 50%, the convention is likely inverted
                        if free_pct > 50.0:
                            if self.verbose > 0: 
                                print(f"⚠️  WARNING: Detected inverted R-free convention!")
                                print(f"   Original: 0={n_free} ({free_pct:.1f}%), other={n_work} ({100-free_pct:.1f}%)")
                                print(f"   Flipping: 0 → work, other → free")
                            
                            # Flip the convention: 0 becomes 1, non-zero becomes 0
                            # But preserve -1 for missing/NA values
                            flipped = torch.zeros_like(self.rfree_flags)
                            flipped[self.rfree_flags == 0] = 1  # Old free (0) becomes work (1)
                            flipped[self.rfree_flags > 0] = 0   # Old work (>0) becomes free (0)
                            flipped[self.rfree_flags < 0] = -1  # Preserve NA markers
                            
                            self.rfree_flags = flipped
                            
                            # Recalculate statistics
                            n_free = (self.rfree_flags == 0).sum().item()
                            n_work = (self.rfree_flags != 0).sum().item()
                            free_pct = 100.0 * n_free / len(self.rfree_flags) if len(self.rfree_flags) > 0 else 0
                            unique_flags = torch.unique(self.rfree_flags).tolist()
                            
                            if self.verbose > 0: print(f"   After flip: free={n_free} ({free_pct:.1f}%), work={n_work} ({100-free_pct:.1f}%)")
                        if self.verbose > 1:
                            print(f"Found R-free flags: {col}")
                            print(f"  Unique flag values: {unique_flags}")
                            print(f"  Convention: 0=test(free), other=work")
                            print(f"  Test set: {n_free} reflections ({free_pct:.1f}%)")
                            print(f"  Work set: {n_work} reflections ({100-free_pct:.1f}%)")
                        
                        # Warn if free set is still unusually large or small after flipping
                        if free_pct > 10 or free_pct < 1:
                            print(f"  ⚠️  WARNING: Free set percentage ({free_pct:.1f}%) is unusual (typical: 2-5%)")
                            print(f"     This may indicate incomplete flags or non-standard partitioning.")
                        
                        return
                    except Exception as e:
                        print(f"Warning: Could not load R-free flags from {col}: {e}")
                        continue
        
        print("No R-free flags found (optional for experimental data)")
        
        # Automatically generate R-free flags if none found
        print("\n⚠️  WARNING: No R-free flags found - generating new flags automatically")
        self._generate_rfree_flags(free_fraction=0.02, n_bins=20, min_per_bin=100)
    
    def _generate_rfree_flags(self, free_fraction: float = 0.02, n_bins: int = 20, 
                             min_per_bin: int = 100, seed: Optional[int] = None) -> None:
        """
        Generate R-free flags with proper resolution binning.
        
        This ensures that free reflections are evenly distributed across resolution shells,
        which is critical for unbiased validation.
        
        Args:
            free_fraction: Fraction of reflections to mark as free (default: 0.02 = 2%)
            n_bins: Target number of resolution bins (default: 20)
            min_per_bin: Minimum number of reflections per resolution bin (default: 100)
            seed: Random seed for reproducibility (default: None)
            
        The algorithm:
        1. Bin reflections by resolution
        2. Ensure bins have at least min_per_bin reflections or 1% of data
        3. Randomly select free_fraction of reflections from each bin
        4. This ensures even distribution across all resolution ranges
        """
        if self.resolution is None:
            raise ValueError("Resolution information required to generate R-free flags")
        
        print(f"Generating R-free flags:")
        print(f"  Target free fraction: {free_fraction*100:.1f}%")
        print(f"  Target bins: {n_bins}")
        print(f"  Minimum per bin: {min_per_bin} reflections")
        
        # Set random seed for reproducibility
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
        
        n_refl = len(self.resolution)
        
        # Create resolution bins
        bin_indices, actual_n_bins = self._create_resolution_bins(n_bins=n_bins, min_per_bin=min_per_bin)
        
        print(f"  Created {actual_n_bins} resolution bins")
        
        # Initialize all flags as work set (1)
        flags = torch.ones(n_refl, dtype=torch.int32)
        
        # Sample free reflections from each bin
        total_free = 0
        for bin_idx in range(actual_n_bins):
            bin_mask = bin_indices == bin_idx
            bin_size = bin_mask.sum().item()
            
            if bin_size == 0:
                continue
            
            # Number of free reflections in this bin
            # Ensure at least 1, but respect the free_fraction
            n_free_in_bin = max(1, int(bin_size * free_fraction))
            
            # Get indices of reflections in this bin
            bin_refl_indices = torch.where(bin_mask)[0]
            
            # Randomly select free reflections
            perm = torch.randperm(bin_size)[:n_free_in_bin]
            free_indices = bin_refl_indices[perm]
            
            # Mark as free (0)
            flags[free_indices] = 0
            total_free += n_free_in_bin
        
        self.rfree_flags = flags
        self.rfree_source = "Generated (resolution-binned)"
        
        n_free = (flags == 0).sum().item()
        n_work = (flags != 0).sum().item()
        free_pct = 100.0 * n_free / n_refl
        
        print(f"  ✓ Generated flags: {n_free} free ({free_pct:.1f}%), {n_work} work ({100-free_pct:.1f}%)")
        print(f"  Flags are resolution-binned for unbiased validation")
    
    def _create_resolution_bins(self, n_bins: int = 20, min_per_bin: int = 100) -> Tuple[torch.Tensor, int]:
        """
        Create resolution bins with target number of bins.
        
        Args:
            n_bins: Target number of resolution bins (default: 20)
            min_per_bin: Minimum reflections per bin (default: 100)
            
        Returns:
            Tuple of (bin_indices, n_bins) where:
                - bin_indices: Tensor of shape (N,) with bin index for each reflection
                - n_bins: Actual number of bins created (may be less than target if dataset is small)
        """
        n_refl = len(self.resolution)
        
        # Calculate how many bins we can actually create given min_per_bin constraint
        max_possible_bins = max(1, n_refl // min_per_bin)
        actual_n_bins = min(n_bins, max_possible_bins)
        
        if actual_n_bins < n_bins:
            print(f"  Note: Adjusted bins from {n_bins} to {actual_n_bins} (min {min_per_bin} refl/bin)")
        
        # Sort reflections by resolution
        sorted_res, sort_indices = torch.sort(self.resolution)
        
        # Create bins with approximately equal number of reflections
        bin_indices = torch.zeros(n_refl, dtype=torch.int32)
        reflections_per_bin = n_refl / actual_n_bins
        
        for i, idx in enumerate(sort_indices):
            # Assign to bin based on position in sorted list
            bin_idx = min(int(i / reflections_per_bin), actual_n_bins - 1)
            bin_indices[idx] = bin_idx
        
        # Print bin statistics
        print(f"  Resolution bins:")
        for bin_idx in range(min(actual_n_bins, 20)):  # Show all bins (up to 20)
            bin_mask = bin_indices == bin_idx
            if bin_mask.sum() > 0:
                bin_res = self.resolution[bin_mask]
                print(f"    Bin {bin_idx:2d}: {bin_mask.sum():6d} refl, "
                      f"resolution {bin_res.min():.2f}-{bin_res.max():.2f} Å")
        
        if actual_n_bins > 20:
            print(f"    ... ({actual_n_bins - 20} more bins)")
        
        return bin_indices, actual_n_bins
    
    def regenerate_rfree_flags(self, free_fraction: float = 0.02, n_bins: int = 20,
                               min_per_bin: int = 100, seed: Optional[int] = None, 
                               force: bool = False) -> None:
        """
        Regenerate R-free flags (public interface).
        
        Args:
            free_fraction: Fraction of reflections to mark as free (default: 0.02 = 2%)
            n_bins: Target number of resolution bins (default: 20)
            min_per_bin: Minimum reflections per resolution bin (default: 100)
            seed: Random seed for reproducibility (default: None)
            force: If True, regenerate even if flags already exist (default: False)
            
        Example:
            # Generate 2% free reflections with 20 bins and reproducible seed
            data.regenerate_rfree_flags(free_fraction=0.02, n_bins=20, seed=42)
            
            # Generate 5% free with 10 bins
            data.regenerate_rfree_flags(free_fraction=0.05, n_bins=10, force=True)
        """
        if self.rfree_flags is not None and not force:
            print("⚠️  WARNING: R-free flags already exist!")
            print(f"   Current source: {self.rfree_source}")
            print("   Use force=True to overwrite existing flags")
            return
        
        if self.rfree_flags is not None and force:
            print("⚠️  WARNING: Overwriting existing R-free flags")
            print(f"   Old source: {self.rfree_source}")
        
        self._generate_rfree_flags(free_fraction=free_fraction, n_bins=n_bins, 
                                   min_per_bin=min_per_bin, seed=seed)
    
    def _calculate_resolution(self) -> None:
        """Calculate resolution for each reflection."""
        if self.hkl is not None and self.cell is not None:
            hkl_np = self.hkl.cpu().numpy()
            cell_np = self.cell.cpu().numpy()
            s = math_np.get_scattering_vectors(hkl_np, cell_np)
            resolution = 1.0 / np.linalg.norm(s, axis=1)
            self.resolution = torch.tensor(resolution, dtype=torch.float32)
    
    def get_structure_factors(self, as_complex: bool = False) -> torch.Tensor:
        """
        Get structure factors, optionally as complex numbers.
        
        Args:
            as_complex: If True and phases available, return F*exp(i*phi)
            
        Returns:
            Tensor of amplitudes or complex structure factors
        """
        if self.F is None:
            raise ValueError("No amplitude data loaded")
        
        if as_complex and self.phase is not None:
            return self.F * torch.exp(1j * self.phase)
        else:
            return self.F
    
    def get_structure_factors_with_sigma(self) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Get structure factor amplitudes and their uncertainties.
        
        Returns:
            Tuple of (F, F_sigma) where:
                - F: Structure factor amplitudes, shape (N,)
                - F_sigma: Uncertainties (None if not available), shape (N,)
        
        Example:
            F, sigma_F = data.get_structure_factors_with_sigma()
            if sigma_F is not None:
                weighted_residual = (F_obs - F_calc) / sigma_F
        """
        if self.F is None:
            raise ValueError("No amplitude data loaded")
        
        return self.F, self.F_sigma
    
    def get_hkl(self):
        """Return Miller indices as a tensor of shape (N, 3)."""
        if self.hkl is None:
            raise ValueError("No Miller indices loaded")
        return self.hkl

    def filter_by_resolution(self, d_min: Optional[float] = None, 
                            d_max: Optional[float] = None) -> 'ReflectionData':
        """
        Filter reflections by resolution range.
        
        Args:
            d_min: Minimum resolution (highest resolution, e.g., 1.5 Å)
            d_max: Maximum resolution (lowest resolution, e.g., 50.0 Å)
            
        Returns:
            New ReflectionData object with filtered data
        """
        if self.resolution is None:
            raise ValueError("No resolution information available")
        
        mask = torch.ones(len(self.hkl), dtype=torch.bool)
        
        if d_min is not None:
            mask &= self.resolution >= d_min
        if d_max is not None:
            mask &= self.resolution <= d_max
        
        print(f"Filtering: {mask.sum()}/{len(mask)} reflections in range "
              f"[{d_max if d_max else 'inf'} - {d_min if d_min else 'inf'}] Å")
        
        filtered = self.__select__(mask)
        

        
        return filtered
    
    def cut_res(self, highres: Optional[float] = None, 
                lowres: Optional[float] = None) -> 'ReflectionData':
        """
        Filter reflections by resolution range (alias for filter_by_resolution).
        
        This method uses the more intuitive naming where:
        - res_min is the minimum resolution (high resolution limit, small d-spacing)
        - res_max is the maximum resolution (low resolution limit, large d-spacing)
        
        Args:
            res_min: Minimum resolution / high resolution cutoff (e.g., 1.5 Å)
                    Keeps reflections with d >= res_min
            res_max: Maximum resolution / low resolution cutoff (e.g., 50.0 Å)
                    Keeps reflections with d <= res_max
            
        Returns:
            New ReflectionData object with filtered data
            
        Example:
            # Keep reflections between 50 Å and 1.5 Å
            filtered = data.cut_res(res_min=1.5, res_max=50.0)
            
            # Keep only high-resolution data (< 2 Å)
            high_res = data.cut_res(res_min=1.0, res_max=2.0)
        """
        return self.filter_by_resolution(d_min=highres, d_max=lowres)
    
    def get_rfree_masks(self) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Get boolean masks for work and test (free) sets.
        
        Returns:
            Tuple of (work_mask, test_mask) where:
                - work_mask: Boolean tensor for work set (flag != 0)
                - test_mask: Boolean tensor for test/free set (flag == 0)
                - Both are None if no R-free flags are available
        
        Example:
            work_mask, test_mask = data.get_rfree_masks()
            if work_mask is not None:
                F_work = data.F[work_mask]
                F_test = data.F[test_mask]
                # Calculate R-factors separately
        """
        if self.rfree_flags is None:
            return None, None
        
        work_mask = self.rfree_flags != 0
        test_mask = self.rfree_flags == 0
        
        return work_mask, test_mask
    
    def get_work_set(self) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Get structure factors for the work set (R-free flag != 0).
        
        Returns:
            Tuple of (F_work, sigma_work) or full dataset if no R-free flags
        """
        if self.rfree_flags is None:
            print("WARNING: No R-free flags available, returning full dataset")
            return self.F, self.F_sigma
        
        work_mask = self.rfree_flags != 0
        F_work = self.F[work_mask] if self.F is not None else None
        sigma_work = self.F_sigma[work_mask] if self.F_sigma is not None else None
        
        return F_work, sigma_work
    
    def get_test_set(self) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Get structure factors for the test set (R-free flag == 0).
        
        Returns:
            Tuple of (F_test, sigma_test)
        
        Raises:
            ValueError if no R-free flags are available
        """
        if self.rfree_flags is None:
            raise ValueError("No R-free flags available in dataset")
        
        test_mask = self.rfree_flags == 0
        F_test = self.F[test_mask] if self.F is not None else None
        sigma_test = self.F_sigma[test_mask] if self.F_sigma is not None else None
        
        return F_test, sigma_test
    
    def __len__(self) -> int:
        """Return number of reflections."""
        return len(self.hkl) if self.hkl is not None else 0
    
    def __repr__(self) -> str:
        """String representation."""
        if self.hkl is None:
            return "ReflectionData(empty)"
        
        parts = [f"ReflectionData(n={len(self.hkl)}"]
        if self.amplitude_source:
            parts.append(f"F={self.amplitude_source}")
        if self.phase_source:
            parts.append(f"φ={self.phase_source}")
        if self.resolution is not None:
            parts.append(f"d={self.resolution.min():.2f}-{self.resolution.max():.2f}Å")
        parts.append(f"sg={self.spacegroup}")
        
        return ", ".join(parts) + ")"

    def __call__(self):
        """

        Return core reflection data: (hkl, F, F_sigma, rfree_flags).

        Returns:

            Tuple of (hkl, F, F_sigma, rfree_flags)

        hkl: Tensor of shape (N, 3)
        F: Tensor of shape (N,)
        F_sigma: Tensor of shape (N,) or None 
        rfree_flags: Tensor of shape (N,) or None (1 is work, 0 is free)

        """


        assert self.hkl is not None, "No Miller indices loaded in ReflectionData"
        return self.hkl, self.F, self.F_sigma, self.rfree_flags

    def __select__(self,mask:torch.Tensor):
        """Select reflections based on a boolean mask."""
        selected = ReflectionData()
        selected.hkl = self.hkl[mask] if self.hkl is not None else None
        selected.F = self.F[mask] if self.F is not None else None
        selected.F_sigma = self.F_sigma[mask] if self.F_sigma is not None else None
        selected.phase = self.phase[mask] if self.phase is not None else None
        selected.fom = self.fom[mask] if self.fom is not None else None
        selected.rfree_flags = self.rfree_flags[mask] if self.rfree_flags is not None else None
        selected.cell = self.cell.clone() if self.cell is not None else None
        selected.spacegroup = self.spacegroup
        selected.resolution = self.resolution[mask] if self.resolution is not None else None
        selected.amplitude_source = self.amplitude_source
        selected.intensity_source = self.intensity_source
        selected.phase_source = self.phase_source
        selected.rfree_source = self.rfree_source
        selected.source = self
        return selected

    def unpack(self):
        while self.source is not None:
            self = self.source
        return self        

    def sanitize_F(self):
        """Cut NA values from F and F_sigma."""
        mask = torch.ones(len(self.F), dtype=torch.bool)
        if self.F is not None:
            if self.verbose > 0: print('found nan F values: ', torch.isnan(self.F).sum().item())
            mask &= ~torch.isnan(self.F)
        if self.F_sigma is not None:
            if self.verbose > 0: print('found nan F_sigma values: ', torch.isnan(self.F_sigma).sum().item())
            mask &= ~torch.isnan(self.F_sigma)
        if mask.sum() < len(mask):
            if self.verbose > 0: print(f"Sanitizing: Removing {len(mask) - mask.sum().item()} reflections with NaN F or F_sigma")
            self = self.__select__(mask)
        return self

    def check_all_data_types(self):
        for key in self.__dict__:
            if self.__dict__[key] is not None and isinstance(self.__dict__[key], torch.Tensor):
                print(f"{key}: {self.__dict__[key].dtype}, shape: {self.__dict__[key].shape}")
            elif self.__dict__[key] is not None:
                print(f"{key}: {type(self.__dict__[key])}, value: {self.__dict__[key]}")
            else:
                print(f"{key}: None")
    
    def validate_hkl(self, hkl_ref: torch.Tensor) -> Tuple['ReflectionData', torch.Tensor]:
        """
        Validate and filter reflections against a reference HKL set.
        
        This method filters the current dataset to only include reflections that are 
        present in the reference HKL tensor, and returns a boolean mask indicating 
        which reference reflections are present in this dataset.
        
        Args:
            hkl_ref: Reference Miller indices tensor of shape (N, 3) with dtype int32
            
        Returns:
            Tuple of (filtered_data, presence_mask) where:
                - filtered_data: New ReflectionData object containing only reflections 
                  that are present in hkl_ref
                - presence_mask: Boolean tensor of shape (N,) indicating which reflections 
                  from hkl_ref are present in this dataset (True = present, False = absent)
        
        Example:
            # Filter data to match reference HKL list
            filtered_data, present_in_data = data.validate_hkl(reference_hkl)
            print(f"Kept {len(filtered_data)} reflections out of {len(data)}")
            print(f"{present_in_data.sum()} reference reflections found in dataset")
        """
        if self.hkl is None:
            raise ValueError("No Miller indices loaded in ReflectionData")
        
        if not isinstance(hkl_ref, torch.Tensor):
            raise TypeError(f"hkl_ref must be a torch.Tensor, got {type(hkl_ref)}")
        
        if hkl_ref.shape[-1] != 3:
            raise ValueError(f"hkl_ref must have shape (N, 3), got {hkl_ref.shape}")
        
        # Ensure hkl_ref is 2D and int32
        if hkl_ref.dim() == 1:
            hkl_ref = hkl_ref.unsqueeze(0)
        hkl_ref = hkl_ref.to(dtype=torch.int32)
        
        n_ref = len(hkl_ref)
        n_data = len(self.hkl)
        
        # Convert to numpy for efficient isin-based lookup
        # We'll use structured arrays to compare HKL triplets as single entities
        hkl_ref_np = hkl_ref.cpu().numpy()
        hkl_data_np = self.hkl.cpu().numpy()
        
        # Create structured arrays to treat each (h,k,l) triplet as a single comparable unit
        # This allows numpy to efficiently check membership
        hkl_ref_structured = np.ascontiguousarray(hkl_ref_np).view(
            np.dtype((np.void, hkl_ref_np.dtype.itemsize * hkl_ref_np.shape[1]))
        )
        hkl_data_structured = np.ascontiguousarray(hkl_data_np).view(
            np.dtype((np.void, hkl_data_np.dtype.itemsize * hkl_data_np.shape[1]))
        )
        
        # Find which data reflections are present in the reference
        # np.isin is highly optimized and uses hash-based lookup internally
        data_mask_np = np.isin(hkl_data_structured, hkl_ref_structured)
        data_mask = torch.from_numpy(data_mask_np.flatten())
        
        # Find which reference reflections are present in the dataset
        presence_mask_np = np.isin(hkl_ref_structured, hkl_data_structured)
        presence_mask = torch.from_numpy(presence_mask_np.flatten())

        # Filter the dataset to only include reflections in the reference
        filtered_data = self.__select__(data_mask)
        
        if self.verbose > 0:
            print(f"HKL validation:")
            print(f"  Dataset reflections: {n_data}")
            print(f"  Reference reflections: {n_ref}")
            print(f"  Kept in dataset: {data_mask.sum().item()} ({100*data_mask.sum().item()/n_data:.1f}%)")
            print(f"  Found in dataset: {presence_mask.sum().item()} ({100*presence_mask.sum().item()/n_ref:.1f}%)")
        
        return filtered_data, presence_mask
            