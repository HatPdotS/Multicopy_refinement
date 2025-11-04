import pandas as pd
import numpy as np
from multicopy_refinement import math_numpy as math_np
from multicopy_refinement import math_torch
import gemmi
import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, List
import reciprocalspaceship as rs
from multicopy_refinement.model_ft import ModelFT
from multicopy_refinement.utils import TensorMasks

class ReflectionData(nn.Module):
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
    
    def __init__(self, verbose: int = 1, device: str = 'cpu'):
        """
        Initialize empty ReflectionData object.
        
        Args:
            verbose: Verbosity level for logging
            device: Device to store tensors on ('cpu' or 'cuda' or 'cuda:0', etc.)
        """
        super().__init__()
        
        self.verbose: int = verbose  # Verbosity level for logging
        self.device = torch.device(device)  # Device for all tensors
        
        # Register tensor buffers (will be moved to GPU with .cuda())
        # Miller indices
        self.register_buffer('hkl', None)  # Shape: (N, 3), dtype: int32
        
        # Amplitudes/Intensities
        self.register_buffer('F', None)  # Structure factor amplitudes, shape: (N,)
        self.register_buffer('F_sigma', None)  # Uncertainties, shape: (N,)
        self.register_buffer('I', None)  # Intensities, shape: (N,)
        self.register_buffer('I_sigma', None)  # Uncertainties, shape: (N,)
        self.masks = TensorMasks()  # Dictionary of boolean masks for filtering reflections 1 is included 0 is excluded

        # Phases
        self.register_buffer('phase', None)  # Phases in radians, shape: (N,)
        self.register_buffer('fom', None)  # Figure of merit, shape: (N,)
        
        # R-free flags
        self.register_buffer('rfree_flags', None)  # R-free test set flags, shape: (N,), dtype: int32
        self.rfree_source: Optional[str] = None  # Name of the R-free column
        
        # Outlier flags
        self.register_buffer('outlier_flags', None)  # Outlier flags, shape: (N,), dtype: bool
        self.outlier_detection_params: Optional[Dict] = None  # Parameters used for outlier detection
        
        # Metadata
        self.register_buffer('cell', None)  # Unit cell parameters [a,b,c,alpha,beta,gamma]
        self.spacegroup: Optional[str] = None
        self.register_buffer('resolution', None)  # Resolution per reflection, shape: (N,)
        
        # Data source tracking
        self.amplitude_source: Optional[str] = None
        self.intensity_source: Optional[str] = None
        self.phase_source: Optional[str] = None
        
        # Original dataframe for reference
        self._raw_data: Optional[rs.DataSet] = None
        self.source = None
    
    def cuda(self, device=None):
        """Move ReflectionData to CUDA, including masks child module."""
        super().cuda(device)
        self.device = torch.device('cuda') if device is None else torch.device(device)
        # Explicitly move masks since it's a child module
        if hasattr(self, 'masks'):
            self.masks.cuda(device)
        if self.verbose > 1:
            print(f"ReflectionData moved to device: {self.device}")
        return self
    
    def cpu(self):
        """Move ReflectionData to CPU, including masks child module."""
        super().cpu()
        self.device = torch.device('cpu')
        # Explicitly move masks since it's a child module
        if hasattr(self, 'masks'):
            self.masks.cpu()
        if self.verbose > 1:
            print(f"ReflectionData moved to cpu")
        return self
        
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
        cell_tensor = torch.tensor([
            dataset.cell.a, dataset.cell.b, dataset.cell.c,
            dataset.cell.alpha, dataset.cell.beta, dataset.cell.gamma
        ], dtype=torch.float32, device=self.device)
        self.register_buffer('cell', cell_tensor)
        self.spacegroup = str(dataset.spacegroup)

        if self.verbose > 0: print(f"Unit cell: {self.cell.tolist()}")
        if self.verbose > 0: print(f"Space group: {self.spacegroup}")

        self._extract_amplitudes_and_intensities()

        # Extract Miller indices
        dataset_reset = self.dataset.reset_index()
        hkl_tensor = torch.tensor(
            dataset_reset[['H', 'K', 'L']].values.astype(np.int32),
            dtype=torch.int32, device=self.device
        )
        self.register_buffer('hkl', hkl_tensor)
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
                self._generate_rfree_flags(free_fraction=0.02, n_bins=20, min_per_bin=100)
            if self.resolution is not None:
                print(f"Resolution range: {self.resolution.min():.2f} - {self.resolution.max():.2f} Å")
            print("="*60 + "\n")
        self = self.sanitize_F()
        self.masks['example'] = torch.ones(len(self.hkl), dtype=torch.bool, device=self.device)
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
            I_tensor = torch.tensor(self.dataset[intensity_col].to_numpy(), dtype=torch.float32, device=self.device)
            self.register_buffer('I', I_tensor)
            self.intensity_source = intensity_col
            
            # Find corresponding sigma with comprehensive search
            sigma_col = self._find_sigma_column(self.dataset, intensity_col, is_intensity=True)
            if sigma_col is not None:
                I_sigma_tensor = torch.tensor(self.dataset[sigma_col].to_numpy(), dtype=torch.float32, device=self.device)
                self.register_buffer('I_sigma', I_sigma_tensor)
                if self.verbose > 0: print(f"Found intensity data: {intensity_col} (σ: {sigma_col})")
                
            else:
                self.register_buffer('I_sigma', None)
                if self.verbose > 0: print(f"Found intensity data: {intensity_col} (no σ found)")

            # Convert to amplitudes using French-Wilson
            if self.verbose > 0: print(f"Converting intensities to amplitudes using French-Wilson...")
            F_tensor, F_sigma_tensor = math_torch.french_wilson_conversion(self.I, self.I_sigma)
            # Move to correct device if not already there
            F_tensor = F_tensor.to(self.device)
            if F_sigma_tensor is not None:
                F_sigma_tensor = F_sigma_tensor.to(self.device)
            self.register_buffer('F', F_tensor)
            self.register_buffer('F_sigma', F_sigma_tensor)
            self.amplitude_source = f"{intensity_col} (French-Wilson)"

            if self.verbose > 0: print(f"  ✓ I({intensity_col}) → F (French-Wilson) with {'σ' if self.F_sigma is not None else 'no σ'}")

        elif amplitude_col is not None:
            self.dataset.dropna(subset=[amplitude_col], inplace=True)
            # Use amplitude data
            F_tensor = torch.tensor(self.dataset[amplitude_col].to_numpy(), dtype=torch.float32, device=self.device)
            self.register_buffer('F', F_tensor)
            self.amplitude_source = amplitude_col
            
            # Find corresponding sigma
            sigma_col = self._find_sigma_column(self.dataset, amplitude_col, is_intensity=False)
            if sigma_col is not None:
                F_sigma_tensor = torch.tensor(self.dataset[sigma_col].to_numpy(), dtype=torch.float32, device=self.device)
                self.register_buffer('F_sigma', F_sigma_tensor)
                if self.verbose > 0: print(f"Found amplitude data: {amplitude_col} (σ: {sigma_col})")
            else:
                self.register_buffer('F_sigma', None)
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
                    phase_tensor = torch.tensor(np.deg2rad(phase_deg), dtype=torch.float32, device=self.device)
                    self.register_buffer('phase', phase_tensor)
                    self.phase_source = col
                    
                    # Look for figure of merit
                    fom_variants = ['FOM', 'fom', 'FigureOfMerit', 'FOMB', 'FOMC']
                    for fom_col in fom_variants:
                        if fom_col in available_cols:
                            fom_tensor = torch.tensor(dataset[fom_col].to_numpy(), dtype=torch.float32, device=self.device)
                            self.register_buffer('fom', fom_tensor)
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
                        
                        rfree_tensor = torch.tensor(flags, dtype=torch.int32, device=self.device)
                        self.register_buffer('rfree_flags', rfree_tensor)
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
                            
                            self.register_buffer('rfree_flags', flipped)
                            
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
        bin_indices, actual_n_bins = self.get_bins(n_bins=n_bins, min_per_bin=min_per_bin)
        
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
        
        # Move to correct device and register
        flags_tensor = flags.to(dtype=torch.int32, device=self.device)
        self.register_buffer('rfree_flags', flags_tensor)
        self.rfree_source = "Generated (resolution-binned)"
        
        n_free = (flags == 0).sum().item()
        n_work = (flags != 0).sum().item()
        free_pct = 100.0 * n_free / n_refl
        
        print(f"  ✓ Generated flags: {n_free} free ({free_pct:.1f}%), {n_work} work ({100-free_pct:.1f}%)")
        print(f"  Flags are resolution-binned for unbiased validation")
    
    def get_bins(self, n_bins: int = 20, min_per_bin: int = 100) -> Tuple[torch.Tensor, int]:
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
        
        if actual_n_bins < n_bins and self.verbose > 0:
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

        if self.verbose > 1:
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
        self.register_buffer('bin_indices', bin_indices)
        self.register_buffer('n_bins', torch.tensor(actual_n_bins, dtype=torch.int32))
        return bin_indices, actual_n_bins
    
    def mean_res_per_bin(self) -> torch.Tensor:
        """
        Calculate mean resolution for each bin.
        
        Returns:
            List of mean resolutions per bin
        """
        if not hasattr(self, 'bin_indices') or not hasattr(self, 'resolution'):
            raise ValueError("Bins have not been created yet")
        
        mean_resolutions = torch.zeros(self.n_bins, dtype=torch.float32, device=self.device)
        count_per_bin = torch.zeros(self.n_bins, dtype=torch.int32, device=self.device)
        mask = self.masks()
        mean_resolutions = torch.scatter_add(mean_resolutions, 0, self.bin_indices[mask].to(torch.int64), self.resolution[mask])
        count_per_bin = torch.scatter_add(count_per_bin, 0, self.bin_indices[mask].to(torch.int64), torch.ones_like(self.resolution[mask], dtype=torch.int32))
        mean_resolutions = mean_resolutions / count_per_bin.clamp(min=1).float()
        return mean_resolutions

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
            resolution_tensor = torch.tensor(resolution, dtype=torch.float32, device=self.device)
            self.register_buffer('resolution', resolution_tensor)
    
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
        return self.hkl[self.masks()]

    def filter_by_resolution(self, d_min: Optional[float] = None, 
                            d_max: Optional[float] = None) -> 'ReflectionData':
        """
        Filter reflections by resolution range.
        adds a boolean mask to self.masks for the specified resolution range.

        Args:
            d_min: Minimum resolution (highest resolution, e.g., 1.5 Å)
            d_max: Maximum resolution (lowest resolution, e.g., 50.0 Å)
            
        Returns:
            self (for method chaining)
        """
        if self.resolution is None:
            raise ValueError("No resolutionlo information available")
        
        mask = torch.ones(len(self.hkl), dtype=torch.bool)
        
        if d_min is not None:
            mask &= self.resolution >= d_min
        if d_max is not None:
            mask &= self.resolution <= d_max
        
        print(f"Filtering: {mask.sum()}/{len(mask)} reflections in range "
              f"[{d_max if d_max else 'inf'} - {d_min if d_min else 'inf'}] Å")
        
        self.masks['resolution'] = mask

        return self
    
    def get_mask(self):
        return self.masks()
    
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

    def get_max_res(self) -> Optional[float]:
        """Return maximum resolution (lowest d-spacing) in Å."""
        if self.resolution is None:
            return None
        return float(self.resolution.min().item())

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

    def forward(self, mask:bool=True)-> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """

        Return core reflection data: (hkl, F, F_sigma, rfree_flags).

        Returns:

            Tuple of (hkl, F, F_sigma, rfree_flags)

        hkl: Tensor of shape (N, 3)
        F: Tensor of shape (N,)
        F_sigma: Tensor of shape (N,) or None 
        rfree_flags: Tensor of shape (N,) or None (1 is work, 0 is free)

        """
        hkl, F, F_sigma, rfree_flags = self.hkl, self.F, self.F_sigma, self.rfree_flags
        if mask:
            to_mask = self.masks()
            hkl, F = hkl[to_mask], F[to_mask]
            if F_sigma is not None:
                F_sigma = F_sigma[to_mask]
            if rfree_flags is not None:
                rfree_flags = rfree_flags[to_mask]
            rfree_flags = rfree_flags.to(torch.bool)
        return hkl, F, F_sigma, rfree_flags
    

    def __select__(self,mask:torch.Tensor, op=None)-> 'ReflectionData':
        """Select reflections based on a boolean mask."""
        # Create new instance with same device
        selected = ReflectionData(verbose=self.verbose, device=self.device)
        
        # Register buffers for the selected data
        if self.hkl is not None:
            selected.register_buffer('hkl', self.hkl[mask])
        if self.I is not None:
            selected.register_buffer('I', self.I[mask])
        if self.F is not None:
            selected.register_buffer('F', self.F[mask])
        if self.F_sigma is not None:
            selected.register_buffer('F_sigma', self.F_sigma[mask])
        if self.phase is not None:
            selected.register_buffer('phase', self.phase[mask])
        if self.fom is not None:
            selected.register_buffer('fom', self.fom[mask])
        if self.rfree_flags is not None:
            selected.register_buffer('rfree_flags', self.rfree_flags[mask])
        if self.cell is not None:
            selected.register_buffer('cell', self.cell.clone())
        if self.resolution is not None:
            selected.register_buffer('resolution', self.resolution[mask])
        
        selected.spacegroup = self.spacegroup
        selected.amplitude_source = self.amplitude_source
        selected.intensity_source = self.intensity_source
        selected.phase_source = self.phase_source
        selected.rfree_source = self.rfree_source
        selected.verbose = self.verbose
        selected.source = self
        selected.dataset = self.dataset.iloc[mask.cpu().numpy()].copy() if self.dataset is not None else None
        self.last_op = op
        return selected

    def unpack(self):
        """
        Unpack to the original source data.
        Traverses the source chain to the original data and validates HKL.
        Marks reflections not present in the original data in self.flagged
        """
        current_hkl = self.hkl
        while self.source is not None:
            self = self.source
        _, valid_hkl = self.validate_hkl(current_hkl)
        self.flagged = ~valid_hkl
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
    
    def find_outliers(self, model: ModelFT, scaler, z_threshold: float = 4.0) -> torch.Tensor:
        """
        Identify outlier reflections based on log-ratio distribution.
        
        Uses the fact that log(F_obs) - log(F_calc) should be normally distributed.
        Outliers are reflections where |log_ratio - mean| > z_threshold * std_dev.
        
        Args:
            model: ModelFT object to compute structure factors
            scaler: Scaler object to scale calculated structure factors  
            z_threshold: Z-score threshold to classify outliers (default: 4.0)
            
        Returns:
            torch.Tensor: Boolean mask where True indicates outliers
        """
        hkl, F_obs, _, _ = self.forward(mask=False)
        log_ratio = self.get_log_ratio(model, scaler)
        eps = 1e-10
        
        # Remove any infinite or NaN values for statistics
        valid_mask = torch.isfinite(log_ratio)
        if valid_mask.sum() == 0:
            if self.verbose > 0:
                print("Warning: No valid log-ratios found for outlier detection")
            return torch.zeros_like(F_obs, dtype=torch.bool, device=self.device)
        
        to_use = valid_mask & self.masks()

        log_ratio_valid = log_ratio[to_use]
        
        # Compute mean and standard deviation of log-ratio distribution
        mean_log_ratio = torch.mean(log_ratio_valid)
        std_log_ratio = torch.std(log_ratio_valid, unbiased=True)
        
        # Identify outliers using Z-score criterion
        z_scores = torch.abs(log_ratio - mean_log_ratio) / (std_log_ratio + eps)
        outlier_mask = z_scores > z_threshold
        
        # Set invalid ratios as outliers too
        outlier_mask = outlier_mask | ~valid_mask
        
        if self.verbose > 0:
            n_outliers = outlier_mask.sum().item()
            n_total = len(F_obs)
            print(f"Outlier detection: {n_outliers}/{n_total} ({100*n_outliers/n_total:.2f}%) outliers found")
            print(f"  Log-ratio statistics: mean={mean_log_ratio:.3f}, std={std_log_ratio:.3f}")
            print(f"  Z-score threshold: {z_threshold:.1f}")
        
        # Ensure outlier_mask is on correct device and register
        outlier_mask = outlier_mask.to(self.device)
        self.masks['outliers'] = ~outlier_mask
        if self.verbose > 0: print(f"Outlier detection: {outlier_mask.sum().item()} reflections flagged as outliers out of {len(outlier_mask)}.")
    
    def get_log_ratio(self, model: ModelFT, scaler) -> torch.Tensor:
        # Get observed and calculated structure factors
        eps = 1e-10
        hkl, F_obs, _ , _ = self.forward(mask=False)
        F_calc_complex = model.forward(hkl)  # Complex structure factors
        F_calc_scaled = torch.abs(scaler(F_calc_complex,use_mask=False))  # Scaled amplitudes
        # Avoid log of zero by adding small epsilon
        F_obs_safe = torch.clamp(F_obs, min=eps)
        F_calc_safe = torch.clamp(F_calc_scaled, min=eps)
        # Compute log-ratio distribution: log(F_obs) - log(F_calc)
        log_ratio = torch.log(F_obs_safe) - torch.log(F_calc_safe)
        return log_ratio

    def get_outlier_statistics(self) -> Dict:
        """Get statistics about flagged outliers."""
        if self.outlier_flags is None:
            return {'n_outliers': 0, 'n_total': 0, 'fraction_outliers': 0.0}
        
        n_outliers = self.outlier_flags.sum().item()
        n_total = len(self.outlier_flags)
        
        stats = {
            'n_outliers': n_outliers,
            'n_total': n_total,
            'fraction_outliers': n_outliers / n_total if n_total > 0 else 0.0,
            'detection_params': self.outlier_detection_params
        }
        
        if self.resolution is not None:
            # Add resolution-dependent statistics
            outlier_resolutions = self.resolution[self.outlier_flags] if n_outliers > 0 else torch.tensor([])
            if len(outlier_resolutions) > 0:
                stats['outlier_resolution_stats'] = {
                    'min': outlier_resolutions.min().item(),
                    'max': outlier_resolutions.max().item(),
                    'mean': outlier_resolutions.mean().item(),
                    'median': outlier_resolutions.median().item()
                }
        
        return stats
    
    def unpack_one(self):
        """
        Unpack one level of source.
        Does not recurse fully.
        Also does not flag. 
        """
        if self.source is not None:
            return self.source
        return self
    
    def get_lognormal_sigma(self, F: Optional[torch.Tensor] = None, 
                           sigma_F: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Convert Gaussian parameters (F, sigma_F) to lognormal sigma parameter.
        
        For a lognormal distribution, if X ~ LogNormal(μ, σ²), then:
        - E[X] = exp(μ + σ²/2)
        - Var(X) = exp(2μ + σ²)(exp(σ²) - 1)
        
        Given observed F (mean) and sigma_F (standard deviation), we can solve for σ.
        
        Args:
            F: Structure factor amplitudes (uses self.F if None)
            sigma_F: Standard deviations (uses self.F_sigma if None)
            
        Returns:
            torch.Tensor: Sigma parameter for lognormal distribution
        """
        if F is None:
            F = self.F
        if sigma_F is None:
            sigma_F = self.F_sigma
            
        if F is None or sigma_F is None:
            raise ValueError("F and sigma_F must be provided or available in self")
            
        return gaussian_to_lognormal_sigma(F, sigma_F)


