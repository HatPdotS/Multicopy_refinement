from torchref.model.model import Model
import torch
import numpy as np
from torchref.math_functions.math_torch import find_relevant_voxels, vectorized_add_to_map, vectorized_add_to_map_aniso\
,ifft,extract_structure_factor_from_grid,fft,get_real_grid, find_grid_size,hash_tensors
import gemmi
import torchref.math_torch.get_scattering_factor_torch as gsf
from torchref.symmetrie.map_symmetry import MapSymmetry
from typing import Optional, Tuple
from torchref.utils import TensorDict

class ModelFT(Model):
    """
    ModelFT is a purpose-built subclass of model for Fourier Transform (FT) based 
    electron density map calculations and structure factor refinement.
    
    Key differences from base model:
    - Uses ITC92 parametrization for electron density calculations
    - Builds electron density maps in real space
    - Computes structure factors via FFT
    - No residue-level caching - uses direct atom access via get_iso/get_aniso
    """

    def __init__(self, *args, max_res=1.0, radius_angstrom=4.0, gridsize: Optional[Tuple[int, int, int]] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.map = None  # Electron density map
        self.max_res = max_res 
        self.radius_angstrom = radius_angstrom  # Radius in Angstroms for density calc
        self._cache = TensorDict()  # Cache for structure factors
        if gridsize is not None:
            self.register_buffer("gridsize", torch.tensor(gridsize, dtype=torch.int32))
        else:
            self.register_buffer("gridsize", None)

    def load_pdb(self, filename):
        """
        Load a PDB file and initialize the model with FT-specific setup.
        """
        super().load_pdb(filename)
        self._build_parametrization()
        self.setup_grid()
        return self

    def load_cif(self, filename):
        """
        Load a CIF file and initialize the model with FT-specific setup.
        """
        super().load_cif(filename)
        self._build_parametrization()
        self.setup_grid()
        return self
    
    def setup_gridsize(self, max_res=None):
        if max_res is not None:
            self.max_res = max_res
        if self.verbose > 1: print(f"Defining grid size for ={self.max_res} Å")
        return find_grid_size(self.cell, self.max_res)
    
    def _build_parametrization(self):
        """
        Build ITC92 parametrization for all atoms in the model.
        Stores the parametrization dictionary: {element: (A, B, C)}
        """
        if self.verbose > 1: print("Building ITC92 parametrization...")

        self.parametrization = gsf.get_parameterization_extended(self.pdb)
        if self.verbose > 0: print(f"Parametrization built for {len(self.parametrization)} unique atom types")
        if self.verbose > 1: print('Elements with parametrization:', list(self.parametrization.keys()))

        elements = self.pdb.element.tolist()

        self.register_buffer("A", torch.cat([self.parametrization[element][0] for element in elements],dim=0))
        self.register_buffer("B", torch.cat([self.parametrization[element][1] for element in elements],dim=0))

    def get_iso(self):
        """
        Get isotropic atoms with their ITC92 parameters.
        
        Returns:
        --------
        xyz : torch.Tensor (n_atoms, 3)
            Atomic coordinates
        b : torch.Tensor (n_atoms,)
            B-factors
        occupancy : torch.Tensor (n_atoms,)
            Occupancies
        A : torch.Tensor (n_atoms, 5)
            ITC92 A parameters (amplitudes)
        B : torch.Tensor (n_atoms, 5)
            ITC92 B parameters (widths)
        """
        # Get base isotropic data
        xyz, b, occupancy = super().get_iso()
        
        # Get elements for isotropic atoms
        iso_mask = ~self.aniso_flag
        A = self.A[iso_mask]
        B = self.B[iso_mask]
        
        return xyz, b, occupancy, A, B
    
    def get_aniso(self):
        """
        Get anisotropic atoms with their ITC92 parameters.
        
        Returns:
        --------
        xyz : torch.Tensor (n_atoms, 3)
            Atomic coordinates
        u : torch.Tensor (n_atoms, 6)
            Anisotropic U parameters
        occupancy : torch.Tensor (n_atoms,)
            Occupancies
        A : torch.Tensor (n_atoms, 5)
            ITC92 A parameters (amplitudes)
        B : torch.Tensor (n_atoms, 5)
            ITC92 B parameters (widths)
        """
        # Get base anisotropic data
        xyz, u, occupancy = super().get_aniso()
        
        # Get elements for anisotropic atoms
        aniso_mask = self.aniso_flag

        A = self.A[aniso_mask]  
        B = self.B[aniso_mask]
        
        return xyz, u, occupancy, A, B
    
    def setup_grid(self, max_res=None, gridsize=None):
        """
        Setup real-space grid for electron density calculation.
        
        Parameters:
        -----------
        max_res : float
            Maximum resolution for grid spacing (in Angstroms)
        gridsize : tuple, optional
            Explicit grid size (nx, ny, nz)
        """
        if max_res is not None:
            self.max_res = max_res
        if self.verbose > 1: print(f"Setting up grids with max_res={self.max_res} Å")
        if gridsize is not None:
            self.register_buffer("gridsize", torch.tensor(gridsize, dtype=torch.int32, device=self.device))
        else:
            self.register_buffer("gridsize", self.setup_gridsize(max_res=self.max_res).to(dtype=torch.int32).to(device=self.device))


        self.register_buffer("real_space_grid", get_real_grid(self.cell, gridsize=self.gridsize, device=self.device))
        self.register_buffer("voxel_size", self.real_space_grid[2, 2, 2] - self.real_space_grid[1, 1, 1])

        # Initialize map symmetry operator
        if hasattr(self, 'spacegroup') and self.spacegroup is not None:
            self.map_symmetry = MapSymmetry(
                space_group=self.spacegroup,
                map_shape=self.real_space_grid.shape[:-1],
                cell_params=self.cell, verbose=self.verbose, device=self.device
            )

        if self.verbose > 2: 
            print(f"Grid shape: {self.real_space_grid.shape[:-1]}")
            print(f"Voxel size: {self.voxel_size}")


    def get_radius(self, min_radius_Angstrom: float = 4.0):
        """
        Get the radius (in voxels) used for density calculation around each atom.
        """
        if not hasattr(self, 'real_space_grid') or self.real_space_grid is None:
            self.setup_grid(
            )
        voxel_size = self.real_space_grid[1, 1, 1] - self.real_space_grid[0, 0, 0]
        min_radius = torch.ceil(min_radius_Angstrom / torch.min(voxel_size)).to(torch.int32).item()
        if self.verbose > 1:
            print(f"Calculated radius for density calculation: {min_radius} voxels (voxel size: {voxel_size}), this corresponds to at least {min_radius_Angstrom} Å")
        return min_radius
    
    def build_complete_map(self, radius=None, apply_symmetry=True):
        """
        Build electron density map from all atoms.
        Uses get_iso() and get_aniso() to get atom data.
        
        Parameters:
        -----------
        radius : int or None
            Radius (in voxels) around each atom to compute density.
            If None, uses self.radius.
        apply_symmetry : bool, default True
            If True and space group is not P1, apply symmetry operations to the map
        
        Returns:
        --------
        map : torch.Tensor
            Electron density map (with symmetry applied if requested)
        """
        self.map = self.build_initial_map(apply_symmetry=apply_symmetry)

        if self.verbose > 2: print(f"Density map built. Sum: {self.map.sum():.2f}, Max: {self.map.max():.4f}")
        return self.map
    
    def build_initial_map(self, apply_symmetry=True):
        if not 'real_space_grid' in self._buffers:
            self.setup_grid()

        if self.verbose > 2: print(f"Building density map with radius={self.radius_angstrom} angstrom...")

        # Reset map
        self.map = torch.zeros(self.real_space_grid.shape[:-1], dtype=self.dtype_float, device=self.device)
        
        # Add isotropic atoms
        xyz_iso, b_iso, occ_iso, A_iso, B_iso = self.get_iso()
        assert torch.all(torch.isfinite(A_iso)), "Non-finite values found in A_iso during map building."
        assert torch.all(torch.isfinite(B_iso)), "Non-finite values found in B_iso during map building."
        assert torch.all(torch.isfinite(xyz_iso)), "Non-finite values found in xyz_iso during map building."
        assert torch.all(torch.isfinite(b_iso)), "Non-finite values found in b_iso during map building."
        assert torch.all(torch.isfinite(occ_iso)), "Non-finite values found in occ_iso during map building."

        if len(xyz_iso) > 0:
            if self.verbose > 3:
                print(xyz_iso.shape, b_iso.shape, occ_iso.shape, A_iso.shape, B_iso.shape)
            if self.verbose > 2:
                print(f"  Adding {len(xyz_iso)} isotropic atoms...")
            surrounding_coords, voxel_indices = find_relevant_voxels(
                self.real_space_grid, xyz_iso, radius_angstrom=self.radius_angstrom, inv_frac_matrix=self.inv_fractional_matrix
            )
            self.map = vectorized_add_to_map(
                surrounding_coords, voxel_indices, self.map,
                xyz_iso, b_iso,
                self.inv_fractional_matrix, self.fractional_matrix,
                A_iso, B_iso, occ_iso
            )
        assert torch.all(torch.isfinite(self.map)), "Non-finite values found in map after adding isotropic atoms."
        # Add anisotropic atoms
        xyz_aniso, u_aniso, occ_aniso, A_aniso, B_aniso = self.get_aniso()
        
        if len(xyz_aniso) > 0:
            if self.verbose > 2: print(f"  Adding {len(xyz_aniso)} anisotropic atoms...")
            surrounding_coords, voxel_indices = find_relevant_voxels(
                self.real_space_grid, xyz_aniso, radius_angstrom=self.radius_angstrom, inv_frac_matrix=self.inv_fractional_matrix
            )
            self.map = vectorized_add_to_map_aniso(
                surrounding_coords, voxel_indices, self.map,
                xyz_aniso, u_aniso,
                self.inv_fractional_matrix, self.fractional_matrix,
                A_aniso, B_aniso, occ_aniso
            )
        assert torch.all(torch.isfinite(self.map)), "Non-finite values found in map after adding anisotropic atoms."
        # Apply symmetry if requested
        if apply_symmetry and self.map_symmetry is not None:
            if self.verbose > 2: print(f"  Applying {self.map_symmetry.n_ops} symmetry operations...")
            self.map = self.map_symmetry(self.map)
            assert torch.all(torch.isfinite(self.map)), "Non-finite values found in map after applying symmetry."
        return self.map
    
    def save_map(self, filename):
        """
        Save the electron density map to a CCP4 format file.
        
        Parameters:
        -----------
        filename : str
            Output filename for the map
        """
        if self.map is None:
            raise ValueError("No map to save. Call build_density_map() first.")
        
        np_map = self.map.detach().cpu().numpy().astype(np.float32)
        cell = self.cell.tolist()
        if self.verbose > 0:
            print(f"Saving map to {filename}")
            print(f"  Map shape: {self.map.shape}")
            print(f"  Map sum: {self.map.sum():.2f}")
            print(f"  Map range: [{self.map.min():.4f}, {self.map.max():.4f}]")
        
        map_ccp = gemmi.Ccp4Map()
        map_ccp.grid = gemmi.FloatGrid(np_map, gemmi.UnitCell(*cell), gemmi.SpaceGroup('P1'))
        map_ccp.setup(0.0)
        map_ccp.update_ccp4_header()
        map_ccp.write_ccp4_map(filename)
        if self.verbose > 0: print(f"Map saved successfully")
    
    def get_map_statistics(self):
        """Get statistics about the current density map."""
        if self.map is None:
            return None
        
        stats = {
            'shape': self.map.shape,
            'sum': float(self.map.sum()),
            'mean': float(self.map.mean()),
            'std': float(self.map.std()),
            'min': float(self.map.min()),
            'max': float(self.map.max()),
            'n_positive': int((self.map > 0).sum()),
            'n_negative': int((self.map < 0).sum()),
        }
        return stats
    
    def rebuild_map(self, radius=None):
        """
        Rebuild the density map from scratch.
        Convenience method that clears and rebuilds everything.
        
        Parameters:
        -----------
        radius : int or None
            Radius (in voxels) around each atom.
            If None, uses self.radius. If specified, overrides self.radius.
        """
        if self.verbose > 1: print("Rebuilding density map from scratch...")
        return self.build_density_map(radius=radius)
    
    def cuda(self, device=None):
        """Move model and FT-specific data to GPU."""
        super().cuda(device)
        if self.map is not None:
            self.map = self.map.cuda(device)
        return self
    
    def cpu(self):
        """Move model and FT-specific data to CPU."""
        super().cpu()
        if self.map is not None:
            self.map = self.map.cpu()
        return self
    
    def update_pdb(self):
        """
        Update PDB with current atomic parameters.
        """
        super().update_pdb()
    
    def reset_cache(self):
        self._cache = dict()

    def get_structure_factor(self, hkl: torch.Tensor, recalc=False) -> torch.Tensor:
        """
        Get structure factors for given hkl reflections.
        Uses caching to avoid recomputation if parameters haven't changed.
        
        Parameters:
        -----------
        hkl : torch.Tensor (n_reflections, 3)
            Miller indices
        recalc : bool
            If True, forces recalculation even if cached
            
        Returns:
        --------
        sf : torch.Tensor (n_reflections,)
            Complex structure factors
        """
        # Compute current parameter hash
        params = (*self.parameters(),hkl)
        current_param_hash = hash_tensors(params)
    
        key = current_param_hash
        if not recalc and key in self._cache:
            if self.verbose > 2:
                print("Using cached structure factors")
            return self._cache[key]
        
        # Build map and compute structure factors
        self.build_complete_map()
        self.reciprocal_space_grid = ifft(self.map)
        sf = extract_structure_factor_from_grid(self.reciprocal_space_grid, hkl)
        
        self._cache[key] = sf
        
        return sf
    
    def fft(self):
        """Perform FFT on the current reciprocal grid."""
        self.density = fft(self.reciprocal_space_grid)
        return self.density
    
    def forward(self, hkl, recalc=False) -> torch.Tensor:
        """
        Forward pass to compute structure factors for given hkl.
        
        Parameters:
        -----------
        hkl : torch.Tensor (n_reflections, 3)
            Miller indices
            
        Returns:
        --------
        F_calc : torch.Tensor (n_reflections,)
            Calculated complex structure factors
        """
        return self.get_structure_factor(hkl,recalc=recalc)


        



