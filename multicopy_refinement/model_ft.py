from multicopy_refinement.model import Model
import multicopy_refinement.math_numpy as mnp
import torch
import numpy as np
from multicopy_refinement.math_torch import find_relevant_voxels, vectorized_add_to_map, vectorized_add_to_map_aniso,ifft,extract_structure_factor_from_grid,fft,get_real_grid
import gemmi
import multicopy_refinement.get_scattering_factor_torch as gsf
from multicopy_refinement.map_symmetry import MapSymmetry
from torch import nn
from tqdm import tqdm


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

    def __init__(self, *args, max_res=1.0, radius=10, **kwargs):
        super().__init__(*args, **kwargs)
        
        # FT-specific storage
        self.parametrization = None  # ITC92 parameters: {element: (A, B, C)}
        self.max_res = max_res 
        self.radius = radius  # Default radius in voxels for density calculation
        self.real_space_grid = None
        self.map = None
        self.voxel_size = None
        self.map_symmetry = None  # MapSymmetry instance for applying space group symmetry
        self.gridsize = None
    
    def load_pdb_from_file(self, filename ,strip_H=True):
        """
        Load a PDB file and initialize the model with FT-specific setup.
        """
        super().load_pdb_from_file(filename, strip_H=strip_H)
        self._build_parametrization()
    
    def _build_parametrization(self):
        """
        Build ITC92 parametrization for all atoms in the model.
        Stores the parametrization dictionary: {element: (A, B, C)}
        """
        print("Building ITC92 parametrization...")
        self.parametrization = gsf.get_parameterization_extended(self.pdb)
        print(f"Parametrization built for {len(self.parametrization)} unique atom types")
        elements = self.pdb.element.tolist()

        self.A = torch.cat([self.parametrization[element][0] for element in elements],dim=0)
        self.B = torch.cat([self.parametrization[element][1] for element in elements],dim=0)
    
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

    def setup_grids(self, max_res=None, gridsize=None):
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
        print(f"Setting up grids with max_res={self.max_res} Å")
        if gridsize is not None:
            self.gridsize = gridsize
        self.real_space_grid = get_real_grid(self.cell, self.max_res, self.gridsize,device=self.xyz.device)
        self.map = torch.zeros(self.real_space_grid.shape[:-1], dtype=torch.float64, device=self.xyz.device)
        self.voxel_size = self.real_space_grid[1, 1, 1] - self.real_space_grid[0, 0, 0]
        
        # Initialize map symmetry operator
        if hasattr(self, 'spacegroup') and self.spacegroup is not None:
            self.map_symmetry = MapSymmetry(
                space_group=self.spacegroup,
                map_shape=self.map.shape,
                cell_params=self.cell
            )
        
        print(f"Grid shape: {self.map.shape}")
        print(f"Voxel size: {self.voxel_size}")
    
    def build_density_map(self, radius=None, apply_symmetry=True):
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
        if radius is not None:
            self.radius = radius
        
        if self.real_space_grid is None:
            self.setup_grids()
        
        print(f"Building density map with radius={self.radius} voxels...")
        
        # Reset map
        self.map = torch.zeros(self.real_space_grid.shape[:-1], dtype=torch.float64, device=self.map.device)
        
        # Add isotropic atoms
        xyz_iso, b_iso, occ_iso, A_iso, B_iso = self.get_iso()
        
        if len(xyz_iso) > 0:
            print(xyz_iso.shape, b_iso.shape, occ_iso.shape, A_iso.shape, B_iso.shape)
            print(f"  Adding {len(xyz_iso)} isotropic atoms...")
            surrounding_coords, voxel_indices = find_relevant_voxels(
                self.real_space_grid, xyz_iso, radius=self.radius, inv_frac_matrix=self.inv_fractional_matrix
            )
            self.map = vectorized_add_to_map(
                surrounding_coords, voxel_indices, self.map,
                xyz_iso, b_iso,
                self.inv_fractional_matrix, self.fractional_matrix,
                A_iso, B_iso, occ_iso
            )
        
        # Add anisotropic atoms
        xyz_aniso, u_aniso, occ_aniso, A_aniso, B_aniso = self.get_aniso()
        
        if len(xyz_aniso) > 0:
            print(f"  Adding {len(xyz_aniso)} anisotropic atoms...")
            surrounding_coords, voxel_indices = find_relevant_voxels(
                self.real_space_grid, xyz_aniso, radius=self.radius, inv_frac_matrix=self.inv_fractional_matrix
            )
            self.map = vectorized_add_to_map_aniso(
                surrounding_coords, voxel_indices, self.map,
                xyz_aniso, u_aniso,
                self.inv_fractional_matrix, self.fractional_matrix,
                A_aniso, B_aniso, occ_aniso
            )
        
        # Apply symmetry if requested
        if apply_symmetry and self.map_symmetry is not None:
            print(f"  Applying {self.map_symmetry.n_ops} symmetry operations...")
            self.map = self.map_symmetry(self.map)
        
        print(f"Density map built. Sum: {self.map.sum():.2f}, Max: {self.map.max():.4f}")
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
        print(f"Saving map to {filename}")
        print(f"  Map shape: {self.map.shape}")
        print(f"  Map sum: {self.map.sum():.2f}")
        print(f"  Map range: [{self.map.min():.4f}, {self.map.max():.4f}]")
        
        map_ccp = gemmi.Ccp4Map()
        map_ccp.grid = gemmi.FloatGrid(np_map, gemmi.UnitCell(*self.cell), gemmi.SpaceGroup('P1'))
        map_ccp.setup(0.0)
        map_ccp.update_ccp4_header()
        map_ccp.write_ccp4_map(filename)
        print(f"Map saved successfully")
    
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
        print("Rebuilding density map from scratch...")
        return self.build_density_map(radius=radius)
    
    def cuda(self):
        """Move model and FT-specific data to GPU."""
        super().cuda()
        
        # Move ITC92 parametrization tensors
        if hasattr(self, 'A') and self.A is not None:
            self.A = self.A.cuda()
        if hasattr(self, 'B') and self.B is not None:
            self.B = self.B.cuda()
        
        # Move grids
        if self.real_space_grid is not None:
            self.real_space_grid = self.real_space_grid.cuda()
            self.inv_fractional_matrix = self.inv_fractional_matrix.cuda()
            self.fractional_matrix = self.fractional_matrix.cuda()
        
        if self.map is not None:
            self.map = self.map.cuda()
        
        # Move map symmetry operator
        if self.map_symmetry is not None:
            self.map_symmetry = self.map_symmetry.cuda()
        
        return self
    
    def cpu(self):
        """Move model and FT-specific data to CPU."""
        super().cpu()
        
        # Move ITC92 parametrization tensors
        if hasattr(self, 'A') and self.A is not None:
            self.A = self.A.cpu()
        if hasattr(self, 'B') and self.B is not None:
            self.B = self.B.cpu()
        
        # Move grids
        if self.real_space_grid is not None:
            self.real_space_grid = self.real_space_grid.cpu()
            self.inv_fractional_matrix = self.inv_fractional_matrix.cpu()
            self.fractional_matrix = self.fractional_matrix.cpu()
        
        if self.map is not None:
            self.map = self.map.cpu()
        
        # Move map symmetry operator
        if self.map_symmetry is not None:
            self.map_symmetry = self.map_symmetry.cpu()
        
        return self
    
    def update_pdb(self):
        """
        Update PDB with current atomic parameters.
        """
        super().update_pdb()

    def get_structure_factor(self, hkl):
        self.build_density_map()
        self.reciprocal_space_grid = ifft(self.map)
        sf = extract_structure_factor_from_grid(self.reciprocal_space_grid, hkl)
        return sf
    
    def fft(self):
        """Perform FFT on the current reciprocal grid."""
        self.density = fft(self.reciprocal_space_grid)
        return self.density
    
    