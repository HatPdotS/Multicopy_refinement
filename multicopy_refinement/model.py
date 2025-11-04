


import torch
import torch.nn as nn
from typing import Optional, Union
import pdb_tools
import multicopy_refinement.symmetrie as sym
import multicopy_refinement.get_scattering_factor_torch as gsf
import multicopy_refinement.symmetrie as sym
import multicopy_refinement.math_numpy as mnp
import numpy as np
from multicopy_refinement import math_torch

class Model(nn.Module):
    def __init__(self,dtype_float=torch.float32,verbose=1,device=torch.device('cpu')):
        super().__init__()
        self.altloc_pairs = []
        self.verbose = verbose
        self.initialized = False
        self.dtype_float = dtype_float
        self.device = device
    
    def __bool__(self):
        """Return the initialization status when used in boolean context."""
        return self.initialized
    
    def load_pdb_from_file(self,file,strip_H=True):
        self.pdb = pdb_tools.load_pdb_as_pd(file)
        self.pdb = self.pdb.loc[self.pdb['element'] != 'H'].reset_index(drop=True) if strip_H else self.pdb
        self.pdb['index'] = self.pdb.index.to_numpy(dtype=int)
        self.register_buffer('cell',torch.tensor(self.pdb.attrs['cell'],requires_grad=False,dtype=self.dtype_float,device=self.device))
        self.spacegroup = self.pdb.attrs['spacegroup']
        self.spacegroup_function = sym.Symmetry(self.spacegroup)

        # Register buffers for various matrices
        self.register_buffer('inv_fractional_matrix',torch.tensor(mnp.get_inv_fractional_matrix(self.cell),dtype=self.dtype_float,requires_grad=False))
        self.register_buffer('fractional_matrix',torch.tensor(mnp.get_fractional_matrix(self.cell),dtype=self.dtype_float,requires_grad=False))
        self.register_buffer('aniso_flag',torch.tensor(self.pdb['anisou_flag'].values,dtype=torch.bool))
        self.register_buffer('recB', math_torch.reciprocal_basis_matrix(self.cell).to(dtype=self.dtype_float).to(self.device))
        
        # Create MixedTensors for model parameters
        self.xyz = MixedTensor(torch.tensor(self.pdb[['x', 'y', 'z']].values,dtype=self.dtype_float), name='xyz')
        self.b = MixedTensor(torch.tensor(self.pdb['tempfactor'].values,dtype=self.dtype_float), name='b_factor')
        self.u = MixedTensor(torch.tensor(self.pdb[['u11', 'u22', 'u33', 'u12', 'u13', 'u23']].values,dtype=self.dtype_float), name='aniso_U')
        
        # Create OccupancyTensor with residue-level sharing and altloc support
        initial_occ = torch.tensor(self.pdb['occupancy'].values, dtype=self.dtype_float)
        sharing_groups, altloc_groups, refinable_mask = self._create_occupancy_groups(self.pdb, initial_occ)
        self.occupancy = OccupancyTensor(
            initial_values=initial_occ,
            sharing_groups=sharing_groups,
            altloc_groups=altloc_groups,
            refinable_mask=refinable_mask,
            dtype=self.dtype_float,
            device=self.device,
            name='occupancy'
        )

        self.set_default_masks()
        self.register_alternative_conformations()
        self.initialized = True
        return self
    
    def _create_occupancy_groups(self, pdb_df, initial_occ):
        """
        Create sharing groups and altloc groups for occupancy.
        
        Logic:
        1. Group atoms by residue (resname, resseq, chainid, altloc)
        2. Within each residue, if all occupancies are within 0.01 tolerance,
           create a sharing group
        3. Identify alternative conformations and create altloc groups
        4. Only refine occupancies that differ from 1.0
        
        Args:
            pdb_df: PDB DataFrame
            initial_occ: Tensor of initial occupancy values
        
        Returns:
            tuple: (sharing_groups_tensor, altloc_groups, refinable_mask)
                sharing_groups_tensor: Tensor of shape (n_atoms,) where each value is the
                                      collapsed index for that atom
                altloc_groups: List of tuples of atom index lists for alternative conformations
                refinable_mask: Boolean tensor indicating which atoms should be refined
        """
        n_atoms = len(initial_occ)
        altloc_groups = []
        refinable_mask = torch.zeros(n_atoms, dtype=torch.bool)
        
        # Initialize sharing groups tensor - each atom maps to its own index initially
        sharing_groups_tensor = torch.arange(n_atoms, dtype=torch.long)
        
        # First pass: create sharing groups by residue (including altloc)
        grouped = pdb_df.groupby(['resname', 'resseq', 'chainid', 'altloc'])
        
        collapsed_idx = 0
        residue_sharing_groups = {}  # Map (resname, resseq, chainid, altloc) -> collapsed index
        
        for (resname, resseq, chainid, altloc), group in grouped:
            indices = group['index'].tolist()
            
            if len(indices) == 0:
                continue
            
            # Get occupancies for this residue
            residue_occs = initial_occ[indices]
            
            # Check if all occupancies are within tolerance
            occ_min = residue_occs.min().item()
            occ_max = residue_occs.max().item()
            occ_mean = residue_occs.mean().item()
            
            if (occ_max - occ_min) <= 0.01:
                # All atoms in residue have similar occupancy - create sharing group
                if len(indices) > 1:
                    # Assign all atoms in this group to the same collapsed index
                    sharing_groups_tensor[indices] = collapsed_idx
                    residue_sharing_groups[(resname, resseq, chainid, altloc)] = collapsed_idx
                    collapsed_idx += 1
                
                # Only refine if mean occupancy differs from 1.0
                if abs(occ_mean - 1.0) > 0.01:
                    for idx in indices:
                        refinable_mask[idx] = True
            else:
                # Occupancies differ within residue - each atom independent
                # Refine those that differ from 1.0
                for idx in indices:
                    if abs(initial_occ[idx].item() - 1.0) > 0.01:
                        refinable_mask[idx] = True
        
        # Second pass: identify alternative conformations
        # Group by residue without altloc to find residues with multiple conformations
        pdb_with_altlocs = pdb_df[pdb_df['altloc'] != '']
        
        if len(pdb_with_altlocs) > 0:
            grouped_by_residue = pdb_with_altlocs.groupby(['resname', 'resseq', 'chainid'])
            
            for (resname, resseq, chainid), group in grouped_by_residue:
                unique_altlocs = sorted(group['altloc'].unique())
                
                # Only create altloc group if there are multiple conformations
                if len(unique_altlocs) > 1:
                    conformation_atom_lists = []
                    
                    for altloc in unique_altlocs:
                        # Get all atoms for this specific altloc
                        altloc_atoms = group[group['altloc'] == altloc]
                        indices = altloc_atoms['index'].tolist()
                        conformation_atom_lists.append(indices)
                    
                    # Add to altloc_groups
                    altloc_groups.append(tuple(conformation_atom_lists))
        
        # Compact the indices - make them contiguous from 0 to n_collapsed-1
        unique_indices = torch.unique(sharing_groups_tensor, sorted=True)
        index_map = torch.zeros(n_atoms, dtype=torch.long)
        for new_idx, old_idx in enumerate(unique_indices):
            mask = (sharing_groups_tensor == old_idx)
            sharing_groups_tensor[mask] = new_idx
        
        n_collapsed = len(unique_indices)
        
        if self.verbose > 0:
            n_groups = n_collapsed
            n_independent = n_atoms - n_collapsed  # Atoms not sharing with others
            n_refinable = refinable_mask.sum().item()
            n_altloc_groups = len(altloc_groups)
            
            print(f"\nOccupancy Setup:")
            print(f"  Total atoms: {n_atoms}")
            print(f"  Collapsed indices: {n_collapsed}")
            print(f"  Alternative conformation groups: {n_altloc_groups}")
            print(f"  Refinable atoms: {n_refinable}")
            print(f"  Compression ratio: {n_atoms / n_collapsed:.2f}x")
        
        return sharing_groups_tensor, altloc_groups, refinable_mask

    def update_pdb(self):
        self.pdb.loc[:, ['x', 'y', 'z']] = self.xyz().cpu().detach().numpy()
        self.pdb.loc[:, ['u11', 'u22', 'u33', 'u12', 'u13', 'u23']] = self.u().cpu().detach().numpy()
        self.pdb.loc[:, 'tempfactor'] = self.b().cpu().detach().numpy()
        self.pdb.loc[:, 'occupancy'] = self.occupancy().cpu().detach().numpy()
        return self.pdb
    
    def get_vdw_radii(self):
        """
        Get van der Waals radii for all atoms in the model based on their elements.
        Caches the result in self.vdw_radii for future calls.
        
        Returns:
        --------
        self.vdw_radii : torch.Tensor (n_atoms,)
            Van der Waals radii for each atom
        """
        import os
        import pandas as pd
        if hasattr(self, 'vdw_radii'):
            return self.vdw_radii
        elements = self.pdb.loc[(self.occupancy() > 0).detach().cpu().numpy(), 'element']
        path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'caching/files/atomic_vdw_radii.csv')
        vdw_df = pd.read_csv(path, comment='#')   
        vdw_radii = vdw_df.set_index('element').loc[elements]['vdW_Radius_Angstrom'].values
        self.register_buffer('vdw_radii', torch.tensor(vdw_radii, dtype=self.dtype_float, device=self.device))
        return self.vdw_radii

    def cuda(self, device: Optional[Union[int, torch.device]] = None):
        super().cuda(device)
        if self.altloc_pairs:
            self.altloc_pairs = [tuple(tensor.cuda(device) for tensor in group) for group in self.altloc_pairs]
        self.device = torch.device('cuda')
        print(f"Model moved to device: {self.device}")
        return self
    

    def cpu(self):
        super().cpu()
        if self.altloc_pairs:
            self.altloc_pairs = [tuple(tensor.cpu() for tensor in group) for group in self.altloc_pairs]
        self.device = torch.device('cpu')
        print(f"Model moved to device: {self.device}")
        return self
    
    def write_pdb(self, filename):
        self.update_pdb()
        pdb_tools.write_file(self.pdb, filename)

    def get_iso(self):
        xyz = self.xyz()[~self.aniso_flag]
        b = self.b()[~self.aniso_flag]
        occupancy = self.occupancy()[~self.aniso_flag]
        return xyz, b, occupancy

    def set_default_masks(self):
        self.xyz.refine_all()
        b_mask = ~self.b().detach().isnan() 
        self.b.update_refinable_mask(b_mask)
        u_mask = ~self.u().detach().isnan().any(dim=1)
        self.u.update_refinable_mask(u_mask)
        # Occupancy mask is set in _create_occupancy_groups() during initialization

    def get_aniso(self):
        xyz = self.xyz()[self.aniso_flag]
        u = self.u()[self.aniso_flag]
        occupancy = self.occupancy()[self.aniso_flag]
        return xyz, u, occupancy
    
    def parameters(self, recurse = True):
        return super().parameters(recurse)
    
    def named_mixed_tensors(self):
        """
        Iterate over all MixedTensor attributes with their names.
        
        Yields:
            Tuple of (name, MixedTensor)
        """
        for name, module in self.named_modules():
            if isinstance(module, MixedTensor) and module != self:
                yield name, module
    
    def print_parameters_info(self):
        """Print information about all MixedTensor parameters."""
        print("=" * 80)
        print("Model Parameters Summary")
        print("=" * 80)
        for attr_name, mixed_tensor in self.named_mixed_tensors():
            print(f"\n{attr_name}: {mixed_tensor}")
            if mixed_tensor.get_refinable_count() > 0:
                print(f"  Refinable values: min={mixed_tensor.refinable_params.min().item():.4f}, "
                      f"max={mixed_tensor.refinable_params.max().item():.4f}, "
                      f"mean={mixed_tensor.refinable_params.mean().item():.4f}")
        print("=" * 80)

    def register_alternative_conformations(self):
        """
        Identify and register all alternative conformation groups in the structure.
        
        For each residue that has alternative conformations (altloc A, B, C, etc.),
        this method identifies all atoms belonging to each conformation and stores
        their indices as tensors in a tuple.
        
        The result is stored in self.altloc_pairs as a list of tuples, where each
        tuple contains tensors of atom indices for each alternative conformation
        of a residue.
        
        Example:
            For a residue with conformations A and B:
            - Conformation A has atoms at indices [100, 101, 102, ...]
            - Conformation B has atoms at indices [110, 111, 112, ...]
            Result: [(tensor([100, 101, 102, ...]), tensor([110, 111, 112, ...])), ...]
            
            For a residue with conformations A, B, C:
            [(tensor([200, 201, ...]), tensor([210, 211, ...]), tensor([220, 221, ...])), ...]
        """
        # Initialize the list to store alternative conformation groups
        self.altloc_pairs = []
        
        # Get all atoms with alternative conformations (non-empty altloc field)
        pdb_with_altlocs = self.pdb[self.pdb['altloc'] != '']
        
        if len(pdb_with_altlocs) == 0:
            # No alternative conformations in this structure
            return
        
        # Group by residue (resname, resseq, chainid) to find all residues
        # that have alternative conformations
        grouped = pdb_with_altlocs.groupby(['resname', 'resseq', 'chainid'])
        
        for (resname, resseq, chainid), group in grouped:
            # Get all unique altloc identifiers for this residue
            unique_altlocs = sorted(group['altloc'].unique())
            
            # Only register if there are actually multiple conformations
            if len(unique_altlocs) > 1:
                # For each altloc, collect all atom indices belonging to that conformation
                conformation_tensors = []
                for altloc in unique_altlocs:
                    # Get all atoms for this specific altloc
                    altloc_atoms = group[group['altloc'] == altloc]
                    # Get their indices and convert to tensor
                    indices = torch.tensor(altloc_atoms['index'].tolist(), dtype=torch.long)
                    conformation_tensors.append(indices)
                
                # Store as a tuple of tensors
                self.altloc_pairs.append(tuple(conformation_tensors))

    def shake_coords(self, stddev: float):
        """
        Apply random Gaussian noise to atomic coordinates.
        
        This method perturbs the atomic coordinates by adding Gaussian noise
        with a specified standard deviation. The noise is applied to all atoms
        in the model.
        
        Args:
            stddev: Standard deviation of the Gaussian noise to be added (in Å).
        """
        xyz = self.xyz().detach()
        new_xyz = xyz + torch.normal(mean=0.0, std=stddev, size=xyz.shape)
        self.xyz = MixedTensor(new_xyz, refinable_mask=self.xyz.refinable_mask, name='xyz')
   
    def shake_b_factors(self, stddev: float):
        """
        Apply random Gaussian noise to B-factors (temperature factors).
        
        This method perturbs the B-factors by adding Gaussian noise
        with a specified standard deviation. The noise is applied to all atoms
        in the model.
        
        Args:
            stddev: Standard deviation of the Gaussian noise to be added (in 1/Å**2).
        """
        b_factors = self.b().detach()
        new_b = b_factors + torch.normal(mean=0.0, std=stddev, size=b_factors.shape)
        self.b = MixedTensor(new_b, refinable_mask=self.b.refinable_mask, name='b_factor')

class MixedTensor(nn.Module):
    """
    A wrapper class to handle tensors where part of the tensor should be fixed 
    and another part should be optimized during refinement.
    
    This class stores a mask indicating which elements can be refined, and maintains
    both fixed and refinable components separately. The full tensor is reconstructed
    on-the-fly when accessed.
    
    Example:
        >>> # Create a tensor with 100 elements, where only indices 20-30 are refinable
        >>> mask = torch.zeros(100, dtype=torch.bool)
        >>> mask[20:30] = True
        >>> initial_values = torch.randn(100)
        >>> mixed = MixedTensor(initial_values, refinable_mask=mask, requires_grad=True)
        >>> 
        >>> # Use it in optimization
        >>> optimizer = torch.optim.Adam([mixed.refinable_params], lr=0.01)
        >>> loss = (mixed() - target).pow(2).sum()
        >>> loss.backward()
        >>> optimizer.step()
    """
    
    def __init__(
        self, 
        initial_values: torch.Tensor, 
        refinable_mask: Optional[torch.Tensor] = None,
        requires_grad: bool = True,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        name: Optional[str] = None
    ):
        """
        Initialize a MixedTensor.
        
        Args:
            initial_values: Initial tensor values for all elements
            refinable_mask: Boolean mask indicating which elements can be refined.
                          If None, all elements are refinable.
            requires_grad: Whether refinable parameters should have gradients
            dtype: Data type for the tensor (default: same as initial_values)
            device: Device for the tensor (default: same as initial_values)
            name: Optional name for this parameter (useful for debugging/logging)
        """
        super().__init__()
        
        # Store the name
        self._name = name
        
        if dtype is None:
            dtype = initial_values.dtype
        if device is None:
            device = initial_values.device
            
        initial_values = initial_values.to(dtype=dtype, device=device)
        
        # Create refinable mask
        if refinable_mask is None:
            refinable_mask = torch.ones_like(initial_values, dtype=torch.bool)
        else:
            refinable_mask = refinable_mask.to(device=device)
            
        if refinable_mask.shape != initial_values.shape:
            raise ValueError(
                f"refinable_mask shape {refinable_mask.shape} must match "
                f"initial_values shape {initial_values.shape}"
            )
        
        # Store the mask as a buffer (not a parameter, won't be optimized)
        self.register_buffer('refinable_mask', refinable_mask)
        self.register_buffer('fixed_mask', ~refinable_mask)
        
        # Store fixed values as a buffer
        fixed_values = initial_values.clone().detach()
        self.register_buffer('fixed_values', fixed_values)
        
        # Store refinable values as a parameter (will be optimized)
        refinable_values = initial_values[refinable_mask].clone().detach()
        self.refinable_params = nn.Parameter(
            refinable_values, 
            requires_grad=requires_grad
        )
        
        # Store shape for reconstruction
        self.register_buffer('_shape', torch.tensor(initial_values.shape))
    
    def forward(self) -> torch.Tensor:
        """
        Reconstruct and return the full tensor by combining fixed and refinable parts.
        
        Returns:
            Full tensor with fixed values in non-refinable positions and 
            current refinable parameter values in refinable positions.
        """
        # Start with fixed values
        result = self.fixed_values.clone()
        
        # Insert refinable values at their positions
        result[self.refinable_mask] = self.refinable_params
        
        return result
    
    def __call__(self) -> torch.Tensor:
        """Allow instance to be called like a function."""
        return self.forward()
    
    @property
    def shape(self):
        """Return the shape of the full tensor."""
        return tuple(self._shape.tolist())
    
    @property
    def dtype(self):
        """Return the dtype of the tensor."""
        return self.fixed_values.dtype
    
    @property
    def device(self):
        """Return the device of the tensor."""
        return self.fixed_values.device
    
    def get_refinable_count(self) -> int:
        """Return the number of refinable parameters."""
        return self.refinable_mask.sum().item()
    
    def get_fixed_count(self) -> int:
        """Return the number of fixed parameters."""
        return self.fixed_mask.sum().item()
    
    def update_fixed_values(self, new_values: torch.Tensor):
        """
        Update the fixed values (does not affect refinable parameters).
        
        Args:
            new_values: New tensor values. Only fixed positions will be updated.
        """
        if new_values.shape != self.shape:
            raise ValueError(
                f"new_values shape {new_values.shape} must match "
                f"tensor shape {self.shape}"
            )
        self.fixed_values = new_values.to(dtype=self.dtype, device=self.device).detach()
    
    def update_refinable_mask(self, new_mask: torch.Tensor, reset_refinable: bool = False):
        """
        Update which elements are refinable. This is an advanced operation.
        
        Args:
            new_mask: New boolean mask indicating refinable elements
            reset_refinable: If True, reset refinable parameters to current fixed values.
                           If False, keep existing refinable parameter values where possible.
        """
        if new_mask.shape[0] != self.shape[0]:
            raise ValueError(
                f"new_mask shape {new_mask.shape} must match "
                f"tensor shape {self.shape}"
            )
        
        current_full = self.forward().detach()
        
        self.refinable_mask = new_mask.to(device=self.device)
        self.fixed_mask = ~new_mask
        
        if reset_refinable:
            # Reset refinable params to current fixed values
            self.fixed_values = current_full.clone()
            new_refinable = current_full[self.refinable_mask].clone()
        else:
            # Try to preserve existing refinable values where masks overlap
            new_refinable = current_full[self.refinable_mask].clone()
        
        # Replace the parameter
        self.refinable_params = nn.Parameter(
            new_refinable,
            requires_grad=self.refinable_params.requires_grad
        )
    
    def detach(self) -> torch.Tensor:
        """Return a detached copy of the full tensor."""
        return self.forward().detach()
    
    def clone(self) -> 'MixedTensor':
        """Create a deep copy of this MixedTensor."""
        new_mixed = MixedTensor(
            self.forward().detach(),
            self.refinable_mask.clone(),
            requires_grad=self.refinable_params.requires_grad,
            dtype=self.dtype,
            device=self.device,
            name=self.name
        )
        return new_mixed
    
    def clip(self, min_value=None, max_value=None) -> 'MixedTensor':
        """Clip the full tensor values between min_value and max_value."""
        full_tensor = self.forward()
        clipped_tensor = full_tensor
        if min_value is not None:
            clipped_tensor = torch.clamp(clipped_tensor, min=min_value)
        if max_value is not None:
            clipped_tensor = torch.clamp(clipped_tensor, max=max_value)
        new_mixed = MixedTensor(
            clipped_tensor.detach(),
            self.refinable_mask.clone(),
            requires_grad=self.refinable_params.requires_grad,
            dtype=self.dtype,
            device=self.device,
            name=self.name
        )
        return new_mixed
    
    def to(self, *args, **kwargs) -> 'MixedTensor':
        """Move tensor to a different device or dtype."""
        super().to(*args, **kwargs)
        return self
    
    def refine(self, selection: Union[slice, torch.Tensor, tuple], reset_values: bool = False):
        """
        Make a selection of the tensor refinable.
        
        Args:
            selection: Boolean mask, slice, or index selection indicating which 
                      elements should become refinable. Can be:
                      - Boolean tensor of same shape as the full tensor
                      - Slice object (e.g., slice(10, 20))
                      - Tuple of indices for multidimensional tensors
                      - Integer indices
            reset_values: If True, reset the selected elements to their current 
                         fixed values before making them refinable.
        
        Example:
            >>> mixed.make_refinable(slice(10, 20))  # Make elements 10-19 refinable
            >>> mixed.make_refinable(mask)  # Make elements where mask is True refinable
        """
        # Get current full tensor
        current_full = self.forward().detach()
        
        # Create a new mask that combines old refinable + new selection
        new_mask = self.refinable_mask.clone()
        
        if isinstance(selection, torch.Tensor):
            if selection.dtype == torch.bool:
                if selection.shape != self.shape:
                    raise ValueError(
                        f"Boolean selection shape {selection.shape} must match "
                        f"tensor shape {self.shape}"
                    )
                new_mask |= selection.to(device=self.device)
            else:
                # Integer indices
                temp_mask = torch.zeros_like(new_mask)
                temp_mask[selection] = True
                new_mask |= temp_mask
        else:
            # Handle slice or tuple indices
            temp_mask = torch.zeros_like(new_mask)
            temp_mask[selection] = True
            new_mask |= temp_mask
        
        # Update the mask
        self.refinable_mask = new_mask
        self.fixed_mask = ~new_mask
        
        # Update values
        if reset_values:
            self.fixed_values = current_full.clone()
        
        # Reconstruct refinable parameters with new selection
        new_refinable = current_full[self.refinable_mask].clone()
        self.refinable_params = nn.Parameter(
            new_refinable,
            requires_grad=self.refinable_params.requires_grad
        )
    
    def fix(self, selection: Union[slice, torch.Tensor, tuple], freeze_at_current: bool = True):
        """
        Make a selection of the tensor fixed (non-refinable).
        
        Args:
            selection: Boolean mask, slice, or index selection indicating which 
                      elements should become fixed. Can be:
                      - Boolean tensor of same shape as the full tensor
                      - Slice object (e.g., slice(10, 20))
                      - Tuple of indices for multidimensional tensors
                      - Integer indices
            freeze_at_current: If True (default), freeze the selected elements at 
                             their current values. If False, they revert to the 
                             original fixed values.
        
        Example:
            >>> mixed.make_fixed(slice(10, 20))  # Fix elements 10-19
            >>> mixed.make_fixed(mask)  # Fix elements where mask is True
        """
        # Get current full tensor
        current_full = self.forward().detach()
        
        # Create a new mask that removes the selection from refinable
        new_mask = self.refinable_mask.clone()
        
        if isinstance(selection, torch.Tensor):
            if selection.dtype == torch.bool:
                if selection.shape != self.shape:
                    raise ValueError(
                        f"Boolean selection shape {selection.shape} must match "
                        f"tensor shape {self.shape}"
                    )
                new_mask &= ~selection.to(device=self.device)
            else:
                # Integer indices
                temp_mask = torch.zeros_like(new_mask)
                temp_mask[selection] = True
                new_mask &= ~temp_mask
        else:
            # Handle slice or tuple indices
            temp_mask = torch.zeros_like(new_mask)
            temp_mask[selection] = True
            new_mask &= ~temp_mask
        
        # Update the mask
        self.refinable_mask = new_mask
        self.fixed_mask = ~new_mask
        
        # Update fixed values
        if freeze_at_current:
            self.fixed_values = current_full.clone()
        
        # Reconstruct refinable parameters without the fixed selection
        if self.refinable_mask.any():
            new_refinable = current_full[self.refinable_mask].clone()
            self.refinable_params = nn.Parameter(
                new_refinable,
                requires_grad=self.refinable_params.requires_grad
            )
        else:
            # All fixed, create empty parameter
            self.refinable_params = nn.Parameter(
                torch.tensor([], dtype=self.dtype, device=self.device),
                requires_grad=self.refinable_params.requires_grad
            )
    
    def refine_all(self):
        """Make all elements refinable."""
        all_true = torch.ones_like(self.refinable_mask)
        self.refine(all_true)
    
    def fix_all(self, freeze_at_current: bool = True):
        """Make all elements fixed."""
        all_true = torch.ones_like(self.refinable_mask)
        self.fix(all_true, freeze_at_current=freeze_at_current)

    @property
    def name(self) -> Optional[str]:
        """Return the name of this parameter."""
        return self._name
    
    @name.setter
    def name(self, value: str):
        """Set the name of this parameter."""
        self._name = value
    
    def __repr__(self) -> str:
        name_str = f"'{self.name}', " if self.name is not None else ""
        return (
            f"MixedTensor({name_str}shape={self.shape}, dtype={self.dtype}, "
            f"device={self.device}, refinable={self.get_refinable_count()}, "
            f"fixed={self.get_fixed_count()})"
        )
    
    def __str__(self) -> str:
        """More detailed string representation."""
        name_str = f" '{self.name}'" if self.name is not None else ""
        return (
            f"MixedTensor{name_str}:\n"
            f"  Shape: {self.shape}\n"
            f"  Dtype: {self.dtype}\n"
            f"  Device: {self.device}\n"
            f"  Refinable: {self.get_refinable_count()} / {self.refinable_mask.numel()}\n"
            f"  Fixed: {self.get_fixed_count()} / {self.refinable_mask.numel()}\n"
            f"  Requires grad: {self.refinable_params.requires_grad}"
        )
    
    def parameters(self):
        """Return refinable parameters for optimizer."""
        yield self.refinable_params


class OccupancyTensor(MixedTensor):
    """
    A specialized MixedTensor for handling occupancy parameters in crystallographic refinement.
    
    This class handles the specific constraints and requirements for occupancy:
    1. Values are bounded between 0 and 1 using sigmoid reparameterization
    2. Atoms can share occupancies (e.g., all atoms in a residue)
    3. Alternative conformations automatically sum to 1.0 via normalization
    4. Supports per-atom, per-residue, or custom grouping schemes
    5. Memory-efficient: stores only one parameter per sharing group
    6. Static (fixed) occupancies that never change during refinement
    7. Fully vectorized collapse/expand operations using scatter_add and indexing
    
    The internal representation:
    - Stores COLLAPSED logit values (one per unique group, plus one per ungrouped atom)
    - Uses an index tensor (expansion_mask) to map atoms to collapsed indices
    - Collapse: scatter_add to sum values by collapsed index (O(n))
    - Expand: direct indexing collapsed_values[expansion_mask] (O(n))
    - Transforms to [0,1] occupancies via sigmoid function during forward pass
    
    Alternative Conformation Handling (New Normalization Approach):
    - Altloc groups are represented as tensors of shape (N_groups, M_conformations)
    - Grouped by number of conformations: linked_occ_2, linked_occ_3, etc.
    - During forward(), all members are passed through sigmoid, then normalized:
      occupancy_i = sigmoid(logit_i) / sum_j(sigmoid(logit_j))
    - This enforces sum-to-1 constraint while keeping all parameters refinable
    - More stable gradients than placeholder approach
    - Handles arbitrary N-way splits (not just 2-way)
    
    Example:
        >>> # Create occupancy with sharing groups as an index tensor
        >>> # Atoms 0,1 share (index 0), atoms 2,3 share (index 1), atoms 4,5 share (index 2)
        >>> sharing_groups = torch.tensor([0, 0, 1, 1, 2, 2])
        >>> occ = OccupancyTensor(
        ...     initial_values=torch.tensor([1.0, 1.0, 0.7, 0.7, 0.3, 0.3]),
        ...     sharing_groups=sharing_groups,
        ...     altloc_groups=[([2, 3], [4, 5])],  # indices 1 and 2 are altlocs
        ... )
        >>> # All parameters refinable, normalization ensures sum-to-1
        >>> result = occ()  # Atoms 2-3 and 4-5 will sum to 1.0
        >>> # Stored as linked_occ_2 = tensor([[1, 2]]) - 2-way split
    """
    
    def __init__(
        self,
        initial_values: torch.Tensor,
        sharing_groups: Optional[torch.Tensor] = None,
        altloc_groups: Optional[list] = None,
        refinable_mask: Optional[torch.Tensor] = None,
        requires_grad: bool = True,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        name: Optional[str] = None,
        use_sigmoid: bool = True
    ):
        """
        Initialize an OccupancyTensor with collapsed storage and altloc support.
        
        Args:
            initial_values: Initial occupancy values for ALL atoms (should be in [0, 1])
            sharing_groups: Tensor of shape (n_atoms,) where each value is the collapsed index
                          for that atom. If None, each atom has independent occupancy.
                          Example: tensor([0, 0, 0, 1, 1, 2]) means atoms 0,1,2 share one occupancy,
                          atoms 3,4 share another, and atom 5 is independent.
            altloc_groups: List of tuples of atom index lists representing alternative
                          conformations. Each tuple contains the atom indices for each
                          conformation. Example: [([10,11], [12,13]), ([20,21], [22,23])]
                          means atoms 10,11 (conf A) and 12,13 (conf B) are altlocs,
                          and atoms 20,21 (conf C) and 22,23 (conf D) are altlocs.
                          These automatically sum to 1.0.
            refinable_mask: Boolean mask for which ATOMS can be refined (in full tensor space).
                          Applied after sharing groups. If any atom in a group is refinable,
                          the entire group becomes refinable. Altloc placeholders are never
                          directly refinable (computed from other altlocs).
            requires_grad: Whether refinable parameters should have gradients
            dtype: Data type for the tensor
            device: Device for the tensor
            name: Optional name for this parameter
            use_sigmoid: If True, use sigmoid parameterization to bound values to [0,1].
        """
        # Store configuration
        self.use_sigmoid = use_sigmoid
        self._full_shape = initial_values.shape[0]
        
        # Initialize Module first (required before register_buffer)
        nn.Module.__init__(self)
        
        self._name = name or 'occupancy'
        
        if dtype is None:
            dtype = initial_values.dtype
        if device is None:
            device = initial_values.device
        
        # Validate initial values are in valid range
        if self.use_sigmoid:
            if torch.any(initial_values < 0) or torch.any(initial_values > 1):
                raise ValueError("Initial occupancy values must be in range [0, 1]")
        
        # Process sharing groups and altlocs, create expansion mask
        self._setup_sharing_groups_and_expansion(
            initial_values, sharing_groups, altloc_groups, device
        )
        
        # Convert initial occupancies to logit space (full space)
        if self.use_sigmoid:
            clamped_values = torch.clamp(initial_values, min=1e-6, max=1-1e-6)
            logit_values = torch.logit(clamped_values)
        else:
            logit_values = initial_values.clone()
        
        # Collapse logit values using vectorized operation
        collapsed_logits = self._collapse_values_vectorized(logit_values)
        
        # Handle refinable mask - collapse it too
        if refinable_mask is not None:
            if refinable_mask.shape[0] != self._full_shape:
                raise ValueError(
                    f"refinable_mask shape {refinable_mask.shape} must match "
                    f"initial_values shape {initial_values.shape}"
                )
            # Collapse the refinable mask
            collapsed_refinable_mask = self._collapse_mask_vectorized(refinable_mask.to(device=device))
        else:
            collapsed_refinable_mask = torch.ones(self._collapsed_shape, dtype=torch.bool, device=device)
        
        # Note: With the new normalization approach, all altloc members are refinable
        # The sum-to-1 constraint is enforced during forward() via normalization
        
        # Store masks as buffers
        self.register_buffer('refinable_mask', collapsed_refinable_mask)
        self.register_buffer('fixed_mask', ~collapsed_refinable_mask)
        
        # Store fixed values as buffer (collapsed)
        self.register_buffer('fixed_values', collapsed_logits.clone().detach())
        
        # Store refinable values as parameter (collapsed, excluding placeholders)
        refinable_values = collapsed_logits[collapsed_refinable_mask].clone().detach()
        self.refinable_params = nn.Parameter(refinable_values, requires_grad=requires_grad)
        
        # Store collapsed shape
        self.register_buffer('_shape', torch.tensor([self._collapsed_shape]))
    
    def _setup_sharing_groups_and_expansion(
        self, 
        initial_values: torch.Tensor, 
        sharing_groups: Optional[torch.Tensor],
        altloc_groups: Optional[list],
        device: torch.device
    ):
        """
        Setup sharing groups, altlocs, and create expansion mask for memory-efficient storage.
        
        Now uses a simple index tensor approach:
        - sharing_groups is a tensor of shape (n_atoms,) mapping each atom to its collapsed index
        - Collapse: scatter_add to sum values by collapsed index
        - Expand: direct indexing collapsed_values[sharing_groups]
        
        For altlocs, we create tensors of shape (N_pairs, M_conformations) where each row
        contains the collapsed indices for conformations that must sum to 1.0.
        These are stored grouped by number of conformations (2-way, 3-way, etc.).
        
        Args:
            initial_values: Initial occupancy values for all atoms
            sharing_groups: Tensor of shape (n_atoms,) giving collapsed index for each atom
            altloc_groups: List of tuples of atom index lists for alternative conformations
            device: Device to place tensors on
        """
        n_atoms = initial_values.shape[0]
        
        # Use sharing_groups directly as the expansion mask
        if sharing_groups is None:
            # No sharing - each atom maps to its own index
            expansion_mask = torch.arange(n_atoms, dtype=torch.long, device=device)
            self._collapsed_shape = n_atoms
        else:
            # Use the provided index tensor
            expansion_mask = sharing_groups.to(device=device, dtype=torch.long)
            self._collapsed_shape = expansion_mask.max().item() + 1
        
        self.register_buffer('expansion_mask', expansion_mask)
        
        # Process altloc groups: convert to collapsed indices and group by size
        # linked_occupancies[n] = tensor of shape (N_groups, n) where n is number of conformations
        linked_occupancies = {}
        
        if altloc_groups is not None and len(altloc_groups) > 0:
            for altloc_idx, conf_groups in enumerate(altloc_groups):
                n_conformations = len(conf_groups)
                if n_conformations < 2:
                    raise ValueError(f"Altloc group {altloc_idx} must have at least 2 conformations")
                
                # Get collapsed indices for each conformation
                collapsed_indices = []
                for conf_atoms in conf_groups:
                    if isinstance(conf_atoms, (list, tuple)):
                        conf_atoms = torch.tensor(conf_atoms, dtype=torch.long, device=device)
                    else:
                        conf_atoms = conf_atoms.to(device=device, dtype=torch.long)
                    
                    # Get collapsed index for first atom
                    collapsed_idx = expansion_mask[conf_atoms[0]].item()
                    
                    # ASSERT: All atoms in this conformation map to the same collapsed index
                    for atom_idx in conf_atoms:
                        atom_collapsed_idx = expansion_mask[atom_idx].item()
                        if atom_collapsed_idx != collapsed_idx:
                            raise AssertionError(
                                f"Altloc group {altloc_idx}, conformation {len(collapsed_indices)}: "
                                f"atom {atom_idx} maps to collapsed index {atom_collapsed_idx}, "
                                f"but first atom maps to {collapsed_idx}. "
                                f"All atoms in a conformation must share the same collapsed index."
                            )
                    
                    collapsed_indices.append(collapsed_idx)
                
                # Add to the appropriate group based on number of conformations
                if n_conformations not in linked_occupancies:
                    linked_occupancies[n_conformations] = []
                
                linked_occupancies[n_conformations].append(collapsed_indices)
        
        # Convert lists to tensors and register as buffers
        # Store as dictionary with keys like 'linked_occ_2', 'linked_occ_3', etc.
        for n_conf, groups in linked_occupancies.items():
            # Shape: (N_groups, n_conf)
            tensor = torch.tensor(groups, dtype=torch.long, device=device)
            self.register_buffer(f'linked_occ_{n_conf}', tensor)
        
        # Store which sizes we have
        self.linked_occ_sizes = sorted(linked_occupancies.keys())
        
        # Create count buffer for vectorized collapse operations
        # counts[i] = number of atoms that map to collapsed index i
        counts = torch.zeros(self._collapsed_shape, dtype=torch.long, device=device)
        counts.scatter_add_(0, expansion_mask, torch.ones_like(expansion_mask))
        self.register_buffer('collapse_counts', counts)
    
    def _collapse_values_vectorized(self, full_values: torch.Tensor) -> torch.Tensor:
        """
        Collapse full tensor to collapsed storage using vectorized scatter_add.
        
        Uses the expansion_mask directly for O(n) collapse operation.
        
        Args:
            full_values: Tensor in full space (one value per atom)
        
        Returns:
            Tensor in collapsed space (one value per group + ungrouped atoms)
        """
        # Sum values at each collapsed index using scatter_add
        collapsed_sum = torch.zeros(
            self._collapsed_shape, 
            dtype=full_values.dtype, 
            device=full_values.device
        )
        collapsed_sum.scatter_add_(0, self.expansion_mask, full_values)
        
        # Divide by counts to get mean (avoid division by zero)
        collapsed = collapsed_sum / self.collapse_counts.float().clamp(min=1)
        
        return collapsed
    
    def _collapse_mask_vectorized(self, full_mask: torch.Tensor) -> torch.Tensor:
        """
        Collapse boolean mask to collapsed storage using vectorized operations.
        
        If ANY atom in a collapsed position is refinable, the position is refinable.
        
        Args:
            full_mask: Boolean mask in full space
        
        Returns:
            Boolean mask in collapsed space
        """
        # Use scatter_add with float tensors, then check if any > 0
        collapsed_sum = torch.zeros(
            self._collapsed_shape,
            dtype=torch.float,
            device=full_mask.device
        )
        collapsed_sum.scatter_add_(0, self.expansion_mask, full_mask.float())
        
        # If sum > 0, at least one atom in that collapsed position was True
        collapsed = collapsed_sum > 0
        
        return collapsed
    
    def _collapse_values(self, full_values: torch.Tensor) -> torch.Tensor:
        """
        Legacy collapse function - redirects to vectorized version.
        Kept for backward compatibility.
        """
        return self._collapse_values_vectorized(full_values)
    
    def _collapse_mask(self, full_mask: torch.Tensor) -> torch.Tensor:
        """
        Legacy collapse mask function - redirects to vectorized version.
        Kept for backward compatibility.
        """
        return self._collapse_mask_vectorized(full_mask)
    
    def _expand_values(self, collapsed_values: torch.Tensor) -> torch.Tensor:
        """
        Expand collapsed storage to full tensor using expansion mask.
        
        Args:
            collapsed_values: Tensor in collapsed space
        
        Returns:
            Tensor in full space (one value per atom)
        """
        return collapsed_values[self.expansion_mask]
    
    def forward(self) -> torch.Tensor:
        """
        Reconstruct full occupancy tensor with sigmoid transformation and altloc constraints.
        
        For alternative conformations, we apply sigmoid then normalize within each group
        to enforce the sum-to-1 constraint. This is done separately for each group size
        (2-way, 3-way, etc.) for efficiency.
        
        Returns:
            Full occupancy tensor with values in [0, 1] (shape: [n_atoms])
        """
        # Get collapsed logit values (combining fixed and refinable)
        result = self.fixed_values.clone()
        result[self.refinable_mask] = self.refinable_params
        
        # Apply sigmoid transformation to get raw occupancies
        if self.use_sigmoid:
            collapsed_occs = torch.sigmoid(result)
        else:
            collapsed_occs = result.clone()
        
        # Handle linked occupancies: normalize within each altloc group
        # Process each group size separately (2-way, 3-way, etc.)
        if hasattr(self, 'linked_occ_sizes') and len(self.linked_occ_sizes) > 0:
            # Start with a copy of collapsed_occs that we'll update
            updated_occs = collapsed_occs.clone()
            
            for n_conf in self.linked_occ_sizes:
                # Get the tensor of linked indices: shape (N_groups, n_conf)
                linked_indices = getattr(self, f'linked_occ_{n_conf}')
                
                # Gather occupancies for all linked groups: shape (N_groups, n_conf)
                linked_occs = collapsed_occs[linked_indices]
                
                # Normalize: divide each by the sum across conformations
                # Shape: (N_groups, 1) for broadcasting
                sums = linked_occs.sum(dim=1, keepdim=True).clamp(min=1e-10)
                normalized_occs = linked_occs / sums
                
                # this should be vectorized assignment back to updated_occs

                # Update the linked indices in the new tensor
                for group_idx in range(linked_indices.shape[0]):
                    for conf_idx in range(n_conf):
                        collapsed_idx = linked_indices[group_idx, conf_idx]
                        updated_occs[collapsed_idx] = normalized_occs[group_idx, conf_idx]
            
            collapsed_occs = updated_occs
        
        # Expand to full space
        full_occs = self._expand_values(collapsed_occs)
        
        return full_occs
    
    @property
    def shape(self):
        """Return the shape of the FULL tensor (not collapsed)."""
        return (self._full_shape,)
    
    @property
    def collapsed_shape(self):
        """Return the shape of the collapsed internal storage."""
        return (self._collapsed_shape,)
    
    def clamp(self, min_value: float = 0.0, max_value: float = 1.0) -> 'OccupancyTensor':
        """
        Clamp occupancy values to specified range and return a new OccupancyTensor.
        
        Args:
            min_value: Minimum occupancy value (default: 0.0)
            max_value: Maximum occupancy value (default: 1.0)
        
        Returns:
            New OccupancyTensor with clamped values
        """
        # Get current occupancy values in full space
        current_occ = self.forward().detach()
        
        # Clamp in occupancy space
        clamped_occ = torch.clamp(current_occ, min=min_value, max=max_value)
        
        # Reconstruct refinable mask in full space
        full_refinable_mask = self._expand_values(self.refinable_mask.float()).bool()
        
        # Create new OccupancyTensor
        new_occ = OccupancyTensor(
            initial_values=clamped_occ,
            sharing_groups=self.expansion_mask.clone(),
            refinable_mask=full_refinable_mask,
            requires_grad=self.refinable_params.requires_grad,
            dtype=self.dtype,
            device=self.device,
            name=self.name,
            use_sigmoid=self.use_sigmoid
        )
        
        return new_occ
    
    def set_group_occupancy(self, group_idx: int, value: float):
        """
        Set the occupancy for all atoms in a specific collapsed group.
        
        Args:
            group_idx: Collapsed index of the group
            value: Occupancy value to set (must be in [0, 1])
        """
        if group_idx < 0 or group_idx >= self._collapsed_shape:
            raise ValueError(f"Invalid group index {group_idx}")
        
        if value < 0 or value > 1:
            raise ValueError(f"Occupancy value must be in [0, 1], got {value}")
        
        # Convert value to logit space
        clamped_value = np.clip(value, 1e-6, 1-1e-6)
        logit_value = np.log(clamped_value / (1 - clamped_value))
        logit_tensor = torch.tensor(logit_value, dtype=self.dtype, device=self.device)
        
        # The group occupies collapsed_idx = group_idx (groups are first in collapsed storage)
        collapsed_idx = group_idx
        
        # Get current collapsed logits
        result = self.fixed_values.clone()
        result[self.refinable_mask] = self.refinable_params.data
        
        # Update the collapsed value for this group
        result[collapsed_idx] = logit_tensor
        
        # Update fixed values and refinable params
        self.fixed_values = result.clone().detach()
        if self.refinable_mask[collapsed_idx]:
            # This group is refinable, update refinable params
            self.refinable_params.data = result[self.refinable_mask].clone()
    
    def get_group_occupancy(self, group_idx: int) -> float:
        """
        Get the current occupancy value for a collapsed group.
        
        Args:
            group_idx: Collapsed index of the group
        
        Returns:
            Current occupancy value for the group
        """
        if group_idx < 0 or group_idx >= self._collapsed_shape:
            raise ValueError(f"Invalid group index {group_idx}")
        
        # Get current occupancies in full space
        occupancies = self.forward()
        
        # Find first atom that maps to this collapsed index
        atom_idx = (self.expansion_mask == group_idx).nonzero()[0].item()
        return occupancies[atom_idx].item()
    
    @staticmethod
    def from_residue_groups(initial_values: torch.Tensor, 
                           pdb_dataframe,
                           refinable_mask: Optional[torch.Tensor] = None,
                           **kwargs) -> 'OccupancyTensor':
        """
        Create an OccupancyTensor where all atoms in each residue share the same occupancy.
        
        This is a common use case where all atoms in a residue should have the same occupancy.
        
        Args:
            initial_values: Initial occupancy values for all atoms
            pdb_dataframe: Pandas DataFrame with PDB data (must have 'resname', 'resseq', 'chainid')
            refinable_mask: Optional mask for refinable atoms
            **kwargs: Additional arguments passed to OccupancyTensor constructor
        
        Returns:
            OccupancyTensor with residue-based sharing groups
        """
        # Group atoms by residue
        grouped = pdb_dataframe.groupby(['resname', 'resseq', 'chainid', 'altloc'])
        
        n_atoms = len(initial_values)
        sharing_groups_tensor = torch.arange(n_atoms, dtype=torch.long)
        collapsed_idx = 0
        
        for (resname, resseq, chainid, altloc), group in grouped:
            indices = group['index'].tolist()
            if len(indices) > 1:  # Only create group if more than one atom
                sharing_groups_tensor[indices] = collapsed_idx
                collapsed_idx += 1
        
        # Compact the indices
        unique_indices = torch.unique(sharing_groups_tensor, sorted=True)
        for new_idx, old_idx in enumerate(unique_indices):
            mask = (sharing_groups_tensor == old_idx)
            sharing_groups_tensor[mask] = new_idx
        
        return OccupancyTensor(
            initial_values=initial_values,
            sharing_groups=sharing_groups_tensor,
            refinable_mask=refinable_mask,
            name='occupancy',
            **kwargs
        )
    
    def __repr__(self) -> str:
        name_str = f"'{self.name}', " if self.name is not None else ""
        n_groups = self._collapsed_shape
        return (
            f"OccupancyTensor({name_str}shape={self.shape}, dtype={self.dtype}, "
            f"device={self.device}, refinable={self.get_refinable_count()}, "
            f"fixed={self.get_fixed_count()}, collapsed_groups={n_groups}, "
            f"use_sigmoid={self.use_sigmoid})"
        )