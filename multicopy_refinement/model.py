


import torch
import torch.nn as nn
from typing import Optional, Union
import pdb_tools
import multicopy_refinement.symmetrie as sym
import multicopy_refinement.get_scattering_factor_torch as gsf
import multicopy_refinement.symmetrie as sym
import multicopy_refinement.math_numpy as mnp
import numpy as np

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.altloc_pairs = []
    
    def load_pdb_from_file(self,file,strip_H=True):
        self.pdb = pdb_tools.load_pdb_as_pd(file)
        self.pdb = self.pdb.loc[self.pdb['element'] != 'H'].reset_index(drop=True) if strip_H else self.pdb
        self.pdb['index'] = self.pdb.index.to_numpy(dtype=int)
        self.cell = np.array(self.pdb.attrs['cell'])
        self.spacegroup = self.pdb.attrs['spacegroup']
        self.spacegroup_function = sym.Symmetry(self.spacegroup)
        self.inv_fractional_matrix = torch.tensor(mnp.get_inv_fractional_matrix(self.cell))
        self.fractional_matrix = torch.tensor(mnp.get_fractional_matrix(self.cell))
        self.aniso_flag = torch.tensor(self.pdb['anisou_flag'].values,dtype=torch.bool)
        # MixedTensor is already a nn.Module, so don't wrap it in Parameter()
        self.xyz = MixedTensor(torch.tensor(self.pdb[['x', 'y', 'z']].values), name='xyz')
        self.b = MixedTensor(torch.tensor(self.pdb['tempfactor'].values), name='b_factor')
        self.u = MixedTensor(torch.tensor(self.pdb[['u11', 'u22', 'u33', 'u12', 'u13', 'u23']].values), name='aniso_U')
        self.occupancy = MixedTensor(torch.tensor(self.pdb['occupancy'].values), name='occupancy')
        self.set_default_masks()
        self.register_alternative_conformations()

    def update_pdb(self):
        self.pdb.loc[:, ['x', 'y', 'z']] = self.xyz().cpu().detach().numpy()
        self.pdb.loc[:, ['u11', 'u22', 'u33', 'u12', 'u13', 'u23']] = self.u().cpu().detach().numpy()
        self.pdb.loc[:, 'tempfactor'] = self.b().cpu().detach().numpy()
        self.pdb.loc[:, 'occupancy'] = self.occupancy().cpu().detach().numpy()
        return self.pdb
    
    def cuda(self, device: Optional[Union[int, torch.device]] = None):
        super().cuda(device)
        self.inv_fractional_matrix = self.inv_fractional_matrix.cuda(device)
        self.fractional_matrix = self.fractional_matrix.cuda(device)
        self.aniso_flag = self.aniso_flag.cuda(device)
        if self.altloc_pairs:
            self.altloc_pairs = [tuple(tensor.cuda(device) for tensor in group) for group in self.altloc_pairs]
        return self
    
    def cpu(self):
        super().cpu()
        self.inv_fractional_matrix = self.inv_fractional_matrix.cpu()
        self.fractional_matrix = self.fractional_matrix.cpu()
        self.aniso_flag = self.aniso_flag.cpu()
        if self.altloc_pairs:
            self.altloc_pairs = [tuple(tensor.cpu() for tensor in group) for group in self.altloc_pairs]
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
        occupancy_refine_mask = self.occupancy().detach() < 1.0
        self.occupancy.update_refinable_mask(occupancy_refine_mask)

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

    def enforce_occ_alternative_conformations(self):
        """
        Enforce occupancy constraints for alternative conformations.
        
        This method applies two constraints:
        1. Mean occupancy per conformation: All atoms within the same conformation 
           (altloc) of a residue get the same mean occupancy value.
        2. Sum to one: The mean occupancies across all conformations of a residue 
           are normalized to sum to 1.0.
        
        This ensures:
        - All atoms in conformation A of a residue have the same occupancy
        - All atoms in conformation B of a residue have the same occupancy
        - The sum of mean occupancies across conformations equals 1.0
        
        Example:
            Before: Conf A atoms: [0.6, 0.65, 0.62], Conf B atoms: [0.4, 0.35, 0.38]
            After:  Conf A atoms: [0.62, 0.62, 0.62], Conf B atoms: [0.38, 0.38, 0.38]
                    (0.62 + 0.38 = 1.0)
        
        Note: This method modifies the occupancy values in-place by updating the
        underlying refinable and fixed parameters in the occupancy MixedTensor.
        """
        if not hasattr(self, 'altloc_pairs') or len(self.altloc_pairs) == 0:
            # No alternative conformations to enforce
            return
        
        # Get current occupancies (this creates a new tensor with current values)
        current_occ = self.occupancy().detach().clone()
        
        for group in self.altloc_pairs:
            # Step 1: Compute mean occupancy for each conformation
            mean_occupancies = []
            for conf_indices in group:
                # Get occupancies for this conformation
                conf_occ = current_occ[conf_indices]
                # Compute mean occupancy
                mean_occ = conf_occ.mean()
                mean_occupancies.append(mean_occ)
            
            # Step 2: Normalize mean occupancies to sum to 1.0
            mean_occupancies_tensor = torch.stack(mean_occupancies)
            total = mean_occupancies_tensor.sum()
            
            # Avoid division by zero
            if total > 0:
                normalized_means = mean_occupancies_tensor / total
            else:
                # If all occupancies are zero, distribute equally
                normalized_means = torch.ones_like(mean_occupancies_tensor) / len(mean_occupancies_tensor)
            
            # Step 3: Set all atoms in each conformation to their normalized mean
            for conf_indices, norm_mean in zip(group, normalized_means):
                current_occ[conf_indices] = norm_mean
        
        # Update the occupancy MixedTensor with new values
        # We need to update both refinable and fixed values
        refinable_mask = self.occupancy.refinable_mask
        
        # Update fixed values
        self.occupancy.fixed_values = current_occ.clone()
        
        # Update refinable parameters
        if refinable_mask.any():
            self.occupancy.refinable_params.data = current_occ[refinable_mask].clone()

    def sanitize_occupancies(self):
        """
        Sanitize occupancy values to ensure they are within valid bounds [0.0, 1.0].
        
        This method clamps all occupancy values to be within the range [0.0, 1.0].
        It updates both the fixed and refinable parameters in the occupancy MixedTensor.
        """
        self.occupancy = self.occupancy.clamp(0.0, 1.0)
        self.enforce_occ_alternative_conformations()

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