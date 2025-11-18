
'''
A base model class for atomic structure models using PyTorch.
'''

import torch
import torch.nn as nn
from typing import Optional, Union
from torchref.io import file_writers
from torchref.utils.utils import sanitize_pdb_dataframe
import torchref.symmetrie.symmetrie as sym
import torchref.math_functions.math_numpy as mnp
from torchref.math_functions import math_torch
from torchref.model.parameter_wrappers import MixedTensor, OccupancyTensor, PositiveMixedTensor
from torchref.io import cif_readers, legacy_format_readers
from torchref.utils.debug_utils import DebugMixin

class Model(DebugMixin, nn.Module):
    def __init__(self,dtype_float=torch.float32,verbose=1,device=torch.device('cpu'), strip_H: bool =True):
        super().__init__()
        self.altloc_pairs = []
        self.verbose = verbose
        self.initialized = False
        self.dtype_float = dtype_float
        self.device = device
        self.strip_H = strip_H
    
    def __bool__(self):
        """Return the initialization status when used in boolean context."""
        return self.initialized
    
    def load(self, reader):
        self.pdb, cell, self.spacegroup = reader()
        self.pdb = self.pdb.loc[self.pdb['element'] != 'H'].reset_index(drop=True) if self.strip_H else self.pdb
        self.pdb.dropna(subset=['x', 'y', 'z', 'tempfactor', 'occupancy'], inplace=True)
        self.pdb['index'] = self.pdb.index.to_numpy(dtype=int)
        
        self.register_buffer('cell',torch.tensor(cell,requires_grad=False,dtype=self.dtype_float,device=self.device))
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

    def load_pdb(self,file):
        '''
        Load atomic model from PDB file.
        '''
        reader = legacy_format_readers.PDB(verbose=self.verbose).read(file)   
        return self.load(reader)
    
    def load_cif(self, file):
        """
        Load atomic model from mmCIF file.
        
        Args:
            file: Path to CIF/mmCIF file
            
        Returns:
            self (for method chaining)
        """
        
        if self.verbose > 0:
            print(f"Loading CIF file: {file}")
        
        # Read CIF file
        cif_reader = cif_readers.ModelCIFReader(file)

        return self.load(cif_reader)
    
    def _create_occupancy_groups(self, pdb_df, initial_occ):
        """
        Create sharing groups and altloc groups for occupancy.
        
        Logic:
        1. First identify alternative conformations (multiple altlocs per residue)
        2. For altloc groups: ALL atoms in each conformation share one collapsed index
        3. For non-altloc residues: group by similar occupancy (within 0.01 tolerance)
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
        collapsed_idx = 0
        
        # First pass: identify and process alternative conformations
        # For altloc atoms: ALL atoms in a conformation MUST share the same collapsed index
        # regardless of their individual occupancy values
        pdb_with_altlocs = pdb_df[pdb_df['altloc'] != '']
        altloc_residues = set()  # Track which residues have altlocs
        
        if len(pdb_with_altlocs) > 0:
            grouped_by_residue = pdb_with_altlocs.groupby(['resname', 'resseq', 'chainid'])
            
            for (resname, resseq, chainid), group in grouped_by_residue:
                unique_altlocs = sorted(group['altloc'].unique())
                
                # Only process if there are multiple conformations
                if len(unique_altlocs) > 1:
                    altloc_residues.add((resname, resseq, chainid))
                    conformation_atom_lists = []
                    
                    for altloc in unique_altlocs:
                        # Get all atoms for this specific altloc
                        altloc_atoms = group[group['altloc'] == altloc]
                        indices = altloc_atoms['index'].tolist()
                        
                        # Assign ALL atoms in this conformation to the same collapsed index
                        sharing_groups_tensor[indices] = collapsed_idx
                        
                        # Check if any atom in this conformation has occupancy != 1.0
                        for idx in indices:
                            if abs(initial_occ[idx].item() - 1.0) > 0.01:
                                refinable_mask[idx] = True
                        
                        conformation_atom_lists.append(indices)
                        collapsed_idx += 1
                    
                    # Add to altloc_groups
                    altloc_groups.append(tuple(conformation_atom_lists))
        
        # Second pass: process non-altloc residues
        # Group by residue, and create sharing groups based on occupancy similarity
        grouped = pdb_df.groupby(['resname', 'resseq', 'chainid', 'altloc'])
        
        for (resname, resseq, chainid, altloc), group in grouped:
            # Skip if this residue has alternative conformations (already processed)
            if (resname, resseq, chainid) in altloc_residues:
                continue
            
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
                sharing_groups_tensor[indices] = collapsed_idx
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
        
        # Compact the indices - make them contiguous from 0 to n_collapsed-1
        unique_indices = torch.unique(sharing_groups_tensor, sorted=True)
        index_map = torch.zeros(n_atoms, dtype=torch.long)
        for new_idx, old_idx in enumerate(unique_indices):
            mask = (sharing_groups_tensor == old_idx)
            sharing_groups_tensor[mask] = new_idx
        
        n_collapsed = len(unique_indices)
        
        if self.verbose > 1:
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
        elements = self.pdb.loc[:, 'element']
        path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'caching/files/atomic_vdw_radii.csv')
        vdw_df = pd.read_csv(path, comment='#')   
        vdw_df['element'] = vdw_df['element'].str.strip().str.capitalize()
        elements = elements.str.strip().str.capitalize()
        elements_not_in = elements[~elements.isin(vdw_df['element'])]
        if len(elements_not_in) > 0:
            # Add missing elements with default vdW radius 1.9 Å
            missing = sorted(set(e.strip().capitalize() for e in elements_not_in))
            if missing:
                add_df = pd.DataFrame({'element': missing,
                                       'vdW_Radius_Angstrom': [1.9] * len(missing)})
                vdw_df = pd.concat([vdw_df, add_df], ignore_index=True)


        vdw_radii = vdw_df.set_index('element').loc[elements]['vdW_Radius_Angstrom'].values
        self.register_buffer('vdw_radii', torch.tensor(vdw_radii, dtype=self.dtype_float, device=self.device))
        assert len(self.vdw_radii) == len(self.pdb), f"vdW radii length mismatch with number of atoms {len(self.vdw_radii)} != {len(self.pdb)}"
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
    
    def copy(self):
        """
        Create a deep copy of the Model with all its parameters and submodules.
        
        This method creates a complete independent copy of the model including:
        - PDB DataFrame (deep copy)
        - All registered buffers (cell, fractional matrices, aniso_flag, etc.)
        - MixedTensor parameters (xyz, b, u) with their refinable masks
        - OccupancyTensor with sharing groups and altloc groups
        - Alternative conformation pairs
        - Spacegroup and symmetry information
        - VdW radii (if computed)
        
        Returns:
            Model: A new Model instance with copied data
            
        Example:
            >>> model = Model().load_pdb('structure.pdb')
            >>> model_copy = model.copy()
            >>> # model_copy is independent, changes won't affect model
        """
        import copy as copy_module
        
        if not self.initialized:
            raise RuntimeError("Cannot copy an uninitialized Model. Load data first.")
        
        # Create new model instance with same configuration
        model_copy = Model(
            dtype_float=self.dtype_float,
            verbose=self.verbose,
            device=self.device,
            strip_H=self.strip_H
        )
        
        # Deep copy the PDB DataFrame
        model_copy.pdb = self.pdb.copy(deep=True)
        
        # Copy scalar attributes
        model_copy.spacegroup = self.spacegroup
        model_copy.initialized = True
        
        # Copy spacegroup function
        model_copy.spacegroup_function = sym.Symmetry(self.spacegroup)
        
        # Copy all registered buffers
        for buffer_name in ['cell', 'inv_fractional_matrix', 'fractional_matrix', 
                           'aniso_flag', 'recB']:
            if hasattr(self, buffer_name):
                buffer_value = getattr(self, buffer_name)
                if buffer_value is not None:
                    model_copy.register_buffer(buffer_name, buffer_value.clone())
        
        # Copy vdw_radii if it exists
        if hasattr(self, 'vdw_radii'):
            model_copy.register_buffer('vdw_radii', self.vdw_radii.clone())
        
        # Copy MixedTensor parameters (xyz, b, u)
        # Create tensors first, then update refinable masks (as done in set_default_masks)
        model_copy.xyz = MixedTensor(
            self.xyz().detach().clone(),
            name='xyz'
        )
        if self.xyz.refinable_mask is not None:
            model_copy.xyz.update_refinable_mask(self.xyz.refinable_mask.clone())
        
        model_copy.b = MixedTensor(
            self.b().detach().clone(),
            name='b_factor'
        )
        if self.b.refinable_mask is not None:
            model_copy.b.update_refinable_mask(self.b.refinable_mask.clone())
        
        model_copy.u = MixedTensor(
            self.u().detach().clone(),
            name='aniso_U'
        )
        if self.u.refinable_mask is not None:
            model_copy.u.update_refinable_mask(self.u.refinable_mask.clone())
        
        # Copy OccupancyTensor
        # Use the same approach as creating a new occupancy tensor in load()
        initial_occ = self.occupancy().detach().clone()
        sharing_groups, altloc_groups, refinable_mask = self._create_occupancy_groups(model_copy.pdb, initial_occ)
        model_copy.occupancy = OccupancyTensor(
            initial_values=initial_occ,
            sharing_groups=sharing_groups,
            altloc_groups=altloc_groups,
            refinable_mask=refinable_mask,
            dtype=self.dtype_float,
            device=self.device,
            name='occupancy'
        )
        
        # Copy alternative conformation pairs
        if self.altloc_pairs:
            model_copy.altloc_pairs = [
                tuple(tensor.clone() for tensor in group) 
                for group in self.altloc_pairs
            ]
        else:
            model_copy.altloc_pairs = []
        
        if self.verbose > 0:
            print(f"✓ Model copied successfully ({len(model_copy.pdb)} atoms)")
        
        return model_copy
    
    def write_pdb(self, filename):
        self.update_pdb()
        self.pdb = sanitize_pdb_dataframe(self.pdb)
        file_writers.write_file(self.pdb, filename)

    def get_iso(self):
        xyz = self.xyz()[~self.aniso_flag]
        b = self.b()[~self.aniso_flag]
        occupancy = self.occupancy()[~self.aniso_flag]
        return xyz, b, occupancy

    def set_default_masks(self):
        self.xyz_mask = torch.ones(len(self.pdb), dtype=torch.bool)
        self.xyz.update_refinable_mask(self.xyz_mask)
        self.b_mask = ~self.b().detach().isnan() 
        self.b.update_refinable_mask(self.b_mask)
        self.u_mask = ~self.u().detach().isnan().any(dim=1)
        self.u.update_refinable_mask(self.u_mask)
        self.occupancy_mask = self.occupancy() < 0.999
        self.occupancy.update_refinable_mask(self.occupancy_mask)

    def freeze(self, target: str):
        if target == 'xyz':
            self.xyz.fix_all()
        elif target == 'b':
            self.b.fix_all()
        elif target == 'u':
            self.u.fix_all()
        elif target == 'occupancy':
            self.occupancy.freeze_all()  # OccupancyTensor uses freeze_all() not fix_all()
    
    def freeze_all(self):
        self.freeze('xyz')
        self.freeze('b')
        self.freeze('u')
        self.freeze('occupancy')

    def unfreeze_all(self):
        self.unfreeze('xyz')
        self.unfreeze('b')
        self.unfreeze('u')
        self.unfreeze('occupancy')

    def unfreeze(self, target: str):
        if target == 'xyz':
            self.xyz.update_refinable_mask(self.xyz_mask)
        elif target == 'b':
            self.b.update_refinable_mask(self.b_mask)
        elif target == 'u':
            self.u.update_refinable_mask(self.u_mask)
        elif target == 'occupancy':
            # OccupancyTensor uses unfreeze_all() or update_refinable_mask() with full atom space mask
            self.occupancy.update_refinable_mask(self.occupancy_mask, in_compressed_space=False)

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

    def adp_loss(self):
        """
        Compute the ADP (B-factor) regularization loss.
        
        This loss encourages B-factors to have similar values across the structure,
        helping to prevent overfitting during refinement.
        
        Returns:
            torch.Tensor: Scalar tensor representing the ADP loss.
        """
        b_current = self.b()
        b_mean = torch.mean(b_current)
        loss = torch.mean((b_current - b_mean) ** 2)
        return loss