"""
Restraints Class for Crystallographic Model Refinement

This module provides a comprehensive restraints handler for crystallographic models.
It parses CIF (Crystallographic Information File) restraints dictionaries and builds
efficient tensor-based representations for bond lengths, angles, and torsion angles
across an entire molecular structure.

The restraints are stored in a format optimized for PyTorch operations:
- Bond lengths: (N, 2) tensor of atom indices
- Angles: (N, 3) tensor of atom indices  
- Torsions: (N, 4) tensor of atom indices

Each restraint type also stores the expected values and sigma (uncertainty) values.
"""

import torch
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from multicopy_refinement.restraints_torch import read_cif


class Restraints:
    """
    Restraints handler for crystallographic model refinement.
    
    This class parses CIF restraints dictionaries and builds efficient tensor 
    representations for the entire molecular structure. It stores restraints
    for bond lengths, angles, and torsion angles with their expected values
    and uncertainties (sigma values).
    
    Attributes:
        model: Reference to the Model instance
        cif_path: Path to the CIF restraints dictionary file
        cif_dict: Parsed CIF dictionary with restraints for each residue type
        
        # Bond length restraints (N_bonds total)
        bond_indices: torch.Tensor of shape (N_bonds, 2) - atom pair indices
        bond_references: torch.Tensor of shape (N_bonds,) - expected bond lengths
        bond_sigmas: torch.Tensor of shape (N_bonds,) - bond length uncertainties
        
        # Angle restraints (N_angles total)
        angle_indices: torch.Tensor of shape (N_angles, 3) - atom triplet indices
        angle_references: torch.Tensor of shape (N_angles,) - expected angles (degrees)
        angle_sigmas: torch.Tensor of shape (N_angles,) - angle uncertainties
        
        # Torsion angle restraints (N_torsions total)
        torsion_indices: torch.Tensor of shape (N_torsions, 4) - atom quartet indices
        torsion_references: torch.Tensor of shape (N_torsions,) - expected torsions (degrees)
        torsion_sigmas: torch.Tensor of shape (N_torsions,) - torsion uncertainties
        
    Example:
        >>> from multicopy_refinement.model import Model
        >>> from multicopy_refinement.restraints_new import Restraints
        >>> 
        >>> # Load model
        >>> model = Model()
        >>> model.load_pdb_from_file('structure.pdb')
        >>> 
        >>> # Create restraints
        >>> restraints = Restraints(model, 'restraints.cif')
        >>> 
        >>> # Access restraint information
        >>> print(f"Total bonds: {restraints.bond_indices.shape[0]}")
        >>> print(f"Total angles: {restraints.angle_indices.shape[0]}")
        >>> print(f"Total torsions: {restraints.torsion_indices.shape[0]}")
    """
    
    def __init__(self, model, cif_path: str):
        """
        Initialize the Restraints handler.
        
        Args:
            model: Model instance containing the atomic structure
            cif_path: Path to the CIF restraints dictionary file
        """
        self.model = model
        self.cif_path = cif_path
        
        self.unique_residues = model.pdb.resname.unique()
        # Check unique residue have more than one atom
        self.unique_residues = [residue for residue in self.unique_residues if self.model.pdb.loc[self.model.pdb['resname'] == residue,'name'].nunique() > 1]
        # Parse the CIF file
        self.cif_dict = read_cif(cif_path)

        self.missing_residues = [res for res in self.unique_residues if res not in self.cif_dict]
        print(self.missing_residues)
        
        
        # Identify restraint dictionary keys
        self._identify_restraint_keys()
        
        # Initialize restraint tensors
        self.bond_indices = None
        self.bond_references = None
        self.bond_sigmas = None
        
        self.angle_indices = None
        self.angle_references = None
        self.angle_sigmas = None
        
        self.torsion_indices = None
        self.torsion_references = None
        self.torsion_sigmas = None
        
        # Build restraints for the entire structure
        self.build_restraints()
    
    def _identify_restraint_keys(self):
        """
        Identify the CIF dictionary keys for different restraint types.
        
        CIF dictionaries may use different naming conventions. This method
        searches for the appropriate keys for bonds, angles, and torsions.
        """
        # Get all unique keys across all residue types
        nested_keys = list(set([key for comp_dict in self.cif_dict.values() 
                               for key in comp_dict.keys()]))
        
        # Find the key for bond length restraints
        self.bond_keys = []
        for key in nested_keys:
            if 'bond' in key.lower():
                self.bond_keys.append(key)

        # Find the key for angle restraints
        self.angle_keys = []
        for key in nested_keys:
            if 'angle' in key.lower() and 'tor' not in key.lower():
                self.angle_keys.append(key)
        
        # Find the key for torsion angle restraints
        self.torsion_keys = []
        for key in nested_keys:
            if 'tor' in key.lower():
                self.torsion_keys.append(key)
                
    
    def build_restraints(self):
        """
        Build all restraints for the entire structure.
        
        This method iterates through all residues in the model and builds
        restraints for bond lengths, angles, and torsions. The results are
        stored as tensors on the same device as the model coordinates.
        """
        # Build each restraint type
        self._build_bond_restraints()
        self._build_angle_restraints()
        self._build_torsion_restraints()
        
        # Move tensors to the same device as model
        device = self.model.xyz().device
        self._move_to_device(device)
    
    def _build_bond_restraints(self):
        """
        Build bond length restraints for the entire structure.
        
        Iterates through all chains and residues, matching atoms with the
        restraints defined in the CIF dictionary. Only includes bonds where
        both atoms are present in the residue.
        """
        if self.bond_keys is None:
            print("Warning: No bond restraints found in CIF dictionary")
            return
        
        bond_idx1_list = []
        bond_idx2_list = []
        bond_ref_list = []
        bond_sigma_list = []
        
        pdb = self.model.pdb
        
        # Iterate through all chains
        for chain_id in pdb['chainid'].unique():
            chain = pdb[pdb['chainid'] == chain_id]
            
            # Iterate through all residues in the chain
            for resseq in chain['resseq'].unique():
                residue = pdb[(pdb['resseq'] == resseq) & (pdb['chainid'] == chain_id)]
                
                # Get residue name
                resname = residue['resname'].values[0]
                
                # Check if restraints exist for this residue type
                if resname not in self.cif_dict:
                    continue
                
                cif_residue = self.cif_dict[resname]
                
                for bond_key in self.bond_keys:
                    if bond_key in cif_residue:
                        bond_key = bond_key
                        break
                
                cif_bonds = cif_residue[bond_key]
                
                # Filter to only include bonds where both atoms are present
                atom_names = residue['name'].values
                usable_bonds = cif_bonds[
                    cif_bonds['atom_id_1'].isin(atom_names) & 
                    cif_bonds['atom_id_2'].isin(atom_names)
                ]
                
                if len(usable_bonds) == 0:
                    continue
                
                # Create a mapping from atom names to indices
                residue_indexed = residue.set_index('name')
                
                # Get atom indices
                idx1 = residue_indexed.loc[usable_bonds['atom_id_1'], 'index'].values
                idx2 = residue_indexed.loc[usable_bonds['atom_id_2'], 'index'].values
                
                # Find the column names for reference values and sigmas
                value_col = [col for col in usable_bonds.columns if 'value' in col and 'dist' in col][0]
                esd_col = [col for col in usable_bonds.columns if 'esd' in col and 'dist' in col][0]
                
                # Get reference distances and sigmas
                references = usable_bonds[value_col].values.astype(float)
                sigmas = usable_bonds[esd_col].values.astype(float)
                
                # Replace zero sigmas with a small value to avoid division by zero
                sigmas[sigmas == 0] = 1e-4
                
                # Append to lists
                bond_idx1_list.append(idx1)
                bond_idx2_list.append(idx2)
                bond_ref_list.append(references)
                bond_sigma_list.append(sigmas)
        
        # Concatenate all bond restraints
        if len(bond_idx1_list) > 0:
            bond_idx1 = np.concatenate(bond_idx1_list, dtype=int)
            bond_idx2 = np.concatenate(bond_idx2_list, dtype=int)
            
            # Create (N, 2) tensor for bond indices
            self.bond_indices = torch.tensor(
                np.stack([bond_idx1, bond_idx2], axis=1), 
                dtype=torch.long
            )
            self.bond_references = torch.tensor(
                np.concatenate(bond_ref_list), 
                dtype=torch.float32
            )
            self.bond_sigmas = torch.tensor(
                np.concatenate(bond_sigma_list), 
                dtype=torch.float32
            )
        else:
            print("Warning: No bond restraints were built")
    
    def _build_angle_restraints(self):
        """
        Build angle restraints for the entire structure.
        
        Iterates through all chains and residues, matching atom triplets with
        the restraints defined in the CIF dictionary. Only includes angles where
        all three atoms are present in the residue.
        """
        if not self.angle_keys:
            print("Warning: No angle restraints found in CIF dictionary")
            return
        
        angle_idx1_list = []
        angle_idx2_list = []
        angle_idx3_list = []
        angle_ref_list = []
        angle_sigma_list = []
        
        pdb = self.model.pdb
        
        # Iterate through all chains
        for chain_id in pdb['chainid'].unique():
            chain = pdb[pdb['chainid'] == chain_id]
            
            # Iterate through all residues in the chain
            for resseq in chain['resseq'].unique():
                residue = pdb[(pdb['resseq'] == resseq) & (pdb['chainid'] == chain_id)]
                            
                # Get residue name
                resname = residue['resname'].values[0]
                
                # Check if restraints exist for this residue type
                if resname not in self.cif_dict:
                    continue
                
                cif_residue = self.cif_dict[resname]
                
                # Check if angle restraints exist for this residue type
                for angle_key in self.angle_keys:
                    if angle_key in cif_residue:
                        angle_key = angle_key
                        break
                
                cif_angles = cif_residue[angle_key]
                
                # Filter to only include angles where all atoms are present
                atom_names = residue['name'].values
                usable_angles = cif_angles[
                    cif_angles['atom_id_1'].isin(atom_names) & 
                    cif_angles['atom_id_2'].isin(atom_names) &
                    cif_angles['atom_id_3'].isin(atom_names)
                ]
                
                if len(usable_angles) == 0:
                    continue
                
                # Create a mapping from atom names to indices
                residue_indexed = residue.set_index('name')
                
                # Get atom indices
                idx1 = residue_indexed.loc[usable_angles['atom_id_1'], 'index'].values
                idx2 = residue_indexed.loc[usable_angles['atom_id_2'], 'index'].values
                idx3 = residue_indexed.loc[usable_angles['atom_id_3'], 'index'].values
                
                # Find the column names for reference values and sigmas
                value_col = [col for col in usable_angles.columns if 'value' in col and 'angle' in col][0]
                esd_col = [col for col in usable_angles.columns if 'esd' in col and 'angle' in col][0]
                
                # Get reference angles and sigmas
                references = usable_angles[value_col].values.astype(float)
                sigmas = usable_angles[esd_col].values.astype(float)
                
                # Replace zero sigmas with a small value to avoid division by zero
                sigmas[sigmas == 0] = 1e-4
                
                # Append to lists
                angle_idx1_list.append(idx1)
                angle_idx2_list.append(idx2)
                angle_idx3_list.append(idx3)
                angle_ref_list.append(references)
                angle_sigma_list.append(sigmas)
        
        # Concatenate all angle restraints
        if len(angle_idx1_list) > 0:
            angle_idx1 = np.concatenate(angle_idx1_list, dtype=int)
            angle_idx2 = np.concatenate(angle_idx2_list, dtype=int)
            angle_idx3 = np.concatenate(angle_idx3_list, dtype=int)
            
            # Create (N, 3) tensor for angle indices
            self.angle_indices = torch.tensor(
                np.stack([angle_idx1, angle_idx2, angle_idx3], axis=1), 
                dtype=torch.long
            )
            self.angle_references = torch.tensor(
                np.concatenate(angle_ref_list), 
                dtype=torch.float32
            )
            self.angle_sigmas = torch.tensor(
                np.concatenate(angle_sigma_list), 
                dtype=torch.float32
            )
        else:
            print("Warning: No angle restraints were built")
    
    def _build_torsion_restraints(self):
        """
        Build torsion angle restraints for the entire structure.
        
        Iterates through all chains and residues, matching atom quartets with
        the restraints defined in the CIF dictionary. Only includes torsions where
        all four atoms are present in the residue.
        """
        if not self.torsion_keys:
            print("Warning: No torsion restraints found in CIF dictionary")
            return
        
        torsion_idx1_list = []
        torsion_idx2_list = []
        torsion_idx3_list = []
        torsion_idx4_list = []
        torsion_ref_list = []
        torsion_sigma_list = []
        
        pdb = self.model.pdb
        
        # Iterate through all chains
        for chain_id in pdb['chainid'].unique():
            chain = pdb[pdb['chainid'] == chain_id]
            
            # Iterate through all residues in the chain
            for resseq in chain['resseq'].unique():
                residue = pdb[(pdb['resseq'] == resseq) & (pdb['chainid'] == chain_id)]
                
                # Get residue name
                resname = residue['resname'].values[0]
                
                # Check if restraints exist for this residue type
                if resname not in self.cif_dict:
                    continue
                
                cif_residue = self.cif_dict[resname]
                
                # select correct torsion key
                for torsion_key in self.torsion_keys:
                    if torsion_key in cif_residue:
                        torsion_key = torsion_key
                        break
                
                cif_torsions = cif_residue[torsion_key]
                
                # Filter to only include torsions where all atoms are present
                atom_names = residue['name'].values
                usable_torsions = cif_torsions[
                    cif_torsions['atom_id_1'].isin(atom_names) & 
                    cif_torsions['atom_id_2'].isin(atom_names) &
                    cif_torsions['atom_id_3'].isin(atom_names) &
                    cif_torsions['atom_id_4'].isin(atom_names)
                ]
                
                if len(usable_torsions) == 0:
                    continue
                
                # Create a mapping from atom names to indices
                residue_indexed = residue.set_index('name')
                
                # Get atom indices
                idx1 = residue_indexed.loc[usable_torsions['atom_id_1'], 'index'].values
                idx2 = residue_indexed.loc[usable_torsions['atom_id_2'], 'index'].values
                idx3 = residue_indexed.loc[usable_torsions['atom_id_3'], 'index'].values
                idx4 = residue_indexed.loc[usable_torsions['atom_id_4'], 'index'].values
                
                # Find the column names for reference values and sigmas
                value_col = [col for col in usable_torsions.columns if 'value' in col and 'angle' in col][0]
                esd_col = [col for col in usable_torsions.columns if 'esd' in col and 'angle' in col][0]
                
                # Get reference torsion angles and sigmas
                references = usable_torsions[value_col].values.astype(float)
                sigmas = usable_torsions[esd_col].values.astype(float)
                
                # Replace zero sigmas with a small value to avoid division by zero
                sigmas[sigmas == 0] = 1e-4
                
                # Append to lists
                torsion_idx1_list.append(idx1)
                torsion_idx2_list.append(idx2)
                torsion_idx3_list.append(idx3)
                torsion_idx4_list.append(idx4)
                torsion_ref_list.append(references)
                torsion_sigma_list.append(sigmas)
        
        # Concatenate all torsion restraints
        if len(torsion_idx1_list) > 0:
            torsion_idx1 = np.concatenate(torsion_idx1_list, dtype=int)
            torsion_idx2 = np.concatenate(torsion_idx2_list, dtype=int)
            torsion_idx3 = np.concatenate(torsion_idx3_list, dtype=int)
            torsion_idx4 = np.concatenate(torsion_idx4_list, dtype=int)
            
            # Create (N, 4) tensor for torsion indices
            self.torsion_indices = torch.tensor(
                np.stack([torsion_idx1, torsion_idx2, torsion_idx3, torsion_idx4], axis=1), 
                dtype=torch.long
            )
            self.torsion_references = torch.tensor(
                np.concatenate(torsion_ref_list), 
                dtype=torch.float32
            )
            self.torsion_sigmas = torch.tensor(
                np.concatenate(torsion_sigma_list), 
                dtype=torch.float32
            )
        else:
            print("Warning: No torsion restraints were built")
    
    def _move_to_device(self, device):
        """
        Move all restraint tensors to the specified device.
        
        Args:
            device: PyTorch device (cpu, cuda, etc.)
        """
        if self.bond_indices is not None:
            self.bond_indices = self.bond_indices.to(device)
            self.bond_references = self.bond_references.to(device)
            self.bond_sigmas = self.bond_sigmas.to(device)
        
        if self.angle_indices is not None:
            self.angle_indices = self.angle_indices.to(device)
            self.angle_references = self.angle_references.to(device)
            self.angle_sigmas = self.angle_sigmas.to(device)
        
        if self.torsion_indices is not None:
            self.torsion_indices = self.torsion_indices.to(device)
            self.torsion_references = self.torsion_references.to(device)
            self.torsion_sigmas = self.torsion_sigmas.to(device)
    
    def cuda(self, device: Optional[int] = None):
        """
        Move all restraint tensors to CUDA device.
        
        Args:
            device: CUDA device index (default: None for current device)
            
        Returns:
            self for chaining
        """
        cuda_device = torch.device('cuda' if device is None else f'cuda:{device}')
        self._move_to_device(cuda_device)
        return self
    
    def cpu(self):
        """
        Move all restraint tensors to CPU.
        
        Returns:
            self for chaining
        """
        self._move_to_device(torch.device('cpu'))
        return self
    
    def __repr__(self):
        """String representation of the Restraints object."""
        n_bonds = 0 if self.bond_indices is None else self.bond_indices.shape[0]
        n_angles = 0 if self.angle_indices is None else self.angle_indices.shape[0]
        n_torsions = 0 if self.torsion_indices is None else self.torsion_indices.shape[0]
        
        device = 'not initialized'
        if self.bond_indices is not None:
            device = str(self.bond_indices.device)
        elif self.angle_indices is not None:
            device = str(self.angle_indices.device)
        elif self.torsion_indices is not None:
            device = str(self.torsion_indices.device)
        
        return (f"Restraints(\n"
                f"  cif_path='{self.cif_path}',\n"
                f"  n_bonds={n_bonds},\n"
                f"  n_angles={n_angles},\n"
                f"  n_torsions={n_torsions},\n"
                f"  device={device}\n"
                f")")
    
    def summary(self):
        """Print a detailed summary of all restraints."""
        print("=" * 80)
        print("Restraints Summary")
        print("=" * 80)
        print(f"CIF file: {self.cif_path}")
        print(f"Residue types in dictionary: {len(self.cif_dict)}")
        print()
        
        if self.bond_indices is not None:
            print(f"Bond Length Restraints: {self.bond_indices.shape[0]}")
            print(f"  Shape: {self.bond_indices.shape}")
            print(f"  Reference distances: min={self.bond_references.min():.3f}, "
                  f"max={self.bond_references.max():.3f}, "
                  f"mean={self.bond_references.mean():.3f}")
            print(f"  Sigmas: min={self.bond_sigmas.min():.4f}, "
                  f"max={self.bond_sigmas.max():.4f}, "
                  f"mean={self.bond_sigmas.mean():.4f}")
        else:
            print("Bond Length Restraints: None")
        print()
        
        if self.angle_indices is not None:
            print(f"Angle Restraints: {self.angle_indices.shape[0]}")
            print(f"  Shape: {self.angle_indices.shape}")
            print(f"  Reference angles: min={self.angle_references.min():.2f}°, "
                  f"max={self.angle_references.max():.2f}°, "
                  f"mean={self.angle_references.mean():.2f}°")
            print(f"  Sigmas: min={self.angle_sigmas.min():.4f}°, "
                  f"max={self.angle_sigmas.max():.4f}°, "
                  f"mean={self.angle_sigmas.mean():.4f}°")
        else:
            print("Angle Restraints: None")
        print()
        
        if self.torsion_indices is not None:
            print(f"Torsion Angle Restraints: {self.torsion_indices.shape[0]}")
            print(f"  Shape: {self.torsion_indices.shape}")
            print(f"  Reference torsions: min={self.torsion_references.min():.2f}°, "
                  f"max={self.torsion_references.max():.2f}°, "
                  f"mean={self.torsion_references.mean():.2f}°")
            print(f"  Sigmas: min={self.torsion_sigmas.min():.4f}°, "
                  f"max={self.torsion_sigmas.max():.4f}°, "
                  f"mean={self.torsion_sigmas.mean():.4f}°")
        else:
            print("Torsion Angle Restraints: None")
        print("=" * 80)
