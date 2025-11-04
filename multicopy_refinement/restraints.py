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

from networkx import sigma
import torch
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from multicopy_refinement.restraints_helper import read_cif,find_cif_file_in_library,read_link_definitions
from multicopy_refinement.model import Model
from torch.special import i0  # modified Bessel function of the first kind
from torch.nn import Module


class Restraints(Module):
    """
    Restraints handler for crystallographic model refinement.
    
    This class parses CIF restraints dictionaries and builds efficient tensor 
    representations for the entire molecular structure. It stores restraints
    for bond lengths, angles, torsion angles, and planes with their expected values
    and uncertainties (sigma values).
    
    The restraints are organized in a hierarchical dictionary structure:
        restraints[restraint_type][origin][property]
    
    Where:
        - restraint_type: 'bond', 'angle', 'torsion', 'plane'
        - origin: 
            - For bonds/angles/torsions: 'intra' (intra-residue), 'peptide', 'disulfide', 'phi', 'psi', 'omega'
            - For planes: '3_atoms', '4_atoms', ..., '10_atoms' (grouped by atom count, minimum 3)
        - property: 'indices', 'references', 'sigmas', 'periods' (for torsions only)
    
    For backward compatibility, the old flat attribute names are preserved as properties.
    
    Attributes:
        model: Reference to the Model instance
        cif_path: Path to the CIF restraints dictionary file
        cif_dict: Parsed CIF dictionary with restraints for each residue type
        restraints: Hierarchical dictionary containing all restraints
        
    Example:
        >>> from multicopy_refinement.model import Model
        >>> from multicopy_refinement.restraints import Restraints
        >>> 
        >>> # Load model
        >>> model = Model()
        >>> model.load_pdb_from_file('structure.pdb')
        >>> 
        >>> # Create restraints
        >>> restraints = Restraints(model)
        >>> 
        >>> # Access via hierarchical structure (new way)
        >>> bond_indices = restraints.restraints['bond']['intra']['indices']
        >>> angle_refs = restraints.restraints['angle']['peptide']['references']
        >>> torsion_periods = restraints.restraints['torsion']['intra']['periods']
        >>> plane_4atom_indices = restraints.restraints['plane']['4_atoms']['indices']
        >>> 
        >>> # Or use backward-compatible properties (old way)
        >>> bond_indices = restraints.bond_indices  # intra-residue bonds
        >>> bond_indices_inter = restraints.bond_indices_inter  # peptide bonds
    """
    
    def __init__(self, model: Model, cif_path: Optional[str] = None, verbose: int = 1):
        """
        Initialize the Restraints handler.
        
        Args:
            model: Model instance containing the atomic structure
            cif_path: Path to the CIF restraints dictionary file
            verbose: Verbosity level (0=silent, 1=normal, 2=detailed)
        """
        super().__init__()  
        self.model = model
        self.cif_path = cif_path
        self.verbose = verbose  
        self.unique_residues = model.pdb.resname.unique()
        # Check unique residue have more than one atom
        self.unique_residues = [residue for residue in self.unique_residues if self.model.pdb.loc[self.model.pdb['resname'] == residue,'name'].nunique() > 1]
        # Parse the CIF file
        if cif_path:
            self.cif_dict = read_cif(cif_path)
        else:
            self.cif_dict = {}
        
        self.missing_residues = [res for res in self.unique_residues if res not in self.cif_dict]
        additional_files_to_load = [find_cif_file_in_library(res) for res in self.missing_residues]

        for cif_file in additional_files_to_load:
            if cif_file is not None:
                if self.verbose > 1: print(cif_file)
                additional_cif_dict = read_cif(cif_file)
                self.cif_dict.update(additional_cif_dict)
        
        self.missing_residues = [res for res in self.unique_residues if res not in self.cif_dict]
        
        if len(self.missing_residues) > 1: 
            if verbose > 0: print(f"Warning: The following residues are missing from the CIF dictionary and will have no restraints applied: {self.missing_residues}")

        # Load link definitions for inter-residue restraints (peptide bonds, etc.)
        if verbose > 1: print("Loading link definitions from monomer library...")
        self.link_dict, self.link_list = read_link_definitions()
        if verbose > 1: print(f"Loaded {len(self.link_dict)} link types")

        # Identify restraint dictionary keys
        self._identify_restraint_keys()
        
        # Initialize hierarchical restraints dictionary
        # Structure: restraints[restraint_type][origin][property]
        # restraint_type: 'bond', 'angle', 'torsion', 'plane'
        # origin: 'intra', 'peptide', 'disulfide', 'phi', 'psi', 'omega', etc.
        # property: 'indices', 'references', 'sigmas', 'periods' (for torsions)
        # For planes: origin is atom count like '3_atoms', '4_atoms', etc.
        self.restraints = {
            'bond': {},
            'angle': {},
            'torsion': {},
            'plane': {}
        }
        
        # Build restraints for the entire structure
        self.build_restraints()
    
    def _identify_restraint_keys(self):
        """
        Identify the CIF dictionary keys for different restraint types.
        
        CIF dictionaries may use different naming conventions. This method
        searches for the appropriate keys for bonds, angles, torsions, and planes.
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
            if '_chem_comp_tor' in key.lower():
                self.torsion_keys.append(key)
        
        # Find the key for plane restraints
        self.plane_keys = []
        for key in nested_keys:
            if 'plane' in key.lower():
                self.plane_keys.append(key)
        
        if self.verbose > 0: 
            print(f"Identified restraint keys:")
            print(f"  Bond keys: {self.bond_keys}")
            print(f"  Angle keys: {self.angle_keys}")
            print(f"  Torsion keys: {self.torsion_keys}")
            print(f"  Plane keys: {self.plane_keys}")

    
    def build_restraints(self):
        """
        Build all restraints for the entire structure.
        
        This method iterates through all residues in the model and builds
        restraints for bond lengths, angles, torsions, and planes. The results are
        stored as tensors on the same device as the model coordinates.
        """
        # Build intra-residue restraints
        self._build_bond_restraints()
        self._build_angle_restraints()
        self._build_torsion_restraints()
        self._build_plane_restraints()
        
        # Build inter-residue restraints (peptide bonds, disulfide bonds, etc.)
        self._build_peptide_bond_restraints()
        self._build_disulfide_bond_restraints()
        
        # Move tensors to the same device as model
        device = self.model.xyz().device
        self._move_to_device(device)

    def expand_altloc(self, residue):
        alt_conf = residue['altloc'].unique()
        if ' ' in alt_conf:
            residue = residue.copy()
            residue_no_alt = residue.loc[residue['altloc'] == ' ']
            for alt in alt_conf:
                if alt == ' ':
                    continue
                residue_alt = residue.loc[residue['altloc'] == alt]
                residue = pd.concat([residue_no_alt, residue_alt], ignore_index=True)
            residue = residue.loc[residue['altloc'] == ' ']
        for alt_loc in residue['altloc'].unique():
            residue_alt = residue.loc[residue['altloc'] == alt_loc]
            yield residue_alt
            

    def _build_bond_restraints(self):
        """
        Build bond length restraints for the entire structure.
        
        Iterates through all chains and residues, matching atoms with the
        restraints defined in the CIF dictionary. Only includes bonds where
        both atoms are present in the residue.
        """

        def __build_bonds(residue):
            # Filter to only include bonds where both atoms are present
            atom_names = residue['name'].values
            usable_bonds = cif_bonds[
                cif_bonds['atom_id_1'].isin(atom_names) & 
                cif_bonds['atom_id_2'].isin(atom_names)
            ]
            
            if len(usable_bonds) == 0:
                return
            
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
                if any(residue['altloc'] != ' '):
                    for residue_alt in self.expand_altloc(residue):
                        __build_bonds(residue_alt)
                else:
                    __build_bonds(residue)

                
        
        # Concatenate all bond restraints
        if len(bond_idx1_list) > 0:
            bond_idx1 = np.concatenate(bond_idx1_list, dtype=int)
            bond_idx2 = np.concatenate(bond_idx2_list, dtype=int)
            
            # Store in hierarchical dictionary
            self.restraints['bond']['intra'] = {
                'indices': torch.tensor(
                    np.stack([bond_idx1, bond_idx2], axis=1), 
                    dtype=torch.long
                ),
                'references': torch.tensor(
                    np.concatenate(bond_ref_list), 
                    dtype=torch.float32
                ),
                'sigmas': torch.tensor(
                    np.concatenate(bond_sigma_list), 
                    dtype=torch.float32
                )
            }
            
            assert (self.restraints['bond']['intra']['indices'].shape[0] == 
                    self.restraints['bond']['intra']['references'].shape[0] == 
                    self.restraints['bond']['intra']['sigmas'].shape[0]), \
                    "Inconsistent bond restraint shapes"
        else:
            print("Warning: No bond restraints were built")
    
    def _build_angle_restraints(self):
        def __build_angle(residue):
            atom_names = residue['name'].values
            usable_angles = cif_angles.loc[
                cif_angles['atom_id_1'].isin(atom_names) & 
                cif_angles['atom_id_2'].isin(atom_names) &
                cif_angles['atom_id_3'].isin(atom_names)
            ]
            if len(usable_angles) == 0:
                return
            
            # Create a mapping from atom names to indices
            residue_indexed = residue.set_index('name')
            
            # Get atom indices
            idx1 = residue_indexed.loc[usable_angles['atom_id_1'], 'index'].values
            idx2 = residue_indexed.loc[usable_angles['atom_id_2'], 'index'].values
            idx3 = residue_indexed.loc[usable_angles['atom_id_3'], 'index'].values
            if len(idx1) != len(idx2) or len(idx1) != len(idx3):
                print("Warning: Mismatched angle indices lengths, skipping these angles")
                pdb1 = residue_indexed.loc[usable_angles['atom_id_1']]
                pdb2 = residue_indexed.loc[usable_angles['atom_id_2']]
                pdb3 = residue_indexed.loc[usable_angles['atom_id_3']]
                print(pdb1)
                print(pdb2)
                print(pdb3)
                print(usable_angles)
                raise ValueError("Mismatched angle indices lengths")
                
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

                # Handle alternate conformations
                if any(residue['altloc'] != ' '):
                    for residue_alt in self.expand_altloc(residue):
                        __build_angle(residue_alt)
                else:
                    __build_angle(residue)

        # Concatenate all angle restraints
        if len(angle_idx1_list) > 0:
            angle_idx1 = np.concatenate(angle_idx1_list, dtype=int)
            angle_idx2 = np.concatenate(angle_idx2_list, dtype=int)
            angle_idx3 = np.concatenate(angle_idx3_list, dtype=int)
            
            # Store in hierarchical dictionary
            self.restraints['angle']['intra'] = {
                'indices': torch.tensor(
                    np.stack([angle_idx1, angle_idx2, angle_idx3], axis=1), 
                    dtype=torch.long
                ),
                'references': torch.tensor(
                    np.concatenate(angle_ref_list), 
                    dtype=torch.float32
                ),
                'sigmas': torch.tensor(
                    np.concatenate(angle_sigma_list), 
                    dtype=torch.float32
                )
            }
            
            assert (self.restraints['angle']['intra']['indices'].shape[0] == 
                    self.restraints['angle']['intra']['references'].shape[0] == 
                    self.restraints['angle']['intra']['sigmas'].shape[0]), \
                    "Inconsistent angle restraint shapes"
        else:
            print("Warning: No angle restraints were built")
    
    def _build_torsion_restraints(self):
        def __build_torsion(residue):
            atom_names = residue['name'].values
            usable_torsions = cif_torsions[
                cif_torsions['atom_id_1'].isin(atom_names) & 
                cif_torsions['atom_id_2'].isin(atom_names) &
                cif_torsions['atom_id_3'].isin(atom_names) &
                cif_torsions['atom_id_4'].isin(atom_names)
            ]
            
            if len(usable_torsions) == 0:
                return
            
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
            
            # Get reference torsion angles, sigmas, and periods
            references = usable_torsions[value_col].values.astype(float)
            sigmas = usable_torsions[esd_col].values.astype(float)
            
            # Get period if available, default to 1 if not present
            if 'period' in usable_torsions.columns:
                periods = usable_torsions['period'].values.astype(int)
            else:
                periods = np.ones(len(usable_torsions), dtype=int)
            
            # Filter out constrained torsions (sigma ≈ 0)
            # These are aromatic ring torsions marked as CONST in CIF files
            # Their planarity is enforced by plane restraints, not torsions
            valid_mask = sigmas > 0.01  # 0.01 degree threshold
            
            if not np.all(valid_mask):
                n_filtered = (~valid_mask).sum()
                if self.verbose > 2:
                    filtered_ids = usable_torsions[~valid_mask]['id'].values if 'id' in usable_torsions.columns else ['unknown'] * n_filtered
                    print(f"Filtering {n_filtered} constrained torsions (sigma≈0): {', '.join(filtered_ids[:5])}")
                
                # Apply filter to all arrays
                idx1 = idx1[valid_mask]
                idx2 = idx2[valid_mask]
                idx3 = idx3[valid_mask]
                idx4 = idx4[valid_mask]
                references = references[valid_mask]
                sigmas = sigmas[valid_mask]
                periods = periods[valid_mask]
            
            # Safety check: ensure no zero sigmas remain
            if np.any(sigmas == 0):
                if self.verbose > 0:
                    print(f"Warning: Found {(sigmas == 0).sum()} torsions with sigma=0 after filtering, setting to 0.1°")
                sigmas[sigmas == 0] = 0.1
            
            # Skip if no valid torsions remain
            if len(idx1) == 0:
                return
            
            # Append to lists
            torsion_idx1_list.append(idx1)
            torsion_idx2_list.append(idx2)
            torsion_idx3_list.append(idx3)
            torsion_idx4_list.append(idx4)
            torsion_ref_list.append(references)
            torsion_sigma_list.append(sigmas)
            torsion_period_list.append(periods)
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
        torsion_period_list = []
        
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
                # Handle alternate conformations
                if any(residue['altloc'] != ' '):
                    for residue_alt in self.expand_altloc(residue):
                        __build_torsion(residue_alt)
                else:
                    __build_torsion(residue)
                # Filter to only include torsions where all atoms are present

        
        # Concatenate all torsion restraints
        if len(torsion_idx1_list) > 0:
            torsion_idx1 = np.concatenate(torsion_idx1_list, dtype=int)
            torsion_idx2 = np.concatenate(torsion_idx2_list, dtype=int)
            torsion_idx3 = np.concatenate(torsion_idx3_list, dtype=int)
            torsion_idx4 = np.concatenate(torsion_idx4_list, dtype=int)
            
            # Store in hierarchical dictionary
            self.restraints['torsion']['intra'] = {
                'indices': torch.tensor(
                    np.stack([torsion_idx1, torsion_idx2, torsion_idx3, torsion_idx4], axis=1), 
                    dtype=torch.long
                ),
                'references': torch.tensor(
                    np.concatenate(torsion_ref_list), 
                    dtype=torch.float32
                ),
                'sigmas': torch.tensor(
                    np.concatenate(torsion_sigma_list), 
                    dtype=torch.float32
                ),
                'periods': torch.tensor(
                    np.concatenate(torsion_period_list),
                    dtype=torch.long
                )
            }
            
            assert (self.restraints['torsion']['intra']['indices'].shape[0] == 
                    self.restraints['torsion']['intra']['references'].shape[0] == 
                    self.restraints['torsion']['intra']['sigmas'].shape[0] == 
                    self.restraints['torsion']['intra']['periods'].shape[0]), \
                    "Inconsistent torsion restraint shapes"
        else:
            print("Warning: No torsion restraints were built")
    
    def _build_plane_restraints(self):
        """
        Build plane restraints for the entire structure.
        
        Planes have variable atom counts (minimum 3 atoms required, typically 3-10 atoms). 
        This method groups planes by their atom count and stores them in separate dictionaries:
        restraints['plane']['3_atoms'], restraints['plane']['4_atoms'], etc.
        
        Planes with fewer than 3 atoms are skipped as they are mathematically invalid.
        
        Each plane is stored with:
        - indices: (N, max_atoms) tensor of atom indices (padded with -1 for unused slots)
        - sigmas: (N, max_atoms) tensor of sigma values
        - atom_counts: (N,) tensor indicating how many atoms each plane actually has
        """
        if not self.plane_keys:
            if self.verbose > 1:
                print("No plane restraints found in CIF dictionary")
            return
        
        # Dictionary to collect planes by atom count
        # planes_by_count[atom_count] = {'atom_indices': [...], 'sigmas': [...]}
        planes_by_count = {}
        
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
                
                # Check if plane restraints exist for this residue type
                plane_key = None
                for key in self.plane_keys:
                    if key in cif_residue:
                        plane_key = key
                        break
                
                if plane_key is None:
                    continue
                
                cif_planes = cif_residue[plane_key]
                
                # Handle alternate conformations
                residue_variants = []
                if any(residue['altloc'] != ' '):
                    residue_variants = list(self.expand_altloc(residue))
                else:
                    residue_variants = [residue]
                
                for residue_alt in residue_variants:
                    atom_names = residue_alt['name'].values
                    
                    # Filter to only include plane atoms that are present
                    usable_planes = cif_planes[cif_planes['atom_id'].isin(atom_names)]
                    
                    if len(usable_planes) == 0:
                        continue
                    
                    # Create a mapping from atom names to indices
                    residue_indexed = residue_alt.set_index('name')
                    
                    # Group by plane_id
                    for plane_id in usable_planes['plane_id'].unique():
                        plane_atoms = usable_planes[usable_planes['plane_id'] == plane_id]
                        
                        # Get atom indices and sigmas for this plane
                        atom_ids = plane_atoms['atom_id'].values
                        atom_indices = residue_indexed.loc[atom_ids, 'index'].values
                        
                        # Find the column name for sigma/esd
                        esd_col = [col for col in plane_atoms.columns if 'esd' in col][0]
                        sigmas = plane_atoms[esd_col].values.astype(float)
                        
                        # Replace zero sigmas with a small value
                        sigmas[sigmas == 0] = 1e-4
                        
                        # Get the atom count for this plane
                        atom_count = len(atom_indices)
                        
                        # Skip planes with fewer than 3 atoms (invalid)
                        if atom_count < 3:
                            if self.verbose > 3:
                                print(f"Warning: Skipping plane {plane_id} with only {atom_count} atom(s) - planes require at least 3 atoms")
                            continue
                        
                        # Initialize the list for this atom count if needed
                        if atom_count not in planes_by_count:
                            planes_by_count[atom_count] = {
                                'atom_indices': [],
                                'sigmas': []
                            }
                        
                        # Append to the appropriate list
                        planes_by_count[atom_count]['atom_indices'].append(atom_indices)
                        planes_by_count[atom_count]['sigmas'].append(sigmas)
        
        # Convert to tensors and store in hierarchical structure
        if len(planes_by_count) == 0:
            if self.verbose > 1:
                print("No plane restraints were built")
            return
        
        for atom_count, plane_data in planes_by_count.items():
            atom_indices_list = plane_data['atom_indices']
            sigmas_list = plane_data['sigmas']
            
            if len(atom_indices_list) > 0:
                # Stack into arrays
                indices_array = np.stack(atom_indices_list, axis=0)
                sigmas_array = np.stack(sigmas_list, axis=0)
                
                # Store in hierarchical dictionary
                key = f'{atom_count}_atoms'
                self.restraints['plane'][key] = {
                    'indices': torch.tensor(indices_array, dtype=torch.long),
                    'sigmas': torch.tensor(sigmas_array, dtype=torch.float32)
                }
                
                if self.verbose > 1:
                    print(f"Built {len(atom_indices_list)} plane restraints with {atom_count} atoms")
    
    def _build_peptide_bond_restraints(self):
        """
        Build peptide bond restraints between consecutive residues.
        
        This method iterates through all chains and identifies consecutive residues
        (resseq and resseq+1). For each consecutive pair, it adds TRANS peptide
        bond, angle, and torsion restraints.
        
        The restraints are stored separately from intra-residue restraints in:
        - bond_indices_inter, angle_indices_inter, torsion_indices_inter
        - bond_references_inter, angle_references_inter, torsion_references_inter
        - bond_sigmas_inter, angle_sigmas_inter, torsion_sigmas_inter
        """
        if 'TRANS' not in self.link_dict:
            if self.verbose > 0:
                print("Warning: TRANS link not found in link dictionary, skipping peptide bonds")
            return
        
        trans_link = self.link_dict['TRANS']
        
        # Get bond, angle, torsion, and plane definitions
        trans_bonds = trans_link.get('bonds', None)
        trans_angles = trans_link.get('angles', None)
        trans_torsions = trans_link.get('torsions', None)
        trans_planes = trans_link.get('planes', None)
        
        if trans_bonds is None:
            if self.verbose > 0:
                print("Warning: No bond restraints in TRANS link definition")
            return
        
        # Get the C-N bond parameters
        c_n_bond = trans_bonds[
            (trans_bonds['atom_1_comp_id'] == '1') & 
            (trans_bonds['atom_id_1'] == 'C') &
            (trans_bonds['atom_2_comp_id'] == '2') & 
            (trans_bonds['atom_id_2'] == 'N')
        ]
        
        if len(c_n_bond) == 0:
            if self.verbose > 0:
                print("Warning: C-N bond not found in TRANS link definition")
            return
        
        # Get bond parameters
        bond_length = float(c_n_bond['value_dist'].values[0])
        bond_sigma = float(c_n_bond['value_dist_esd'].values[0])
        
        if self.verbose > 1:
            print(f"Peptide bond parameters: C-N = {bond_length:.3f} Å ± {bond_sigma:.4f} Å")
        
        # Lists to accumulate peptide bond restraints
        bond_idx1_list = []
        bond_idx2_list = []
        bond_ref_list = []
        bond_sigma_list = []
        
        # Lists for angle restraints
        angle_idx1_list = []
        angle_idx2_list = []
        angle_idx3_list = []
        angle_ref_list = []
        angle_sigma_list = []
        
        # Lists for backbone torsion angles (phi, psi, omega tracked separately)
        phi_idx1_list = []
        phi_idx2_list = []
        phi_idx3_list = []
        phi_idx4_list = []
        phi_period_list = []
        
        psi_idx1_list = []
        psi_idx2_list = []
        psi_idx3_list = []
        psi_idx4_list = []
        psi_period_list = []
        
        omega_idx1_list = []
        omega_idx2_list = []
        omega_idx3_list = []
        omega_idx4_list = []
        omega_ref_list = []
        omega_sigma_list = []
        omega_period_list = []
        omega_is_proline_list = []  # Track if following residue is proline
        
        # Lists for plane restraints (organized by plane_id)
        # TRANS has plan-1 and plan-2, each with 4 atoms
        plane_indices_by_id = {}  # Dict of plane_id -> list of atom index lists
        plane_sigma_by_id = {}    # Dict of plane_id -> sigma value
        
        # Parse plane definitions if available
        if trans_planes is not None:
            # Group planes by plane_id
            for plane_id in trans_planes['plane_id'].unique():
                plane_atoms = trans_planes[trans_planes['plane_id'] == plane_id]
                plane_indices_by_id[plane_id] = []
                # All atoms in same plane should have same sigma
                plane_sigma_by_id[plane_id] = float(plane_atoms['dist_esd'].values[0])
                
                if self.verbose > 1:
                    atom_list = ', '.join([f"{row['atom_comp_id']}:{row['atom_id']}" 
                                          for _, row in plane_atoms.iterrows()])
                    print(f"Plane {plane_id}: {atom_list} (σ={plane_sigma_by_id[plane_id]:.3f} Å)")
        
        pdb = self.model.pdb
        
        # Only consider protein residues (ATOM records, not HETATM)
        protein_atoms = pdb[pdb['ATOM'] == 'ATOM']
        
        # Iterate through all chains
        for chain_id in protein_atoms['chainid'].unique():
            chain = protein_atoms[protein_atoms['chainid'] == chain_id]
            
            # Get sorted list of residue sequence numbers
            resseq_list = sorted(chain['resseq'].unique())
            
            # Iterate through consecutive residue pairs
            for i in range(len(resseq_list) - 1):
                resseq_i = resseq_list[i]
                resseq_next = resseq_list[i + 1]
                
                # Skip if not consecutive (chain break)
                if resseq_next != resseq_i + 1:
                    if self.verbose > 1:
                        print(f"Skipping chain break: {chain_id}:{resseq_i} → {chain_id}:{resseq_next}")
                    continue
                
                # Get residues
                residue_i = chain[chain['resseq'] == resseq_i]
                residue_next = chain[chain['resseq'] == resseq_next]
                
                # Get residue names
                resname_next = residue_next['resname'].values[0]
                is_proline = (resname_next == 'PRO')
                
                # Create atom lookup dictionaries for both residues
                # Handle alternate conformations by preferring ' ' then 'A'
                def get_atom_index(residue, atom_name):
                    atoms = residue[residue['name'] == atom_name]
                    if len(atoms) == 0:
                        return None
                    if ' ' in atoms['altloc'].values:
                        return atoms[atoms['altloc'] == ' '].iloc[0]['index']
                    elif 'A' in atoms['altloc'].values:
                        return atoms[atoms['altloc'] == 'A'].iloc[0]['index']
                    else:
                        return atoms.iloc[0]['index']
                
                # Get atom indices for bond (C-N)
                c_idx = get_atom_index(residue_i, 'C')
                n_idx = get_atom_index(residue_next, 'N')
                
                if c_idx is None or n_idx is None:
                    continue
                
                # Add peptide bond restraint
                bond_idx1_list.append(c_idx)
                bond_idx2_list.append(n_idx)
                bond_ref_list.append(bond_length)
                bond_sigma_list.append(bond_sigma)
                
                # Add angle restraints if available
                if trans_angles is not None:
                    for _, angle_row in trans_angles.iterrows():
                        comp1 = angle_row['atom_1_comp_id']
                        comp2 = angle_row['atom_2_comp_id']
                        comp3 = angle_row['atom_3_comp_id']
                        atom1_name = angle_row['atom_id_1']
                        atom2_name = angle_row['atom_id_2']
                        atom3_name = angle_row['atom_id_3']
                        
                        # Get atom indices based on which residue they belong to
                        res1 = residue_i if comp1 == '1' else residue_next
                        res2 = residue_i if comp2 == '1' else residue_next
                        res3 = residue_i if comp3 == '1' else residue_next
                        
                        idx1 = get_atom_index(res1, atom1_name)
                        idx2 = get_atom_index(res2, atom2_name)
                        idx3 = get_atom_index(res3, atom3_name)
                        
                        if idx1 is not None and idx2 is not None and idx3 is not None:
                            angle_idx1_list.append(idx1)
                            angle_idx2_list.append(idx2)
                            angle_idx3_list.append(idx3)
                            angle_ref_list.append(float(angle_row['value_angle']))
                            angle_sigma_list.append(float(angle_row['value_angle_esd']))
                
                # Add torsion restraints if available - separate phi, psi, and omega
                if trans_torsions is not None:
                    for _, torsion_row in trans_torsions.iterrows():
                        comp1 = torsion_row['atom_1_comp_id']
                        comp2 = torsion_row['atom_2_comp_id']
                        comp3 = torsion_row['atom_3_comp_id']
                        comp4 = torsion_row['atom_4_comp_id']
                        atom1_name = torsion_row['atom_id_1']
                        atom2_name = torsion_row['atom_id_2']
                        atom3_name = torsion_row['atom_id_3']
                        atom4_name = torsion_row['atom_id_4']
                        torsion_id = torsion_row['id']
                        
                        # Get atom indices based on which residue they belong to
                        res1 = residue_i if comp1 == '1' else residue_next
                        res2 = residue_i if comp2 == '1' else residue_next
                        res3 = residue_i if comp3 == '1' else residue_next
                        res4 = residue_i if comp4 == '1' else residue_next
                        
                        idx1 = get_atom_index(res1, atom1_name)
                        idx2 = get_atom_index(res2, atom2_name)
                        idx3 = get_atom_index(res3, atom3_name)
                        idx4 = get_atom_index(res4, atom4_name)
                        
                        if idx1 is not None and idx2 is not None and idx3 is not None and idx4 is not None:
                            # Get period if available, default to 0
                            period = int(torsion_row['period']) if 'period' in torsion_row and pd.notna(torsion_row['period']) else 0
                            
                            # Separate phi, psi, and omega angles
                            if torsion_id == 'psi':
                                # psi: N(i) - CA(i) - C(i) - N(i+1)
                                psi_idx1_list.append(idx1)
                                psi_idx2_list.append(idx2)
                                psi_idx3_list.append(idx3)
                                psi_idx4_list.append(idx4)
                                psi_period_list.append(period)
                            elif torsion_id == 'phi':
                                # phi: C(i-1) - N(i) - CA(i) - C(i)
                                # In our case: C(i) - N(i+1) - CA(i+1) - C(i+1)
                                phi_idx1_list.append(idx1)
                                phi_idx2_list.append(idx2)
                                phi_idx3_list.append(idx3)
                                phi_idx4_list.append(idx4)
                                phi_period_list.append(period)
                            elif torsion_id == 'omega':
                                # omega: CA(i) - C(i) - N(i+1) - CA(i+1)
                                omega_idx1_list.append(idx1)
                                omega_idx2_list.append(idx2)
                                omega_idx3_list.append(idx3)
                                omega_idx4_list.append(idx4)
                                omega_ref_list.append(float(torsion_row['value_angle']))
                                omega_sigma_list.append(float(torsion_row['value_angle_esd']))
                                omega_period_list.append(period)
                                omega_is_proline_list.append(is_proline)
                
                # Add plane restraints if available
                if trans_planes is not None:
                    for plane_id in plane_indices_by_id.keys():
                        plane_atoms = trans_planes[trans_planes['plane_id'] == plane_id]
                        
                        # Collect atom indices for this plane
                        atom_indices = []
                        all_found = True
                        
                        for _, plane_atom_row in plane_atoms.iterrows():
                            comp_id = plane_atom_row['atom_comp_id']
                            atom_name = plane_atom_row['atom_id']
                            
                            # Get the correct residue based on comp_id
                            residue = residue_i if comp_id == '1' else residue_next
                            
                            atom_idx = get_atom_index(residue, atom_name)
                            
                            if atom_idx is None:
                                # Skip this plane if any atom is missing (e.g., H in proline)
                                all_found = False
                                break
                            
                            atom_indices.append(atom_idx)
                        
                        # Add plane if all atoms were found
                        if all_found:
                            plane_indices_by_id[plane_id].append(atom_indices)
        
        # Create tensors for inter-residue bond restraints (peptide bonds)
        if len(bond_idx1_list) > 0:
            self.restraints['bond']['peptide'] = {
                'indices': torch.tensor(
                    np.stack([np.array(bond_idx1_list), np.array(bond_idx2_list)], axis=1),
                    dtype=torch.long
                ),
                'references': torch.tensor(
                    np.array(bond_ref_list),
                    dtype=torch.float32
                ),
                'sigmas': torch.tensor(
                    np.array(bond_sigma_list),
                    dtype=torch.float32
                )
            }
            
            if self.verbose > 0:
                print(f"Built {len(bond_idx1_list)} peptide bond restraints")
        else:
            if self.verbose > 0:
                print("Warning: No peptide bond restraints were built")
        
        # Create tensors for inter-residue angle restraints (peptide angles)
        if len(angle_idx1_list) > 0:
            self.restraints['angle']['peptide'] = {
                'indices': torch.tensor(
                    np.stack([np.array(angle_idx1_list), np.array(angle_idx2_list), np.array(angle_idx3_list)], axis=1),
                    dtype=torch.long
                ),
                'references': torch.tensor(
                    np.array(angle_ref_list),
                    dtype=torch.float32
                ),
                'sigmas': torch.tensor(
                    np.array(angle_sigma_list),
                    dtype=torch.float32
                )
            }
            
            if self.verbose > 0:
                print(f"Built {len(angle_idx1_list)} peptide angle restraints")
        
        # Create tensors for backbone torsion angles (phi, psi, omega)
        if len(phi_idx1_list) > 0:
            self.restraints['torsion']['phi'] = {
                'indices': torch.tensor(
                    np.stack([np.array(phi_idx1_list), np.array(phi_idx2_list), 
                             np.array(phi_idx3_list), np.array(phi_idx4_list)], axis=1),
                    dtype=torch.long
                ),
                'periods': torch.tensor(
                    np.array(phi_period_list),
                    dtype=torch.long
                )
            }
            if self.verbose > 0:
                print(f"Built {len(phi_idx1_list)} phi angle indices (C-N-CA-C)")
        
        if len(psi_idx1_list) > 0:
            self.restraints['torsion']['psi'] = {
                'indices': torch.tensor(
                    np.stack([np.array(psi_idx1_list), np.array(psi_idx2_list), 
                             np.array(psi_idx3_list), np.array(psi_idx4_list)], axis=1),
                    dtype=torch.long
                ),
                'periods': torch.tensor(
                    np.array(psi_period_list),
                    dtype=torch.long
                )
            }
            if self.verbose > 0:
                print(f"Built {len(psi_idx1_list)} psi angle indices (N-CA-C-N)")
        
        if len(omega_idx1_list) > 0:
            self.restraints['torsion']['omega'] = {
                'indices': torch.tensor(
                    np.stack([np.array(omega_idx1_list), np.array(omega_idx2_list), 
                             np.array(omega_idx3_list), np.array(omega_idx4_list)], axis=1),
                    dtype=torch.long
                ),
                'references': torch.tensor(
                    np.array(omega_ref_list),
                    dtype=torch.float32
                ),
                'sigmas': torch.tensor(
                    np.array(omega_sigma_list),
                    dtype=torch.float32
                ),
                'periods': torch.tensor(
                    np.array(omega_period_list),
                    dtype=torch.long
                ),
                'is_proline': torch.tensor(
                    np.array(omega_is_proline_list),
                    dtype=torch.bool
                )
            }
            if self.verbose > 0:
                n_proline = sum(omega_is_proline_list)
                print(f"Built {len(omega_idx1_list)} omega angle restraints (CA-C-N-CA, ~180°)")
                print(f"  {n_proline} omega angles preceding proline residues")
        
        # Create tensors for peptide plane restraints
        # Combine all planes from different plane_ids (plan-1, plan-2, etc.)
        if trans_planes is not None and len(plane_indices_by_id) > 0:
            all_plane_indices = []
            all_plane_sigmas = []
            
            for plane_id, plane_list in plane_indices_by_id.items():
                if len(plane_list) > 0:
                    sigma = plane_sigma_by_id[plane_id]
                    for atom_indices in plane_list:
                        all_plane_indices.append(atom_indices)
                        # Create sigma array with same length as number of atoms
                        all_plane_sigmas.append([sigma] * len(atom_indices))
            
            if len(all_plane_indices) > 0:
                # Determine atom count (should be 4 for peptide planes)
                atom_count = len(all_plane_indices[0])
                key = f'{atom_count}_atoms'
                
                # Check if this key exists, if not create it, otherwise append
                plane_indices_array = np.array(all_plane_indices, dtype=np.int64)
                plane_sigmas_array = np.array(all_plane_sigmas, dtype=np.float32)
                
                if key in self.restraints['plane']:
                    # Append to existing planes
                    existing_indices = self.restraints['plane'][key]['indices'].cpu().numpy()
                    existing_sigmas = self.restraints['plane'][key]['sigmas'].cpu().numpy()
                    
                    plane_indices_array = np.vstack([existing_indices, plane_indices_array])
                    plane_sigmas_array = np.vstack([existing_sigmas, plane_sigmas_array])
                
                self.restraints['plane'][key] = {
                    'indices': torch.tensor(plane_indices_array, dtype=torch.long),
                    'sigmas': torch.tensor(plane_sigmas_array, dtype=torch.float32)
                }
                
                if self.verbose > 0:
                    print(f"Built {len(all_plane_indices)} peptide plane restraints ({atom_count} atoms each)")
                    for plane_id, plane_list in plane_indices_by_id.items():
                        if len(plane_list) > 0:
                            print(f"  {plane_id}: {len(plane_list)} planes")
    
    def _build_disulfide_bond_restraints(self):
        """
        Build disulfide bond restraints between cysteine residues.
        
        This method identifies disulfide bonds by:
        1. Finding all cysteine residues (or residues with SG atoms)
        2. Computing distances between all pairs of SG atoms
        3. Creating disulfide restraints for SG-SG pairs closer than 2.0 Å
        
        The restraints are appended to the existing inter-residue restraints.
        This includes bonds, angles (CB-SG-SG), and torsions (CB-SG-SG-CB).
        """
        if 'disulf' not in self.link_dict:
            if self.verbose > 0:
                print("Warning: disulf link not found in link dictionary, skipping disulfide bonds")
            return
        
        disulf_link = self.link_dict['disulf']
        
        # Get bond, angle, and torsion definitions
        disulf_bonds = disulf_link.get('bonds', None)
        disulf_angles = disulf_link.get('angles', None)
        disulf_torsions = disulf_link.get('torsions', None)
        
        if disulf_bonds is None:
            if self.verbose > 0:
                print("Warning: No bond restraints in disulf link definition")
            return
        
        # Get the SG-SG bond parameters
        sg_sg_bond = disulf_bonds[
            (disulf_bonds['atom_id_1'] == 'SG') & 
            (disulf_bonds['atom_id_2'] == 'SG')
        ]
        
        if len(sg_sg_bond) == 0:
            if self.verbose > 0:
                print("Warning: SG-SG bond not found in disulf link definition")
            return
        
        # Get bond parameters
        bond_length = float(sg_sg_bond['value_dist'].values[0])
        bond_sigma = float(sg_sg_bond['value_dist_esd'].values[0])
        
        if self.verbose > 1:
            print(f"Disulfide bond parameters: SG-SG = {bond_length:.3f} Å ± {bond_sigma:.4f} Å")
        
        # Find all SG atoms (from cysteine residues)
        pdb = self.model.pdb
        sg_atoms = pdb[(pdb['name'] == 'SG') & (pdb['ATOM'] == 'ATOM')]
        
        if len(sg_atoms) == 0:
            if self.verbose > 1:
                print("No SG atoms found, skipping disulfide bonds")
            return
        
        if self.verbose > 1:
            print(f"Found {len(sg_atoms)} SG atoms")
        
        # Get coordinates of all SG atoms
        xyz = self.model.xyz()
        sg_indices = sg_atoms['index'].values
        sg_coords = xyz[sg_indices]
        
        # Get residue identifiers (chain + resseq) for each SG atom
        sg_residues = (sg_atoms['chainid'].astype(str) + '_' + sg_atoms['resseq'].astype(str)).values
        
        # Compute pairwise distances between all SG atoms
        distances = torch.cdist(sg_coords, sg_coords)
        
        # Find pairs closer than 2.0 Å (but not the same atom)
        threshold = 2.0
        close_pairs = torch.where((distances < threshold) & (distances > 0.1))
        
        # Filter out pairs from the same residue and only keep i < j
        valid_pairs = []
        for i, j in zip(close_pairs[0].cpu().numpy(), close_pairs[1].cpu().numpy()):
            if i < j and sg_residues[i] != sg_residues[j]:
                valid_pairs.append((i, j))
        
        if len(valid_pairs) == 0:
            if self.verbose > 1:
                print("No disulfide bonds found (no SG atoms from different residues within 2.0 Å)")
            return
        
        # Lists for disulfide restraints
        bond_idx1_list = []
        bond_idx2_list = []
        bond_ref_list = []
        bond_sigma_list = []
        
        angle_idx1_list = []
        angle_idx2_list = []
        angle_idx3_list = []
        angle_ref_list = []
        angle_sigma_list = []
        
        torsion_idx1_list = []
        torsion_idx2_list = []
        torsion_idx3_list = []
        torsion_idx4_list = []
        torsion_ref_list = []
        torsion_sigma_list = []
        torsion_period_list = []
        
        # Process each disulfide bond
        for i_local, j_local in valid_pairs:
            sg1_idx = sg_indices[i_local]
            sg2_idx = sg_indices[j_local]
            
            # Get the residues containing these SG atoms
            residue1 = pdb[pdb['index'] == sg1_idx].iloc[0]
            residue2 = pdb[pdb['index'] == sg2_idx].iloc[0]
            
            chain1 = residue1['chainid']
            resseq1 = residue1['resseq']
            chain2 = residue2['chainid']
            resseq2 = residue2['resseq']
            
            # Get all atoms from both residues
            res1_atoms = pdb[(pdb['chainid'] == chain1) & (pdb['resseq'] == resseq1)]
            res2_atoms = pdb[(pdb['chainid'] == chain2) & (pdb['resseq'] == resseq2)]
            
            # Helper function to get atom index
            def get_atom_index(residue, atom_name):
                atoms = residue[residue['name'] == atom_name]
                if len(atoms) == 0:
                    return None
                if ' ' in atoms['altloc'].values:
                    return atoms[atoms['altloc'] == ' '].iloc[0]['index']
                elif 'A' in atoms['altloc'].values:
                    return atoms[atoms['altloc'] == 'A'].iloc[0]['index']
                else:
                    return atoms.iloc[0]['index']
            
            # Add bond restraint (SG-SG)
            bond_idx1_list.append(sg1_idx)
            bond_idx2_list.append(sg2_idx)
            bond_ref_list.append(bond_length)
            bond_sigma_list.append(bond_sigma)
            
            # Add angle restraints if available
            if disulf_angles is not None:
                for _, angle_row in disulf_angles.iterrows():
                    comp1 = angle_row['atom_1_comp_id']
                    comp2 = angle_row['atom_2_comp_id']
                    comp3 = angle_row['atom_3_comp_id']
                    atom1_name = angle_row['atom_id_1']
                    atom2_name = angle_row['atom_id_2']
                    atom3_name = angle_row['atom_id_3']
                    
                    # Get atom indices based on which residue they belong to
                    res1 = res1_atoms if comp1 == '1' else res2_atoms
                    res2_mid = res1_atoms if comp2 == '1' else res2_atoms
                    res3 = res1_atoms if comp3 == '1' else res2_atoms
                    
                    idx1 = get_atom_index(res1, atom1_name)
                    idx2 = get_atom_index(res2_mid, atom2_name)
                    idx3 = get_atom_index(res3, atom3_name)
                    
                    if idx1 is not None and idx2 is not None and idx3 is not None:
                        angle_idx1_list.append(idx1)
                        angle_idx2_list.append(idx2)
                        angle_idx3_list.append(idx3)
                        angle_ref_list.append(float(angle_row['value_angle']))
                        angle_sigma_list.append(float(angle_row['value_angle_esd']))
            
            # Add torsion restraints if available (CB-SG-SG-CB)
            if disulf_torsions is not None:
                for _, torsion_row in disulf_torsions.iterrows():
                    comp1 = torsion_row['atom_1_comp_id']
                    comp2 = torsion_row['atom_2_comp_id']
                    comp3 = torsion_row['atom_3_comp_id']
                    comp4 = torsion_row['atom_4_comp_id']
                    atom1_name = torsion_row['atom_id_1']
                    atom2_name = torsion_row['atom_id_2']
                    atom3_name = torsion_row['atom_id_3']
                    atom4_name = torsion_row['atom_id_4']
                    
                    # Get atom indices based on which residue they belong to
                    res1_tor = res1_atoms if comp1 == '1' else res2_atoms
                    res2_tor = res1_atoms if comp2 == '1' else res2_atoms
                    res3_tor = res1_atoms if comp3 == '1' else res2_atoms
                    res4_tor = res1_atoms if comp4 == '1' else res2_atoms
                    
                    idx1 = get_atom_index(res1_tor, atom1_name)
                    idx2 = get_atom_index(res2_tor, atom2_name)
                    idx3 = get_atom_index(res3_tor, atom3_name)
                    idx4 = get_atom_index(res4_tor, atom4_name)
                    
                    if idx1 is not None and idx2 is not None and idx3 is not None and idx4 is not None:
                        torsion_idx1_list.append(idx1)
                        torsion_idx2_list.append(idx2)
                        torsion_idx3_list.append(idx3)
                        torsion_idx4_list.append(idx4)
                        torsion_ref_list.append(float(torsion_row['value_angle']))
                        torsion_sigma_list.append(float(torsion_row['value_angle_esd']))
                        # Get period if available, default to 0
                        period = int(torsion_row['period']) if 'period' in torsion_row and pd.notna(torsion_row['period']) else 0
                        torsion_period_list.append(period)
        
        # Append to existing inter-residue restraints
        # Bonds
        if len(bond_idx1_list) > 0:
            new_bond_indices = torch.tensor(
                np.stack([np.array(bond_idx1_list), np.array(bond_idx2_list)], axis=1),
                dtype=torch.long
            )
            new_bond_references = torch.tensor(
                np.array(bond_ref_list),
                dtype=torch.float32
            )
            new_bond_sigmas = torch.tensor(
                np.array(bond_sigma_list),
                dtype=torch.float32
            )
            
            # Append to peptide bonds (or create disulfide as separate origin)
            if 'disulfide' in self.restraints['bond']:
                # Append to existing disulfide bonds
                self.restraints['bond']['disulfide']['indices'] = torch.cat(
                    [self.restraints['bond']['disulfide']['indices'], new_bond_indices], dim=0)
                self.restraints['bond']['disulfide']['references'] = torch.cat(
                    [self.restraints['bond']['disulfide']['references'], new_bond_references], dim=0)
                self.restraints['bond']['disulfide']['sigmas'] = torch.cat(
                    [self.restraints['bond']['disulfide']['sigmas'], new_bond_sigmas], dim=0)
            else:
                # Create new disulfide bond restraints
                self.restraints['bond']['disulfide'] = {
                    'indices': new_bond_indices,
                    'references': new_bond_references,
                    'sigmas': new_bond_sigmas
                }
            
            if self.verbose > 0:
                print(f"Built {len(bond_idx1_list)} disulfide bond restraints")
                if self.verbose > 1:
                    # Show which residues are bonded
                    for idx1, idx2 in zip(bond_idx1_list, bond_idx2_list):
                        atom1 = pdb.iloc[idx1]
                        atom2 = pdb.iloc[idx2]
                        print(f"  Disulfide: {atom1['chainid']}:{atom1['resname']}{atom1['resseq']} -- "
                              f"{atom2['chainid']}:{atom2['resname']}{atom2['resseq']}")
        
        # Angles
        if len(angle_idx1_list) > 0:
            new_angle_indices = torch.tensor(
                np.stack([np.array(angle_idx1_list), np.array(angle_idx2_list), np.array(angle_idx3_list)], axis=1),
                dtype=torch.long
            )
            new_angle_references = torch.tensor(
                np.array(angle_ref_list),
                dtype=torch.float32
            )
            new_angle_sigmas = torch.tensor(
                np.array(angle_sigma_list),
                dtype=torch.float32
            )
            
            # Append to disulfide angles
            if 'disulfide' in self.restraints['angle']:
                self.restraints['angle']['disulfide']['indices'] = torch.cat(
                    [self.restraints['angle']['disulfide']['indices'], new_angle_indices], dim=0)
                self.restraints['angle']['disulfide']['references'] = torch.cat(
                    [self.restraints['angle']['disulfide']['references'], new_angle_references], dim=0)
                self.restraints['angle']['disulfide']['sigmas'] = torch.cat(
                    [self.restraints['angle']['disulfide']['sigmas'], new_angle_sigmas], dim=0)
            else:
                self.restraints['angle']['disulfide'] = {
                    'indices': new_angle_indices,
                    'references': new_angle_references,
                    'sigmas': new_angle_sigmas
                }
            
            if self.verbose > 0:
                print(f"Built {len(angle_idx1_list)} disulfide angle restraints")
        
        # Torsions
        if len(torsion_idx1_list) > 0:
            new_torsion_indices = torch.tensor(
                np.stack([np.array(torsion_idx1_list), np.array(torsion_idx2_list), 
                         np.array(torsion_idx3_list), np.array(torsion_idx4_list)], axis=1),
                dtype=torch.long
            )
            new_torsion_references = torch.tensor(
                np.array(torsion_ref_list),
                dtype=torch.float32
            )
            new_torsion_sigmas = torch.tensor(
                np.array(torsion_sigma_list),
                dtype=torch.float32
            )
            new_torsion_periods = torch.tensor(
                np.array(torsion_period_list),
                dtype=torch.long
            )
            
            # Append to disulfide torsions
            if 'disulfide' in self.restraints['torsion']:
                self.restraints['torsion']['disulfide']['indices'] = torch.cat(
                    [self.restraints['torsion']['disulfide']['indices'], new_torsion_indices], dim=0)
                self.restraints['torsion']['disulfide']['references'] = torch.cat(
                    [self.restraints['torsion']['disulfide']['references'], new_torsion_references], dim=0)
                self.restraints['torsion']['disulfide']['sigmas'] = torch.cat(
                    [self.restraints['torsion']['disulfide']['sigmas'], new_torsion_sigmas], dim=0)
                self.restraints['torsion']['disulfide']['periods'] = torch.cat(
                    [self.restraints['torsion']['disulfide']['periods'], new_torsion_periods], dim=0)
            else:
                self.restraints['torsion']['disulfide'] = {
                    'indices': new_torsion_indices,
                    'references': new_torsion_references,
                    'sigmas': new_torsion_sigmas,
                    'periods': new_torsion_periods
                }
            
            if self.verbose > 0:
                print(f"Built {len(torsion_idx1_list)} disulfide torsion restraints")
    
    def _move_to_device(self, device):
        """
        Move all restraint tensors to the specified device.
        
        Args:
            device: PyTorch device (cpu, cuda, etc.)
        """
        # Iterate through hierarchical restraints structure
        for restraint_type in ['bond', 'angle', 'torsion', 'plane']:
            if restraint_type in self.restraints:
                for origin, properties in self.restraints[restraint_type].items():
                    if isinstance(properties, dict):
                        for prop_name, tensor in properties.items():
                            if tensor is not None and isinstance(tensor, torch.Tensor):
                                self.restraints[restraint_type][origin][prop_name] = tensor.to(device)
    
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
        # Helper function to get count from hierarchical structure
        def get_count(rtype, origin):
            indices = self.restraints.get(rtype, {}).get(origin, {}).get('indices')
            return 0 if indices is None else indices.shape[0]
        
        n_bonds = get_count('bond', 'intra')
        n_angles = get_count('angle', 'intra')
        n_torsions = get_count('torsion', 'intra')
        
        n_bonds_peptide = get_count('bond', 'peptide')
        n_bonds_disulfide = get_count('bond', 'disulfide')
        n_angles_peptide = get_count('angle', 'peptide')
        n_angles_disulfide = get_count('angle', 'disulfide')
        n_torsions_disulfide = get_count('torsion', 'disulfide')
        
        n_phi = get_count('torsion', 'phi')
        n_psi = get_count('torsion', 'psi')
        n_omega = get_count('torsion', 'omega')
        
        # Count planes by atom count
        n_planes = 0
        plane_info = []
        for key in self.restraints.get('plane', {}).keys():
            count = get_count('plane', key)
            if count > 0:
                n_planes += count
                plane_info.append(f"{key}={count}")
        
        plane_str = ", ".join(plane_info) if plane_info else "none"
        
        # Get device from first available tensor
        device = 'not initialized'
        for rtype in ['bond', 'angle', 'torsion', 'plane']:
            for origin in self.restraints.get(rtype, {}).keys():
                indices = self.restraints[rtype][origin].get('indices')
                if indices is not None:
                    device = str(indices.device)
                    break
            if device != 'not initialized':
                break
        
        return (f"Restraints(\n"
                f"  cif_path='{self.cif_path}',\n"
                f"  intra-residue: bonds={n_bonds}, angles={n_angles}, torsions={n_torsions},\n"
                f"  planes: {plane_str} (total={n_planes}),\n"
                f"  peptide: bonds={n_bonds_peptide}, angles={n_angles_peptide},\n"
                f"  disulfide: bonds={n_bonds_disulfide}, angles={n_angles_disulfide}, torsions={n_torsions_disulfide},\n"
                f"  backbone: phi={n_phi}, psi={n_psi}, omega={n_omega},\n"
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
        
        # Helper function to print restraint details
        def print_restraint_section(rtype, origin, title):
            data = self.restraints.get(rtype, {}).get(origin, {})
            indices = data.get('indices')
            
            if indices is None:
                print(f"{title}: None")
                return
            
            print(f"{title}: {indices.shape[0]}")
            print(f"  Shape: {indices.shape}")
            
            refs = data.get('references')
            sigmas = data.get('sigmas')
            periods = data.get('periods')
            
            if refs is not None:
                unit = "Å" if rtype == 'bond' else "°"
                print(f"  References: min={refs.min():.3f}{unit}, "
                      f"max={refs.max():.3f}{unit}, mean={refs.mean():.3f}{unit}")
            
            if sigmas is not None:
                unit = "Å" if rtype == 'bond' else "°"
                print(f"  Sigmas: min={sigmas.min():.4f}{unit}, "
                      f"max={sigmas.max():.4f}{unit}, mean={sigmas.mean():.4f}{unit}")
            
            if periods is not None:
                unique_periods = periods.unique().tolist()
                print(f"  Periods: {unique_periods}")
        
        # Intra-residue restraints
        print("INTRA-RESIDUE RESTRAINTS:")
        print("-" * 80)
        print_restraint_section('bond', 'intra', "Bonds")
        print()
        print_restraint_section('angle', 'intra', "Angles")
        print()
        print_restraint_section('torsion', 'intra', "Torsions")
        print()
        
        # Peptide restraints
        print("PEPTIDE LINK RESTRAINTS:")
        print("-" * 80)
        print_restraint_section('bond', 'peptide', "Bonds")
        print()
        print_restraint_section('angle', 'peptide', "Angles")
        print()
        
        # Disulfide restraints
        if 'disulfide' in self.restraints.get('bond', {}):
            print("DISULFIDE LINK RESTRAINTS:")
            print("-" * 80)
            print_restraint_section('bond', 'disulfide', "Bonds")
            print()
            print_restraint_section('angle', 'disulfide', "Angles")
            print()
            print_restraint_section('torsion', 'disulfide', "Torsions")
            print()
        
        # Backbone torsion angles
        print("BACKBONE TORSION ANGLES:")
        print("-" * 80)
        
        phi_data = self.restraints.get('torsion', {}).get('phi', {})
        if phi_data.get('indices') is not None:
            print(f"Phi: {phi_data['indices'].shape[0]}")
            print(f"  Definition: C(i) - N(i+1) - CA(i+1) - C(i+1)")
            if phi_data.get('periods') is not None:
                print(f"  Period: {phi_data['periods'][0].item()}")
        else:
            print("Phi: None")
        print()
        
        psi_data = self.restraints.get('torsion', {}).get('psi', {})
        if psi_data.get('indices') is not None:
            print(f"Psi: {psi_data['indices'].shape[0]}")
            print(f"  Definition: N(i) - CA(i) - C(i) - N(i+1)")
            if psi_data.get('periods') is not None:
                print(f"  Period: {psi_data['periods'][0].item()}")
        else:
            print("Psi: None")
        print()
        
        omega_data = self.restraints.get('torsion', {}).get('omega', {})
        if omega_data.get('indices') is not None:
            print(f"Omega: {omega_data['indices'].shape[0]}")
            print(f"  Definition: CA(i) - C(i) - N(i+1) - CA(i+1)")
            if omega_data.get('references') is not None:
                refs = omega_data['references']
                print(f"  References: min={refs.min():.2f}°, max={refs.max():.2f}°, mean={refs.mean():.2f}°")
            if omega_data.get('sigmas') is not None:
                sigs = omega_data['sigmas']
                print(f"  Sigmas: min={sigs.min():.4f}°, max={sigs.max():.4f}°, mean={sigs.mean():.4f}°")
            if omega_data.get('periods') is not None:
                print(f"  Period: {omega_data['periods'][0].item()}")
        else:
            print("Omega: None")
        
        # Plane restraints
        if len(self.restraints.get('plane', {})) > 0:
            print()
            print("PLANE RESTRAINTS:")
            print("-" * 80)
            for atom_count_key in sorted(self.restraints['plane'].keys()):
                plane_data = self.restraints['plane'][atom_count_key]
                indices = plane_data.get('indices')
                if indices is not None:
                    print(f"{atom_count_key}: {indices.shape[0]} planes")
                    print(f"  Shape: {indices.shape}")
                    sigmas = plane_data.get('sigmas')
                    if sigmas is not None:
                        print(f"  Sigmas: min={sigmas.min():.4f}Å, "
                              f"max={sigmas.max():.4f}Å, mean={sigmas.mean():.4f}Å")
                    print()
        
        print("=" * 80)
    
    def _get_all_indices(self, restraint_type, keys_to_merge = None):
        """
        Helper to gather all indices of a given restraint type across all origins.
        
        Args:
            restraint_type: 'bond', 'angle', or 'torsion'
            
        Returns:
            Concatenated tensor of all indices, or None if none exist
        """
        indices_list = []
        for origin, data in self.restraints.get(restraint_type, {}).items():
            indices = data.get('indices')
            if indices is not None:
                if keys_to_merge is None:
                    indices_list.append(indices)
                elif origin in keys_to_merge:
                    indices_list.append(indices)
        
        if not indices_list:
            return None
        
        return torch.cat(indices_list, dim=0)
    
    def _get_all_property(self, restraint_type, property_name, keys_to_merge = None):
        """
        Helper to gather all values of a given property across all origins.
        
        Args:
            restraint_type: 'bond', 'angle', or 'torsion'
            property_name: 'references', 'sigmas', or 'periods'
            
        Returns:
            Concatenated tensor of all property values, or None if none exist
        """
        values_list = []
        for origin, data in self.restraints.get(restraint_type, {}).items():
            values = data.get(property_name)
            if values is not None:
                if keys_to_merge is None:
                    values_list.append(values)
                elif origin in keys_to_merge:
                    values_list.append(values)
        
        if not values_list:
            return None
        
        return torch.cat(values_list, dim=0)

    def bond_lengths(self,idx):
        """Compute current bond lengths from atomic coordinates."""

        if idx is None:
            return torch.tensor([], device=self.model.xyz().device)
        xyz = self.model.xyz()
        pos1 = xyz[idx[:, 0], :]
        pos2 = xyz[idx[:, 1], :]
        return torch.linalg.norm(pos2 - pos1, dim=-1)
    
    def nll_bonds(self):
        """
        Compute negative log-likelihood for bond length restraints.
        
        For Gaussian distribution: NLL = -log(P(x|μ,σ))
        NLL = 0.5 * ((x - μ) / σ)^2 + log(σ) + 0.5 * log(2π)
        
        This is the true NLL where exp(-NLL) = probability density.
        
        Returns:
            nll_bonds: Tensor of shape (n_bonds,) with negative log-likelihood values
        """
        if not 'all' in self.restraints['bond']:
            self.cat_dict()

        idx = self.restraints['bond']['all']['indices']
        bond_references = self.restraints['bond']['all']['references']
        sigmas = self.restraints['bond']['all']['sigmas']

        # Get current bond lengths
        bond_lengths = self.bond_lengths(idx)
        
        # Compute full NLL for Gaussian distribution
        diffs = bond_lengths - bond_references
        log_2pi = torch.log(torch.tensor(2.0 * torch.pi, device=sigmas.device))
        nll_bonds = 0.5 * (diffs / sigmas) ** 2 + torch.log(sigmas) + 0.5 * log_2pi
        
        return nll_bonds

    def angles(self, idx):
        """
        Compute current angle values for all angle restraints (intra + inter residue).
        
        Returns:
            angles: Tensor of shape (n_angles,) with current angle values in degrees
        """
        xyz = self.model.xyz()
        pos1 = xyz[idx[:, 0], :]
        pos2 = xyz[idx[:, 1], :]
        pos3 = xyz[idx[:, 2], :]
        
        # Compute vectors
        v1 = pos1 - pos2  # Vector from atom2 to atom1
        v2 = pos3 - pos2  # Vector from atom2 to atom3
        
        # Compute angle using dot product
        # cos(θ) = (v1 · v2) / (|v1| * |v2|)
        dot_product = torch.sum(v1 * v2, dim=-1)
        norm1 = torch.linalg.norm(v1, dim=-1)
        norm2 = torch.linalg.norm(v2, dim=-1)
        
        # Clamp to avoid numerical issues with arccos
        cos_angle = torch.clamp(dot_product / (norm1 * norm2), -1.0, 1.0)
        
        # Return angle in degrees
        angles_rad = torch.acos(cos_angle)
        angles_deg = torch.rad2deg(angles_rad)
        
        return angles_deg
    
    def nll_angles(self):
        """
        Compute negative log-likelihood for angle restraints.
        
        For Gaussian distribution: NLL = -log(P(x|μ,σ))
        NLL = 0.5 * ((x - μ) / σ)^2 + log(σ) + 0.5 * log(2π)
        
        This is the true NLL where exp(-NLL) = probability density.
        
        Returns:
            nll_angles: Tensor of shape (n_angles,) with negative log-likelihood values
        """
        if not 'all' in self.restraints['angle']:
            self.cat_dict()
        # Get current angles
        idx = self.restraints['angle']['all']['indices']
        references = self.restraints['angle']['all']['references'].deg2rad()
        sigmas = self.restraints['angle']['all']['sigmas'].deg2rad()

        all_deviations = self.angles(idx).deg2rad()
    
        # Compute full NLL for Gaussian distribution
        diffs = all_deviations - references
        log_2pi = torch.log(torch.tensor(2.0 * torch.pi, device=sigmas.device))
        nll_angles = 0.5 * (diffs / sigmas) ** 2 + torch.log(sigmas) + 0.5 * log_2pi
        
        return nll_angles
    
    def cat_dict(self):
        self.restraints['bond']['all'] = {
            'indices': self._get_all_indices('bond'),
            'references': self._get_all_property('bond', 'references'),
            'sigmas': self._get_all_property('bond', 'sigmas')
        }
        self.restraints['angle']['all'] = {
            'indices': self._get_all_indices('angle'),
            'references': self._get_all_property('angle', 'references'),
            'sigmas': self._get_all_property('angle', 'sigmas')
        }
        self.restraints['torsion']['all'] = {
            'indices': self._get_all_indices('torsion',['intra','disulfide']),
            'references': self._get_all_property('torsion', 'references',['intra','disulfide']),
            'sigmas': self._get_all_property('torsion', 'sigmas',['intra','disulfide']),
            'periods': self._get_all_property('torsion', 'periods',['intra','disulfide'])
        }

    def torsions(self,idx):
        """
        Compute current torsion angle values for all torsion restraints (intra + inter residue).
        
        Returns:
            torsions: Tensor of shape (n_torsions,) with current torsion values in degrees
        """
        xyz = self.model.xyz()

        pos1 = xyz[idx[:, 0], :]
        pos2 = xyz[idx[:, 1], :]
        pos3 = xyz[idx[:, 2], :]
        pos4 = xyz[idx[:, 3], :]
        
        # Compute torsion angles using vector math
        b1 = pos2 - pos1
        b2 = pos3 - pos2
        b3 = pos4 - pos3
        
        # Normalize b2 for projection
        b2_norm = torch.linalg.norm(b2, dim=-1, keepdim=True)
        b2_unit = b2 / b2_norm
        
        # Compute normals to planes
        n1 = torch.cross(b1, b2, dim=-1)
        n2 = torch.cross(b2, b3, dim=-1)
        
        # Normalize normals
        n1_unit = n1 / torch.linalg.norm(n1, dim=-1, keepdim=True)
        n2_unit = n2 / torch.linalg.norm(n2, dim=-1, keepdim=True)
        
        # Compute angle between normals
        m1 = torch.cross(n1_unit, b2_unit, dim=-1)
        
        x = torch.sum(n1_unit * n2_unit, dim=-1)
        y = torch.sum(m1 * n2_unit, dim=-1)
        
        torsions_rad = torch.atan2(y, x)
        torsions_deg = torch.rad2deg(torsions_rad)
        return torsions_deg
    
    def torsion_deviations(self, wrapped=True):
        """
        Compute deviations between calculated and expected torsion angles.
        
        Args:
            wrapped: If True (default), wrap deviations accounting for periodicity.
                    If False, return raw deviations (calculated - expected).
        
        Returns:
            deviations: Tensor of shape (n_torsions,) with deviations in degrees.
                       For wrapped=True, deviations are in range appropriate for the period.
        
        Note:
            Expected values from CIF library are discrete (typically -60°, 0°, 60°, 90°, 180°)
            while calculated values from structure are continuous. This is correct!
            Use wrapped=True for meaningful comparison and visualization.
        """
        if not 'all' in self.restraints['torsion']:
            self.cat_dict()
            
        idx = self.restraints['torsion']['all']['indices']
        expected = self.restraints['torsion']['all']['references']
        periods = self.restraints['torsion']['all']['periods']
        calculated = self.torsions(idx)
        
        if not wrapped:
            # Simple difference
            return calculated - expected
        else:
            # Apply periodicity and wrap to meaningful range
            diff_rad = (calculated - expected) * torch.pi / 180.0
            diff_periodic = periods.float() * diff_rad
            
            # Wrap to [-pi, pi]
            diff_wrapped = torch.atan2(torch.sin(diff_periodic), torch.cos(diff_periodic))
            
            # Undo periodicity scaling and convert back to degrees
            deviations_deg = (diff_wrapped / periods.float()) * 180.0 / torch.pi
            
            return deviations_deg
    
    def nll_torsions(self):
        """
        Compute negative log-likelihood for torsion angle restraints.
        
        For von Mises distribution: NLL = -log(P(θ|μ,κ))
        NLL = -κ*cos(θ-μ) + log(I₀(κ)) + log(2π)
        
        where κ = 1/σ² is the concentration parameter and I₀ is the modified
        Bessel function of the first kind.
        
        Note on periodicity:
        - Period indicates n-fold rotational symmetry (e.g., period=6 for benzene)
        - We handle this by finding the minimum angular distance considering symmetry
        - For period=n, angles differing by 360°/n are equivalent
        
        This is the true NLL where exp(-NLL) = probability density.
        
        Returns:
            nll: Tensor of shape (n_torsions,) with negative log-likelihood values
        """
        if not 'all' in self.restraints['torsion']:
            self.cat_dict()
        idx = self.restraints['torsion']['all']['indices']
        expectation = self.restraints['torsion']['all']['references']
        sigmas = self.restraints['torsion']['all']['sigmas']
        periods = self.restraints['torsion']['all']['periods']
    
        torsion_angle = self.torsions(idx)

        # Calculate angular difference
        diff = torsion_angle - expectation
        diff_rad = diff * torch.pi / 180.0
        
        # For n-fold symmetry, find minimum equivalent difference
        # Period=n means angles are equivalent modulo 360°/n
        if periods.max() > 1:
            # Create offsets for each period: [0, 360/n, 2*360/n, ..., (n-1)*360/n]
            max_period = periods.max().item()
            device = diff_rad.device
            
            # For each torsion, check all equivalent angles
            diff_wrapped_best = torch.zeros_like(diff_rad)
            
            for i in range(len(diff_rad)):
                period = periods[i].item()
                if period == 1:
                    # No symmetry, just wrap to [-pi, pi]
                    diff_wrapped_best[i] = torch.atan2(torch.sin(diff_rad[i]), torch.cos(diff_rad[i]))
                else:
                    # Check all equivalent angles due to n-fold symmetry
                    equivalent_diffs = []
                    for k in range(period):
                        offset = k * (2.0 * torch.pi / period)
                        equiv_diff = diff_rad[i] - offset
                        # Wrap to [-pi, pi]
                        equiv_diff_wrapped = torch.atan2(torch.sin(equiv_diff), torch.cos(equiv_diff))
                        equivalent_diffs.append(equiv_diff_wrapped)
                    
                    # Take the one with minimum absolute value
                    equivalent_diffs_tensor = torch.stack(equivalent_diffs)
                    min_idx = torch.argmin(torch.abs(equivalent_diffs_tensor))
                    diff_wrapped_best[i] = equivalent_diffs_tensor[min_idx]
        else:
            # All periods are 1, simple wrapping
            diff_wrapped_best = torch.atan2(torch.sin(diff_rad), torch.cos(diff_rad))

        # Compute full NLL for von Mises distribution with numerical stability
        sigmas_rad = sigmas * torch.pi / 180.0
        kappa = torch.clamp(1.0 / (sigmas_rad**2), min=1e-3, max=1e4)
        
        # Compute log(I_0(kappa)) using stable approximation
        log_i0_kappa = torch.zeros_like(kappa)
        small_kappa_mask = kappa < 50.0
        large_kappa_mask = ~small_kappa_mask
        
        if small_kappa_mask.any():
            log_i0_kappa[small_kappa_mask] = torch.log(i0(kappa[small_kappa_mask]))
        
        if large_kappa_mask.any():
            kappa_large = kappa[large_kappa_mask]
            log_i0_kappa[large_kappa_mask] = kappa_large - 0.5 * torch.log(2.0 * torch.pi * kappa_large)
        
        log_2pi = torch.log(torch.tensor(2.0 * torch.pi, device=sigmas.device))
        
        # Full von Mises NLL: -κ*cos(diff) + log(I_0(κ)) + log(2π)
        nll = -kappa * torch.cos(diff_wrapped_best) + log_i0_kappa + log_2pi
        
        return nll

    def nll_omega(self, eps_kappa=1e-3):
            """
            Compute negative log-likelihood for omega torsion angle restraints.
            
            Uses mixture model for cis/trans conformations:
            - Trans (180°): 97% probability (93% for pre-proline)
            - Cis (0°): 3% probability (7% for pre-proline)
            
            NLL = -log(P_trans * P_vonMises(θ|180°,κ) + P_cis * P_vonMises(θ|0°,κ))
            
            This is the true NLL where exp(-NLL) = probability density.

            Returns:
                nll_omega: Tensor of shape (n_omega,) with negative log-likelihood
            """
            if not 'omega' in self.restraints['torsion']:
                if self.verbose > 0:
                    print("No omega restraints found.")
                return torch.tensor([], device=self.model.xyz().device)

            
            idx = self.restraints['torsion']['omega']['indices']
            sigmas = self.restraints['torsion']['omega']['sigmas'].deg2rad()
            is_proline = self.restraints['torsion']['omega'].get('is_proline', None)
            if is_proline is None:
                is_proline = torch.zeros(len(idx), dtype=torch.bool, device=idx.device)

            # Get current omega angles
            omega_angles = self.torsions(idx).deg2rad()
            
            mu_trans = torch.pi  # 180 degrees
            mu_cis = torch.tensor(0.0, dtype=torch.float32, device=omega_angles.device)  # 0 degrees
            kappa = torch.clamp(1.0 / (sigmas ** 2), min=eps_kappa, max=1e6)
            
            # Compute angular differences (wrapped to [-pi, pi])
            diff_trans = torch.atan2(torch.sin(omega_angles - mu_trans), torch.cos(omega_angles - mu_trans))
            diff_cis   = torch.atan2(torch.sin(omega_angles - mu_cis),   torch.cos(omega_angles - mu_cis))

            # Compute log(I_0(kappa)) with numerical stability
            log_i0_kappa = torch.zeros_like(kappa)
            small_kappa_mask = kappa < 50.0
            large_kappa_mask = ~small_kappa_mask
            
            if small_kappa_mask.any():
                log_i0_kappa[small_kappa_mask] = torch.log(i0(kappa[small_kappa_mask]))
            
            if large_kappa_mask.any():
                kappa_large = kappa[large_kappa_mask]
                log_i0_kappa[large_kappa_mask] = kappa_large - 0.5 * torch.log(2.0 * torch.pi * kappa_large)
            
            log_2pi = torch.log(torch.tensor(2.0 * torch.pi, device=sigmas.device))
            
            # von Mises log-probability for each component
            # log P_vonMises = κ*cos(diff) - log(I_0(κ)) - log(2π)
            logp_trans = kappa * torch.cos(diff_trans) - log_i0_kappa - log_2pi
            logp_cis   = kappa * torch.cos(diff_cis) - log_i0_kappa - log_2pi
            
            # Mixture probabilities
            prob_cis = torch.zeros_like(logp_cis)
            prob_cis[:] = 0.03
            prob_cis[is_proline] = 0.07
            prob_trans = 1.0 - prob_cis
            
            # Mixture log-probability: log(P_trans * P_trans + P_cis * P_cis)
            log_prob_trans = torch.log(prob_trans)
            log_prob_cis = torch.log(prob_cis)
            
            log_mixture_trans = log_prob_trans + logp_trans
            log_mixture_cis = log_prob_cis + logp_cis
            
            # NLL = -log P(data)
            total_log_prob = torch.logaddexp(log_mixture_trans, log_mixture_cis)
            nll_omega = -total_log_prob
            
            return nll_omega

    def nll_planes(self):
        """
        Compute negative log-likelihood for plane restraints.
        
        For each plane:
        1. Fit the best plane to the atomic coordinates using SVD
        2. Compute perpendicular distances of each atom from the fitted plane
        3. Calculate NLL using Gaussian distribution for each atom's deviation
        
        The plane is defined by its normal vector n and a point on the plane.
        Distance from point p to plane: d = |n · (p - p0)|
        
        For Gaussian distribution: NLL = 0.5 * (d / σ)^2 + log(σ) + 0.5 * log(2π)
        
        Returns:
            nll_planes: Tensor of shape (total_plane_atoms,) with NLL for each atom
        """
        if len(self.restraints.get('plane', {})) == 0:
            if self.verbose > 1:
                print("No plane restraints found.")
            return torch.tensor([], device=self.model.xyz().device)
        
        xyz = self.model.xyz()
        all_deviations = []
        all_sigmas = []
        
        # Process each atom count group
        for atom_count_key in self.restraints['plane'].keys():
            plane_data = self.restraints['plane'][atom_count_key]
            indices = plane_data['indices']  # Shape: (n_planes, n_atoms)
            sigmas = plane_data['sigmas']    # Shape: (n_planes, n_atoms)
            
            if indices.shape[0] == 0:
                continue
            
            # Get coordinates for all planes: (n_planes, n_atoms, 3)
            plane_coords = xyz[indices]
            
            # Compute plane center (centroid) for each plane
            center = plane_coords.mean(dim=1, keepdim=True)  # (n_planes, 1, 3)
            
            # Center the coordinates
            centered = plane_coords - center  # (n_planes, n_atoms, 3)
            
            # Fit plane using SVD
            # For each plane, we want to find the normal vector n such that
            # the sum of squared distances to the plane is minimized
            # This is the eigenvector corresponding to the smallest eigenvalue
            
            # Reshape for batch SVD: (n_planes, n_atoms, 3)
            # SVD gives us U, S, V where centered = U @ S @ V^T
            # The last column of V (or V^T row) is the normal to the best-fit plane
            
            try:
                # Compute SVD for each plane
                U, S, Vt = torch.linalg.svd(centered, full_matrices=False)
                # Vt has shape (n_planes, 3, 3) for full_matrices=False
                # The last row of Vt (or last column of V) is the normal vector
                normals = Vt[:, -1, :]  # Shape: (n_planes, 3)
                
            except RuntimeError as e:
                # Handle SVD failures (e.g., for degenerate planes)
                if self.verbose > 0:
                    print(f"Warning: SVD failed for some planes in {atom_count_key}: {e}")
                # Fall back to cross product for triangular planes or skip
                continue
            
            # Compute distances from each atom to its plane
            # Distance = |n · (p - p0)| where p0 is the center
            # Since we already centered, distance = |n · centered_p|
            
            # normals: (n_planes, 3)
            # centered: (n_planes, n_atoms, 3)
            # We want dot product for each atom: (n_planes, n_atoms)
            
            # Expand normals for broadcasting: (n_planes, 1, 3)
            normals_expanded = normals.unsqueeze(1)
            
            # Compute dot products: (n_planes, n_atoms)
            distances = torch.abs(torch.sum(centered * normals_expanded, dim=2))
            
            # Collect deviations and sigmas
            all_deviations.append(distances.flatten())
            all_sigmas.append(sigmas.flatten())
        
        if len(all_deviations) == 0:
            if self.verbose > 1:
                print("No valid plane restraints to compute NLL.")
            return torch.tensor([], device=xyz.device)
        
        # Concatenate all deviations and sigmas
        deviations = torch.cat(all_deviations)
        sigmas_flat = torch.cat(all_sigmas)
        
        # Compute NLL for Gaussian distribution
        log_2pi = torch.log(torch.tensor(2.0 * torch.pi, device=sigmas_flat.device))
        nll = 0.5 * (deviations / sigmas_flat) ** 2 + torch.log(sigmas_flat) + 0.5 * log_2pi
        
        return nll

    def loss(self,weights=None):
        """
        Compute total negative log-likelihood loss from all restraints.
        
        Returns:
            total_nll: Scalar tensor with total negative log-likelihood
        """

        default_values = {
            'bond': 1.0,
            'angle': 1.0,
            'torsion': 1.0,
            'omega': 1.0,
            'plane': 1.0
        }
    
        if weights:
            default_values.update(weights)  

        nll_bonds = torch.mean(self.nll_bonds())
        if self.verbose > 2: print(f"Mean NLL Bonds wo weights: {nll_bonds.item():.4f}")
        nll_angles = torch.mean(self.nll_angles())
        if self.verbose > 2: print(f"Mean NLL Angles wo weights: {nll_angles.item():.4f}")
        nll_torsions = self.nll_torsions()
        if torch.any(torch.isnan(nll_torsions)):
            print(torch.sum(torch.isnan(nll_torsions)))
            raise ValueError("NaN values found in torsion NLL computation.")
        nll_torsions = torch.mean(nll_torsions)
        if self.verbose > 2: print(f"Mean NLL Torsions wo weights: {nll_torsions.item():.4f}")
        nll_omega = torch.mean(self.nll_omega())
        if self.verbose > 2: print(f"Mean NLL Omega wo weights: {nll_omega.item():.4f}")
        nll_planes = torch.mean(self.nll_planes())
        if self.verbose > 2: print(f"Mean NLL Planes wo weights: {nll_planes.item():.4f}")

        total_nll = (nll_bonds * default_values['bond'] +
                     nll_angles * default_values['angle'] +
                     nll_torsions * default_values['torsion'] +
                     nll_omega * default_values['omega'] +
                     nll_planes * default_values['plane']) / (
                         default_values['bond'] + 
                         default_values['angle'] + 
                         default_values['torsion'] + 
                         default_values['omega'] + 
                         default_values['plane'])

        return total_nll
