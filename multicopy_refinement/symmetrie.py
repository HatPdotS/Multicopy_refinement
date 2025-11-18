import torch
import json
import torch.nn as nn
from multicopy_refinement.debug_utils import DebugMixin
# Dictionary storing all non-crystallographic symmetry operations
# Format: {spacegroup_canonical_name: (rotation_matrices, translation_vectors)}

import os
symmetry_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'caching/files/gemmi_symmetry_operations.pt')
SYMMETRY_OPERATIONS = torch.load(symmetry_path)

# Dictionary mapping different space group names/aliases to canonical identifiers
# This allows for flexible space group name input while maintaining consistency
# Loaded from JSON file for easier maintenance

mapping_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'caching/files/spacegroup_name_mapping.json')

with open(mapping_path, 'r') as f: SPACEGROUP_NAME_MAPPING = json.load(f)

class Symmetry(DebugMixin, nn.Module):
    def __init__(self, space_group, dtype=torch.float64,device=torch.device('cpu')):
        super(Symmetry, self).__init__()
        self.device = device
        self.space_group = space_group.strip().replace(' ','')
        self.canonical_space_group = self._resolve_space_group_name(self.space_group)
        matrices, translations = self._get_ops(self.canonical_space_group)
        matrices = matrices.to(dtype).to(self.device)  # Ensure matrices are of the correct dtype
        translations = translations.to(dtype).to(self.device)  # Ensure translations are of the correct dtype
        self.register_buffer('matrices', matrices)
        self.register_buffer('translations', translations)


    def _resolve_space_group_name(self, space_group):
        """
        Resolve space group name to canonical identifier using the name mapping.
        Uses space-removed canonicalization for flexible matching.
        
        Parameters:
        -----------
        space_group : str
            Input space group name (with any common variations/aliases)
            
        Returns:
        --------
        str
            Canonical space group identifier used in SYMMETRY_OPERATIONS
            
        Raises:
        -------
        ValueError
            If space group name is not recognized
        """
        # First try direct lookup
        if space_group in SPACEGROUP_NAME_MAPPING:
            return SPACEGROUP_NAME_MAPPING[space_group]
        
        # Try with spaces removed (canonical form)
        canonical_input = space_group.replace(' ', '')
        
        # Try direct lookup of canonicalized name in SYMMETRY_OPERATIONS
        if canonical_input in SYMMETRY_OPERATIONS:
            return canonical_input
        
        # Try case-insensitive lookup with spaces removed in name mapping
        for key, value in SPACEGROUP_NAME_MAPPING.items():
            if key.replace(' ', '').upper() == canonical_input.upper():
                return value
        
        # If still not found, check SYMMETRY_OPERATIONS directly (case-insensitive)
        for key in SYMMETRY_OPERATIONS:
            if key.replace(' ', '').upper() == canonical_input.upper():
                return key
        
        available_names = list(SYMMETRY_OPERATIONS.keys())[:20]  # Show first 20
        raise ValueError(f'Space group "{space_group}" not recognized. '
                       f'Available space groups (first 20): {available_names}...')

    def _get_ops(self, canonical_space_group):
        """
        Get symmetry operations for the canonical space group name.
        
        Parameters:
        -----------
        canonical_space_group : str
            Canonical space group identifier
            
        Returns:
        --------
        tuple
            (rotation_matrices, translation_vectors) as torch tensors
            
        Raises:
        -------
        ValueError
            If canonical space group is not implemented
        """
        if canonical_space_group in SYMMETRY_OPERATIONS:
            matrices, translations = SYMMETRY_OPERATIONS[canonical_space_group]
            # Return deep copies to avoid modifying the stored tensors
            return matrices.clone(), translations.clone()
        else:
            available_groups = list(SYMMETRY_OPERATIONS.keys())
            raise ValueError(f'Space group "{canonical_space_group}" not implemented. '
                           f'Available space groups: {available_groups}')

    def apply(self, fractional_coords):
        """
        Apply symmetry operations to fractional coordinates.

        Parameters:
        -----------
        fractional_coords : torch.Tensor
            Input tensor of shape (N, 3) representing fractional coordinates
        Returns:
        --------
        torch.Tensor
            Transformed coordinates of shape (3, N, ops) where ops is the number of symmetry operations
        """
        coords = fractional_coords.reshape(3, -1).to(self.matrices.device)  # (3, N)
        coords = coords.unsqueeze(0)  # (1, 3, N)
        transformed = torch.matmul(self.matrices, coords) + self.translations.unsqueeze(2)
        # transformed: (ops, 3, N)
        return transformed.permute(1, 2, 0)  # (3, N, ops)

    def forward(self, fractional_coords):
        return self.apply(fractional_coords)

    def __repr__(self):
        return f'Symmetry(space_group={self.space_group}, canonical={self.canonical_space_group})'