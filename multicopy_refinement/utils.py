import torch
import gemmi
import numpy as np


def save_map(array, cell, filename):
    """
    Save a 3D map to a CCP4 file.
    Parameters:
    - array: 3D numpy array or torch tensor representing the map.
    - cell: Unit cell parameters as a list, tuple, numpy array, or gemmi
      UnitCell object.
    - filename: Output CCP4 file name.
    """

    if isinstance(array, torch.Tensor):
        np_map = array.detach().cpu().numpy().astype(np.float32)
    else:
        np_map = array.astype(np.float32)
    if isinstance(cell, gemmi.UnitCell):
        cell = cell.parameters
    elif isinstance(cell, np.ndarray):
        cell = cell.tolist()
    elif isinstance(cell, list):
        cell = cell
    elif isinstance(cell, tuple):
        cell = list(cell)
    elif isinstance(cell, torch.Tensor):
        cell = cell.tolist()
    map_ccp = gemmi.Ccp4Map()
    map_ccp.grid = gemmi.FloatGrid(np_map, gemmi.UnitCell(*cell), gemmi.SpaceGroup('P1'))
    map_ccp.setup(0.0)
    map_ccp.update_ccp4_header()
    map_ccp.write_ccp4_map(filename)
    print(f"Map saved successfully")

    return True

import torch
import torch.nn as nn

class TensorDict(nn.Module):
    """
    A dictionary-like container for PyTorch tensors that:
    - Supports standard dict syntax
    - Automatically moves with the module
    - Registers tensors as buffers so they are included in state_dict
    """
    def __init__(self):
        super().__init__()
        self._keys = []

    def __setitem__(self, key: str, tensor: torch.Tensor):
        name = f"_buf_{key}"
        if not hasattr(self, name):
            # Register as buffer
            self.register_buffer(name, tensor)
            self._keys.append(key)
        else:
            # Update existing buffer in-place
            getattr(self, name).data.copy_(tensor)

    def __getitem__(self, key: str) -> torch.Tensor:
        name = f"_buf_{key}"
        if not hasattr(self, name):
            raise KeyError(key)
        return getattr(self, name)

    def __contains__(self, key: str):
        return key in self._keys

    def keys(self):
        return self._keys.copy()

    def values(self):
        return [getattr(self, f"_buf_{k}") for k in self._keys]

    def items(self):
        return [(k, getattr(self, f"_buf_{k}")) for k in self._keys]

    def __len__(self):
        return len(self._keys)

    def __repr__(self):
        return f"TensorDict({{"+", ".join(f'{k}: {getattr(self, f"_buf_{k}")}' for k in self._keys)+"}})"
    


class TensorMasks(TensorDict):
    """
    A specialized TensorDict for managing boolean masks.
    Ensures all tensors are of boolean dtype.
    """

    def __init__(self):
        super().__init__()
        self._cache = TensorDict()
        self.updated = True

    def __setitem__(self, key: str, tensor: torch.Tensor):
        if tensor.dtype != torch.bool:
            raise ValueError("All masks must be of boolean dtype.")
        super().__setitem__(key, tensor)
        self.updated = True
    
    def forward(self):
        """
        Return the current masks.
        """
        
        if self.updated:
            combined_mask = self.get_combined_mask()
            self._cache['combined'] = combined_mask
            self.updated = False
        return self._cache['combined']

    def get_combined_mask(self) -> torch.Tensor:
        """
        Combine all masks using logical AND.
        Caches the result for efficiency.
        Returns:
            torch.Tensor: Combined boolean mask.
        """

        combined_mask = torch.ones_like(self[self._keys[0]], dtype=torch.bool)
        try:
            for key in self._keys:
                combined_mask &= self[key]
        except Exception as e:
            for key in self._keys:
                print(f"'{key}': {self[key].shape}, {self[key].dtype}, {self[key].device}")
            print(f"Error combining masks: {e}")
        return combined_mask.to(torch.bool)
