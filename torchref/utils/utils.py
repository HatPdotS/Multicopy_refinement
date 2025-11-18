import torch
import gemmi
import numpy as np
from typing import Optional, List, Dict, Any, Union
from pathlib import Path
import pandas as pd


class ModuleReference:
    """
    A wrapper class to hold references to PyTorch modules without registering them.
    
    When you assign a nn.Module to an attribute of another nn.Module, PyTorch
    automatically registers it as a submodule, which adds its parameters to the
    parent's parameter tree. This wrapper prevents that automatic registration.
    
    This is useful when you want to:
    - Hold references to modules without including their parameters
    - Avoid circular dependencies in the module tree
    - Reference external modules that should be managed separately
    
    Example:
        >>> model = MyModel()
        >>> scaler = Scaler()
        >>> scaler._model = ModuleReference(model)  # Won't register as submodule
        >>> # Access the module via .module property
        >>> output = scaler._model.module(input_data)
    """
    
    def __init__(self, module):
        """
        Wrap a module to prevent automatic registration.
        
        Args:
            module: The PyTorch module to wrap
        """
        # Store in __dict__ directly to avoid any attribute interception
        object.__setattr__(self, '_wrapped_module', module)
    
    @property
    def module(self):
        """Access the wrapped module."""
        return object.__getattribute__(self, '_wrapped_module')
    
    def __getattr__(self, name):
        """Forward attribute access to the wrapped module."""
        return getattr(self.module, name)
    
    def __call__(self, *args, **kwargs):
        """Forward calls to the wrapped module."""
        return self.module(*args, **kwargs)
    
    def __repr__(self):
        return f"ModuleReference({self.module.__class__.__name__})"


class CIFReader:

    """
    A dictionary-like reader for CIF/mmCIF files.
    
    Loops are stored as pandas DataFrames.
    Other data is stored in a hierarchical dictionary structure.
    """
    
    def __init__(self, filepath: Optional[str] = None):
        """
        Initialize CIF reader.
        
        Args:
            filepath: Optional path to CIF file to load immediately
        """
        self.data = {}
        self.filepath = None
        
        if filepath:
            self.load(filepath)
    
    def load(self, filepath: str):
        """
        Load and parse a CIF file.
        
        Args:
            filepath: Path to CIF file
        """
        self.filepath = Path(filepath)
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        self._parse(content)
    
    def _parse(self, content: str):
        """
        Parse CIF file content.
        
        Args:
            content: String content of CIF file
        """
        lines = content.split('\n')
        i = 0
        
        while i < len(lines):
            line = lines[i].strip()
            
            # Skip empty lines and comments
            if not line or line.startswith('#'):
                i += 1
                continue
            
            # Check for data block (usually just one in mmCIF)
            if line.startswith('data_'):
                i += 1
                continue
            
            # Check for loop
            if line.startswith('loop_'):
                i = self._parse_loop(lines, i + 1)
                continue
            
            # Parse single key-value pairs
            if line.startswith('_'):
                i = self._parse_keyvalue(lines, i)
                continue
            
            i += 1
    
    def _parse_loop(self, lines: List[str], start_idx: int) -> int:
        """
        Parse a loop structure into a pandas DataFrame.
        
        Args:
            lines: All lines of the file
            start_idx: Starting line index (after 'loop_')
        
        Returns:
            Index of the next line to process
        """
        # Collect column names
        columns = []
        i = start_idx
        
        while i < len(lines):
            line = lines[i].strip()
            if not line or line.startswith('#'):
                i += 1
                continue
            
            if line.startswith('_'):
                columns.append(line)
                i += 1
            else:
                break
        
        if not columns:
            return i
        
        # Collect data rows
        data_rows = []
        current_row = []
        in_multiline = False
        multiline_value = []
        
        while i < len(lines):
            line = lines[i]
            stripped = line.strip()
            
            # Check if we've reached the end of the loop
            if not in_multiline and (not stripped or stripped.startswith('_') or 
                                    stripped.startswith('loop_') or stripped.startswith('data_')):
                if current_row:
                    data_rows.append(current_row)
                break
            
            # Handle multiline strings (starting with semicolon)
            if not in_multiline and line.startswith(';'):
                in_multiline = True
                multiline_value = [line[1:]]  # Remove leading semicolon
                i += 1
                continue
            
            if in_multiline:
                if line.startswith(';'):
                    # End of multiline string
                    in_multiline = False
                    current_row.append('\n'.join(multiline_value))
                    multiline_value = []
                    
                    # Check if row is complete
                    if len(current_row) == len(columns):
                        data_rows.append(current_row)
                        current_row = []
                else:
                    multiline_value.append(line)
                i += 1
                continue
            
            # Parse regular data line
            if stripped and not stripped.startswith('#'):
                tokens = self._tokenize_line(stripped)
                for token in tokens:
                    current_row.append(token)
                    
                    # Check if row is complete
                    if len(current_row) == len(columns):
                        data_rows.append(current_row)
                        current_row = []
            
            i += 1
        
        # Create DataFrame
        if data_rows:
            df = pd.DataFrame(data_rows, columns=columns)
            
            # Store in hierarchical dictionary based on category
            # Extract category from first column name (e.g., _atom_site.id -> atom_site)
            if columns:
                category = self._extract_category(columns[0])
                if category:
                    self.data[category] = df
        
        return i
    
    def _parse_keyvalue(self, lines: List[str], start_idx: int) -> int:
        """
        Parse a single key-value pair.
        
        Args:
            lines: All lines of the file
            start_idx: Starting line index
        
        Returns:
            Index of the next line to process
        """
        line = lines[start_idx].strip()
        
        # Handle multiline values
        if start_idx + 1 < len(lines) and lines[start_idx + 1].startswith(';'):
            key = line
            value_lines = []
            i = start_idx + 2
            
            while i < len(lines):
                if lines[i].startswith(';'):
                    break
                value_lines.append(lines[i])
                i += 1
            
            value = '\n'.join(value_lines)
            self._store_keyvalue(key, value)
            return i + 1
        
        # Handle single line key-value
        parts = line.split(None, 1)
        if len(parts) == 2:
            key, value = parts
            # Remove quotes if present
            if value.startswith("'") and value.endswith("'"):
                value = value[1:-1]
            elif value.startswith('"') and value.endswith('"'):
                value = value[1:-1]
            
            self._store_keyvalue(key, value)
        
        return start_idx + 1
    
    def _tokenize_line(self, line: str) -> List[str]:
        """
        Tokenize a data line, handling quoted strings.
        
        Args:
            line: Line to tokenize
        
        Returns:
            List of tokens
        """
        tokens = []
        current_token = []
        in_quotes = False
        quote_char = None
        
        i = 0
        while i < len(line):
            char = line[i]
            
            # Handle quotes
            if char in ('"', "'") and not in_quotes:
                in_quotes = True
                quote_char = char
                i += 1
                continue
            
            if char == quote_char and in_quotes:
                in_quotes = False
                quote_char = None
                if current_token:
                    tokens.append(''.join(current_token))
                    current_token = []
                i += 1
                continue
            
            # Handle whitespace outside quotes
            if char.isspace() and not in_quotes:
                if current_token:
                    tokens.append(''.join(current_token))
                    current_token = []
                i += 1
                continue
            
            current_token.append(char)
            i += 1
        
        if current_token:
            tokens.append(''.join(current_token))
        
        return tokens
    
    def _extract_category(self, key: str) -> str:
        """
        Extract category from a CIF key (e.g., '_atom_site.id' -> 'atom_site').
        
        Args:
            key: CIF key
        
        Returns:
            Category name
        """
        if key.startswith('_'):
            key = key[1:]
        
        if '.' in key:
            return key.split('.')[0]
        
        return key
    
    def _store_keyvalue(self, key: str, value: str):
        """
        Store a key-value pair in the hierarchical dictionary.
        
        Args:
            key: CIF key (e.g., '_entry.id')
            value: Value to store
        """
        # Extract category and attribute
        if key.startswith('_'):
            key = key[1:]
        
        if '.' in key:
            category, attribute = key.split('.', 1)
            
            if category not in self.data:
                self.data[category] = {}
            
            if isinstance(self.data[category], dict):
                self.data[category][attribute] = value
        else:
            self.data[key] = value
    
    def write(self, filepath: str):
        """
        Write the CIF data back to a file.
        
        Args:
            filepath: Output file path
        """
        with open(filepath, 'w') as f:
            f.write('data_structure\n')
            f.write('#\n')
            
            # Write single key-value pairs first
            for category, content in sorted(self.data.items()):
                if isinstance(content, dict):
                    for key, value in sorted(content.items()):
                        # Handle multiline values
                        if '\n' in str(value):
                            f.write(f'_{category}.{key}\n')
                            f.write(';\n')
                            f.write(str(value))
                            f.write('\n;\n')
                        else:
                            # Quote values with spaces
                            if ' ' in str(value):
                                f.write(f"_{category}.{key} '{value}'\n")
                            else:
                                f.write(f'_{category}.{key} {value}\n')
                    f.write('#\n')
            
            # Write loops (DataFrames)
            for category, content in sorted(self.data.items()):
                if isinstance(content, pd.DataFrame):
                    f.write('loop_\n')
                    
                    # Write column names
                    for col in content.columns:
                        f.write(f'{col}\n')
                    
                    # Write data rows
                    for _, row in content.iterrows():
                        row_values = []
                        for val in row:
                            val_str = str(val)
                            # Quote values with spaces or special characters
                            if ' ' in val_str or any(c in val_str for c in ['"', "'"]):
                                row_values.append(f"'{val_str}'")
                            else:
                                row_values.append(val_str)
                        f.write(' '.join(row_values) + '\n')
                    
                    f.write('#\n')
    
    # Dictionary-like interface
    def __getitem__(self, key: str) -> Union[pd.DataFrame, Dict, Any]:
        """Get item by key."""
        return self.data[key]
    
    def __setitem__(self, key: str, value: Union[pd.DataFrame, Dict, Any]):
        """Set item by key."""
        self.data[key] = value
    
    def __contains__(self, key: str) -> bool:
        """Check if key exists."""
        return key in self.data
    
    def __len__(self) -> int:
        """Return number of top-level categories."""
        return len(self.data)
    
    def keys(self):
        """Return dictionary keys."""
        return self.data.keys()
    
    def values(self):
        """Return dictionary values."""
        return self.data.values()
    
    def items(self):
        """Return dictionary items."""
        return self.data.items()
    
    def get(self, key: str, default=None):
        """Get item with default value."""
        return self.data.get(key, default)
    
    def __repr__(self) -> str:
        """String representation."""
        categories = list(self.keys())
        loops = [k for k, v in self.items() if isinstance(v, pd.DataFrame)]
        dicts = [k for k, v in self.items() if isinstance(v, dict)]
        
        return (f"CIFReader(categories={len(categories)}, "
                f"loops={len(loops)}, "
                f"key-value_groups={len(dicts)})")
    
    def summary(self):
        """Print a summary of the CIF contents."""
        print(f"CIF File: {self.filepath}")
        print(f"Total categories: {len(self.data)}")
        print("\nLoops (DataFrames):")
        for key, value in sorted(self.items()):
            if isinstance(value, pd.DataFrame):
                print(f"  {key}: {len(value)} rows × {len(value.columns)} columns")
        
        print("\nKey-Value Groups (Dictionaries):")
        for key, value in sorted(self.items()):
            if isinstance(value, dict):
                print(f"  {key}: {len(value)} items")

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


def sanitize_pdb_dataframe(pdb: pd.DataFrame, verbose: int = 0) -> pd.DataFrame:
    """
    Sanitize a PDB DataFrame to ensure unique atom identifiers.
    
    This function fixes common issues in PDB/CIF files:
    1. HETATM records (especially waters) with duplicate resseq values (e.g., all 0)
    2. Residue names longer than 3 characters (truncates to 3)
    3. Ensures unique (chainid, resseq, name, altloc) combinations
    
    Args:
        pdb: DataFrame with PDB data (must have columns: ATOM, chainid, resseq, name, altloc, resname, serial)
        verbose: Verbosity level (0=silent, 1=info, 2=debug)
        
    Returns:
        Sanitized DataFrame with unique atom identifiers
        
    Example:
        >>> from torchref.model import Model
        >>> from torchref.utils import sanitize_pdb_dataframe
        >>> model = Model()
        >>> model.load_cif('structure.cif')
        >>> model.pdb = sanitize_pdb_dataframe(model.pdb, verbose=1)
    """
    pdb = pdb.copy()
    
    if verbose > 0:
        print("Sanitizing PDB DataFrame...")
        print(f"  Initial atoms: {len(pdb)}")
    
    # 1. Standardize residue names to max 3 characters
    long_resnames = pdb['resname'].str.len() > 3
    if long_resnames.any():
        n_long = long_resnames.sum()
        if verbose > 0:
            unique_long = pdb.loc[long_resnames, 'resname'].unique()
            print(f"  Truncating {n_long} atoms with resname > 3 chars: {unique_long[:5]}")
        pdb.loc[long_resnames, 'resname'] = pdb.loc[long_resnames, 'resname'].str[:3]
    
    # 2. Fix duplicate atom identifiers by reassigning resseq
    # Check for duplicates
    dup_mask = pdb.duplicated(subset=['chainid', 'resseq', 'name', 'altloc'], keep=False)
    
    if dup_mask.any():
        n_dup = dup_mask.sum()
        if verbose > 0:
            print(f"  Found {n_dup} atoms with duplicate identifiers")
        
        # Group by (chainid, resname, ATOM) to handle each group separately
        # This ensures we only renumber within the same molecule type and chain
        for (chainid, resname, atom_type), group in pdb.groupby(['chainid', 'resname', 'ATOM']):
            group_indices = group.index
            
            # Check if this group has duplicates
            group_dup_mask = group.duplicated(subset=['chainid', 'resseq', 'name', 'altloc'], keep=False)
            
            if group_dup_mask.any():
                # Find the maximum resseq in this chain to start numbering from there
                chain_data = pdb[pdb['chainid'] == chainid]
                max_resseq = chain_data['resseq'].max()
                
                # Start numbering from max_resseq + 1
                new_resseq_start = max_resseq + 1 if pd.notna(max_resseq) and max_resseq > 0 else 1
                
                # Assign new sequential resseq values to all atoms in this group
                # Group by (serial) to keep atoms of the same residue together
                unique_serials = group['serial'].unique()
                residue_counter = new_resseq_start
                
                for serial in unique_serials:
                    serial_mask = pdb['serial'] == serial
                    pdb.loc[serial_mask, 'resseq'] = residue_counter
                    residue_counter += 1
                
                if verbose > 1:
                    n_fixed = len(unique_serials)
                    print(f"    Fixed {n_fixed} {resname} residues in chain {chainid} (resseq {new_resseq_start}-{residue_counter-1})")
        
        # Verify duplicates are fixed
        final_dup_mask = pdb.duplicated(subset=['chainid', 'resseq', 'name', 'altloc'], keep=False)
        if final_dup_mask.any():
            remaining_dups = final_dup_mask.sum()
            if verbose > 0:
                print(f"  WARNING: Still have {remaining_dups} duplicate identifiers after sanitization")
                dups = pdb[final_dup_mask].sort_values(['chainid', 'resseq', 'name'])
                print(dups[['ATOM', 'serial', 'name', 'resname', 'chainid', 'resseq', 'altloc']].head(10))
        else:
            if verbose > 0:
                print(f"  ✓ All duplicate identifiers resolved")
    else:
        if verbose > 0:
            print(f"  ✓ No duplicate atom identifiers found")
    
    if verbose > 0:
        print(f"  Final atoms: {len(pdb)}")
    
    return pdb
