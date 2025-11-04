"""
Generate space group name mapping JSON file from gemmi cache keys.

This script creates a JSON file mapping common space group name variations to the 
canonical names used in the gemmi cache. This allows flexible space group name 
input while maintaining consistency with gemmi's naming conventions.

The mapping includes:
- Canonical names (e.g., 'P21')
- Lowercase variants (e.g., 'p21')
- Extended notation (e.g., 'P1211', 'P 1 21 1')
- Subscript notation (e.g., 'P4_1' -> 'P41')
- Bar notation (e.g., 'P1bar' -> 'P-1')
- Historical/alternative settings (e.g., 'P121' -> 'P2')

Output: caching/files/spacegroup_name_mapping.json

Usage:
    python generate_spacegroup_name_mapping.py
"""

import torch
import json
import re
import os


def generate_spacegroup_name_mapping():
    """Generate and save space group name mapping to JSON file."""
    
    # Get the cache file path relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cache_file = os.path.join(script_dir, '../files/gemmi_symmetry_operations.pt')
    output_file = os.path.join(script_dir, '../files/spacegroup_name_mapping.json')
    
    # Load cached operations
    print(f"Loading symmetry operations from {cache_file}...")
    cached_ops = torch.load(cache_file)
    canonical_names = sorted(cached_ops.keys())
    print(f"Found {len(canonical_names)} canonical space groups")
    
    # Build the mapping
    mapping = {}
    
    # For each canonical name, add various aliases
    for canonical in canonical_names:
        # Add the canonical name itself
        mapping[canonical] = canonical
        
        # Add lowercase version
        mapping[canonical.lower()] = canonical
        
        # Add version with spaces
        spaced = canonical.replace('-', ' -')
        if spaced != canonical:
            mapping[spaced] = canonical
        
        # Add version with underscores for subscripts
        if '_' not in canonical and any(c.isdigit() for c in canonical):
            # Match patterns like P41, P42, P43, I41, P31, P32, P61-P65
            matches = list(re.finditer(r'([A-Z])(\d{2})', canonical))
            for match in matches:
                letter = match.group(1)
                digits = match.group(2)
                if digits in ['41', '42', '43', '31', '32', '61', '62', '63', '64', '65']:
                    alias = canonical.replace(letter + digits, letter + digits[0] + '_' + digits[1])
                    mapping[alias] = canonical
    
    # Add common crystallographic aliases (monoclinic unique axis b, cell choice 1)
    monoclinic_aliases = {
        'P121': 'P2', 'p121': 'P2', 'P 1 2 1': 'P2', 'P12': 'P2',
        'P1211': 'P21', 'p1211': 'P21', 'P 1 21 1': 'P21',
        'C121': 'C2', 'c121': 'C2', 'C 1 2 1': 'C2', 'C12': 'C2',
        'P1m1': 'Pm', 'p1m1': 'Pm', 'P 1 m 1': 'Pm', 'P1m': 'Pm',
        'P1c1': 'Pc', 'p1c1': 'Pc', 'P 1 c 1': 'Pc', 'P1c': 'Pc',
        'C1m1': 'Cm', 'c1m1': 'Cm', 'C 1 m 1': 'Cm', 'C1m': 'Cm',
        'C1c1': 'Cc', 'c1c1': 'Cc', 'C 1 c 1': 'Cc', 'C1c': 'Cc',
        'P12/m1': 'P2/m', 'p12/m1': 'P2/m', 'P 1 2/m 1': 'P2/m', 'P12/m': 'P2/m',
        'P121/m1': 'P21/m', 'p121/m1': 'P21/m', 'P 1 21/m 1': 'P21/m', 'P121/m': 'P21/m',
        'C12/m1': 'C2/m', 'c12/m1': 'C2/m', 'C 1 2/m 1': 'C2/m', 'C12/m': 'C2/m',
        'P12/c1': 'P2/c', 'p12/c1': 'P2/c', 'P 1 2/c 1': 'P2/c', 'P12/c': 'P2/c',
        'P121/c1': 'P21/c', 'p121/c1': 'P21/c', 'P 1 21/c 1': 'P21/c', 'P121/c': 'P21/c',
        'C12/c1': 'C2/c', 'c12/c1': 'C2/c', 'C 1 2/c 1': 'C2/c', 'C12/c': 'C2/c',
    }
    
    for alias, target in monoclinic_aliases.items():
        if target in canonical_names:
            mapping[alias] = target
    
    # Add 'bar' notation for inversion symbols
    for name in canonical_names:
        if '-' in name:
            bar_version = name.replace('-', '') + 'bar'
            mapping[bar_version] = name
    
    # Add orthorhombic aliases
    if 'Aea2' in canonical_names:
        mapping['Aba2'] = 'Aea2'
        mapping['aba2'] = 'Aea2'
        mapping['A b a 2'] = 'Aea2'
    
    # Verify all mappings point to valid canonical names
    invalid = [k for k, v in mapping.items() if v not in canonical_names]
    if invalid:
        print(f'WARNING: Invalid mappings found: {invalid}')
        return False
    
    print(f'Generated {len(mapping)} valid name mappings')
    print(f'Covering {len(set(mapping.values()))} unique space groups')
    
    # Save to JSON file
    with open(output_file, 'w') as f:
        json.dump(mapping, f, indent=2, sort_keys=True)
    
    print(f'\n✓ Saved name mappings to {output_file}')
    return True


if __name__ == '__main__':
    success = generate_spacegroup_name_mapping()
    if not success:
        exit(1)
