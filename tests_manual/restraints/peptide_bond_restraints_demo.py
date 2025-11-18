"""
Demo: How to Add Inter-Residue (Peptide Bond) Restraints

Currently MISSING from the restraints.py implementation!

For proteins, we need to add restraints between consecutive residues
in each chain for the peptide bond linkages.

Standard peptide bond geometry:
- C(i) - N(i+1): 1.329 Å (±0.014 Å)
- CA(i) - C(i) - N(i+1): 116.2° (±2.0°)
- C(i) - N(i+1) - CA(i+1): 121.7° (±1.8°)
- CA(i) - C(i) - N(i+1) - CA(i+1) (omega): 180° (±5°) for trans peptide
"""

import numpy as np
import torch
import pandas as pd

def add_peptide_bond_restraints(model, bond_indices, bond_refs, bond_sigmas):
    """
    Add peptide bond restraints between consecutive residues in each chain.
    
    For each peptide bond between residue i and i+1:
    - Add C(i) - N(i+1) bond restraint
    
    Standard values:
    - Distance: 1.329 Å
    - Sigma: 0.014 Å
    """
    pdb = model.pdb
    
    new_bonds = []
    
    # Process each chain separately
    for chain_id in pdb['chainid'].unique():
        chain = pdb[pdb['chainid'] == chain_id].copy()
        
        # Get unique residues, sorted by resseq
        residues = chain.groupby('resseq').first().sort_index()
        resseq_list = residues.index.tolist()
        
        # Iterate through consecutive residues
        for i in range(len(resseq_list) - 1):
            resseq_i = resseq_list[i]
            resseq_j = resseq_list[i + 1]
            
            # Get residues
            res_i = chain[chain['resseq'] == resseq_i]
            res_j = chain[chain['resseq'] == resseq_j]
            
            # Skip if either is HETATM
            if res_i['ATOM'].iloc[0] == 'HETATM' or res_j['ATOM'].iloc[0] == 'HETATM':
                continue
            
            # Find C atom in residue i
            c_atoms = res_i[res_i['name'] == 'C']
            if len(c_atoms) == 0:
                continue
            c_idx = c_atoms['index'].iloc[0]
            
            # Find N atom in residue i+1  
            n_atoms = res_j[res_j['name'] == 'N']
            if len(n_atoms) == 0:
                continue
            n_idx = n_atoms['index'].iloc[0]
            
            # Add peptide bond restraint
            new_bonds.append({
                'idx1': c_idx,
                'idx2': n_idx,
                'reference': 1.329,  # Å
                'sigma': 0.014       # Å
            })
    
    if len(new_bonds) == 0:
        return bond_indices, bond_refs, bond_sigmas
    
    # Convert to arrays
    new_bond_indices = np.array([[b['idx1'], b['idx2']] for b in new_bonds])
    new_bond_refs = np.array([b['reference'] for b in new_bonds])
    new_bond_sigmas = np.array([b['sigma'] for b in new_bonds])
    
    # Concatenate with existing bonds
    bond_indices = torch.cat([
        bond_indices,
        torch.tensor(new_bond_indices, dtype=torch.long)
    ])
    bond_refs = torch.cat([
        bond_refs,
        torch.tensor(new_bond_refs, dtype=torch.float)
    ])
    bond_sigmas = torch.cat([
        bond_sigmas,
        torch.tensor(new_bond_sigmas, dtype=torch.float)
    ])
    
    print(f"Added {len(new_bonds)} peptide bond restraints")
    
    return bond_indices, bond_refs, bond_sigmas


def add_peptide_backbone_angles(model, angle_indices, angle_refs, angle_sigmas):
    """
    Add backbone angle restraints for peptide bonds.
    
    For each peptide bond between residue i and i+1:
    - Add CA(i) - C(i) - N(i+1) angle restraint (116.2°)
    - Add C(i) - N(i+1) - CA(i+1) angle restraint (121.7°)
    """
    pdb = model.pdb
    new_angles = []
    
    for chain_id in pdb['chainid'].unique():
        chain = pdb[pdb['chainid'] == chain_id].copy()
        residues = chain.groupby('resseq').first().sort_index()
        resseq_list = residues.index.tolist()
        
        for i in range(len(resseq_list) - 1):
            resseq_i = resseq_list[i]
            resseq_j = resseq_list[i + 1]
            
            res_i = chain[chain['resseq'] == resseq_i]
            res_j = chain[chain['resseq'] == resseq_j]
            
            if res_i['ATOM'].iloc[0] == 'HETATM' or res_j['ATOM'].iloc[0] == 'HETATM':
                continue
            
            # Get atoms for CA(i) - C(i) - N(i+1)
            ca_i = res_i[res_i['name'] == 'CA']
            c_i = res_i[res_i['name'] == 'C']
            n_j = res_j[res_j['name'] == 'N']
            
            if len(ca_i) > 0 and len(c_i) > 0 and len(n_j) > 0:
                new_angles.append({
                    'idx1': ca_i['index'].iloc[0],
                    'idx2': c_i['index'].iloc[0],
                    'idx3': n_j['index'].iloc[0],
                    'reference': 116.2,  # degrees
                    'sigma': 2.0
                })
            
            # Get atoms for C(i) - N(i+1) - CA(i+1)
            ca_j = res_j[res_j['name'] == 'CA']
            
            if len(c_i) > 0 and len(n_j) > 0 and len(ca_j) > 0:
                new_angles.append({
                    'idx1': c_i['index'].iloc[0],
                    'idx2': n_j['index'].iloc[0],
                    'idx3': ca_j['index'].iloc[0],
                    'reference': 121.7,  # degrees
                    'sigma': 1.8
                })
    
    if len(new_angles) == 0:
        return angle_indices, angle_refs, angle_sigmas
    
    # Convert and concatenate
    new_angle_indices = np.array([[a['idx1'], a['idx2'], a['idx3']] for a in new_angles])
    new_angle_refs = np.array([a['reference'] for a in new_angles])
    new_angle_sigmas = np.array([a['sigma'] for a in new_angles])
    
    angle_indices = torch.cat([
        angle_indices,
        torch.tensor(new_angle_indices, dtype=torch.long)
    ])
    angle_refs = torch.cat([
        angle_refs,
        torch.tensor(new_angle_refs, dtype=torch.float)
    ])
    angle_sigmas = torch.cat([
        angle_sigmas,
        torch.tensor(new_angle_sigmas, dtype=torch.float)
    ])
    
    print(f"Added {len(new_angles)} backbone angle restraints")
    
    return angle_indices, angle_refs, angle_sigmas


def add_peptide_omega_torsions(model, torsion_indices, torsion_refs, torsion_sigmas):
    """
    Add omega (ω) torsion restraints for peptide bonds.
    
    Omega is the CA(i) - C(i) - N(i+1) - CA(i+1) dihedral
    Standard trans-peptide: 180° (±5°)
    """
    pdb = model.pdb
    new_torsions = []
    
    for chain_id in pdb['chainid'].unique():
        chain = pdb[pdb['chainid'] == chain_id].copy()
        residues = chain.groupby('resseq').first().sort_index()
        resseq_list = residues.index.tolist()
        
        for i in range(len(resseq_list) - 1):
            resseq_i = resseq_list[i]
            resseq_j = resseq_list[i + 1]
            
            res_i = chain[chain['resseq'] == resseq_i]
            res_j = chain[chain['resseq'] == resseq_j]
            
            if res_i['ATOM'].iloc[0] == 'HETATM' or res_j['ATOM'].iloc[0] == 'HETATM':
                continue
            
            # Get atoms for CA(i) - C(i) - N(i+1) - CA(i+1)
            ca_i = res_i[res_i['name'] == 'CA']
            c_i = res_i[res_i['name'] == 'C']
            n_j = res_j[res_j['name'] == 'N']
            ca_j = res_j[res_j['name'] == 'CA']
            
            if len(ca_i) > 0 and len(c_i) > 0 and len(n_j) > 0 and len(ca_j) > 0:
                new_torsions.append({
                    'idx1': ca_i['index'].iloc[0],
                    'idx2': c_i['index'].iloc[0],
                    'idx3': n_j['index'].iloc[0],
                    'idx4': ca_j['index'].iloc[0],
                    'reference': 180.0,  # degrees (trans-peptide)
                    'sigma': 5.0
                })
    
    if len(new_torsions) == 0:
        return torsion_indices, torsion_refs, torsion_sigmas
    
    # Convert and concatenate
    new_torsion_indices = np.array([[t['idx1'], t['idx2'], t['idx3'], t['idx4']] for t in new_torsions])
    new_torsion_refs = np.array([t['reference'] for t in new_torsions])
    new_torsion_sigmas = np.array([t['sigma'] for t in new_torsions])
    
    torsion_indices = torch.cat([
        torsion_indices,
        torch.tensor(new_torsion_indices, dtype=torch.long)
    ])
    torsion_refs = torch.cat([
        torsion_refs,
        torch.tensor(new_torsion_refs, dtype=torch.float)
    ])
    torsion_sigmas = torch.cat([
        torsion_sigmas,
        torch.tensor(new_torsion_sigmas, dtype=torch.float)
    ])
    
    print(f"Added {len(new_torsions)} omega torsion restraints")
    
    return torsion_indices, torsion_refs, torsion_sigmas


if __name__ == "__main__":
    print(__doc__)
    print("\nTo use these functions, add them to the Restraints class:")
    print("1. Call add_peptide_bond_restraints() after _build_bond_restraints()")
    print("2. Call add_peptide_backbone_angles() after _build_angle_restraints()")
    print("3. Call add_peptide_omega_torsions() after _build_torsion_restraints()")
