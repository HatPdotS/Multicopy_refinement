import pandas as pd
import numpy as np
from multicopy_refinement import math_numpy as math_np
from multicopy_refinement import math_torch
import gemmi
import torch
import pdb_tools
from torch import tensor

def read_crystfel_hkl(path):
    dtypes = {'h': int, 'k': int, 'l': int, 'I': float, 'phase': float, 'sigma': float, 'nmeas': int}
    ref = pd.read_csv(path, sep=r'\s+', names=['h','k','l','I','phase','sigma','nmeas'],skiprows=3,on_bad_lines='skip',dtype=dtypes).dropna()
    return ref

def read_mtz(path):
    import reciprocalspaceship as rs
    mtz = rs.read_mtz(path).reset_index()
    mtz.rename(columns={'H': 'h', 'K': 'k', 'L': 'l'}, inplace=True)
    dtypes = {'h': int, 'k': int, 'l': int}
    for key in mtz.keys():
        if key in dtypes:
            mtz[key] = mtz[key].astype(dtypes[key])
        else:
            mtz[key] = mtz[key].astype(np.float64)
    return mtz


def superpose_pdbs(pdb1, npdb2, selection=(None,None,None,None)):
    """
    Superpose two pdb files using the gemmi library.
    The selection parameter is a tuple of (chain1, chain2, res1, res2) to select specific chains and residues.
    If None, all chains and residues are used.
    """
    pdb1 = pdb_tools.load_pdb_as_pd(pdb1)
    pdb2 = pdb_tools.load_pdb_as_pd(npdb2)
    sel = np.ones(pdb1.shape[0], dtype=bool)
    if selection[0] is not None:
        sel &= pdb1['chain'].isin(selection[0])
    if selection[1] is not None:
       sel &= pdb1['resseq'].isin(selection[1])
    if selection[2] is not None:
        sel &= pdb1['resname'].isin(selection[2])
    if selection[3] is not None:
        sel &= pdb1['name'].isin(selection[3])
    from multicopy_refinement import math_numpy
    alignment,rmsd = math_numpy.superpose_vectors_robust(pdb1.loc[sel,['x','y','z']].values, pdb2.loc[sel,['x','y','z']].values)
    print(rmsd)
    pdb2.loc[:,['x','y','z']] = math_numpy.apply_transformation(pdb2.loc[:,['x','y','z']].values, alignment)
    pdb_tools.write_file(pdb2, npdb2.replace('.pdb','_superposed.pdb'))


def bin_hkl(hkl,cell,bins=20):
    hkl['res'] = get_resolution(hkl[['h','k','l']].values,cell)
    sorted_res = np.sort(hkl.res)[::-1]
    sorted_res[0] = sorted_res[0] + 10
    sorted_res[-1] = 0
    reflection_to_target = sorted_res.shape[0] // bins
    bins = sorted_res[::reflection_to_target]
    bins[-1] = 0
    hkl['bin'] = np.digitize(hkl.res,bins)
    return hkl

def get_resolution(hkl, unit_cell):
    if isinstance(unit_cell, list):
        unit_cell = np.array(unit_cell)
    elif isinstance(unit_cell, tuple):
        unit_cell = np.array(unit_cell)
    elif isinstance(unit_cell, np.ndarray):
        unit_cell = unit_cell
    elif isinstance(unit_cell, gemmi.UnitCell):
        unit_cell = np.array([unit_cell.a, unit_cell.b, unit_cell.c, unit_cell.alpha, unit_cell.beta, unit_cell.gamma])
    elif isinstance(unit_cell, torch.Tensor):
        unit_cell = unit_cell.detach().cpu().numpy()
    else:
        raise ValueError("unit_cell must be a list, tuple, pytorch tensor, numpy array or gemmi.UnitCell")
    s = math_np.get_scattering_vectors(hkl, unit_cell)
    return 1 / np.sum(s**2, axis=1)**0.5


def find_best_data_keys(hkl_df):
    """
    Find the best available intensity or structure factor data in the MTZ dataframe
    Returns a tuple of (data_key, sigma_key, is_intensity)
    """
    # Comprehensive list of possible keys, in order of preference
    intensity_keys = ['I', 'IOBS', 'I-obs', 'IMEAN', 'I(+)', 'IPLUS', 'I-pk', 'I_pk', 
                     'IHLI', 'I-obs-filtered', 'I_full', 'IOBS_full', 'IP', 'IO']
    
    amplitude_keys = ['F', 'FOBS', 'F-obs', 'F-obs-filtered', 'FMEAN', 'F(+)', 
                     'FPLUS', 'FP', 'F-pk', 'F_pk', 'FO', 'FODD','F-model']
    
    # Check for intensity keys first (preferred)
    for key in intensity_keys:
        if key in hkl_df:
            sigma_key = f"SIG{key}" if f"SIG{key}" in hkl_df else None
            print(f"Using intensity data: {key}")
            return key, sigma_key, True
    
    # Then check for amplitude keys
    for key in amplitude_keys:
        if key in hkl_df:
            sigma_key = f"SIG{key}" if f"SIG{key}" in hkl_df else None
            if sigma_key is None and 'SIGF' in hkl_df:
                sigma_key = 'SIGF'
            print(f"Using amplitude data: {key}")
            return key, sigma_key, False
    
    # If we get here, no suitable data was found
    print("Available columns:", list(hkl_df.columns))
    raise ValueError("No suitable intensity or structure factor data found in MTZ file")

def get_f(hkl_df):
    key, sigma_key, is_intensity = find_best_data_keys(hkl_df)
    F_I = tensor(hkl_df[key].to_numpy())
    if sigma_key is not None:
        sigma = tensor(hkl_df[sigma_key].to_numpy())
    else:
        sigma = None
    if is_intensity:
        F,F_sigma = math_torch.french_wilson_conversion(F_I, sigma)
        return F, F_sigma
    else:
        F = tensor(hkl_df[key].to_numpy())
        if sigma_key is not None:
            F_sigma = tensor(hkl_df[sigma_key].to_numpy())
        else:
            F_sigma = None
        return F, F_sigma
        