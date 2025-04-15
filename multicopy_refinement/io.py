import pandas as pd
import numpy as np
import multicopy_refinement.math_numpy as math_np
import gemmi
import torch

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