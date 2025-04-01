import torch
import numpy as np
import multicopy_refinement.math_numpy as math_np
from multicopy_refinement.numba_functions import add_coord_to_mask



def squeeze(F_obs, Fcalc_model, model, hkl, cell,max_res=0.8, exclusion_radius=1.2,n_iter=5):
    recgrid, xyz_real_grid = get_grids(cell,max_res=max_res)
    mask = build_mask(model, xyz_real_grid, cell, exclusion_radius)
    # write_numpy_to_ccp4(mask.astype(float),'mask.ccp4',cell)
    diff = calc_difference(F_obs, Fcalc_model)
    for _ in range(n_iter):
        rec_space = put_hkl_on_grid(recgrid,diff,hkl)
        real_space = np.fft.fftn(rec_space)
        real_space[mask] = 0
        real_space[real_space < 0] = 0
        # real_space = smooth_solvent(real_space)
        rec_space_updated = np.fft.ifftn(real_space)
        f_solvent = extract_hkl(rec_space_updated,hkl)
        diff = calc_difference_with_solvent(F_obs, Fcalc_model, f_solvent)
    return f_solvent

def squeeze_torch(F_obs, Fcalc_model, model, hkl, cell,max_res=0.8, exclusion_radius=1.2,n_iter=5):
    recgrid, xyz_real_grid = get_grids(cell,max_res=max_res)
    mask = build_mask_torch(model, xyz_real_grid, cell, exclusion_radius)
    mask = torch.tensor(mask)
    # write_numpy_to_ccp4(mask.astype(float),'mask.ccp4',cell)
    diff = calc_difference_torch(F_obs, Fcalc_model)
    for _ in range(n_iter):
        rec_space = put_hkl_on_grid_torch(recgrid,diff,hkl)
        real_space = torch.fft.fftn(rec_space)
        real_space[mask] = 0
        real_space[torch.abs(real_space) < 0] = 0
        # real_space = smooth_solvent(real_space)
        rec_space_updated = torch.fft.ifftn(real_space)
        f_solvent = extract_hkl(rec_space_updated,hkl)
        diff = calc_difference_with_solvent_torch(F_obs, Fcalc_model, f_solvent)
    return f_solvent

def smooth_solvent(solvent):
    from scipy.ndimage import gaussian_filter
    return gaussian_filter(solvent, sigma=1)

def calc_difference(f_obs, f_calc):
    f_obs_diff = f_obs - np.abs(f_calc)
    phases = np.angle(f_calc)
    diff = f_obs_diff * np.exp(1j * phases)
    return diff

def calc_difference_torch(f_obs, f_calc):
    f_obs_diff = f_obs - torch.abs(f_calc)
    phases = torch.angle(f_calc)
    diff = f_obs_diff * torch.exp(1j * phases)
    return diff

def calc_difference_with_solvent(f_obs, f_calc, f_solvent):
    f_obs_diff = f_obs - np.abs(f_calc)
    f_comb = f_calc + f_solvent
    phases = np.angle(f_comb)
    diff = f_obs_diff * np.exp(1j * phases)
    return diff

def calc_difference_with_solvent_torch(f_obs, f_calc, f_solvent):
    f_obs_diff = f_obs - torch.abs(f_calc)
    f_comb = f_calc + f_solvent
    phases = torch.angle(f_comb)
    diff = f_obs_diff * torch.exp(1j * phases)
    return diff

def extract_hkl(rec_space,hkl):
    f = rec_space[hkl[:,0],hkl[:,1],hkl[:,2]]
    return f

def build_mask(model_coords, grid, cell, radius):
    inv_fractional_matrix = math_np.get_inv_fractional_matrix(cell)
    fractional_matrix = math_np.get_fractional_matrix(cell)
    mask = np.zeros(grid.shape[:3], dtype=bool)
    for coord in model_coords:
        mask = add_coord_to_mask(mask, grid, coord, inv_fractional_matrix, fractional_matrix, radius)
    return mask

def build_mask_torch(model_coords, grid, cell, radius):
    inv_fractional_matrix = math_np.get_inv_fractional_matrix(cell)
    fractional_matrix = math_np.get_fractional_matrix(cell)
    mask = np.zeros(grid.shape[:3], dtype=bool)
    for coord in model_coords:
        coord = coord.detach().numpy()
        mask = add_coord_to_mask(mask, grid, coord, inv_fractional_matrix, fractional_matrix, radius)
    return mask

def put_hkl_on_grid(real_space_grid,f,hkl):
    rec_space = np.zeros(real_space_grid.shape[:3],dtype=np.complex128)
    rec_space[hkl[:,0],hkl[:,1],hkl[:,2]] = f
    return rec_space

def put_hkl_on_grid_torch(real_space_grid,diff,hkl):
    rec_space = torch.zeros(real_space_grid.shape[:3],dtype=torch.complex128)
    f = diff
    rec_space[hkl[:,0],hkl[:,1],hkl[:,2]] = f
    return rec_space

def get_grids(cell,max_res=0.8):
    nsteps = np.astype(np.floor(cell[:3] / max_res * 3), int)
    x = np.linspace(0, 1, nsteps[0])
    y = np.linspace(0, 1, nsteps[1])
    z = np.linspace(0, 1, nsteps[2])
    x, y, z = np.meshgrid(x, y, z, indexing='ij')
    array_shape = x.shape
    x = x.reshape((*x.shape, 1))
    y = y.reshape((*y.shape, 1))
    z = z.reshape((*z.shape, 1))
    xyz = np.concatenate((x, y, z), axis=3).reshape(-1, 3)
    xyz_real_grid = math_np.fractional_to_cartesian(xyz, cell)
    xyz_real_grid = xyz_real_grid.reshape((*array_shape, 3))
    recgrid = np.zeros(array_shape, dtype=float)
    return recgrid, xyz_real_grid