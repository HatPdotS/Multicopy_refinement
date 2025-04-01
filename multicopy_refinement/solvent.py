import numpy as np
import source.math_numpy as math_np

def scale_with_solvent(fcalc,f_solvent,ref,s):
    from scipy.optimize import minimize
    def loss(x, fcalc, f_solvent, ref,s):
        return np.sum(np.abs(x[0] * np.abs(fcalc + x[1] * f_solvent * np.exp(-s*x[2])) - ref))
    x0 = [1, 1,0]
    bounds = [(0, None), (0, None),(-20, 20)]
    res = minimize(loss, x0 = x0, args=(fcalc, f_solvent, ref,s),bounds=bounds)
    f_composite = res.x[0] * (fcalc + res.x[1] * f_solvent * np.exp(-s*res.x[2]))
    return f_composite

def get_mask(pdb,cell,max_res=0.8,radius=1.1):
    nsteps = np.astype(np.floor(cell[:3] / max_res * 5), int)
    x = np.linspace(0, 1, nsteps[0])
    y = np.linspace(0, 1, nsteps[1])
    z = np.linspace(0, 1, nsteps[2])
    x, y, z = np.meshgrid(x, y, z, indexing='ij')
    array_shape = x.shape
    x = x.reshape((*x.shape, 1))
    y = y.reshape((*y.shape, 1))
    z = z.reshape((*z.shape, 1))
    fractional_matrix = math_np.get_fractional_matrix(cell)
    inv_frac_matrix = math_np.get_inv_fractional_matrix(cell)
    xyz = np.concatenate((x, y, z), axis=3).reshape(-1, 3)
    xyz_real_size = math_np.fractional_to_cartesian(xyz, cell)
    xyz_real_size = xyz_real_size.reshape((*array_shape, 3))
    mask = np.zeros(array_shape, dtype=bool)
    from source.numba_functions import add_coord_to_mask
    for coord in pdb[['x','y','z']].values:
        mask = add_coord_to_mask(mask,xyz_real_size,coord,inv_frac_matrix,fractional_matrix,radius=radius)
    return mask

def smooth_solvent(solvent):
    from scipy.ndimage import gaussian_filter
    return gaussian_filter(solvent, sigma=2)

def calculate_solvent(pdb,cell,max_res=0.8,rho_solvent=0.333):
    import gemmi
    mask = get_mask(pdb,cell,max_res)
    solvent = np.zeros_like(mask,dtype=float)
    solvent[~mask] = rho_solvent
    solvent = smooth_solvent(solvent)   
    grid = gemmi.FloatGrid(solvent.shape[0], solvent.shape[1], solvent.shape[2])
    unit_cell = gemmi.UnitCell(*cell)
    grid.spacegroup = gemmi.SpaceGroup('P1')
    grid.set_unit_cell(unit_cell)
    grid_array = np.array(grid, copy=False)  # Get a view into gemmi's grid
    grid_array[...] = solvent  # Copy NumPy array data into gemmi's grid
    sf_grid = gemmi.transform_map_to_f_phi(grid)
    return sf_grid
