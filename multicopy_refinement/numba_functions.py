import numba as nb
import numpy as np  

@nb.njit(cache=True)
def unravel_index(idx, shape):
    n_dims = len(shape)
    indices = np.empty(n_dims, dtype=np.int64)
    for i in range(n_dims - 1, -1, -1):
        indices[i] = idx % shape[i]
        idx //= shape[i]
    return indices

@nb.njit(cache=True)
def get_starting_positions(xyz: np.array,coords: np.array,frac_matrix: np.array,inv_frac_matrix: np.array):
    diff = xyz - coords
    diff = smallest_diff(diff.copy(),inv_frac_matrix,frac_matrix)
    idx = np.argmin(diff)
    idxs = unravel_index(idx,diff.shape)
    return idxs

@nb.njit(cache=True)
def get_starting_postion_fast(coords: np.array,inv_frac_matrix: np.array,shape_xyz: np.array):
    fractional_coords = np.dot(inv_frac_matrix,coords)
    closest_idx = np.round(fractional_coords * shape_xyz)
    closest_idx_int = closest_idx.astype(np.int64)
    for i in range(3):
        closest_idx_int[i] = check_shape(closest_idx_int[i], shape_xyz[i])
    return closest_idx_int

@nb.njit(cache=True)
def check_shape(coord: np.int64,dim: np.int64):
    if coord < 0:
        coord += dim
    elif coord >= dim:
        coord -= dim
    return coord

@nb.njit(cache=True)
def smallest_diff(diff: np.array,inv_frac_matrix: np.array,frac_matrix: np.array):
    diff_shape = diff.shape
    diff = diff.reshape(-1,3)
    diff_frac = np.dot(inv_frac_matrix,diff.T)
    translation = np.round(diff_frac)
    diff -= np.dot(frac_matrix,translation).T
    return np.sum(diff ** 2,axis=-1).reshape(diff_shape[:-1])

@nb.njit(cache=True)
def add_coord_to_mask(grid: np.array,xyz: np.array,coord: np.array,inv_frac_matrix: np.array,frac_matrix: np.array,radius: int = 1):
    i_start, j_start, k_start = get_starting_postion_fast(coord.copy(), inv_frac_matrix.copy(), np.array(xyz.shape[:3]))
    radius_squared = radius ** 2
    coords_to_check = [(i_start, j_start, k_start)]
    coords_checked = set()
    while coords_to_check:
        i, j, k = coords_to_check.pop()
        x_y_z = xyz[i, j, k]
        diff_vector = x_y_z - coord
        diff = smallest_diff(diff_vector.reshape(-1,3), inv_frac_matrix,frac_matrix)
        if diff <= radius_squared:
            grid[i, j, k] = True
            coords_checked.add((i, j, k))
            for di, dj, dk in ((1, 0, 0), (0, 1, 0), (0, 0, 1), (-1, 0, 0), (0, -1, 0), (0, 0, -1)):
                i_new, j_new, k_new = i + di, j + dj, k + dk
                i_new = check_shape(i_new, grid.shape[0])
                j_new = check_shape(j_new, grid.shape[1])
                k_new = check_shape(k_new, grid.shape[2])
                if 0 <= i_new < grid.shape[0] and 0 <= j_new < grid.shape[1] and 0 <= k_new < grid.shape[2]:
                    if (i_new, j_new, k_new) not in coords_checked:
                        coords_to_check.append((i_new, j_new, k_new))
    return grid

def get_recgrid_bin_edge(x1,s1):
    half_point = x1.shape[0]//2
    x_temp = np.concatenate((x1[half_point:],x1[:half_point]))
    print(x_temp)
    idx = np.digitize(s1,x_temp) - half_point
    return idx

def align_and_interpolate_recgrid_values(s_vector, recgrid, s1,s2,s3):
    x1 = get_recgrid_bin_edge(s1.flatten(),s_vector[:,0])
    x2 = get_recgrid_bin_edge(s2.flatten(),s_vector[:,1])
    x3 = get_recgrid_bin_edge(s3.flatten(),s_vector[:,2])
    print(s1[x1[0]],s2[x2[0]],s3[x3[0]],s_vector[0])
