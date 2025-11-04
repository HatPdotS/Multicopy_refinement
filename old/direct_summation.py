import numpy as np
import pandas as pd
import pdb_tools
from numpy import fft
from scipy.fft import fftn, fftshift
from multicopy_refinement import get_scattering_factor as gsf
import torch
import multicopy_refinement.restraints_helper as restraints
import multicopy_refinement.math_numpy as math_np
import multicopy_refinement.math_torch as math_torch  
import multicopy_refinement.symmetrie as sym
from multicopy_refinement.squeeze import squeeze_torch,squeeze
from multicopy_refinement import symmetrie_np as sym_np

def direct_summation(df,hkl,cell,space_group='P1'):
    df_iso = df.loc[~df.anisou_flag]
    df_aniso = df.loc[df.anisou_flag]
    fcalc = np.zeros(hkl.shape[0],dtype=complex)
    if df_iso.shape[0] > 0:
        fcalc += direct_summation_iso(df_iso,hkl,cell,space_group)
    if df_aniso.shape[0] > 0:
        fcalc_aniso = direct_summation_aniso(df_aniso,hkl,cell)
        fcalc += fcalc_aniso
    return fcalc

def direct_summation_iso(df,hkl,cell,space_group='P1'):
    s = math_np.get_s(hkl, cell)
    scattering_factors = gsf.get_scattering_factors(df,s)
    fractional_coords = math_np.convert_coords_to_fractional(df, cell)
    hkl = hkl.astype(np.float32)
    fractional_coords = sym_np.apply_space_group(fractional_coords.T,space_group).astype(np.float32)
    fractional_shape = fractional_coords.shape
    fractional_coords = fractional_coords.reshape(3,-1)
    # fractional_coords = sym_np.apply_space_group(fractional_coords,space_group)
    dot_product = np.dot(hkl, fractional_coords).reshape(hkl.shape[0],fractional_shape[1],-1)
    occ = df.occupancy.values.astype(np.float32).reshape(1,-1)
    B_factors = df.tempfactor.values.astype(np.float32).reshape(1,-1)
    s = s.reshape(-1,1).astype(np.float32)
    B = -B_factors * (s ** 2) / 4
    exp_B = np.exp(B)
    # return np.sum(scattering_factors.values * exp_B * occ *  np.cos(2 * np.pi * dot_product), axis=1)  
    pidot = 2 * np.pi * dot_product
    return np.sum(scattering_factors * exp_B * occ *  np.sum(1j * np.sin(pidot)+np.cos(pidot),axis=-1), axis=1)

def direct_summation_aniso(df,hkl,cell,space_group='P1'):
    s = math_np.get_s(hkl, cell)
    scattering_factors = gsf.get_scattering_factors(df,s)
    fractional_coords = math_np.convert_coords_to_fractional(df, cell)
    hkl = hkl.astype(np.float32)
    fractional_coords = sym_np.apply_space_group(fractional_coords.T,space_group).astype(np.float32)
    fractional_shape = fractional_coords.shape
    fractional_coords = fractional_coords.reshape(3,-1)
    # fractional_coords = sym_np.apply_space_group(fractional_coords,space_group)
    dot_product = np.dot(hkl, fractional_coords).reshape(hkl.shape[0],fractional_shape[1],-1)
    occ = df.occupancy.values.astype(np.float32).reshape(1,-1)
    s_vector = math_np.get_scattering_vectors(hkl, cell)
    U = df[["u11", "u22", "u33", "u12", "u13", "u23"]].values.astype(float)
    U_matrix = np.array([
            [U[:, 0],U[:,3], U[:,4]],
            [U[:,3], U[:,1], U[:,5]],
            [U[:,4], U[:,5], U[:,2]]
        ])
    U_dot_s = np.einsum('ijk,jl->ikl', U_matrix, s_vector.T)  # Shape (3, M, N)
    StUS = np.einsum('il,ikl->kl', s_vector.T, U_dot_s)  # Shape (M, N)
    StUS = StUS.T
    B = -2 * (np.pi**2) *StUS 
    exp_B = np.exp(B)
    # return np.sum(scattering_factors.values * exp_B * occ *  np.cos(2 * np.pi * dot_product), axis=1)  
    pidot = 2 * np.pi * dot_product
    return np.sum(scattering_factors * exp_B * occ *  np.sum(1j * np.sin(pidot)+np.cos(pidot),axis=-1), axis=1)

def shift_fractional_coords(fractional_coords):
    fractional_coords_to_shift = fractional_coords < 0
    fractional_coords[fractional_coords_to_shift] += 1
    fractional_coords_to_shift = fractional_coords > 1
    fractional_coords[fractional_coords_to_shift] -= 1
    return fractional_coords

def convert_tensor_dict_to_structure(all_the_data,pdb,cell):
    real_space_coords = all_the_data['real_space_coords'].detach().numpy()
    pdb.loc[:,['x','y','z']] = real_space_coords
    if all_the_data['present_iso']:
        occ = all_the_data['occ_iso'].detach().numpy()
        tempfactor = all_the_data['tempfactor'].detach().numpy()
        pdb.loc[~pdb.anisou_flag,'occupancy'] = occ.flatten()
        pdb.loc[~pdb.anisou_flag,'tempfactor'] = tempfactor.flatten()
    if all_the_data['present_aniso']:
        occ = all_the_data['occ_aniso'].detach().numpy()
        U = all_the_data['U'].detach().numpy()
        pdb.loc[pdb.anisou_flag,'occupancy'] = occ.flatten()
        pdb.loc[pdb.anisou_flag,["u11", "u22", "u33", "u12", "u13", "u23"]] = U
    coords = pdb[['x','y','z']].values
    coords = coords + all_the_data['xyz_offset'].detach().numpy()
    coords = math_np.rotate_coords_numpy(coords,*all_the_data['rotation'].detach().numpy())
    pdb.cell = cell
    pdb.spacegroup = all_the_data['spacegroup']
    return pdb

def prep_data_for_torch_refinement(df,mtz,mtz_key_to_use,hkl,cell,rfree_flags=None,rfree_fraction=0.1):
    all_the_data = get_data_for_fcalc(df,hkl,cell,mtz.spacegroup.short_name())
    if rfree_flags is not None:
        all_the_data['rfree_flags'] = torch.tensor(rfree_flags)
    else:
        all_the_data['rfree_flags'] = torch.tensor(np.random.random(hkl.shape[0]) < rfree_fraction,dtype=bool)
    all_the_data['outlier_flags'] = torch.tensor(np.zeros(hkl.shape[0],dtype=bool),dtype=bool)
    all_the_data['data'] = torch.tensor(mtz[mtz_key_to_use].values.astype(float),requires_grad=False)
    print('Spacegroup:',all_the_data['spacegroup'])
    return all_the_data

def get_data_for_fcalc(df,hkl,cell,spacegroup=None):
    all_the_data = dict()
    s_vector = math_np.get_scattering_vectors(hkl, cell)
    s = np.sum(s_vector ** 2,axis=1) ** 0.5
    df_iso = df.loc[~df.anisou_flag]
    df_aniso = df.loc[df.anisou_flag]
    if spacegroup is None:
        spacegroup = str(df.spacegroup)
    all_the_data['spacegroup'] = str(spacegroup)
    all_the_data['aniso_mask'] = torch.tensor(df.anisou_flag.values,dtype=bool)
    all_the_data['real_space_coords'] = torch.tensor(df[['x','y','z']].values.astype(float),requires_grad=True)
    all_the_data['cell']  = cell
    all_the_data['xyz_offset']  = torch.tensor([0.,0.,0.],requires_grad=True)
    all_the_data['rotation'] = torch.tensor([0.,0.],requires_grad=True)
    if df_iso.shape[0] > 0:
        all_the_data['occ_iso'] = torch.tensor(df_iso.occupancy.values.astype(float).reshape(1,-1),requires_grad=True)
        all_the_data['tempfactor'] = torch.tensor(df_iso.tempfactor.values.astype(float).reshape(1,-1),requires_grad=True)
        all_the_data['present_iso'] = True
    else:
        all_the_data['tempfactor'] = None
        all_the_data['occ_iso'] = None
        all_the_data['present_iso'] = False
    if df_aniso.shape[0] > 0:
        all_the_data['occ_aniso'] = torch.tensor(df_aniso.occupancy.values.astype(float).reshape(1,-1),requires_grad=True)
        all_the_data['U'] = torch.tensor(df_aniso[["u11", "u22", "u33", "u12", "u13", "u23"]].values.astype(float),requires_grad=True)
        all_the_data['present_aniso'] = True
    else:
        all_the_data['occ_aniso'] = None
        all_the_data['U'] = None
        all_the_data['present_aniso'] = False
    df_structure_factor_fit = pd.read_feather('/das/work/p17/p17490/Peter/manual_refinement/fitted_structure_factors/fitted_structure_factors.feather')
    for atomtype in df.element.unique():
        key = 'scattering_factor_' + atomtype
        all_the_data[key] = torch.tensor(df_structure_factor_fit[atomtype].values.astype(float),requires_grad=True)
    all_the_data['atom_types'] = df.element.values
    all_the_data['hkl'] = torch.from_numpy(hkl)
    all_the_data['s'] = torch.from_numpy(s)
    all_the_data['s_vector'] = torch.from_numpy(s_vector)
    return all_the_data

def get_fcalc_torch(df,hkl,cell):
    all_the_data = get_data_for_fcalc(df,hkl,cell)
    f_calc = torch_fcalc(all_the_data)
    return f_calc

def torch_refinement(df,hkl,cell,mtz,mtz_key,cif,restraints_weight=1,
                     excluded_parameters=['occ_iso','occ_aniso'],
                     additional_parameters=[],max_iter=50,learning_rate=0.001,rfree_fraction=0.1,rfree_flags=None,rfree_gap_target=5,noise_amplitude=0):
    if df.spacegroup != None:
        if mtz.spacegroup != df.spacegroup:
            print('spacegroup mismatch using mtz spacegroup')
    cif = restraints.read_cif(cif)
    res = restraints.build_restraints(cif,df)
    all_the_data = prep_data_for_torch_refinement(df,mtz,mtz_key,hkl,cell,rfree_flags,rfree_fraction)
    all_the_data['scale'] = torch.tensor([1.,0.,0.,0.],requires_grad=True)
    # all_the_data_rfree = prep_data_for_torch_refinement(df,hkl,cell)
    standard_includes_iso = ['occ_iso',
                         'real_space_coords','tempfactor','aniso_mask']
    standard_includes_aniso = ['occ_aniso',
                         'real_space_coords','U','aniso_mask']
    includes = ['scale']
    if all_the_data['present_iso']:
        includes += standard_includes_iso
    if all_the_data['present_aniso']:
        includes += standard_includes_aniso
    includes += additional_parameters
    includes = [x for x in includes if x not in excluded_parameters]
    includes = list(set(includes))
    additional_labels = [key for key in all_the_data.keys() if not key in includes]
    included_tensors = [all_the_data[key] for key in includes]
    additional_tensors = [all_the_data[key] for key in additional_labels]
    optimizer = torch.optim.Adam(included_tensors, lr=learning_rate)
    f_solvent = squeeze(all_the_data['data'].detach().clone().numpy()
                              ,torch_fcalc(all_the_data).detach().clone().numpy()
                              ,all_the_data['real_space_coords'].detach().clone().numpy()
                              , all_the_data['hkl'].detach().clone().numpy()
                              ,all_the_data['cell'],max_res=0.8)
    f_solvent = torch.tensor(f_solvent,dtype=torch.complex128)
    for i in range(max_iter):
        optimizer.zero_grad()
        loss, rwork, rfree, f_calc = loss_function(included_tensors,includes,additional_tensors,additional_labels,res,restraints_weight,f_solvent)
        if i % 5 == 0:
            f_solvent = squeeze(all_the_data['data'].detach().clone().numpy()
                            ,f_calc.detach().clone().numpy()
                            ,all_the_data['real_space_coords'].detach().clone().numpy()
                            , all_the_data['hkl'].detach().clone().numpy()
                            ,all_the_data['cell'],max_res=0.8)
            f_solvent = torch.tensor(f_solvent,dtype=torch.complex128)
        loss.backward()
        rfree_gap = (rfree-rwork) * 100
        for tensor_id,tensor in zip(includes,included_tensors):
            if tensor_id == 'scale':
                continue
            tensor_grad = tensor.grad
            if tensor_grad is None:
                continue
            noise_to_add = torch.rand_like(tensor) * noise_amplitude * rwork
            tensor.grad +=  noise_to_add
        print(f'step {i}:',loss.item(),rfree_gap.item(),rwork.item(),rfree.item(),'n_outliers:',torch.sum(all_the_data['outlier_flags']).item())
        if rfree_gap > rfree_gap_target:
            break
        optimizer.step()
        
    fcalc = torch_fcalc(all_the_data)
    fcalc += f_solvent
    scale = all_the_data['scale']
    s = all_the_data['s']
    scale_function = scale[0] + scale[1] * s + scale[2] * s**2 + scale[3] * s**3
    f = scale_function * fcalc

    pdb = convert_tensor_dict_to_structure(all_the_data,df,cell)
    return f, pdb, all_the_data['outlier_flags']

def loss_function(refinement_vals,refinement_label,additonal_parameters,additional_labels,res,res_weight,f_solvent):
    all_the_data = dict(zip(additional_labels,additonal_parameters))
    additional_parameters = dict(zip(refinement_label,refinement_vals))
    all_the_data.update(additional_parameters)
    f_calc = torch_fcalc(all_the_data)
    f_comp = f_calc + f_solvent
    update_outlier_mask(all_the_data,f_comp)

    ref = all_the_data['data']
    scale = all_the_data['scale']
    s = all_the_data['s']
    scale_function = scale[0] + scale[1] * s + scale[2] * s**2 + scale[3] * s**3

    r_work_mask = ~all_the_data['rfree_flags'] & ~all_the_data['outlier_flags']
    r_free_mask = all_the_data['rfree_flags'] & ~all_the_data['outlier_flags']
    ref_work = ref[r_work_mask]
    ref_free = ref[r_free_mask]
    f_comp = torch.abs(scale_function * f_comp)
    f_calc_work = f_comp[r_work_mask]
    f_calc_free = f_comp[r_free_mask]

    real_space_coords = all_the_data['real_space_coords']
    res_result = restraints.calculate_restraints_all(real_space_coords,res)

    rwork = math_torch.get_rfactor_torch(ref_work,f_calc_work)
    rfree = math_torch.get_rfactor_torch(ref_free,f_calc_free)
    return torch.sum(torch.abs(f_calc_work-ref_work)) + res_weight * res_result, rwork, rfree, f_calc

def get_scattering_factor_from_approximation(s,approx):
        y = torch.zeros_like(s,requires_grad=True) + approx[-1]
        for i in range(5):
            y += approx[2*i]*torch.exp(-((s)/approx[2*i+1])**2) 
        if torch.isnan(y).any():
            print(approx)

            raise ValueError('nan in scattering factor')
        return y

def update_outlier_mask(all_the_data,fcalc,z=2):
    fobs = all_the_data['data']
    mask = math_torch.calc_outliers(fobs,fcalc,z)
    all_the_data['outlier_flags'] = mask

def torch_fcalc(all_the_data):
    cell = all_the_data['cell']
    s = all_the_data['s']
    structure_factor_keys = [key for key in all_the_data.keys() if key.startswith('scattering_factor')]
    approximated_scattering_factors = {}
    for key in structure_factor_keys:
        keynew = key.split('_')[-1]
        approximated_scattering_factors[keynew] = get_scattering_factor_from_approximation(s,all_the_data[key])
    scattering_factors = torch.vstack([approximated_scattering_factors[key] for key in all_the_data['atom_types']])
    f_calc = torch.zeros(all_the_data['hkl'].shape[0],dtype=torch.complex128)
    real_space_coords = all_the_data['real_space_coords']
    real_space_coords = real_space_coords + all_the_data['xyz_offset']
    real_space_coords = math_torch.rotate_coords_torch(real_space_coords,*all_the_data['rotation'])
    fractional_coords = math_torch.cartesian_to_fractional_torch(all_the_data['real_space_coords'],cell).T
    space_group = all_the_data['spacegroup']
    if all_the_data['present_iso']:
        scattering_factors_iso = scattering_factors[~all_the_data['aniso_mask']]
        iso_fractional_coords = fractional_coords[:,~all_the_data['aniso_mask']]
        f_calc += iso_structure_factor_torched(all_the_data['hkl'],all_the_data['s'],
                                               iso_fractional_coords,all_the_data['occ_iso'],
                                               scattering_factors_iso.T,all_the_data['tempfactor'],space_group)
    if all_the_data['present_aniso']:
        scattering_factors_aniso = scattering_factors[all_the_data['aniso_mask']]
        aniso_fractional_coords = fractional_coords[:,all_the_data['aniso_mask']]
        f_calc_aniso = aniso_structure_factor_torched(all_the_data['hkl'],all_the_data['s_vector'],
                                                      aniso_fractional_coords,all_the_data['occ_aniso'],
                                                      scattering_factors_aniso.T,all_the_data['U'],space_group)
        f_calc += f_calc_aniso
    return f_calc

def aniso_structure_factor_torched(hkl,s_vector,fractional_coords,occ,scattering_factors,U,space_group):
    fractional_coords = sym.apply_space_group(fractional_coords,space_group)
    dot_product = torch.einsum('ik,kjs->ijs',hkl.to(torch.float64), fractional_coords)
    U_row1 = torch.stack([U[:,0],U[:,3], U[:,4]],dim=0)
    U_row2 = torch.stack([U[:,3], U[:,1], U[:,5]],dim=0)
    U_row3 = torch.stack([U[:,4], U[:,5], U[:,2]],dim=0)
    U_matrix = torch.stack([U_row1,U_row2,U_row3],dim=0)
    U_dot_s = torch.einsum('ijk,lj->ikl', U_matrix, s_vector)  # Shape (3, M, N)
    StUS = torch.einsum('li,ikl->lk', s_vector, U_dot_s)  # Shape (M, N)
    B = -2 * (np.pi**2) *StUS 
    exp_B = torch.exp(B)
    terms = scattering_factors * exp_B * occ
    terms = terms.reshape((*terms.shape,1))
    return torch.sum(terms * torch.exp(2j * np.pi * dot_product), axis=(1,2))

def iso_structure_factor_torched(hkl,s,fractional_coords,occ,scattering_factors,tempfactor,space_group):
    fractional_coords = sym.apply_space_group(fractional_coords,space_group)
    dot_product = torch.einsum('ik,kjs->ijs',hkl.to(torch.float64), fractional_coords)
    tempfactor = tempfactor.reshape(1,-1)
    s = s.reshape(-1,1)
    B = -tempfactor * (s ** 2) / 4
    exp_B = torch.exp(B)
    terms = scattering_factors * exp_B * occ
    terms = terms.reshape((*terms.shape,1))
    return torch.sum(terms * torch.exp(2j * np.pi * dot_product), axis=(1,2))

def scale_to_reference(fcalc,ref):
    from scipy.optimize import minimize
    def loss(x, fcalc, ref):
        return np.sum(np.abs(np.abs(x * fcalc) - ref))
    x0 = 1
    bounds = [(0, None)]
    res = minimize(loss, x0 = x0, args=(fcalc, ref),bounds=bounds)
    fcalc = res.x[0] * fcalc
    return fcalc

def extract_structure_factors_gemmi(gemmi_recgrid,hkl):
    f = []
    for h,k,l in hkl:
        f.append(gemmi_recgrid.get_value(h,k,l))
    return np.array(f)

def write_numpy_to_ccp4(array, filename, unit_cell):
    """
    Writes a NumPy array to a CCP4 map file using gemmi.
    
    Parameters:
        array (np.ndarray): 3D NumPy array to save.
        filename (str): Output filename for the CCP4 map.
        unit_cell_dims (tuple): Unit cell dimensions (a, b, c, alpha, beta, gamma).
        grid_start (tuple): The starting indices of the grid (default is (0, 0, 0)).
    """
    import gemmi
    # Ensure the array is in the correct format
    if array.ndim != 3:
        raise ValueError("Input array must be 3D.")
    
    # Create a gemmi.FloatGrid and copy the NumPy array
    map = gemmi.Ccp4Map()
    grid = gemmi.FloatGrid(array.shape[0], array.shape[1], array.shape[2])
    unit_cell = gemmi.UnitCell(*unit_cell)
    grid.set_unit_cell(unit_cell)  # Set the unit cell dimensions
    # Set the grid data
    grid_array = np.array(grid, copy=False)  # Get a view into gemmi's grid
    grid_array[...] = array  # Copy NumPy array data into gemmi's grid
    map.grid = grid
    map.grid.spacegroup = gemmi.SpaceGroup('P1')
    # Set the grid start position
    
    # Write out the grid as a CCP4 map
    map.update_ccp4_header()
    map.write_ccp4_map(filename)
    print(f"Map saved to {filename}")


def get_scattering_factors_for_atom_types(hkl,cell):
    df = pd.read_feather('/das/work/p17/p17490/Peter/manual_refinement/Scattering_table_as_Excel_corrected.feather')
    s = math_np.get_s(hkl,cell)
    idx = df.index.values.copy()
    idx_diff = (idx[1:] - idx[:-1]) / 2
    idx[:-1] += idx_diff
    s_pos = np.digitize(s, idx)
    df = df.iloc[s_pos]
    df[['h','k','l']] = hkl
    df.set_index(['h','k','l'], inplace=True)
    return df

def scale_with_resolution_dependence(fcalc,ref,s):
    from scipy.optimize import minimize
    def loss(x, fcalc, ref, s):
        scale_function = x[0] + x[1] * s + x[2] * s**2 + x[3] * s**3 #+ x[4] * s**4 + x[5] * s**5 + x[6] * s**6 + x[7] * s**7 + x[8] * s**8 + x[9] * s**9 + x[10] * s**10
        f = np.abs(scale_function * fcalc)
        return np.sum(np.abs(f - ref))
    x0 = [1, 0, 0, 0]
    bounds = [(0,None),(None,None),(None,None),(None,None)]
    res = minimize(loss, x0 = x0, args=(fcalc, ref, s),bounds=bounds)
    return fcalc * (res.x[0] + res.x[1] * s + res.x[2] * s**2 + res.x[3] * s**3)

def scale_with_resolution_dependence_and_solvent(fcalc,f_solvent,ref,s):
    from scipy.optimize import minimize
    def loss(x, fcalc, f_solvent, ref, s):
        scale_function = x[0] + x[1] * s + x[2] * s**2 + x[3] * s**3 #+ x[4] * s**4 + x[5] * s**5 + x[6] * s**6 + x[7] * s**7 + x[8] * s**8 + x[9] * s**9 + x[10] * s**10
        f_solvent_scaled = x[4] * f_solvent * np.exp(-s*x[5])
        f = np.abs(scale_function * fcalc + f_solvent_scaled)
        return np.sum(np.abs(f - ref))
    x0 = [1, 0, 0, 0, 1, 0]
    bounds = [(0,None),(None,None),(None,None),(None,None),(0,None),(-20,20)]
    res = minimize(loss, x0 = x0, args=(fcalc,f_solvent, ref, s),bounds=bounds)
    return fcalc * (res.x[0] + res.x[1] * s + res.x[2] * s**2 + res.x[3] * s**3)

def save_difference_map(fcalc,ref,hkl,cell,filename='difference_map.mtz'):
    diff_f = np.abs(fcalc) - ref
    phases = np.angle(fcalc) / np.pi * 180
    import reciprocalspaceship as rs
    x2fobs_mfc = 2 * np.abs(ref) - np.abs(fcalc) 
    dataset = rs.DataSet({'Fcalc': np.abs(fcalc), 'Fobs': ref,'2FobsFcalc': x2fobs_mfc, 'Fobs-Fcalc': diff_f, 'Phases': phases},index=pd.MultiIndex.from_arrays(hkl.T,names=['H','K','L']),cell=cell,spacegroup='P1')
    dataset.infer_mtz_dtypes(inplace=True)
    dataset.write_mtz(filename)



