import torch
from multicopy_refinement import math_numpy as math_np
import numpy as np
import multicopy_refinement.symmetrie as sym

def cartesian_to_fractional_torch(cart_coords, unit_cell):
    B_inv = math_np.get_inv_fractional_matrix(unit_cell)
    B_inv = torch.tensor(B_inv)
    fractional_vector = torch.einsum('ik,kj->ij',cart_coords,B_inv.T)
    return fractional_vector

def fractional_to_cartesian_torch(fractional_coords, unit_cell):
    B = math_np.get_fractional_matrix(unit_cell)
    B = torch.tensor(B)
    cart_coords = torch.einsum('ik,kj->ij',fractional_coords,B.T)
    return cart_coords

def rotate_coords_torch(coords,phi,rho):
    phi = phi * np.pi / 180
    rho = rho * np.pi / 180
    rot_matrix = torch.tensor([[torch.cos(phi),-torch.sin(phi),0],
                               [torch.sin(phi)*torch.cos(rho),torch.cos(phi)*torch.cos(rho),-torch.sin(rho)],
                               [torch.sin(phi)*torch.sin(rho),torch.cos(phi)*torch.sin(rho),torch.cos(rho)]],dtype=torch.float64)
    return torch.einsum('ij,kj->ki',rot_matrix,coords)

def get_rfactor_torch(fobs,fcalc):
    fobs = torch.abs(fobs)
    fcalc = torch.abs(fcalc)
    return torch.sum(torch.abs(fobs - fcalc)) / torch.sum(fobs)

def calc_outliers(fobs,fcalc,z):
    fobs = torch.abs(fobs)
    fcalc = torch.abs(fcalc)
    diff = torch.abs(fobs - fcalc) / fobs
    std = torch.std(diff)
    outliers = diff >  z * std
    return outliers

def apply_transformation(points, transformation_matrix):
    """Apply 4x4 transformation matrix to 3D points"""
    # Convert to homogeneous coordinates
    homo_points = torch.hstack((points, torch.ones((points.shape[0], 1))))
    # Apply transformation
    transformed = torch.dot(homo_points, transformation_matrix.T)
    # Return 3D coordinates
    return transformed[:, :3]

def aniso_structure_factor_torched(hkl,s_vector,fractional_coords,occ,scattering_factors,U,space_group):
    fractional_coords = sym.apply_space_group(fractional_coords,space_group)
    dot_product = torch.einsum('ik,kjs->ijs',hkl.to(torch.float64), fractional_coords)
    U_row1 = torch.stack([U[:,0],U[:,3], U[:,4]],dim=0)
    U_row2 = torch.stack([U[:,3], U[:,1], U[:,5]],dim=0)
    U_row3 = torch.stack([U[:,4], U[:,5], U[:,2]],dim=0)
    U_matrix = torch.stack([U_row1,U_row2,U_row3],dim=0)
    U_dot_s = torch.einsum('ijk,lj->ikl', U_matrix, s_vector)  # Shape (3, M, N)
    StUS = torch.einsum('li,ikl->lk', s_vector, U_dot_s)  # Shape (M, N)
    B = -2 * (np.pi**2) * StUS 
    exp_B = torch.exp(B)
    terms = scattering_factors * exp_B * occ
    pidot = 2 * np.pi * dot_product
    sin_cos = torch.sum(1j * torch.sin(pidot) + torch.cos(pidot),axis=-1)
    return torch.sum(terms * sin_cos, axis=(1))

def iso_structure_factor_torched(hkl,s,fractional_coords,occ,scattering_factors,tempfactor,space_group):
    fractional_coords = sym.apply_space_group(fractional_coords,space_group)
    dot_product = torch.einsum('ik,kjs->ijs',hkl.to(torch.float64), fractional_coords)
    tempfactor = tempfactor.reshape(1,-1)
    s = s.reshape(-1,1)
    B = -tempfactor * (s ** 2) / 4
    exp_B = torch.exp(B)
    terms = scattering_factors * exp_B * occ
    pidot = 2 * np.pi * dot_product
    sin_cos = torch.sum(1j * torch.sin(pidot) + torch.cos(pidot),axis=-1)
    return torch.sum(terms * sin_cos, axis=(1))