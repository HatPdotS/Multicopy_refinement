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
    homo_points = torch.hstack((points, torch.ones((points.shape[0], 1),device=points.device)))
    last_row = torch.tensor([0,0,0,1],device=points.device)
    transformation_matrix = torch.vstack((transformation_matrix,last_row))
    # Apply transformation
    transformed = torch.matmul(homo_points, transformation_matrix.T)
    # Return 3D coordinates
    return transformed[:, :3]

def core_deformation(core_correction,s):
    return 1.0 - core_correction * torch.exp(-s*s/0.5)

def aniso_structure_factor_torched(hkl,s_vector,fractional_coords,occ,scattering_factors,U,space_group):
    fractional_coords = space_group(fractional_coords.T)
    fractional_shape = fractional_coords.shape
    fractional_coords = fractional_coords.reshape(3,-1)
    dot_product = torch.matmul(hkl.to(torch.float64), fractional_coords).reshape(hkl.shape[0],fractional_shape[1],-1)
    U_row1 = torch.stack([U[:,0],U[:,3], U[:,4]],dim=0)
    U_row2 = torch.stack([U[:,3], U[:,1], U[:,5]],dim=0)
    U_row3 = torch.stack([U[:,4], U[:,5], U[:,2]],dim=0)
    U_matrix = torch.stack([U_row1,U_row2,U_row3],dim=0)
    U_dot_s = torch.einsum('jik,li->jkl', U_matrix, s_vector)  # Shape (3, M, N)
    StUS = torch.einsum('li,ikl->lk', s_vector, U_dot_s)  # Shape (M, N)
    B = -2 * (np.pi**2) * StUS 
    exp_B = torch.exp(B)
    terms = scattering_factors * exp_B * occ
    pidot = 2 * np.pi * dot_product
    sin_cos = torch.sum(1j * torch.sin(pidot) + torch.cos(pidot),axis=-1)
    return torch.sum(terms * sin_cos, axis=(1))

def iso_structure_factor_torched(hkl,s,fractional_coords,occ,scattering_factors,tempfactor,space_group):
    fractional_coords = space_group(fractional_coords.T)
    fractional_shape = fractional_coords.shape
    fractional_coords = fractional_coords.reshape(3,-1)
    dot_product = torch.matmul(hkl.to(torch.float64), fractional_coords).reshape(hkl.shape[0],fractional_shape[1],-1)
    tempfactor = tempfactor.reshape(1,-1)
    s = s.reshape(-1,1)
    B = -tempfactor * (s ** 2) / 4
    exp_B = torch.exp(B)
    terms = scattering_factors * exp_B * occ
    pidot = 2 * np.pi * dot_product
    sin_cos = torch.sum(1j * torch.sin(pidot) + torch.cos(pidot),axis=-1)
    return torch.sum(terms * sin_cos, axis=(1))

def superpose_vectors_robust_torch(ref_coords, mov_coords, weights=None, max_iterations=10):
    if weights is None:
        weights = torch.ones((ref_coords.shape[0],1), device=ref_coords.device)
    weights = weights / torch.sum(weights)

    mobile_coords_current = mov_coords.clone()
    best_matrix = torch.eye(4, device=mobile_coords_current.device, dtype=mobile_coords_current.dtype)
    best_rmsd = torch.tensor(float('inf'))

    
    for iteration in range(max_iterations):
        # Calculate centroids
        target_centroid = torch.sum(weights * ref_coords, axis=0)
        mobile_centroid = torch.sum(weights * mobile_coords_current, axis=0)
        
        # Center coordinates
        target_centered = ref_coords - target_centroid
        mobile_centered = mobile_coords_current - mobile_centroid
        
        # Calculate the covariance matrix with weights
        covariance = torch.zeros((3, 3),dtype=mobile_coords_current.dtype, device=mobile_coords_current.device)
        for i in range(len(weights)):
            covariance += weights[i] * torch.outer(mobile_centered[i], target_centered[i])
        
        # SVD of covariance matrix
    
        U, S, Vt = torch.linalg.svd(covariance)
        
        # Check for reflection case (determinant < 0)
        det = torch.linalg.det(torch.matmul(Vt.T, U.T))
        correction = torch.eye(3, dtype=mobile_coords_current.dtype, device=mobile_coords_current.device)
        if det < 0:
            correction[2, 2] = -1
            
        # Calculate rotation matrix
        rotation_matrix = torch.matmul(torch.matmul(Vt.T, correction), U.T)
        
        # FIXED: Calculate translation correctly
        # The correct way is: translation = target_centroid - (rotation_matrix @ mobile_centroid)
        # In NumPy notation with row vectors, this is:
        rotated_mobile_centroid = torch.matmul(mobile_centroid, rotation_matrix.T)
        translation = target_centroid - rotated_mobile_centroid
        
        # Compute 4x4 transformation matrix
        transformation_matrix = torch.zeros((3,4), device=mobile_coords_current.device, dtype=mobile_coords_current.dtype)
        transformation_matrix[:, :3] = rotation_matrix
        transformation_matrix[:, 3] = translation
        
        # Apply transformation and calculate RMSD
        # Using the correct transformation application
        mobile_transformed = torch.matmul(mov_coords, rotation_matrix.T) + translation
        
        squared_diffs = torch.sum((ref_coords - mobile_transformed)**2, axis=1)
        rmsd = torch.sqrt(torch.sum(weights * squared_diffs))
        
        if rmsd < best_rmsd:
            best_rmsd = rmsd
            best_matrix = transformation_matrix
        # Update mobile coords for next iteration if doing iterative refinement
        if max_iterations > 1:
            mobile_coords_current = mobile_transformed
        return best_matrix
    
def align_torch(xyz1,xyz2,idx_to_move=None):
    if idx_to_move is not None:
        transformation_matrix1 = superpose_vectors_robust_torch(xyz1[idx_to_move],xyz2[idx_to_move])
    else:
        transformation_matrix1 = superpose_vectors_robust_torch(xyz1,xyz2)
    transformation_matrix = transformation_matrix1
    xyz_moved = apply_transformation(xyz2, transformation_matrix)
    return xyz_moved

def get_alignement_matrix(xyz1,xyz2,idx_to_move=None):
    if idx_to_move is not None:
        transformation_matrix = superpose_vectors_robust_torch(xyz1[idx_to_move],xyz2[idx_to_move])
    else:
        transformation_matrix = superpose_vectors_robust_torch(xyz1,xyz2)
    return transformation_matrix

def anharmonic_correction(hkl,c):
    h1, h2, h3 = hkl[:, 0], hkl[:, 1], hkl[:, 2]
    # These third-order terms specifically address toroidal features
    C111, C222, C333, C112, C122, C113, C133, C223, C233, C123 = c
    # For toroidal features around z-axis, C111 and C222 are most important
    third_order = (
        C111 * h1**3 + 
        C222 * h2**3 + 
        C333 * h3**3 +
        3 * C112 * h1**2 * h2 +
        3 * C122 * h1 * h2**2 +
        3 * C113 * h1**2 * h3 +
        3 * C133 * h1 * h3**2 +
        3 * C223 * h2**2 * h3 +
        3 * C233 * h2 * h3**2 +
        6 * C123 * h1 * h2 * h3
    ) * (-8j * torch.pi**3) / 6e7
    return torch.exp(third_order)

def aniso_structure_factor_torched_no_complex(hkl,s_vector,fractional_coords,occ,scattering_factors,U,space_group):
    fractional_coords = space_group(fractional_coords.T)
    fractional_shape = fractional_coords.shape
    fractional_coords = fractional_coords.reshape(3,-1)
    dot_product = torch.matmul(hkl.to(torch.float64), fractional_coords).reshape(hkl.shape[0],fractional_shape[1],-1)
    U_row1 = torch.stack([U[:,0],U[:,3], U[:,4]],dim=0)
    U_row2 = torch.stack([U[:,3], U[:,1], U[:,5]],dim=0)
    U_row3 = torch.stack([U[:,4], U[:,5], U[:,2]],dim=0)
    U_matrix = torch.stack([U_row1,U_row2,U_row3],dim=0)
    U_dot_s = torch.einsum('jik,li->jkl', U_matrix, s_vector)  # Shape (3, M, N)
    StUS = torch.einsum('li,ikl->lk', s_vector, U_dot_s)  # Shape (M, N)
    B = -2 * (np.pi**2) * StUS 
    exp_B = torch.exp(B)
    terms = scattering_factors * exp_B * occ
    pidot = 2 * np.pi * dot_product
    complex = torch.sum(torch.sum(torch.sin(pidot),axis=-1) * terms,axis=1)
    real = torch.sum(torch.sum(torch.cos(pidot),axis=-1) * terms,axis=1)
    return torch.vstack((real,complex))

def iso_structure_factor_torched_no_complex(hkl,s,fractional_coords,occ,scattering_factors,tempfactor,space_group):
    fractional_coords = space_group(fractional_coords.T)
    fractional_shape = fractional_coords.shape
    fractional_coords = fractional_coords.reshape(3,-1)
    dot_product = torch.matmul(hkl.to(torch.float64), fractional_coords).reshape(hkl.shape[0],fractional_shape[1],-1)
    tempfactor = tempfactor.reshape(1,-1)
    s = s.reshape(-1,1)
    B = -tempfactor * (s ** 2) / 4
    exp_B = torch.exp(B)
    terms = scattering_factors * exp_B * occ
    pidot = 2 * np.pi * dot_product
    complex = torch.sum(torch.sum(torch.sin(pidot),axis=-1) * terms,axis=1)
    real = torch.sum(torch.sum(torch.cos(pidot),axis=-1) * terms,axis=1)
    return torch.vstack((real,complex))

def anharmonic_correction_no_complex(hkl,c):
    h1, h2, h3 = hkl[:, 0], hkl[:, 1], hkl[:, 2]
    # These third-order terms specifically address toroidal features
    C111, C222, C333, C112, C122, C113, C133, C223, C233, C123 = c
    # For toroidal features around z-axis, C111 and C222 are most important
    third_order = (
        C111 * h1**3 + 
        C222 * h2**3 + 
        C333 * h3**3 +
        3 * C112 * h1**2 * h2 +
        3 * C122 * h1 * h2**2 +
        3 * C113 * h1**2 * h3 +
        3 * C133 * h1 * h3**2 +
        3 * C223 * h2**2 * h3 +
        3 * C233 * h2 * h3**2 +
        6 * C123 * h1 * h2 * h3 
    ) *  (-8 * torch.pi**3) / 6e7
    return torch.vstack((torch.cos(third_order), torch.sin(third_order)))

def multiplication_quasi_complex_tensor(a,b):
    real_part = a[0] * b[0] - a[1] * b[1]
    imag_part = a[0] * b[1] + a[1] * b[0]
    return torch.vstack((real_part, imag_part))

def french_wilson_conversion(Iobs, sigma_I=None):
    """
    Convert intensities to structure factor amplitudes using French-Wilson method
    Also converts standard deviations
    
    Parameters:
    -----------
    Iobs : torch.Tensor
        Observed intensity values
    sigma_I : torch.Tensor, optional
        Estimated standard deviations of intensities
        
    Returns:
    --------
    tuple (torch.Tensor, torch.Tensor)
        Structure factor amplitudes and their standard deviations
    """
    # If no sigmas provided, estimate them
    if sigma_I is None:
        sigma_I = torch.sqrt(torch.clamp(Iobs, min=1e-6))
    
    # Determine mean intensity for Wilson prior
    mean_I = torch.mean(torch.clamp(Iobs, min=0))
    
    # Strong reflections: simple square root
    strong_mask = Iobs > 3.0 * sigma_I
    F = torch.zeros_like(Iobs)
    F[strong_mask] = torch.sqrt(Iobs[strong_mask])
    
    # Weak/negative reflections: Bayesian approach
    weak_mask = ~strong_mask
    if weak_mask.any():
        # For weak reflections, use Bayesian estimate
        I_weak = Iobs[weak_mask]
        sigma_weak = sigma_I[weak_mask]
        
        # Simplified Bayesian estimate (posterior mean)
        wilson_param = mean_I / 2.0
        variance_correction = sigma_weak**2 / (2.0 * wilson_param)
        F[weak_mask] = torch.sqrt(torch.clamp(I_weak + variance_correction, min=0))
    
    # Convert sigmas using error propagation formula
    # For F = sqrt(I), σ(F) = σ(I)/(2*F)
    # Avoid division by zero
    sigma_F = torch.zeros_like(sigma_I)
    nonzero_F = F > 1e-6
    sigma_F[nonzero_F] = sigma_I[nonzero_F] / (2.0 * F[nonzero_F])
    
    # For very weak reflections where F approaches zero,
    # use an upper bound approximation to avoid huge sigma values
    tiny_F = (F <= 1e-6) & (sigma_I > 0)
    if tiny_F.any():
        # Approximate using the typical Wilson distribution variance
        sigma_F[tiny_F] = torch.sqrt(wilson_param / 2.0)
    
    return F, sigma_F

def reciprocal_basis_matrix(unit_cell):
    # Extract unit cell parameters
    a, b, c, alpha, beta, gamma = torch.tensor(unit_cell)
    alpha, beta, gamma =  torch.deg2rad(torch.tensor([alpha, beta, gamma], dtype=torch.float64))
    # Compute real-space basis vectors
    cos_alpha, cos_beta, cos_gamma = torch.cos(alpha), torch.cos(beta), torch.cos(gamma)
    sin_gamma = torch.sin(gamma)
    volume = torch.sqrt(1 - cos_alpha**2 - cos_beta**2 - cos_gamma**2 + 2 * cos_alpha * cos_beta * cos_gamma)
    a_vec = torch.tensor([a, 0, 0])
    b_vec = torch.tensor([b * cos_gamma, b * sin_gamma, 0])
    c_vec = torch.tensor([
        c * cos_beta,
        c * (cos_alpha - cos_beta * cos_gamma) / sin_gamma,
        c * volume / sin_gamma
    ])
    # Compute reciprocal basis vectors
    volume_real = torch.dot(a_vec, torch.linalg.cross(b_vec, c_vec))
    a_star = torch.linalg.cross(b_vec, c_vec) / volume_real
    b_star = torch.linalg.cross(c_vec, a_vec) / volume_real
    c_star = torch.linalg.cross(a_vec, b_vec) / volume_real
    # Assemble reciprocal basis matrix
    return torch.stack([a_star, b_star, c_star])

def get_scattering_vectors(hkl, unit_cell):
    recB = reciprocal_basis_matrix(unit_cell)
    s = torch.matmul(hkl.to(torch.float64),recB)
    return s