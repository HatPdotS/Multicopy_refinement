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
    B = torch.tensor(B,dtype=fractional_coords.dtype,device=fractional_coords.device)
    cart_coords = torch.einsum('ik,kj->ij',fractional_coords,B.T)
    return cart_coords

def get_real_grid(cell,max_res=0.8,gridsize=None,device='cpu'):
    if gridsize is not None:
        nsteps = torch.tensor(gridsize,dtype=int,device=device)
    else:
        nsteps = torch.floor(cell[:3] / max_res * 3).to(torch.int,device=device)
    # Place grid points at grid edges: i / N (CCTBX convention)
    # This matches how CCTBX/gemmi create maps
    x = torch.arange(nsteps[0],device=device) / nsteps[0]
    y = torch.arange(nsteps[1],device=device) / nsteps[1]
    z = torch.arange(nsteps[2],device=device) / nsteps[2]
    x, y, z = torch.meshgrid(x, y, z, indexing='ij')
    array_shape = x.shape
    x = x.reshape((*x.shape, 1))
    y = y.reshape((*y.shape, 1))
    z = z.reshape((*z.shape, 1))
    xyz = torch.cat((x, y, z), axis=3).reshape(-1, 3)
    xyz_real_grid = fractional_to_cartesian_torch(xyz, cell)
    xyz_real_grid = xyz_real_grid.reshape((*array_shape, 3))
    return xyz_real_grid

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

def smallest_diff(diff: torch.Tensor,inv_frac_matrix: torch.Tensor,frac_matrix: torch.Tensor):
    diff_shape = diff.shape
    diff = diff.reshape(-1,3)
    diff_frac = torch.matmul(inv_frac_matrix,diff.T)
    translation = torch.round(diff_frac)
    diff = diff - torch.matmul(frac_matrix,translation).T
    return torch.sum(diff ** 2,axis=-1).reshape(diff_shape[:-1])



def smallest_diff_aniso(diff: torch.Tensor,inv_frac_matrix: torch.Tensor,frac_matrix: torch.Tensor):
    diff_shape = diff.shape
    diff = diff.reshape(-1,3)
    diff_frac = torch.matmul(inv_frac_matrix,diff.T)
    translation = torch.round(diff_frac)
    diff -= torch.matmul(frac_matrix,translation).T
    return torch.abs(diff).reshape(diff_shape)

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

def find_relevant_voxels(real_space_grid, xyz, radius=7, inv_frac_matrix=None):
    """
    Vectorized function to identify the surrounding voxels of atoms in a real space grid.
    
    Parameters:
    -----------
    real_space_grid : torch.Tensor, shape (nx, ny, nz, 3)
        Real space grid containing xyz coordinates at each grid point
    xyz : torch.Tensor, shape (N, 3) or (3,)
        Atom coordinates in real space (Cartesian coordinates)
    radius : int
        Half-size of the box (in voxels) around each atom.
        The box will have size (2*radius+1)³
    inv_frac_matrix : torch.Tensor, shape (3, 3), optional
        Matrix to convert Cartesian to fractional coordinates.
        Required for proper handling of non-orthogonal cells.
    
    Returns:
    --------
    tuple of:
        surrounding_coords : torch.Tensor, shape (N, R³, 3)
            Coordinates of surrounding voxels for each atom, where R = 2*radius+1
            Returns the real-space coordinates from the grid for each voxel
        voxel_indices_wrapped : torch.Tensor, shape (N, R³, 3)
            Wrapped voxel indices
    
    Note:
    -----
    Atom coordinates are NOT wrapped here - periodic boundary conditions are handled
    in smallest_diff() which finds the minimum image distance. We only wrap voxel indices
    to ensure they're valid array indices.
    """
    # Ensure xyz is 2D (N, 3)
    if xyz.ndim == 1:
        xyz = xyz.unsqueeze(0)
    
    grid_shape = torch.tensor(real_space_grid.shape[:3], device=xyz.device)
    
    # Get grid origin (first voxel corner)
    grid_origin = real_space_grid[0, 0, 0]
    
    # Convert atom positions to grid indices
    # For non-orthogonal cells, we must use fractional coordinates
    if inv_frac_matrix is not None:
        # Proper way: Cartesian -> Fractional -> Wrap to [0,1] -> Grid indices
        # This ensures atoms outside the unit cell are correctly wrapped
        xyz_frac = torch.matmul(inv_frac_matrix, xyz.T).T  # (N, 3)
        xyz_frac = xyz_frac % 1.0  # Wrap to [0, 1]
        center_idx = torch.round(xyz_frac * grid_shape.unsqueeze(0)).to(torch.int64)
    else:
        # Fallback for orthogonal cells (less accurate for non-orthogonal)
        voxelsize = real_space_grid[3, 3, 3] - real_space_grid[2, 2, 2]
        center_idx = torch.round((xyz - grid_origin.unsqueeze(0)) / voxelsize.unsqueeze(0)).to(torch.int64)
    
    # Generate local offset indices for the box
    offsets = torch.arange(-radius, radius + 1, device=xyz.device)
    offset_x, offset_y, offset_z = torch.meshgrid(offsets, offsets, offsets, indexing='ij')
    local_offsets = torch.stack([offset_x.flatten(), offset_y.flatten(), offset_z.flatten()], dim=1)  # Shape: (R³, 3)
    
    # Broadcast center indices and add offsets
    # center_idx: (N, 3) -> (N, 1, 3)
    # local_offsets: (R³, 3) -> (1, R³, 3)
    # Result: (N, R³, 3)
    voxel_indices = center_idx.unsqueeze(1) + local_offsets.unsqueeze(0)
    
    # Apply periodic boundary conditions to indices (wrap around)
    # This ensures we always get valid array indices
    voxel_indices_wrapped = voxel_indices % grid_shape.unsqueeze(0).unsqueeze(0)
    
    # Extract coordinates from real_space_grid
    # For each atom, get all surrounding voxel coordinates
    surrounding_coords = real_space_grid[
        voxel_indices_wrapped[..., 0],
        voxel_indices_wrapped[..., 1],
        voxel_indices_wrapped[..., 2]
    ]
    
    return surrounding_coords, voxel_indices_wrapped

def vectorized_add_to_map(surrounding_coords, voxel_indices, map, xyz, b, inv_frac_matrix, frac_matrix, A, B,occ):
    """
    Add atoms to density map using ITC92 Gaussian parameterization.
    
    Parameters:
    -----------
    surrounding_coords : torch.Tensor, shape (N_atoms, N_voxels, 3)
        Coordinates of voxels around each atom
    voxel_indices : torch.Tensor, shape (N_atoms, N_voxels, 3)
        Indices of voxels in the map
    map : torch.Tensor, shape (nx, ny, nz)
        Electron density map
    xyz : torch.Tensor, shape (N_atoms, 3)
        Atom positions
    b : torch.Tensor, shape (N_atoms,)
        B-factors (thermal parameters) in Å²
    A : torch.Tensor, shape (N_atoms, 5)
        ITC92 amplitude coefficients for each atom
    B : torch.Tensor, shape (N_atoms, 5)
        ITC92 width coefficients (b parameters) in Å² for each atom
    inv_frac_matrix : torch.Tensor, shape (3, 3)
        Inverse fractionalization matrix
    frac_matrix : torch.Tensor, shape (3, 3)
        Fractionalization matrix
    occ: torch.Tensor, shape (N_atoms,)
        Occupancies for each atom
    """
    # Calculate squared distances with periodic boundary conditions
    # diff_coords shape: (N_atoms, N_voxels)
    diff_coords_squared = smallest_diff(surrounding_coords - xyz.unsqueeze(1), inv_frac_matrix, frac_matrix)
    
    # ITC92 + B-factor in reciprocal space: f(s) = Σ aᵢ exp(-(bᵢ + B)s²)
    # Fourier transform to real space: ρ(r) = Σ aᵢ (π/(bᵢ+B))^(3/2) exp(-π²r²/(bᵢ+B))
    # We can rewrite this as: ρ(r) = Σ aᵢ·Cᵢ exp(-r²/(2σᵢ²))
    # where Cᵢ = (π/(bᵢ+B))^(3/2) and σᵢ² = (bᵢ+B)/(2π²)
    # This way we can use the standard Gaussian form with proper normalization
    
    # B shape: (N_atoms, 4), b shape: (N_atoms,) -> B_total shape: (N_atoms, 4)
    # From comparing to CCTBX implementation we find that B_total = (B + b) / 4 NO IDEA WHY
    B_total = (B + b.unsqueeze(1)) / 4
    
    # Normalization constant: (π/B_total)^(3/2)
    normalization = (np.pi / B_total) ** 1.5
    
    # Scale amplitudes by occupancy and normalization
    A_normalized = A * occ.unsqueeze(1) * normalization
    
    # Calculate Gaussian with exponent: exp(-π²r²/B_total)
    # Note: diff_coords_squared already contains r²
    gaussian_terms = torch.exp(-(np.pi**2) * diff_coords_squared.unsqueeze(2) / B_total.unsqueeze(1))
    
    # Sum over the 4 Gaussian components
    density = torch.sum(A_normalized.unsqueeze(1) * gaussian_terms, dim=2)
    
    # Flatten to (N_atoms * N_voxels,)
    density_flat = density.flatten()
    voxel_indices_flat = voxel_indices.reshape(-1, 3)
    
    # Add to map
    map = scatter_add_nd(density_flat, voxel_indices_flat, map)
    return map

def vectorized_add_to_map_aniso(surrounding_coords, voxel_indices, map, xyz, U, inv_frac_matrix, frac_matrix, A, B, occ):
    """
    Add anisotropic atoms to density map using ITC92 Gaussian parameterization with anisotropic displacement.
    
    For anisotropic atoms, the Gaussian is:
    ρ(r) = Σᵢ Aᵢ * exp(-2π² * Δr^T * (U + Uᵢ) * Δr)
    
    where:
    - U is the atomic displacement parameter tensor (6 components: u11, u22, u33, u12, u13, u23)
    - Uᵢ is the ITC92 Gaussian width tensor derived from Bᵢ parameter
    - Δr is the distance vector from atom center
    
    Parameters:
    -----------
    surrounding_coords : torch.Tensor, shape (N_atoms, N_voxels, 3)
        Coordinates of voxels around each atom
    voxel_indices : torch.Tensor, shape (N_atoms, N_voxels, 3)
        Indices of voxels in the map
    map : torch.Tensor, shape (nx, ny, nz)
        Electron density map
    xyz : torch.Tensor, shape (N_atoms, 3)
        Atom positions in Cartesian coordinates
    U : torch.Tensor, shape (N_atoms, 6)
        Anisotropic displacement parameters in Å² (u11, u22, u33, u12, u13, u23)
        Note: These are already in Cartesian space from PDB
    inv_frac_matrix : torch.Tensor, shape (3, 3)
        Inverse fractionalization matrix
    frac_matrix : torch.Tensor, shape (3, 3)
        Fractionalization matrix
    A : torch.Tensor, shape (N_atoms, 4)
        ITC92 amplitude coefficients for each atom
    B : torch.Tensor, shape (N_atoms, 4)
        ITC92 width coefficients (b parameters) in Å² for each atom
    occ : torch.Tensor, shape (N_atoms,)
        Occupancies for each atom
        
    Returns:
    --------
    map : torch.Tensor
        Updated electron density map
    """
    # Calculate distance vectors with periodic boundary conditions
    # diff_coords shape: (N_atoms, N_voxels, 3)
    diff_coords = surrounding_coords - xyz.unsqueeze(1)
    
    diff_coords = smallest_diff_aniso(diff_coords, inv_frac_matrix, frac_matrix)
    
    # Convert ITC92 B parameters and atomic U to anisotropic formulation
    # 
    # Isotropic version uses: exp(-r² / (2σ²_total)) where σ²_total = σ²_ITC + σ²_atomic
    #   σ²_ITC = B/(4π²)
    #   σ²_atomic = U_PDB (since B-factor = 8π²*U, and σ²_B = B/(8π²) = U)
    #
    # Anisotropic version uses: exp(-2π² * r^T * U_total * r)
    # For isotropic limit: -2π² * u_total * r² = -r² / (2σ²_total)
    # Solving: u_total = 1 / (4π² σ²_total)
    #
    # KEY INSIGHT: We can't convert σ²_ITC and σ²_atomic separately and add!
    # Instead: u_total = 1 / (4π² * (σ²_ITC + σ²_atomic))
    #         = 1 / (4π² * (B/(4π²) + U_PDB))
    #         = 1 / (B + 4π²*U_PDB)
    #
    # For each ITC92 Gaussian component:
    # σ²_total = B[i]/(4π²) + U_PDB[j]
    # u_total[i,j] = 1 / (4π² * (B[i]/(4π²) + U_PDB[j]))
    #              = 1 / (B[i] + 4π²*U_PDB[j])
    #
    # B shape: (N_atoms, 4)
    # U shape: (N_atoms, 6) with format [u11, u22, u33, u12, u13, u23]
    
    # For diagonal terms of U, compute total u values
    # Need to broadcast: B is (N_atoms, 4), U diagonal is (N_atoms, 3)
    # Result should be (N_atoms, 4, 3) for u11, u22, u33 of each Gaussian component
    
    four_pi_sq = 4 * np.pi**2
    
    # Diagonal U terms
    U_diag = U[:, :3]  # (N_atoms, 3) - u11, u22, u33
    
    # Compute u_total for each combination of ITC92 component and U diagonal
    # u_total[atom, component, direction] = 1 / (B[atom, component] + 4π²*U[atom, direction])
    B_expanded = B.unsqueeze(2)  # (N_atoms, 4, 1)
    U_diag_expanded = U_diag.unsqueeze(1)  # (N_atoms, 1, 3)
    
    U_total_diag = 1.0 / (B_expanded + four_pi_sq * U_diag_expanded)  # (N_atoms, 4, 3)
    
    # Build U_total tensor with proper format [u11, u22, u33, u12, u13, u23]
    # Shape: (N_atoms, 4, 6)
    U_total = torch.zeros(B.shape[0], B.shape[1], 6, device=B.device, dtype=B.dtype)
    U_total[:, :, :3] = U_total_diag  # u11, u22, u33
    
    # For off-diagonal terms, use similar approach
    # But for simplicity, if they're close to zero, keep them zero
    # Otherwise would need to handle the full matrix inversion
    # For now: u12_total ≈ U12 / (4π²) as approximation
    U_off_diag = U[:, 3:]  # (N_atoms, 3) - u12, u13, u23
    U_total[:, :, 3:] = (U_off_diag.unsqueeze(1) / four_pi_sq).expand(-1, B.shape[1], -1)  # Approximate
    # U_total shape: (N_atoms, 4, 6)
    
    # Build U matrix for quadratic form: Δr^T * U * Δr
    # U matrix is symmetric: [[u11, u12, u13],
    #                         [u12, u22, u23],
    #                         [u13, u23, u33]]
    # U_total format: [u11, u22, u33, u12, u13, u23]
    # Shape transformations for efficient computation:
    # diff_coords: (N_atoms, N_voxels, 3)
    # U_total: (N_atoms, 4, 6) -> need (N_atoms, 4, 3, 3)
    
    U_matrix = torch.zeros(U_total.shape[0], U_total.shape[1], 3, 3,
                           device=U_total.device, dtype=U_total.dtype)
    U_matrix[:, :, 0, 0] = U_total[:, :, 0]  # u11
    U_matrix[:, :, 1, 1] = U_total[:, :, 1]  # u22
    U_matrix[:, :, 2, 2] = U_total[:, :, 2]  # u33
    U_matrix[:, :, 0, 1] = U_total[:, :, 3]  # u12
    U_matrix[:, :, 1, 0] = U_total[:, :, 3]  # u12 (symmetric)
    U_matrix[:, :, 0, 2] = U_total[:, :, 4]  # u13
    U_matrix[:, :, 2, 0] = U_total[:, :, 4]  # u13 (symmetric)
    U_matrix[:, :, 1, 2] = U_total[:, :, 5]  # u23
    U_matrix[:, :, 2, 1] = U_total[:, :, 5]  # u23 (symmetric)
    # U_matrix shape: (N_atoms, 4, 3, 3)
    
    # Compute quadratic form: Δr^T * U * Δr for each Gaussian component
    # diff_coords: (N_atoms, N_voxels, 3) -> (N_atoms, N_voxels, 1, 3) for broadcasting
    # U_matrix: (N_atoms, 4, 3, 3) -> (N_atoms, 1, 4, 3, 3) for broadcasting
    diff_coords_expanded = diff_coords.unsqueeze(2)  # (N_atoms, N_voxels, 1, 3)
    U_matrix_expanded = U_matrix.unsqueeze(1)  # (N_atoms, 1, 4, 3, 3)
    
    # First: U * Δr -> (N_atoms, N_voxels, 4, 3)
    U_times_diff = torch.einsum('naijk,namk->naij', U_matrix_expanded, diff_coords_expanded)
    
    # Second: Δr^T * (U * Δr) -> (N_atoms, N_voxels, 4)
    quad_form = torch.einsum('namk,namk->nam', diff_coords_expanded, U_times_diff)
    
    # Apply occupancy scaling to amplitudes
    A_scaled = A * occ.unsqueeze(1)  # Shape: (N_atoms, 4)
    
    # Calculate Gaussian density for each component
    # quad_form shape: (N_atoms, N_voxels, 4)
    # Exponent: -2π² * quad_form
    gaussian_terms = torch.exp(-2 * np.pi**2 * quad_form)  # (N_atoms, N_voxels, 4)
    
    # Sum over 4 Gaussian components
    # A_scaled: (N_atoms, 4) -> (N_atoms, 1, 4) for broadcasting
    density = torch.sum(A_scaled.unsqueeze(1) * gaussian_terms, dim=2)  # (N_atoms, N_voxels)
    
    # Flatten to (N_atoms * N_voxels,)
    density_flat = density.flatten()
    voxel_indices_flat = voxel_indices.reshape(-1, 3)
    
    # Add to map
    map = scatter_add_nd(density_flat, voxel_indices_flat, map)
    return map

def scatter_add_nd_super_slow(source, index, map):
    for i in range(source.shape[0]):
        idx = tuple(index[i].tolist())
        map[idx] += source[i]
    return map

def scatter_add_nd(source, index, map):
    """
    Vectorized n-dimensional scatter add operation.
    
    Parameters:
    -----------
    source : torch.Tensor, shape (N,)
        Values to add to the map
    index : torch.Tensor, shape (N, ndim)
        Indices where values should be added
    map : torch.Tensor, shape (d1, d2, ..., dn)
        N-dimensional tensor to add values into
        
    Returns:
    --------
    map : torch.Tensor
        Modified map with values added
    """
    map_shape = torch.tensor(map.shape, device=index.device, dtype=torch.int64)
    
    # Convert n-dimensional indices to flat indices
    # For shape (d1, d2, d3, ..., dn), flat_index = i0 * (d1*d2*...*dn) + i1 * (d2*d3*...*dn) + ... + in
    strides = torch.ones(len(map_shape), device=index.device, dtype=torch.int64)
    for i in range(len(map_shape) - 2, -1, -1):
        strides[i] = strides[i + 1] * map_shape[i + 1]
    
    index_flat = torch.sum(index * strides.unsqueeze(0), dim=-1)
    
    map_flat = map.view(-1)
    map_flat.scatter_add_(0, index_flat, source)
    return map

def place_on_grid(hkls, structure_factor, grid_size, enforce_hermitian: bool = True) -> torch.Tensor:
    """
    Vectorized placement of batched structure factors on reciprocal-space grid.
    
    Args:
        enforce_hermitian: Whether to enforce Hermitian symmetry
        
    Returns:
        Complex tensor grid of structure factors
    """
    batch_mode = True
    if structure_factor.ndim == 1:
        structure_factor = structure_factor.unsqueeze(0)  # Add batch dimension
        batch_mode = False
    B = structure_factor.shape[0]
    device = structure_factor.device
    dtype = structure_factor.dtype
    Nx, Ny, Nz = [int(x) for x in grid_size]
    # Prepare Miller indices and linear indices
    hkls = hkls.to(device=device)
    h = hkls[:, 0].to(torch.int64)
    k = hkls[:, 1].to(torch.int64)
    l = hkls[:, 2].to(torch.int64)
    
    hi = torch.remainder(h, Nx)
    ki = torch.remainder(k, Ny)
    li = torch.remainder(l, Nz)
    lin = (hi * (Ny * Nz) + ki * Nz + li).to(torch.int64)  # (N,)
    # Vectorized scatter-add to grid
    grid = torch.zeros((B, Nx * Ny * Nz), dtype=dtype, device=device)
    grid = grid.index_add(1, lin, structure_factor)  # (B, Nx*Ny*Nz)
    
    if enforce_hermitian:
        hi_sym = torch.remainder(-h, Nx)
        ki_sym = torch.remainder(-k, Ny)
        li_sym = torch.remainder(-l, Nz)
        lin_sym = (hi_sym * (Ny * Nz) + ki_sym * Nz + li_sym).to(torch.int64)
        vals_conj = torch.conj(structure_factor)
        grid = grid.index_add(1, lin_sym, vals_conj)
        
    grid = grid.view(B, Nx, Ny, Nz)
    if not batch_mode:
        grid = grid.squeeze(0)
    return grid

def fft(reciprocal_grid) -> torch.Tensor:
    """
    Perform FFT to obtain real space electron density.
    
    Returns:
        Real-valued tensor of electron density
        
    Raises:
        ValueError: If grid not initialized
    """
    if reciprocal_grid.ndim == 4:
        rs = torch.fft.ifftn(reciprocal_grid, dim=(1, 2, 3)).real
        rs = torch.flip(rs, dims=(1, 2, 3))
        rs = torch.roll(rs, shifts=(1, 1, 1), dims=(1, 2, 3))
    else:
        rs = torch.fft.ifftn(reciprocal_grid, dim=(0, 1, 2)).real
        rs = torch.flip(rs, dims=(0, 1, 2))
        rs = torch.roll(rs, shifts=(1, 1, 1), dims=(0, 1, 2))
    return rs

def ifft(real_space_map) -> torch.Tensor:
    """
    Perform inverse FFT to obtain reciprocal space structure factors.
    
    Returns:
        Complex-valued tensor of structure factors
        
    Raises:
        ValueError: If grid not initialized
    """
    if real_space_map.ndim == 4:
        rg = torch.roll(real_space_map, shifts=(-1, -1, -1), dims=(1, 2, 3))
        rg = torch.flip(rg, dims=(1, 2, 3))
        rg = torch.fft.fftn(rg, dim=(1, 2, 3))
    else:
        rg = torch.roll(real_space_map, shifts=(-1, -1, -1), dims=(0, 1, 2))
        rg = torch.flip(rg, dims=(0, 1, 2))
        rg = torch.fft.fftn(rg, dim=(0, 1, 2))
    return rg

def extract_structure_factor_from_grid(reciprocal_grid, hkls) -> torch.Tensor:
    """
    Extract structure factors from reciprocal space grid at given Miller indices.
    
    Args:
        reciprocal_grid: Complex tensor of shape (Nx, Ny, Nz) or (B, Nx, Ny, Nz)
        hkls: Tensor of Miller indices of shape (N, 3)

    Returns:
        Tensor of structure factors of shape (N,) or (B, N,)

    """
    device = reciprocal_grid.device
    dtype = reciprocal_grid.dtype
    
    # Handle both batched and unbatched input
    if reciprocal_grid.ndim == 3:
        reciprocal_grid = reciprocal_grid.unsqueeze(0)  # Add batch dimension
        squeeze_output = True
    else:
        squeeze_output = False
    
    B, Nx, Ny, Nz = reciprocal_grid.shape
    
    # Convert Miller indices to grid positions using same convention as place_on_grid
    hkls = hkls.to(device=device)
    h = hkls[:, 0].to(torch.int64)
    k = hkls[:, 1].to(torch.int64)
    l = hkls[:, 2].to(torch.int64)
    
    # Map to grid indices with periodic wrapping
    hi = torch.remainder(h, Nx)
    ki = torch.remainder(k, Ny)
    li = torch.remainder(l, Nz)
    
    # Extract structure factors at these positions
    # For batched: (B, Nx, Ny, Nz) -> (B, N)
    structure_factors = reciprocal_grid[:, hi, ki, li]  # (B, N)
    
    if squeeze_output:
        structure_factors = structure_factors.squeeze(0)  # (N,)
    
    return structure_factors