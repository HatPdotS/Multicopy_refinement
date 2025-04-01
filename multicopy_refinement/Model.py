import pdb_tools
import multicopy_refinement.math_numpy as mnp
import numpy as np
import gemmi 
from multicopy_refinement.numba_functions import smallest_diff
from multicopy_refinement import math_numpy as math_np
from multicopy_refinement import squeeze
from multicopy_refinement import direct_summation as ds
from torch import tensor
from torch import nn
import multicopy_refinement.math_torch as math_torch
import torch

class model:
    def __init__(self,model=None):
        self.pdb = model if model else None

    def load_pdb_from_file(self,file,strip_H=True):
        self.pdb = pdb_tools.load_pdb_as_pd(file)
        self.pdb = self.pdb.loc[self.pdb['element'] != 'H'] if strip_H else self.pdb
        self.cell = np.array(self.pdb.attrs['cell'])
        self.spacegroup = self.pdb.attrs['spacegroup']
        self.inv_fractional_matrix = math_np.get_inv_fractional_matrix(self.cell)
        self.fractional_matrix = math_np.get_fractional_matrix(self.cell)
        self.mean_models = []
        self.projection_matrices = []
        self.projected_molecules = []
        self.residues = dict()
        for id, res in self.pdb.groupby(['resseq','chainid','altloc'],as_index=False):
            resseq, chainid, altloc = id
            if altloc == '':
                altloc = None
            if resseq == '':
                resseq = None
            if chainid == '':
                chainid = None
            occ = res.occupancy.mean()
            if occ < 1:
                refine_occ = True
            else:
                refine_occ = False
            id = (chainid,resseq,altloc)
            self.residues[id] = residue(res,id,self.spacegroup,self.cell,occ=occ,refine_occ=refine_occ)

    def setup_grids(self):
        self.real_space_grid = mnp.get_real_grid(self.cell,self.max_res)
        self.real_space_grid = self.real_space_grid.astype(np.float64)
        self.map = np.zeros(self.real_space_grid.shape[:-1],dtype=np.float64)
    
    def build_atom_full(self, xyz, b, A, B):
        # Calculate squared distances
        offset_map = smallest_diff(self.real_space_grid - xyz, 
                                self.inv_fractional_matrix, 
                                self.fractional_matrix).reshape((*self.map.shape, 1))
        
        # Convert B-factor to real space
        b_factor_term = b / (4 * np.pi**2)
        adjusted_b = B / (2 *np.pi)  # B is in Å², so is b_factor_term
        density = np.sum(A * (np.pi/adjusted_b)**(3/2) * np.exp(-offset_map/adjusted_b), axis=-1)
        self.map += density

    def get_f_ds(self,hkl):
        return ds.direct_summation(self.pdb,hkl,self.cell,space_group=self.spacegroup)

    def build_ed(self):
        for i in range(len(self.pdb)):
            row = self.pdb.iloc[i]
            xyz = row[['x','y','z']].values.astype(np.float64).reshape(1,1,1,3)
            element = row['element']
            b = float(row['tempfactor'])
            fe_coef = gemmi.Element(element).it92
            A = np.array(fe_coef.a).reshape(1,1,1,-1).astype(np.float64)
            B = np.array(fe_coef.b).reshape(1,1,1,-1).astype(np.float64)
            self.build_atom_full(xyz,b,A,B)

    def get_selection(self,selection):
        chainid = selection[0]
        resnumber = selection[1]
        atomname = selection[2]
        altloc = selection[3]
        if chainid == None:
            sel = self.pdb
        else:
            sel = self.pdb.loc[self.pdb['chainid'] == chainid]
        if resnumber == None:
            pass
        else:
            sel = sel.loc[sel['resseq'] == resnumber]
        if atomname == None:
            pass
        else:
            sel = sel.loc[sel['name'] == atomname]
        if altloc == None:
            pass
        else:
            sel = sel.loc[sel['altloc']==altloc]
        return sel

    def align_models(self,selections):
        copies = [self.get_selection(sel) for sel in selections]
        model1 = copies[0]
        aligned_models = []
        for model in copies:
            aligned,rmsd = math_np.align_pdbs(model1,model,Atoms=['C8','N9','N10','C11'])
            aligned_models.append(aligned)
        return aligned_models

    def replace_copies_with_mean(self,selections,atoms=None):
        mean_model = self.get_mean_model(selections,atoms=atoms)
        copies = [self.get_selection(sel) for sel in selections]
        self.mean_models.append(mean_model)
        self.projected_molecules.append(selections)
        alignements = []
        for model in copies:
            alignement, rmsd = math_np.get_alignment_matrix(model,mean_model,Atoms=atoms)
            alignements.append(alignement)
        self.projection_matrices.append(alignements)
        
    def get_mean_model(self,selections,atoms=None):
        copies = [self.get_selection(sel).copy() for sel in selections]
        model1 = copies[0]
        xyz_mean = []
        for model in copies:
            aligned,rmsd = math_np.align_pdbs(model1,model,Atoms=atoms)
            xyz_mean.append(aligned[['x','y','z']].values.reshape(-1,3,1))
        mean = np.mean(np.concatenate(xyz_mean,axis=2),axis=2)
        self.mean_model = model1.copy()
        self.mean_model.loc[:,['x','y','z']] = mean

    def write(self,file):
        pdb_tools.write_file(self.pdb,file)

    def get_f_ft(self, hkl):
        # Apply FFT and phase shift for proper origin placement
        recgrid = np.fft.fftn(self.map)
        
        # Apply proper scaling
        volume = np.prod(self.cell[:3])
        grid_points = np.prod(self.map.shape)
        scale_factor = volume / grid_points
        
        # Extract structure factors with proper negative index handling
        self.f = squeeze.extract_hkl(recgrid, hkl) * scale_factor
        
        # Apply phase shift for each reflection based on grid dimensions
        nx, ny, nz = self.map.shape
        phase_shifts = np.zeros(len(hkl), dtype=np.complex128)
        for i, (h, k, l) in enumerate(hkl):
            # Calculate phase shift to center map properly
            shift = np.exp(-2j * np.pi * (h/(2*nx) + k/(2*ny) + l/(2*nz)))
            self.f[i] *= shift
        return self.f

class occupancy_manager(nn.Module):
    def __init__(self, occ_start = None):
        super().__init__()
        if occ_start is None:
            self.occupancy = nn.Parameter(tensor(1))
        else:
            self.occupancy = nn.Parameter(tensor(occ_start))
        self.occupancy.requires_grad = True

    def enforce_boundaries(self):
        self.occupancy.clamp_(min=0, max=1)
        return self.occupancy
    
    def get_occupancy(self):
        return self.occupancy
    
    def get_tensors(self):
        return self.occupancy

class linked_occupancy_manager(nn.Module):
    def __init__(self, molecules,occ_start = None,target_occupancy=1):
        super().__init__()
        self.target_occupancy = target_occupancy
        self.molecules = molecules
        self.n_molecules = len(molecules)
        if occ_start is None:
            self.occupancies = nn.Parameter(tensor(np.ones(self.n_molecules)/self.n_molecules))
        else:   
            self.occupancies = nn.Parameter(tensor(occ_start))
        self.occupancies.requires_grad = True
    
    def enforce_boundaries(self):
        self.occupancies.clamp_(min=0, max=1)
        return self.occupancies.div_(self.occupancies.sum()/self.target_occupancy)
    
    def get_occupancies(self):
        return self.occupancies
    
    def get_tensors(self):
        return self.occupancies

    def get_occupancy(self, id):
        idx = self.molecules.index(id)
        return self.occupancies[idx]
    
class residue(nn.Module):
    def __init__(self,res_pdb,id,spacegroup,cell,occ=None,refine_occ=False):
        super().__init__()
        self.res_pdb = res_pdb
        self.aniso_flag = res_pdb['aniso_flag'].values[0]
        self.spacegroup = spacegroup
        self.cell = cell
        self.refine_occ = refine_occ
        self.id = id
        self.xyz = nn.Parameter(tensor(res_pdb[['x','y','z']].values))
        if occ is None:
            occ = self.res_pdb['occupancy'].values[0]
            self.occ = occupancy_manager(occ)
        elif isinstance(occ, occupancy_manager) or isinstance(occ, linked_occupancy_manager):
            self.occ = occ
        elif isinstance(occ, float):
            self.occ = occupancy_manager(occ_start=occ)
        else:
            raise ValueError("occupancy must be an occupancy_manager or linked_occupancy_manager instance")
        self.b = nn.Parameter(tensor(res_pdb['tempfactor'].values))
        self.u = nn.Parameter(tensor(res_pdb[['u11','u22','u33','u12','u13','u23']].values))
        self.element = res_pdb['element'].values
    
    def get_tensors(self):
        return self.transformation_matrix, self.occ.get_tensors()

    def get_xyz(self):
        return self.xyz
    
    def get_b(self):
        return self.b
    
    def get_occupancy(self):
        return self.occ.get_occupancy()

    def get_tensors_to_refine(self):
        if self.refine_occ:
            return self.xyz, self.occ, self.b
        else:
            return self.xyz, self.b
    
    def get_contribution_f(self):
        # Calculate the contribution of the residue to the structure factor
        return
        

class projected_residue(nn.Module):
    def __init__(self,id,transformation_matrix,projected_molecule: residue,occ=None,refine_occ=False):
        super().__init__()
        self.id = id
        self.projected_molecule = projected_molecule
        self.aniso_flag = projected_molecule.aniso_flag
        self.spacegroup = projected_molecule.spacegroup
        self.cell = projected_molecule.cell
        self.transformation_matrix = nn.Parameter(tensor(transformation_matrix))
        self.refine_occ = refine_occ
        if occ is None:
            self.occ = occupancy_manager()
        elif isinstance(occ, occupancy_manager) or isinstance(occ, linked_occupancy_manager):
            self.occ = occ
        elif isinstance(occ, float):
            self.occ = occupancy_manager(occ_start=occ)
        else:
            raise ValueError("occupancy must be an occupancy_manager or linked_occupancy_manager instance")

    def get_tensors(self):
        return self.transformation_matrix, self.occ.get_tensors()
    
    def get_occupancy(self):
        return self.occ.get_occupancy(self.projected_molecule.id)

    def get_xyz(self):
        xyz = self.projected_molecule.get_xyz()
        return math_torch.apply_transformation(xyz,self.transformation_matrix)

    def get_tensors_to_refine(self):
        if self.refine_occ:
            return self.transformation_matrix, self.occ.get_tensors()
        else:
            return self.transformation_matrix