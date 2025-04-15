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
import pickle


class model(nn.Module):
    def __init__(self,model=None,abs_coeff=None,extinction=0.0):
        super().__init__()
        self.pdb = model if model else None
        if abs_coeff is not None:
            self.abs_coeffs = nn.parameter(torch.tensor(abs_coeff))
        else:
            self.abs_coeffs = nn.Parameter(torch.zeros(6,dtype=torch.float64,requires_grad=True))
        self.extinction_parameter = nn.Parameter(torch.tensor([extinction],dtype=torch.float64,requires_grad=True))
    
    def cuda(self):
        super().cuda()
        for res in self.residues.values():
            res.cuda()

    def load_pdb_from_file(self,file,strip_H=True):
        self.pdb = pdb_tools.load_pdb_as_pd(file)
        self.pdb = self.pdb.loc[self.pdb['element'] != 'H'] if strip_H else self.pdb
        self.cell = np.array(self.pdb.attrs['cell'])
        self.spacegroup = self.pdb.attrs['spacegroup']
        self.inv_fractional_matrix = math_np.get_inv_fractional_matrix(self.cell)
        self.fractional_matrix = math_np.get_fractional_matrix(self.cell)
        self.mean_models = []
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
        self.link_all_alternative_conformations()

    def setup_grids(self):
        self.real_space_grid = mnp.get_real_grid(self.cell,self.max_res)
        self.real_space_grid = self.real_space_grid.astype(np.float64)
        self.map = np.zeros(self.real_space_grid.shape[:-1],dtype=np.float64)

    def link_all_alternative_conformations(self):
        alternative_confomations = self.pdb.loc[self.pdb['altloc'] != '']
        for res in alternative_confomations.groupby(['resseq','chainid'],as_index=False):
            resseq, chainid = res[0]
            selection = (chainid,resseq,None)
            self.link_residues_for_selection([selection])

    def get_residues_matching_selection(self,selection):
        selected_keys = []
        for sel in selection:
            for key in self.residues:
                flag = False
                for i in range(len(sel)):
                    if sel[i] is not None:
                        if sel[i] != key[i]:
                            flag = True
                            break
                if not flag:
                    selected_keys.append(key)
        return list(set(selected_keys))

    def link_residues_for_selection(self,selection,refine_occ=True):
        keys = self.get_residues_matching_selection(selection)
        occs = torch.hstack([self.residues[key].get_occupancy() for key in keys])
        chained_occ = linked_occupancy_manager(keys,occ_start=occs)
        for key in keys:
            self.residues[key].set_occupancy(chained_occ)
            self.residues[key].refine_occ = refine_occ

    def pickle(self,filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    def update_pdb(self):
        for res in self.residues.values():
            chainid, resseq, altloc = res.id
            if altloc == None:
                altloc = ''
            if resseq == None:
                resseq = ''
            if chainid == None:
                chainid = ''
            selection = (self.pdb['resseq'] == resseq) & (self.pdb['chainid'] == chainid) & (self.pdb['altloc'] == altloc)
            self.pdb.loc[selection,['x','y','z']] = res.get_xyz().detach().cpu().numpy()
            self.pdb.loc[selection,'occupancy'] = res.get_occupancy().detach().cpu().numpy()
            if res.anisou_flag:
                self.pdb.loc[selection,['u11','u22','u33','u12','u13','u23']] = res.get_U().detach().cpu().numpy()
            else:
                self.pdb.loc[selection,'tempfactor'] = res.get_b().detach().cpu().numpy()

    def write_pdb(self,file):
        self.update_pdb()
        pdb_tools.write_file(self.pdb,file)
    
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

    def align_models(self,selections,atoms_to_include=None):
        with torch.no_grad():
            residues = self.get_residues_matching_selection(selections)
            res0 = residues[0]
            if atoms_to_include is not None:
                names = self.residues[res0].names
                atoms_to_include = tensor([i in atoms_to_include for i in names])
            aligned = []
            self.mean_res = [] 
            for res in residues:
                aligned.append(math_torch.align_torch(self.residues[res0].get_xyz(),self.residues[res].get_xyz(),atoms_to_include))
        return aligned

    def replace_copies_with_mean(self,selections,atoms_to_superpose=None):
        self.mean_models = []
        with torch.no_grad():
            residues = self.get_residues_matching_selection(selections)
            res0 = residues[0]
            aligned = []
            self.mean_res = [] 
            for res in residues:
                aligned.append(math_torch.align_torch(self.residues[res0].get_xyz(),self.residues[res].get_xyz()))
            mean_xyz = torch.mean(torch.stack(aligned),dim=0)
            if self.residues[res0].anisou_flag:
                mean_U = torch.mean(torch.stack([self.residues[res].get_U() for res in residues]),dim=0)
            mean_tempfactor = torch.mean(torch.stack([self.residues[res].get_b() for res in residues]),dim=0)
            mean_residue = residue_no_pdb(mean_xyz,(-1,-1,-1),self.spacegroup,self.cell,
                                          U=mean_U,b=mean_tempfactor,names=self.residues[res0].names,
                                          resname=self.residues[res0].resname,
                                          anisou_flag=self.residues[res0].anisou_flag,element=self.residues[res0].element)
            self.mean_models.append(mean_residue)
            for res in residues:
                resi = self.residues[res]
                alignement_matrix= math_torch.get_alignement_matrix(resi.get_xyz(),mean_residue.get_xyz())
                res_new = projected_residue(resi.id,alignement_matrix,mean_residue,resi.occ,resi.refine_occ)
                self.residues[res] = res_new
                
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
            self.hidden = nn.Parameter(self._inverse_sigmoid(1))
        else:
            self.hidden = nn.Parameter(self._inverse_sigmoid(occ_start))
        self.hidden.requires_grad = True
    
    def cuda(self):
        super().cuda()
        self.hidden.cuda()
    
    def _inverse_sigmoid(self, y):
        """Convert from [0,1] range to unconstrained parameter"""
        # Handle edge cases to avoid numerical issues
        y = torch.clamp(y, 1e-6, 1-1e-6)
        return torch.log(y / (1 - y))
    
    def _sigmoid(self, x):
        """Convert unconstrained parameter to [0,1] range"""
        return torch.sigmoid(x)
    
    def get_occupancy(self,*args):
        return self._sigmoid(self.hidden)

    def get_tensors(self):
        return self.hidden

class linked_occupancy_manager(occupancy_manager):
    def __init__(self, molecules,occ_start = None,target_occupancy=1):
        nn.Module.__init__(self)
        self.target_occupancy = target_occupancy
        self.molecules = molecules
        self.n_molecules = len(molecules)
        if occ_start is None:
            self.hidden = nn.Parameter(self._inverse_sigmoid(np.ones(self.n_molecules)/self.n_molecules))
        else:   
            try:
                self.hidden = nn.Parameter(self._inverse_sigmoid(occ_start.clone().detach()))
            except:
                self.hidden = nn.Parameter(self._inverse_sigmoid(occ_start))
        self.hidden.requires_grad = True

    def get_occupancies(self):
        occs = self._sigmoid(self.hidden)
        occs = occs / occs.sum()
        if torch.isnan(occs).any():
            print("Occ contains NaN values")
            raise ValueError("Occ contains NaN values")
        return occs

    def get_occupancy(self, id):
        idx = self.molecules.index(id)
        occupancies = self.get_occupancies()
        return occupancies[idx]
    
class residue(nn.Module):
    def __init__(self,res_pdb,id,spacegroup,cell,occ=None,refine_occ=False):
        super().__init__()
        self.res_pdb = res_pdb
        self.anisou_flag = res_pdb['anisou_flag'].values[0]
        self.spacegroup = spacegroup
        self.cell = cell
        self.refine_occ = refine_occ
        self.id = id
        self.names = res_pdb['name'].values
        self.xyz = nn.Parameter(tensor(res_pdb[['x','y','z']].values))
        self.resname = res_pdb['resname'].values[0]
        if occ is None:
            occ = tensor(self.res_pdb['occupancy'].values[0])
            self.occ = occupancy_manager(occ)
        elif isinstance(occ, occupancy_manager) or isinstance(occ, linked_occupancy_manager):
            self.occ = occ
        elif isinstance(occ, float):
            self.occ = occupancy_manager(occ_start=tensor(occ))
        else:
            raise ValueError("occupancy must be an occupancy_manager or linked_occupancy_manager instance")
        self.b = nn.Parameter(tensor(res_pdb['tempfactor'].values))
        self.u = nn.Parameter(tensor(res_pdb[['u11','u22','u33','u12','u13','u23']].values))
        self.element = res_pdb['element'].values
        self.use_anharmonic = False
        self.use_core_deformation = False
    
    def cuda(self):
        super().cuda()
        self.occ.cuda()
    
    def set_core_deformation(self, deformation_factor=0):
        """Adds a parameter to model core electron contraction"""
        self.use_core_deformation = True
        self.core_deformation = nn.Parameter(torch.tensor(float(deformation_factor)))

    def set_occupancy(self,occ):
        if isinstance(occ, occupancy_manager) or isinstance(occ, linked_occupancy_manager):
            self.occ = occ

    def set_anharmonic(self, start=None,scale=0.01):
        self.use_anharmonic = True
        if start is None:
            # Initialize with small random values instead of zeros
            self.anharmonic = nn.Parameter(torch.randn(10) * scale)
        else:
            self.anharmonic = nn.Parameter(torch.tensor(start))
        

    def get_xyz(self):
        return self.xyz
    
    def get_b(self):
        return self.b
    
    def get_U(self):
        if self.anisou_flag:
            return self.u
        else:
            raise ValueError("This residue does not have anisotropic B-factors")


    def get_names(self):
        return self.names
    
    def get_atoms(self):
        return self.element

    def get_occupancy(self):
        return self.occ.get_occupancy(self.id)

    def get_tensors_to_refine(self):
        tensors_to_return = [self.xyz]
        if self.anisou_flag:
            tensors_to_return.append(self.u)
        else:
            tensors_to_return.append(self.b)
        if self.use_core_deformation:
            tensors_to_return.append(self.core_deformation)
        if self.refine_occ:
            tensors_to_return.append(self.occ.get_tensors())
        if self.use_anharmonic:
            tensors_to_return.append(self.anharmonic)
        return tensors_to_return


class projected_residue(nn.Module):
    def __init__(self,id,transformation_matrix,projected_molecule: residue,occ=None,refine_occ=False):
        super().__init__()
        self.id = id
        self.projected_molecule = projected_molecule
        self.anisou_flag = projected_molecule.anisou_flag
        self.spacegroup = projected_molecule.spacegroup
        self.cell = projected_molecule.cell
        if isinstance(transformation_matrix, torch.Tensor):
            self.transformation_matrix = nn.Parameter(transformation_matrix)
        else:
            self.transformation_matrix = nn.Parameter(tensor(transformation_matrix))
        self.refine_occ = refine_occ
        self.use_anharmonic = False
        self.resname = projected_molecule.resname
        if occ is None:
            self.occ = occupancy_manager()
        elif isinstance(occ, occupancy_manager) or isinstance(occ, linked_occupancy_manager):
            self.occ = occ
        elif isinstance(occ, float):
            self.occ = occupancy_manager(occ_start=occ)
        else:
            raise ValueError("occupancy must be an occupancy_manager or linked_occupancy_manager instance")
        self.use_core_deformation = False
    

    def cuda(self):
        super().cuda()
        self.projected_molecule.cuda()

    def set_occupancy(self,occ):
        if isinstance(occ, occupancy_manager) or isinstance(occ, linked_occupancy_manager):
            self.occ = occ

    def set_anharmonic(self, start=None):
        self.use_anharmonic = True
        if start is None:
            self.anharmonic = nn.Parameter(torch.zeros(10))
        else:
            self.anharmonic = nn.Parameter(torch.tensor(start))

    def get_tensors(self):
        return self.transformation_matrix, self.occ.get_tensors()
    
    def get_U(self):
        return self.projected_molecule.get_U()

    def get_atoms(self):
        return self.projected_molecule.get_atoms()

    def get_occupancy(self):
        return self.occ.get_occupancy(self.id)

    def get_xyz(self):
        xyz = self.projected_molecule.get_xyz()
        return math_torch.apply_transformation(xyz,self.transformation_matrix)

    def get_names(self):
        return self.projected_molecule.get_names()
    
    def get_tensors_to_refine(self):
        if self.refine_occ:
            return self.transformation_matrix, self.occ.get_tensors()
        else:
            return self.transformation_matrix
        
class residue_no_pdb(residue):
    def __init__(self,xyz,id,spacegroup,cell,b,names,element,U=None,anisou_flag=False,resname=None,occ=None):
        nn.Module.__init__(self)
        self.res_pdb = None
        self.id = id
        self.element = element
        self.names = names
        self.use_anharmonic = False
        self.use_core_deformation = False
        if isinstance(xyz, torch.Tensor):
            self.xyz =  nn.Parameter(xyz)
        elif isinstance(xyz, np.ndarray):
            self.xyz = nn.Parameter(tensor(xyz))
        else:
            raise ValueError("xyz must be a torch.Tensor or numpy.ndarray")
        self.resname = resname
        if U is not None:
            if isinstance(U, torch.Tensor):
                self.u = nn.Parameter(U)
            elif isinstance(U, np.ndarray):
                self.u = nn.Parameter(tensor(U))
            else:
                raise ValueError("U must be a torch.Tensor or numpy.ndarray")
        if isinstance(b, torch.Tensor):
            self.b = nn.Parameter(b)
        elif isinstance(b, np.ndarray):
            self.b = nn.Parameter(tensor(b))
        else:
            raise ValueError("b must be a torch.Tensor or numpy.ndarray")
        if occ is None:
            occ = tensor(float(1))
            self.occ = occupancy_manager(occ)
        elif isinstance(occ, occupancy_manager) or isinstance(occ, linked_occupancy_manager):
            self.occ = occ
        elif isinstance(occ, float):
            self.occ = occupancy_manager(occ_start=tensor(occ))
        self.anisou_flag = anisou_flag
        self.spacegroup = spacegroup
        self.cell = cell