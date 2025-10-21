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
import multicopy_refinement.get_scattering_factor_torch as gsf
import multicopy_refinement.symmetrie as sym

class model(nn.Module):
    def __init__(self,model=None,abs_coeff=None,extinction=0.0):
        super().__init__()
        self.pdb = model if model else None
        if abs_coeff is not None:
            self.abs_coeffs = nn.parameter(torch.tensor(abs_coeff))
        else:
            self.abs_coeffs = nn.Parameter(torch.zeros(6,dtype=torch.float64,requires_grad=True))
        self.extinction_parameter = nn.Parameter(torch.tensor([extinction],dtype=torch.float64,requires_grad=True))
        self.scale = nn.Parameter(torch.tensor([1.0],dtype=torch.float64,requires_grad=True))
        self.mean_models = []
        self.global_corrections = []
        self.correction_parameters = [self.scale]
        self.use_structure_factor_fast = True
        self._cached_iso_xyz = self._cached_iso_b = None
        self._cached_aniso_xyz = self._cached_aniso_U = self._cached_aniso_scattering = None
    
    def cache_isotropic_residues(self):
        """Cache xyz, b, occ for all isotropic residues (not special/projected)."""
        self._iso_cache_map = []
        xyzs, bs, occs,scattering = [], [], [], []
        start = 0
        for key, res in self.residues.items():
            if res.anisou_flag or res.corrections or isinstance(res, projected_residue):
                continue
            n_atoms = res.get_xyz().shape[0]
            self._iso_cache_map.append((key, start, start + n_atoms))
            xyzs.append(res.get_xyz())
            bs.append(res.get_b())
            occs.append(res.get_occupancy().expand(n_atoms))
            scattering.append(res.scattering_factors)
            res._use_iso_cache = True
            res._iso_cache_indices = (start, start + n_atoms)
            res._parent_model = self  # Store reference to parent model
            start += n_atoms
        if xyzs:
            self._cached_iso_xyz = nn.Parameter(torch.cat(xyzs, dim=0).clone(), requires_grad=True)
            self._cached_iso_b = nn.Parameter(torch.cat(bs, dim=0).clone(), requires_grad=True)
            self._cached_iso_scattering = nn.Parameter(torch.cat(scattering, dim=1).clone(), requires_grad=True)
            print('cached isotropic scattering factors:', self._cached_iso_scattering)
        else:
            self._cached_iso_xyz = self._cached_iso_b = self._cached_iso_scattering = None

    def cache_anisotropic_residues(self):
        """Cache xyz, U, occ for all anisotropic residues (not special/projected)."""
        self._aniso_cache_map = []
        xyzs, Us, occs,scattering = [], [], [], []
        start = 0
        for key, res in self.residues.items():
            if not res.anisou_flag or res.corrections or isinstance(res, projected_residue):
                continue
            n_atoms = res.get_xyz().shape[0]
            self._aniso_cache_map.append((key, start, start + n_atoms))
            xyzs.append(res.get_xyz())
            Us.append(res.get_U())
            scattering.append(res.scattering_factors)
            res._use_aniso_cache = True
            res._aniso_cache_indices = (start, start + n_atoms)
            res._parent_model = self  # Store reference to parent model
            start += n_atoms
        if xyzs:
            self._cached_aniso_xyz = nn.Parameter(torch.cat(xyzs, dim=0).clone(), requires_grad=True)
            self._cached_aniso_U = nn.Parameter(torch.cat(Us, dim=0).clone(), requires_grad=True)
            self._cached_aniso_scattering = nn.Parameter(torch.cat(scattering, dim=1).clone(), requires_grad=True)
        else:
            self._cached_aniso_xyz = self._cached_aniso_U = self._cached_aniso_scattering = None

    def map_iso_cache_back(self):
        """Map cached isotropic values back to residues."""
        if not hasattr(self, "_iso_cache_map") or self._cached_iso_xyz is None:
            return
        for key, start, end in self._iso_cache_map:
            res = self.residues[key]
            res.xyz.data = self._cached_iso_xyz.data[start:end]
            res.b.data = self._cached_iso_b.data[start:end]
            res._use_iso_cache = False
        self._cached_iso_xyz = self._cached_iso_b = self._cached_iso_occ = self._cached_iso_scattering = None
        
    def map_aniso_cache_back(self):
        """Map cached anisotropic values back to residues."""
        if not hasattr(self, "_aniso_cache_map") or self._cached_aniso_xyz is None:
            return
        for key, start, end in self._aniso_cache_map:
            res = self.residues[key]
            res.xyz.data = self._cached_aniso_xyz.data[start:end]
            res.u.data = self._cached_aniso_U.data[start:end]
            res._use_aniso_cache = False
        self._cached_aniso_xyz = self._cached_aniso_U = self._cached_aniso_occ = self._cached_aniso_scattering = None
    
    def clear_caches(self):
        self.map_iso_cache_back()
        self.map_aniso_cache_back()
        
    def absorption_correction(self):
        s_norm = self.scattering_vectors / torch.norm(self.scattering_vectors**2)
        Y00 = torch.ones_like(self.scattering_vectors[:,0])
        Y10 = s_norm[:,2]  # z
        Y11 = s_norm[:,0]  # x
        Y12 = s_norm[:,1]  # y
        Y20 = 1.5 * s_norm[:,2]**2 - 0.5
        Y21 = s_norm[:,0] * s_norm[:,2]
        # Apply correction - using exponential to ensure positive values
        harmonic_sum = (self.abs_coeffs[0] * Y00 + 
                        self.abs_coeffs[1] * Y10 + 
                        self.abs_coeffs[2] * Y11 + 
                        self.abs_coeffs[3] * Y12 + 
                        self.abs_coeffs[4] * Y20 + 
                        self.abs_coeffs[5] * Y21)
        absorption = torch.exp(-harmonic_sum)
        return absorption

    def cuda(self):
        super().cuda()
        self.spacegroup_function = self.spacegroup_function.cuda()
        for res in self.residues.values():
            res.cuda()
    
    def cpu(self):
        super().cpu()
        for res in self.residues.values():
            res.cpu()
    
    def realize_residues(self):
        for key,res in self.residues.items():
            if isinstance(res, projected_residue):
                self.residues[key] = res.convert_to_normal_res()
        self.mean_models = []
    
    def get_structure_factor_not_corrected(self,hkl,scattering_vectors,s):
        """Calculate structure factors for all residues in the model"""
        self.s = s
        self.hkl = hkl
        self.scattering_vectors = scattering_vectors
        f = []
        for res in self.residues.values():
            f.append(res.get_structure_factor(hkl,scattering_vectors,s))
        fs = torch.stack(f,dim=0)
        self.f_calc = torch.sum(fs,dim=0)
        return self.f_calc
    
    def try_deepcopy_model(model):
        import copy
        """Try to create a deep copy using Python's built-in deepcopy"""
        try:
            # Simple attempt with standard library
            new_model = copy.deepcopy(model)
            
            # Test that all tensors are actually independent
            for param_name, param in model.named_parameters():
                new_param = dict(new_model.named_parameters())[param_name]
                
                # Verify independence by modifying original
                old_value = param.data.clone()
                param.data += 1.0
                
                # Check if the new one remained unchanged
                if torch.allclose(new_param.data, param.data):
                    print(f"Warning: Parameter {param_name} was not independent after deepcopy")
                    # Restore original value
                    param.data = old_value
            
            return new_model
        
        except Exception as e:
            print(f"Deepcopy failed: {e}")
            return None

    def load_pdb_from_file(self,file,strip_H=True):
        self.pdb = pdb_tools.load_pdb_as_pd(file)
        self.pdb = self.pdb.loc[self.pdb['element'] != 'H'] if strip_H else self.pdb
        self.cell = np.array(self.pdb.attrs['cell'])
        self.spacegroup = self.pdb.attrs['spacegroup']
        self.spacegroup_function = sym.Symmetry(self.spacegroup)
        self.inv_fractional_matrix = nn.Parameter(torch.tensor(math_np.get_inv_fractional_matrix(self.cell)))
        self.fractional_matrix = nn.Parameter(torch.tensor(math_np.get_fractional_matrix(self.cell)))
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
            occ = res.occupancy.values
            if np.all(occ < 1):
                refine_occ = True
            else:
                refine_occ = False
            id = (chainid,resseq,altloc)
            self.residues[id] = residue(res,id,self.spacegroup,self.cell,occ=occ,refine_occ=refine_occ)
        # self.link_all_alternative_conformations()

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
        keys = [key for key in keys if not key[-1] == None]
        occs = torch.cat([torch.mean(self.residues[key].get_occupancy()).unsqueeze(0) for key in keys])
        natoms = [self.residues[key].n_atoms for key in keys]
        chained_occ = linked_occupancy_manager(keys,occ_start=occs,natoms=natoms,target_occupancy=1)
        for i,key in enumerate(keys):
            self.residues[key].set_occupancy(chained_occ)
            self.residues[key].refine_occ = refine_occ
            self.residues[key].split_residue = True

    def pickle(self,filename):
        self.map_aniso_cache_back()
        self.map_iso_cache_back()
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    def update_pdb(self):
        self.map_aniso_cache_back()
        self.map_iso_cache_back()
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
                                          resname=self.residues[res0].resname,standard_pdb=self.residues[res0].res_pdb,
                                          anisou_flag=self.residues[res0].anisou_flag,element=self.residues[res0].element)
            self.mean_models.append(mean_residue)
            for res in residues:
                resi = self.residues[res]
                alignement_matrix= math_torch.get_alignement_matrix(resi.get_xyz(),mean_residue.get_xyz())
                res_new = projected_residue(resi.id,alignement_matrix,mean_residue,resi.occ,resi.refine_occ)
                self.residues[res] = res_new
    
    def write_out_mean_residue(self,filename):
        for i,mean_res in enumerate(self.mean_models):
            pdb = mean_res.get_pdb()
            outname = filename.replace('.pdb',f'_{i}.pdb')
            pdb_tools.write_file(pdb,outname)
                
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


class uniform_occupancy_manager(nn.Module):
    """
    Occupancy manager for residues where all atoms have the same occupancy.
    Stores a single scalar parameter for efficiency.
    """
    def __init__(self, occ_start = None, n_atoms=None):
        super().__init__()
        self.n_atoms = n_atoms  # Store number of atoms for expansion
        if occ_start is None:
            self.hidden = nn.Parameter(self._inverse_sigmoid(tensor(1.0)))
        else:
            if not isinstance(occ_start, torch.Tensor):
                occ_start = tensor(occ_start)
            self.hidden = nn.Parameter(self._inverse_sigmoid(occ_start))
        self.hidden.requires_grad = True
    
    def cuda(self):
        super().cuda()
    
    def cpu(self):
        super().cpu()

    def _inverse_sigmoid(self, y):
        """Convert from [0,1] range to unconstrained parameter"""
        # Handle edge cases to avoid numerical issues
        y = torch.clamp(y, 1e-6, 1-1e-6)
        return torch.log(y / (1 - y))
    
    def _sigmoid(self, x):
        """Convert unconstrained parameter to [0,1] range"""
        return torch.sigmoid(x)
    
    def get_occupancy(self, *args):
        """Returns expanded tensor of shape (n_atoms,) with same occupancy for all atoms"""
        occ_scalar = self._sigmoid(self.hidden)
        return occ_scalar.unsqueeze(0).expand(self.n_atoms)



    def get_tensors(self):
        return self.hidden


class per_atom_occupancy_manager(nn.Module):
    """
    Occupancy manager for residues with different occupancies per atom.
    Stores a tensor of shape (n_atoms,) with individual occupancies.
    """
    def __init__(self, occ_values):
        super().__init__()
        if not isinstance(occ_values, torch.Tensor):
            occ_values = tensor(occ_values)
        # Store per-atom occupancies as constrained parameters
        self.hidden = nn.Parameter(self._inverse_sigmoid(occ_values))
        self.hidden.requires_grad = True
    
    def cuda(self):
        super().cuda()
    
    def cpu(self):
        super().cpu()

    def _inverse_sigmoid(self, y):
        """Convert from [0,1] range to unconstrained parameter"""
        # Handle edge cases to avoid numerical issues
        y = torch.clamp(y, 1e-6, 1-1e-6)
        return torch.log(y / (1 - y))
    
    def _sigmoid(self, x):
        """Convert unconstrained parameter to [0,1] range"""
        return torch.sigmoid(x)
    
    def get_occupancy(self, *args):
        """Returns tensor of shape (n_atoms,) with individual occupancies"""
        return self._sigmoid(self.hidden)

    def get_tensors(self):
        return self.hidden


# Legacy alias for backward compatibility
occupancy_manager = uniform_occupancy_manager

class linked_occupancy_manager(occupancy_manager):
    def __init__(self, molecules,occ_start = None,target_occupancy=1,natoms=None):
        nn.Module.__init__(self)
        self.target_occupancy = target_occupancy
        self.molecules = molecules
        self.n_molecules = len(molecules)
        self.natoms = natoms
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
        n_atoms = self.natoms[idx]
        occupancies = self.get_occupancies()
        occupancy = occupancies[idx]
        if n_atoms > 1:
            return occupancy.unsqueeze(0).expand(n_atoms)
        return occupancy.unsqueeze(0)

    
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
        self.n_atoms = len(self.xyz)
        self.split_residue = False
        
        if not occ is None:
            if np.allclose(occ[0], occ):
                self.occ = uniform_occupancy_manager(occ_start=occ[0], n_atoms=self.n_atoms)
            else:
                self.occ = per_atom_occupancy_manager(occ_values=occ)
        else:
            self.occ = torch.ones(self.n_atoms,dtype=torch.float64,requires_grad=False)
        
        self.b = nn.Parameter(tensor(res_pdb['tempfactor'].values))
        self.u = nn.Parameter(tensor(res_pdb[['u11','u22','u33','u12','u13','u23']].values))
        self.element = res_pdb['element'].values
        self.use_anharmonic = False
        self.use_core_deformation = False
        if self.anisou_flag:
            self.structure_factor_function = self.anisotropic_structure_factor
        else:
            self.structure_factor_function = self.iso_structure_factor
        B_inv = math_np.get_inv_fractional_matrix(self.cell)
        self.B_inv = nn.Parameter(torch.tensor(B_inv))
        self.corrections = []
        self.spacegroup_function = sym.Symmetry(self.spacegroup)


    def update_pdb(self):
        self.res_pdb[['x','y','z']] = self.get_xyz().detach().cpu().numpy()
        self.res_pdb['occupancy'] = self.get_occupancy().detach().cpu().numpy()
        if self.anisou_flag:
            self.res_pdb[['u11','u22','u33','u12','u13','u23']] = self.get_U().detach().cpu().numpy()
        else:
            self.res_pdb['tempfactor'] = self.get_b().detach().cpu().numpy()

    def get_scattering_factor_parametrizatian(self,parametrization, s):
        self.scattering_factors = nn.Parameter(gsf.calc_scattering_factors_paramtetrization(parametrization, s, self.get_atoms()))
    
    def get_scattering_factor(self,unique):
        self.scattering_factors = nn.Parameter(gsf.get_scattering_factors(unique, self.get_atoms()))

    def get_scattering_parameters(self,):
        return
    
    def anisotropic_structure_factor(self):
        U = self.get_U()
        xyz = self.cartesian_to_fractional()
        occ = self.get_occupancy()
        return math_torch.aniso_structure_factor_torched(self.hkl,self.scattering_vectors,xyz,
                                                         occ,self.scattering_factors,U,
                                                         self.spacegroup_function)

    def iso_structure_factor(self):
        b = self.get_b()
        xyz = self.cartesian_to_fractional()
        occ = self.get_occupancy()
        return math_torch.iso_structure_factor_torched(self.hkl,self.s,xyz,occ,self.scattering_factors,b,self.spacegroup_function)
    
    def get_structure_factor(self,hkl,scattering_vectors,s):
        self.hkl = hkl
        self.scattering_vectors = scattering_vectors
        self.s = s
        f = self.structure_factor_function()
        for correction in self.corrections:
            f = correction(f)
        return f
    
    def anharmonic_correction(self,fcalc):
        return fcalc * math_torch.anharmonic_correction(self.hkl,self.anharmonic)

    def core_deformation_function(self,fcalc):
        return fcalc * math_torch.core_deformation(self.core_deformation,self.s)

    def cuda(self):
        super().cuda()
        self.spacegroup_function = self.spacegroup_function.cuda()
        self.occ.cuda()
    
    def cartesian_to_fractional(self):
        coords = torch.matmul(self.get_xyz(),self.B_inv.T)
        return coords
    
    def set_core_deformation(self, deformation_factor=0):
        """Adds a parameter to model core electron contraction"""
        self.use_core_deformation = True
        self.corrections.append(self.core_deformation_function)
        self.core_deformation = nn.Parameter(torch.tensor(float(deformation_factor)))

    def set_occupancy(self,occ):
        if isinstance(occ, occupancy_manager) or isinstance(occ, linked_occupancy_manager):
            self.occ = occ

    def set_anharmonic(self, start=None,scale=0.01):
        self.use_anharmonic = True
        self.corrections.append(self.anharmonic_correction)
        if start is None:
            # Initialize with small random values instead of zeros
            self.anharmonic = nn.Parameter(torch.randn(10) * scale)
        else:
            self.anharmonic = nn.Parameter(torch.tensor(start))

    def get_xyz(self):
        if hasattr(self, "_use_iso_cache") and self._use_iso_cache:
            start, end = self._iso_cache_indices
            return self._parent_model._cached_iso_xyz[start:end]
        if hasattr(self, "_use_aniso_cache") and self._use_aniso_cache:
            start, end = self._aniso_cache_indices
            return self._parent_model._cached_aniso_xyz[start:end]
        return self.xyz

    def get_b(self):
        if hasattr(self, "_use_iso_cache") and self._use_iso_cache:
            start, end = self._iso_cache_indices
            return self._parent_model._cached_iso_b[start:end]
        return self.b

    def get_U(self):
        if hasattr(self, "_use_aniso_cache") and self._use_aniso_cache:
            start, end = self._aniso_cache_indices
            return self._parent_model._cached_aniso_U[start:end]
        return self.u

    def get_occupancy(self):
        return self.occ.get_occupancy(self.id)

    def get_names(self):
        return self.names
    
    def get_atoms(self):
        return self.element

    def get_tensors_to_refine(self):
        tensors_to_return = []
        # Use cached tensors if available
        if hasattr(self, "_use_iso_cache") and self._use_iso_cache:
            # print("Using isotropic cache for residue", self.id)
            tensors_to_return.append(self._parent_model._cached_iso_xyz)
            tensors_to_return.append(self._parent_model._cached_iso_b)
        elif hasattr(self, "_use_aniso_cache") and self._use_aniso_cache:
            # print("Using anisotropic cache for residue", self.id)
            tensors_to_return.append(self._parent_model._cached_aniso_xyz)
            tensors_to_return.append(self._parent_model._cached_aniso_U)
        else:
            # print("Using normal tensors for residue", self.id)
            tensors_to_return.append(self.xyz)
            if self.anisou_flag:
                tensors_to_return.append(self.u)
            else:
                tensors_to_return.append(self.b)
            if self.use_core_deformation:
                tensors_to_return.append(self.core_deformation)

            if self.use_anharmonic:
                tensors_to_return.append(self.anharmonic)
        if self.refine_occ:
            tensors_to_return.append(self.occ.get_tensors())
        return tensors_to_return

    def get_scattering_factor_parametrizatian(self,parametrization, s):
        self.scattering_factors = nn.Parameter(gsf.calc_scattering_factors_paramtetrization(parametrization, s, self.get_atoms()))
    
    def get_scattering_factor(self,unique):
        self.scattering_factors = nn.Parameter(gsf.get_scattering_factors(unique, self.get_atoms()))

class projected_residue(residue):
    def __init__(self,id,transformation_matrix,projected_molecule: residue,occ=None,refine_occ=False):
        nn.Module.__init__(self)
        self.id = id
        self.projected_molecule = projected_molecule
        self.anisou_flag = projected_molecule.anisou_flag
        self.spacegroup = projected_molecule.spacegroup
        self.cell = projected_molecule.cell
        self.res_pdb = projected_molecule.res_pdb
        if isinstance(transformation_matrix, torch.Tensor):
            self.transformation_matrix = nn.Parameter(transformation_matrix)
        else:
            self.transformation_matrix = nn.Parameter(tensor(transformation_matrix))
        self.refine_occ = refine_occ
        self.use_anharmonic = False
        self.resname = projected_molecule.resname
        if isinstance(occ, occupancy_manager) or isinstance(occ, linked_occupancy_manager):
            self.occ = occ
        elif isinstance(occ, float):
            self.occ = occupancy_manager(occ_start=occ)
        
        self.use_core_deformation = False
        if self.anisou_flag:
            self.structure_factor_function = self.anisotropic_structure_factor
        else:
            self.structure_factor_function = self.iso_structure_factor
        B_inv = math_np.get_inv_fractional_matrix(self.cell)
        self.B_inv = nn.Parameter(torch.tensor(B_inv))
        self.corrections = []
        self.spacegroup_function = sym.get_space_group_function(self.spacegroup)
        self.refine_orientation = True
    
    def convert_to_normal_res(self):
        if not hasattr(self,'res_pdb'):
            self.res_pdb = self.projected_molecule.res_pdb
        self.update_pdb()
        return residue(self.res_pdb,self.id,self.spacegroup,self.cell,occ=self.occ,
                        refine_occ=self.refine_occ)

    
    def get_U(self):
        return self.projected_molecule.get_U()

    def get_atoms(self):
        return self.projected_molecule.get_atoms()

    def get_occupancy(self):
        return self.occ.get_occupancy(self.id)

    def get_xyz(self):
        xyz = self.projected_molecule.get_xyz()
        return math_torch.apply_transformation(xyz,self.transformation_matrix)
    
    def get_b(self):
        return self.projected_molecule.get_b()

    def get_names(self):
        return self.projected_molecule.get_names()
    
    def get_tensors_to_refine(self):
        base_tensors = [self.projected_molecule.xyz]
        if self.refine_orientation:
            base_tensors.append(self.transformation_matrix)
        if self.anisou_flag:
            base_tensors.append(self.projected_molecule.u)
        else:
            base_tensors.append(self.projected_molecule.b)
        if self.refine_occ:
            base_tensors.append(self.occ.get_tensors())
        return base_tensors
      
class residue_no_pdb(residue):
    def __init__(self,xyz,id,spacegroup,cell,b,names,element,standard_pdb,U=None,anisou_flag=False,resname=None,occ=None):
        nn.Module.__init__(self)
        self.res_pdb = standard_pdb
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

    def get_pdb(self):
        self.res_pdb[['x','y','z']] = self.xyz.detach().cpu().numpy()
        self.res_pdb['occupancy'] = self.occ.get_occupancy().detach().cpu().numpy()
        if self.anisou_flag:
            self.res_pdb[['u11','u22','u33','u12','u13','u23']] = self.u.detach().cpu().numpy()
        else:
            self.res_pdb['tempfactor'] = self.b.detach().cpu().numpy()
        return self.res_pdb