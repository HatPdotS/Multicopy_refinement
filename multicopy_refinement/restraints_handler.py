from multicopy_refinement.restraints_torch import read_cif
import torch
import numpy as np

class restraints:
    def __init__(self, cif_path):
        self.cif_path = cif_path
        self.cif_dict = read_cif(cif_path)
        self.nested_keys = list(set([key for i in self.cif_dict.values() for key in i.keys()]))
        self.angle_key = [key for key in self.nested_keys if 'angle' in key]
        self.angle_key = self.angle_key[0] if len(self.angle_key) > 0 else None
        self.bond_len_key = [key for key in self.nested_keys if 'bond' in key]
        self.bond_len_key =self.bond_len_key[0] if len(self.bond_len_key) > 0 else None
        self.torsion_key = [key for key in self.nested_keys if '_tor' in key]
        self.torsion_key = self.torsion_key[0] if len(self.torsion_key) > 0 else None
        self.plane_key = [key for key in self.nested_keys if 'plane' in key]
        self.plane_key = self.plane_key[0] if len(self.plane_key) > 0 else None
    
    def get_deviations(self,residue):
        xyz = residue.xyz
        names = residue.names
        id = residue.resname
        bond_len = self.get_sigma_bond_length(xyz,names,id)
        angle = self.get_sigma_angles(xyz,names,id)
        torsion = self.get_sigma_torsion(xyz,names,id)
        planes = self.get_sigma_planes(xyz,names,id)
        tensors_to_concat = [t for t in [bond_len, angle, torsion, planes] if t is not None and t.numel() > 0]
        if tensors_to_concat:
            loss = torch.cat(tensors_to_concat, dim=0)
        else:
            loss = None
        return loss

    def get_sigma_bond_length(self,xyz,names,id):
        try:
            bond_len_dic = self.cif_dict[id][self.bond_len_key]
        except: return None
        to_include = bond_len_dic['atom_id_1'].isin(names) & bond_len_dic['atom_id_2'].isin(names)
        bond_len_dic = bond_len_dic[to_include]
        key_value = [key for key in bond_len_dic.keys() if 'value' in key][0]
        esd_value = [key for key in bond_len_dic.keys() if 'esd' in key][0]
        indices1 = self.get_name_indices(bond_len_dic['atom_id_1'],names)
        indices2 = self.get_name_indices(bond_len_dic['atom_id_2'],names)
        xyz1 = xyz[indices1]
        xyz2 = xyz[indices2]
        bond_lengths = torch.sqrt(torch.sum((xyz1 - xyz2) **2,axis=1))
        bond_lengths_diff = (bond_lengths - torch.tensor(bond_len_dic[key_value].values.astype(np.float64))) / torch.tensor(bond_len_dic[esd_value].values.astype(np.float64))
        return bond_lengths_diff

    def get_sigma_angles(self,xyz,names,id):
        try:
            angle_dic = self.cif_dict[id][self.angle_key]
        except: return None
        to_include = angle_dic['atom_id_1'].isin(names) & angle_dic['atom_id_2'].isin(names) & angle_dic['atom_id_3'].isin(names)
        angle_dic = angle_dic[to_include]
        key_value = [key for key in angle_dic.keys() if 'value' in key][0]
        esd_value = [key for key in angle_dic.keys() if 'esd' in key][0]
        indices1 = self.get_name_indices(angle_dic['atom_id_1'],names)
        indices2 = self.get_name_indices(angle_dic['atom_id_2'],names)  
        indices3 = self.get_name_indices(angle_dic['atom_id_3'],names)
        xyz1 = xyz[indices1]
        xyz2 = xyz[indices2]
        xyz3 = xyz[indices3]
        v1 = xyz2 - xyz1
        v2 = xyz3 - xyz2
        v1 = v1 / torch.sum(v1**2,axis=1).reshape(-1,1) ** 0.5
        v2 = v2 / torch.sum(v2**2,axis=1).reshape(-1,1) ** 0.5
        angle = torch.arccos(torch.sum(v1*v2,axis=1)) * 180 / np.pi
        angle_diff = torch.abs((angle - torch.tensor(angle_dic[key_value].values.astype(np.float64))))/torch.tensor(angle_dic[esd_value].values.astype(np.float64))
        return angle_diff
    
    def get_sigma_torsion(self,xyz,names,id):
        try:
            torsion_dic = self.cif_dict[id][self.torsion_key]
        except: return None
        try:
            to_include = torsion_dic['atom_id_1'].isin(names) & torsion_dic['atom_id_2'].isin(names) & torsion_dic['atom_id_3'].isin(names) & torsion_dic['atom_id_4'].isin(names)
        except:
            print(torsion_dic)
            print(torsion_dic.keys())
            raise ValueError('Error in torsion dictionary')
        torsion_dic = torsion_dic[to_include]
        key_value = [key for key in torsion_dic.keys() if 'value' in key][0]
        esd_value = [key for key in torsion_dic.keys() if 'esd' in key][0]
        indices1 = self.get_name_indices(torsion_dic['atom_id_1'],names)
        indices2 = self.get_name_indices(torsion_dic['atom_id_2'],names)
        indices3 = self.get_name_indices(torsion_dic['atom_id_3'],names)
        indices4 = self.get_name_indices(torsion_dic['atom_id_4'],names)
        xyz1 = xyz[indices1]
        xyz2 = xyz[indices2]
        xyz3 = xyz[indices3]
        xyz4 = xyz[indices4]
        v1 = xyz2 - xyz1
        v2 = xyz3 - xyz2
        v3 = xyz4 - xyz3
        n1 = torch.linalg.cross(v1,v2)
        n2 = torch.linalg.cross(v2,v3)
        n1 = n1 / torch.sum(n1**2,axis=1).reshape(-1,1) ** 0.5
        n2 = n2 / torch.sum(n2**2,axis=1).reshape(-1,1) ** 0.5
        angle = torch.arccos(torch.sum(n1*n2,axis=1)) * 180 / np.pi
        dif = angle - torch.tensor(torsion_dic[key_value].values.astype(np.float64))
        dif = torch.min(torch.vstack((torch.abs(dif),torch.abs(dif+180),torch.abs(dif-180),torch.abs(dif-360),torch.abs(dif+360))),axis=0)[0]
        dif = torch.abs(dif)/torch.tensor(torsion_dic[esd_value].values.astype(np.float64))
        return dif
    
    def calculate_plane_distances(self,points):
        # 1. Calculate centroid
        centroid = torch.mean(points, dim=0)
        
        # 2. Center the points
        centered_points = points - centroid
        
        # 3. Find best-fit plane using SVD
        u, s, vh = torch.linalg.svd(centered_points)
        
        # The normal vector is the last row of vh
        normal = vh[-1]
        
        # Normalize the normal vector
        normal = normal / torch.norm(normal)
        
        # 4. Calculate the distance of each point from the plane
        distances = torch.abs(torch.matmul(centered_points, normal))
        
        return distances

    def get_sigma_planes(self,xyz,names,id):
        deviations = []
        try:
            plane_dic = self.cif_dict[id][self.plane_key]
        except: return None
        for id,plane in plane_dic.groupby('plane_id'):
            esd_value = [key for key in plane.keys() if 'esd' in key][0]
            to_include = plane['atom_id'].isin(names) 
            plane = plane[to_include]
            if len(plane) < 3: continue
            idx = self.get_name_indices(plane['atom_id'],names)
            points = xyz[idx]
            distances = self.calculate_plane_distances(points)
            distances = torch.abs(distances) / torch.tensor(plane[esd_value].values.astype(np.float64))
            deviations.append(distances)
        deviations = torch.cat(deviations,dim=0)
        return deviations
    
    def get_name_indices(self,atoms,names):
        idx_dict = {}
        for i, name in enumerate(names):
            idx_dict[name] = i
        indices = [idx_dict[atom] for atom in atoms if atom in idx_dict]
        return indices