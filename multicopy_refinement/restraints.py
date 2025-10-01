  import pandas as pd
import numpy as np

def read_cif(cif_path):
    with open(cif_path) as f:
        lines = f.readlines()
    components = read_comp_list(lines)
    residue_data = {}
    for comp_id in components['id']:
        data = read_for_component(lines,comp_id)
        residue_data[comp_id] = data
    return residue_data
    
def split_respecting_quotes(line):
    line_new = ''
    in_quotes = False
    for character in line:
        if character == "'" or character == '"':
            in_quotes = not in_quotes
        if in_quotes and character == ' ':
            continue
        line_new += character
    return line_new.split()

def build_restraints_bondlength(cif,pdb):
    columns1 = []
    columns2 = []
    references = []
    sigmas = []
    for chain_id in pdb['chainid'].unique():
        chain = pdb.loc[pdb['chainid'] == chain_id]
        for resseq in chain['resseq'].unique():
            residue = pdb.loc[(pdb['resseq'] == resseq) & (pdb['chainid'] == chain_id)]
            if residue.ATOM.values[0] == 'HETATM':
                continue
            resname = residue['resname'].values[0]
            if not resname in cif:
                continue
            cif_residue = cif[resname]
            cif_bonds_residue = cif_residue['_chem_comp_bond']
            usable_dict = cif_bonds_residue.loc[cif_bonds_residue['atom_id_1'].isin(residue['name']) & cif_bonds_residue['atom_id_2'].isin(residue['name'])]
            not_found = cif_bonds_residue.loc[~(cif_bonds_residue['atom_id_1'].isin(residue['name']) & cif_bonds_residue['atom_id_2'].isin(residue['name']))]
            residue.set_index('name',inplace=True)
            column1 = residue.loc[usable_dict['atom_id_1'],'index'].values
            column2 = residue.loc[usable_dict['atom_id_2'],'index'].values
            reference = usable_dict['value_dist'].values.astype(float)
            sigma = usable_dict['value_dist_esd'].values.astype(float)
            columns1.append(column1)
            columns2.append(column2)
            references.append(reference)
            sigmas.append(sigma)
    column1 = np.concatenate(columns1,dtype=int)
    column2 = np.concatenate(columns2,dtype=int)
    references = np.concatenate(references,dtype=float)
    sigmas = np.concatenate(sigmas,dtype=float)
    return [column1,column2,references,sigmas]

def build_restraints_angles(cif,pdb):
    columns1 = []
    columns2 = []
    columns3 = []
    references = []
    sigmas = []
    for chain_id in pdb['chainid'].unique():
        chain = pdb.loc[pdb['chainid'] == chain_id]
        for resseq in chain['resseq'].unique():
            residue = pdb.loc[(pdb['resseq'] == resseq) & (pdb['chainid'] == chain_id)]
            if residue.ATOM.values[0] == 'HETATM':
                continue
            resname = residue['resname'].values[0]
            if not resname in cif:
                continue
            cif_residue = cif[resname]
            cif_bonds_residue = cif_residue['_chem_comp_angle']
            usable_dict = cif_bonds_residue.loc[cif_bonds_residue['atom_id_1'].isin(residue['name']) & cif_bonds_residue['atom_id_2'].isin(residue['name'])]
            not_found = cif_bonds_residue.loc[~(cif_bonds_residue['atom_id_1'].isin(residue['name']) & cif_bonds_residue['atom_id_2'].isin(residue['name']))]
            residue.set_index('name',inplace=True)
            column1 = residue.loc[usable_dict['atom_id_1'],'index'].values
            column2 = residue.loc[usable_dict['atom_id_2'],'index'].values
            column3 = residue.loc[usable_dict['atom_id_3'],'index'].values
            reference = usable_dict['value_angle'].values.astype(float)
            sigma = usable_dict['value_angle_esd'].values.astype(float)
            columns1.append(column1)
            columns2.append(column2)
            columns3.append(column3)
            references.append(reference)
            sigmas.append(sigma)
    column1 = np.concatenate(columns1,dtype=int)
    column2 = np.concatenate(columns2,dtype=int)
    column3 = np.concatenate(columns3,dtype=int)
    references = np.concatenate(references,dtype=float)
    sigmas = np.concatenate(sigmas,dtype=float)
    return [column1,column2,column3,references,sigmas]

def build_restraints_torsion(cif,pdb):
    columns1 = []
    columns2 = []
    columns3 = []
    columns4 = []
    references = []
    sigmas = []
    for chain_id in pdb['chainid'].unique():
        chain = pdb.loc[pdb['chainid'] == chain_id]
        for resseq in chain['resseq'].unique():
            residue = pdb.loc[(pdb['resseq'] == resseq) & (pdb['chainid'] == chain_id)]
            if residue.ATOM.values[0] == 'HETATM':
                continue
            resname = residue['resname'].values[0]
            if not resname in cif:
                continue
            cif_residue = cif[resname]
            cif_bonds_residue = cif_residue['_chem_comp_tor']
            usable_dict = cif_bonds_residue.loc[cif_bonds_residue['atom_id_1'].isin(residue['name']) & cif_bonds_residue['atom_id_2'].isin(residue['name'])]
            not_found = cif_bonds_residue.loc[~(cif_bonds_residue['atom_id_1'].isin(residue['name']) & cif_bonds_residue['atom_id_2'].isin(residue['name']))]
            residue.set_index('name',inplace=True)
            column1 = residue.loc[usable_dict['atom_id_1'],'index'].values
            column2 = residue.loc[usable_dict['atom_id_2'],'index'].values
            column3 = residue.loc[usable_dict['atom_id_3'],'index'].values
            column4 = residue.loc[usable_dict['atom_id_4'],'index'].values
            reference = usable_dict['value_angle'].values.astype(float)
            sigma = usable_dict['value_angle_esd'].values.astype(float)
            columns1.append(column1)
            columns2.append(column2)
            columns3.append(column3)
            columns4.append(column4)
            references.append(reference)
            sigmas.append(sigma)
    column1 = np.concatenate(columns1,dtype=int)
    column2 = np.concatenate(columns2,dtype=int)
    column3 = np.concatenate(columns3,dtype=int)
    column4 = np.concatenate(columns4,dtype=int)
    references = np.concatenate(references,dtype=float)
    sigmas = np.concatenate(sigmas,dtype=float)
    return [column1,column2,column3,column4,references,sigmas]

def build_restraints_planes(cif,pdb):
    columns1 = []
    planenrs = []
    references = []
    sigmas = []
    last_plane = 0
    for chain_id in pdb['chainid'].unique():
        chain = pdb.loc[pdb['chainid'] == chain_id]
        for resseq in chain['resseq'].unique():
            residue = pdb.loc[(pdb['resseq'] == resseq) & (pdb['chainid'] == chain_id)]
            if residue.ATOM.values[0] == 'HETATM':
                continue
            resname = residue['resname'].values[0]
            if not resname in cif:
                continue
            cif_residue = cif[resname]
            if not '_chem_comp_plane_atom' in cif_residue:
                continue
            cif_bonds_residue = cif_residue['_chem_comp_plane_atom']
            usable_dict = cif_bonds_residue.loc[cif_bonds_residue['atom_id'].isin(residue['name']) & cif_bonds_residue['atom_id'].isin(residue['name'])]
            not_found = cif_bonds_residue.loc[~(cif_bonds_residue['atom_id'].isin(residue['name']) & cif_bonds_residue['atom_id'].isin(residue['name']))]
            residue.set_index('name',inplace=True)
            column1 = residue.loc[usable_dict['atom_id'],'index'].values
            plane_nr = usable_dict.plane_id.str.split('-').str[1].astype(int).values + last_plane
            last_plane = plane_nr.max()
            sigma = usable_dict['dist_esd'].values.astype(float)
            columns1.append(column1)
            planenrs.append(plane_nr)
            sigmas.append(sigma)
    column1 = np.concatenate(columns1,dtype=int)
    planenrs = np.concatenate(planenrs,dtype=int)
    sigmas = np.concatenate(sigmas,dtype=float)
    return [column1,planenrs,sigmas]

def build_restraints(cif,pdb):
    bondlength = build_restraints_bondlength(cif,pdb)
    angles = build_restraints_angles(cif,pdb)
    torsion = build_restraints_torsion(cif,pdb)
    planes = build_restraints_planes(cif,pdb)
    restraints = dict()
    restraints['bondlength'] = bondlength
    restraints['angles'] = angles
    restraints['torsion'] = torsion
    restraints['planes'] = planes
    return restraints

def calculate_restraints_bondlength(xyz,restraints_bondlength):
    column1 = restraints_bondlength[0]
    column2 = restraints_bondlength[1]
    reference = restraints_bondlength[2]
    sigma = restraints_bondlength[3]
    distances = np.sum((xyz[column1] - xyz[column2])**2, axis = 1) ** 0.5
    return np.sum(np.exp(np.abs(distances - reference)/sigma))

def calculate_restraints_angles(xyz,restraints_angles):
    column1 = restraints_angles[0]
    column2 = restraints_angles[1]
    column3 = restraints_angles[2]
    reference = restraints_angles[3]
    sigma = restraints_angles[4]
    v1 = xyz[column1] - xyz[column2]
    v2 = xyz[column3] - xyz[column2]
    v1 = v1 / np.sum(v1**2,axis=1).reshape(-1,1) ** 0.5
    v2 = v2 / np.sum(v2**2,axis=1).reshape(-1,1) ** 0.5
    angle = np.arccos(np.sum(v1*v2,axis=1)) * 180 / np.pi
    return np.sum(np.exp(np.abs((angle - reference))/sigma))

def calculate_restraints_torsion(xyz,restraints_torsion):
    column1 = restraints_torsion[0]
    column2 = restraints_torsion[1]
    column3 = restraints_torsion[2]
    column4 = restraints_torsion[3]
    reference = restraints_torsion[4]
    sigma = restraints_torsion[5]
    v1 = xyz[column1] - xyz[column2]
    v2 = xyz[column3] - xyz[column2]
    v3 = xyz[column3] - xyz[column4]
    n1 = np.cross(v1,v2)
    n2 = np.cross(v2,v3)
    n1 = n1 / np.linalg.norm(n1,axis=1).reshape(-1,1)
    n2 = n2 / np.linalg.norm(n2,axis=1).reshape(-1,1)
    angle = np.arccos(np.sum(n1*n2,axis=1)) * 180 / np.pi
    dif = angle - reference
    dif = np.min(np.vstack((np.abs(dif),np.abs(dif+180),np.abs(dif-180),np.abs(dif-360),np.abs(dif+360))),axis=0)
    return np.sum(np.exp(np.abs(dif)/sigma))

def calculate_restraints_all(xyz,restraints):
    bondlength = calculate_restraints_bondlength(xyz,restraints['bondlength'])
    angles = calculate_restraints_angles(xyz,restraints['angles'])
    torsion = calculate_restraints_torsion(xyz,restraints['torsion'])
    return bondlength , angles , torsion
    
def read_for_component(lines,comp_id):
    lines = iter(lines)
    for line in lines:
        if line.startswith('data_comp_' + comp_id):
            line = next(lines)
            dfs = {}
            for line in lines:
                if line.startswith('data_comp_') and not line.startswith('data_comp_' + comp_id):
                    break
                if line.strip() == 'loop_':
                    line = next(lines)
                    id = line.split('.')[0].strip()
                    comp_list = [line.split('.')[1].strip()]
                    values = []
                    for line in lines:
                        if line.startswith('#'):
                            break
                        if line.startswith(id):
                            comp_list.append(line.split('.')[1].strip())
                        else:
                            split_items = split_respecting_quotes(line)
                            values.append(split_items)
                    data = pd.DataFrame(values, columns=comp_list)  
                    esd_columns = [col for col in data if col.endswith('_esd')]
                    for col in esd_columns:
                        try:
                            data[col] = data[col].astype(float)
                            data.loc[data[col] == 0,col] = 1e-4
                        except:
                            print('Failed to convert esd column to float for:',col)
                    dfs[id] = data
            return dfs

def read_comp_list(lines):
    lines = iter(lines)
    for line in lines:
        if line.strip() == 'data_comp_list':
            for line in lines:
                if line.strip() == 'loop_':
                    comp_list = []
                    values = []
                    for line in lines:
                        if line.startswith('#'):
                            break
                        if line.startswith('_chem_comp'):
                            comp_list.append(line.split('.')[1].strip())
                        else:
                            split_items = split_respecting_quotes(line)
                            values.append(split_items)
                    data = pd.DataFrame(values, columns=comp_list)  
                    return data
                
