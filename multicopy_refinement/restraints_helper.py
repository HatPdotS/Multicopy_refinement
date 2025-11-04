import pandas as pd
import torch
import numpy as np
from pathlib import Path

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
    """
    Split a line by whitespace, but preserve quoted strings intact.
    Handles both single and double quotes.
    """
    line_new = ''
    in_quotes = False
    quote_char = None
    for character in line:
        if (character == "'" or character == '"') and not in_quotes:
            # Starting a quoted section
            in_quotes = True
            quote_char = character
        elif character == quote_char and in_quotes:
            # Ending a quoted section
            in_quotes = False
            quote_char = None
        elif in_quotes and character == ' ':
            # Skip spaces inside quotes
            continue
        line_new += character
    return line_new.split()

def find_cif_file_in_library(resname):
    """
    Find a CIF file in the external monomer library based on residue name.
    
    The library is organized by first character (e.g., 'ALA' -> 'a/ALA.cif').
    This function works regardless of the current working directory by
    calculating the path relative to this script's location.
    
    Args:
        resname: Residue name (e.g., 'ALA', 'GLY', 'ATP')
    
    Returns:
        Path object pointing to the CIF file, or None if not found
    """
    # Get the directory containing this script
    script_dir = Path(__file__).parent
    
    # Go up one level to the repo root, then into external_monomer_library
    library_root = script_dir.parent / "external_monomer_library"
    
    # The library organizes files by first character (lowercase)
    first_char = resname[0].lower()
    
    # Construct the expected path
    cif_file = library_root / first_char / f"{resname}.cif"
    
    # Check if file exists
    if cif_file.exists():
        return cif_file
    else:
        # Try uppercase version as fallback
        cif_file_upper = library_root / first_char / f"{resname.upper()}.cif"
        if cif_file_upper.exists():
            return cif_file_upper
        return None

def read_link_definitions():
    """
    Read link definitions from mon_lib_list.cif
    
    Returns a dictionary where keys are link IDs (e.g., 'TRANS', 'CIS')
    and values are dictionaries containing:
    - 'info': Basic link information
    - 'bonds': DataFrame of inter-residue bonds
    - 'angles': DataFrame of inter-residue angles
    - 'torsions': DataFrame of inter-residue torsions
    """
    link_file_path = Path(__file__).parent.parent / "external_monomer_library" / 'list' / "mon_lib_list.cif"
    with open(link_file_path) as f:
        lines = f.readlines()
    
    link_dict = {}
    link_list = None
    
    # First, read the link list to get all link IDs
    lines_iter = iter(lines)
    for line in lines_iter:
        if line.strip() == 'data_link_list':
            # Read the link list
            for line in lines_iter:
                if line.strip() == 'loop_':
                    comp_list = []
                    values = []
                    for line in lines_iter:
                        if line.startswith('#') or (line.strip() == '' and len(values) > 0):
                            break
                        if line.startswith('_chem_link.'):
                            comp_list.append(line.split('.')[1].strip())
                        else:
                            split_items = split_respecting_quotes(line.strip())
                            if split_items:
                                values.append(split_items)
                    
                    if len(values) > 0:
                        link_list = pd.DataFrame(values, columns=comp_list)
                        print(f"Found {len(link_list)} link definitions")
                        break
                    break
            break
    
    # Now read each individual link definition
    lines_iter = iter(lines)
    for line in lines_iter:
        if line.startswith('data_link_') and line.strip() != 'data_link_list':
            link_id = line.strip().replace('data_link_', '')
            
            link_data = {}
            
            # Read bonds, angles, torsions for this link
            while True:
                line = next(lines_iter, None)
                if line is None:
                    break
                
                # Stop if we hit the next link
                if line.startswith('data_link_'):
                    break
                
                if line.strip() == 'loop_':
                    # Read the next line to see what type of data this is
                    first_col_line = next(lines_iter)
                    
                    if '_chem_link_bond.' in first_col_line:
                        section_type = 'bonds'
                        prefix = '_chem_link_bond.'
                    elif '_chem_link_angle.' in first_col_line:
                        section_type = 'angles'
                        prefix = '_chem_link_angle.'
                    elif '_chem_link_tor.' in first_col_line:
                        section_type = 'torsions'
                        prefix = '_chem_link_tor.'
                    elif '_chem_link_plane.' in first_col_line:
                        section_type = 'planes'
                        prefix = '_chem_link_plane.'
                    else:
                        # Skip unknown sections
                        continue
                    
                    # Read column names
                    columns = [first_col_line.split('.')[1].strip()]
                    for line in lines_iter:
                        if line.startswith(prefix):
                            columns.append(line.split('.')[1].strip())
                        else:
                            # First data line
                            break
                    
                    # Read data values
                    values = []
                    while line is not None:
                        if line.strip() == '' or line.startswith('loop_') or line.startswith('data_'):
                            break
                        split_items = split_respecting_quotes(line.strip())
                        if split_items and not line.startswith('_'):
                            values.append(split_items)
                        
                        line = next(lines_iter, None)
                    
                    if len(values) > 0:
                        df = pd.DataFrame(values, columns=columns)
                        link_data[section_type] = df
            
            if len(link_data) > 0:
                link_dict[link_id] = link_data
    
    return link_dict, link_list


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
    column1 = torch.tensor(np.concatenate(columns1,dtype=int))
    column2 = torch.tensor(np.concatenate(columns2,dtype=int))
    references = torch.tensor(np.concatenate(references,dtype=float))
    sigmas = torch.tensor(np.concatenate(sigmas,dtype=float))
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
    column1 = torch.tensor(np.concatenate(columns1,dtype=int))
    column2 = torch.tensor(np.concatenate(columns2,dtype=int))
    column3 = torch.tensor(np.concatenate(columns3,dtype=int))
    references = torch.tensor(np.concatenate(references,dtype=float))
    sigmas = torch.tensor(np.concatenate(sigmas,dtype=float))
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
    column1 = torch.tensor(np.concatenate(columns1,dtype=int))
    column2 = torch.tensor(np.concatenate(columns2,dtype=int))
    column3 = torch.tensor(np.concatenate(columns3,dtype=int))
    column4 = torch.tensor(np.concatenate(columns4,dtype=int))
    references = torch.tensor(np.concatenate(references,dtype=float))
    sigmas = torch.tensor(np.concatenate(sigmas,dtype=float))
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
    column1 = torch.tensor(np.concatenate(columns1,dtype=int))
    planenrs = np.concatenate(planenrs,dtype=int)
    sigmas = torch.tensor(np.concatenate(sigmas,dtype=float))
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
    distances = torch.sum((xyz[column1] - xyz[column2])**2, axis = 1) ** 0.5
    return torch.sum(torch.exp((torch.abs(distances - reference)/sigma)**2))

def calculate_restraints_angles(xyz,restraints_angles):
    column1 = restraints_angles[0]
    column2 = restraints_angles[1]
    column3 = restraints_angles[2]
    reference = restraints_angles[3]
    sigma = restraints_angles[4]
    v1 = xyz[column1] - xyz[column2]
    v2 = xyz[column3] - xyz[column2]
    v1 = v1 / torch.sum(v1**2,axis=1).reshape(-1,1) ** 0.5
    v2 = v2 / torch.sum(v2**2,axis=1).reshape(-1,1) ** 0.5
    angle = torch.arccos(torch.sum(v1*v2,axis=1)) * 180 / np.pi
    return torch.sum(torch.exp(torch.abs((angle - reference))/sigma))

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
    n1 = torch.linalg.cross(v1,v2)
    n2 = torch.linalg.cross(v2,v3)
    n1 = n1 / torch.sum(n1**2,axis=1).reshape(-1,1) ** 0.5
    n2 = n2 / torch.sum(n2**2,axis=1).reshape(-1,1) ** 0.5
    angle = torch.arccos(torch.sum(n1*n2,axis=1)) * 180 / np.pi
    dif = angle - reference
    dif = torch.min(torch.vstack((torch.abs(dif),torch.abs(dif+180),torch.abs(dif-180),torch.abs(dif-360),torch.abs(dif+360))),axis=0)[0]
    return torch.sum(torch.exp(torch.abs(dif)/sigma))

def calculate_restraints_all(xyz,restraints):
    bondlength = calculate_restraints_bondlength(xyz,restraints['bondlength'])
    angles = calculate_restraints_angles(xyz,restraints['angles'])
    torsion = calculate_restraints_torsion(xyz,restraints['torsion'])
    return bondlength + angles + torsion
    
def read_for_component(lines,comp_id):
    lines = iter(lines)
    for line in lines:
        if line.startswith('data_comp_' + comp_id):
            line = next(lines)
            dfs = {}
            while True:
                # Check if we've left this component's section
                if line.startswith('data_comp_') and not line.startswith('data_comp_' + comp_id):
                    break
                    
                if line.strip() == 'loop_':
                    line = next(lines)
                    id = line.split('.')[0].strip()
                    comp_list = [line.split('.')[1].strip()]
                    values = []
                    in_data_section = False
                    
                    for line in lines:
                        # Stop on comment line
                        if line.startswith('#'):
                            break
                        # Stop on blank line if we've already seen data
                        if in_data_section and line.strip() == '':
                            break
                        # Stop if we hit a new component section
                        if line.strip().startswith('data_comp_'):
                            break
                        # Stop if we hit a new loop - but don't consume the line!
                        if line.strip() == 'loop_':
                            # Don't break - we'll process this loop in the outer while loop
                            # But we need to exit this inner loop
                            break
                        # Collect column names for this loop
                        if line.startswith(id):
                            comp_list.append(line.split('.')[1].strip())
                        else:
                            # Only process non-empty lines as data
                            split_items = split_respecting_quotes(line.strip())
                            if split_items:
                                values.append(split_items)
                                in_data_section = True

                    try: 
                        data = pd.DataFrame(values, columns=comp_list)
                        dfs[id] = data
                        
                        # Apply esd column handling
                        esd_columns = [col for col in data if col.endswith('_esd')]
                        for col in esd_columns:
                            try:
                                data[col] = data[col].astype(float)
                                data.loc[data[col] == 0,col] = 1e-4
                            except:
                                print(f'Failed to convert esd column to float for: {col}')
                                
                    except Exception as e:
                        print(f'Failed to create dataframe for: {comp_id}')
                        print(f'Columns ({len(comp_list)}): {comp_list}')
                        print(f'Values ({len(values)} rows):')
                        for i, v in enumerate(values[:5]):  # Show first 5
                            print(f'  Row {i} ({len(v)} items): {v}')
                        if len(values) > 5:
                            print(f'  ... and {len(values)-5} more rows')
                        print(f'Error: {e}')
                    
                    # If we broke because of 'loop_', the line variable now contains 'loop_'
                    # and the while loop will process it in the next iteration
                    # If we broke for another reason, we need to read the next line
                    if line.strip() == 'loop_':
                        continue  # Don't read next line, process this loop_ in the while loop
                    
                # Read the next line for the while loop
                try:
                    line = next(lines)
                except StopIteration:
                    break
                    
            return dfs

def read_comp_list(lines):
    lines = iter(lines)
    for line in lines:
        if line.strip() == 'data_comp_list':
            for line in lines:
                if line.strip() == 'loop_':
                    comp_list = []
                    values = []
                    in_data_section = False
                    for line in lines:
                        # Stop on comment line
                        if line.startswith('#'):
                            break
                        # Stop on blank line if we've already seen data
                        if in_data_section and line.strip() == '':
                            break
                        # Stop if we hit a new section marker
                        if line.strip().startswith('data_') or line.strip().startswith('loop_'):
                            break
                        # Collect column names
                        if line.startswith('_chem_comp'):
                            comp_list.append(line.split('.')[1].strip())
                        else:
                            # Only process non-empty lines as data
                            split_items = split_respecting_quotes(line.strip())
                            if split_items:
                                values.append(split_items)
                                in_data_section = True
                    data = pd.DataFrame(values, columns=comp_list)  
                    return data
                
