import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch

def get_scattering_factors_unique(atoms, s):
    scattering_table = pd.read_feather('/das/work/p17/p17490/Peter/manual_refinement/Scattering_table_as_Excel_corrected.feather')
    PSE = list(scattering_table)
    ionized_equivalents= atoms.element
    ionized_nr_equivalents = torch.tensor([PSE.index(ion) for ion in ionized_equivalents])
    ionized_nr_equivalents -= atoms.charge.values
    ionized_element_equivalents = [PSE[ion] for ion in ionized_nr_equivalents]
    unique_atoms = np.unique(ionized_element_equivalents)
    scattering_angle = scattering_table.index.values
    idxs = np.digitize(s, scattering_angle)
    idxs_lower = idxs - 1
    s_lower = torch.tensor(scattering_angle[idxs_lower])
    s_higher = torch.tensor(scattering_angle[idxs])
    atom_dict = {}
    for unique_atom in unique_atoms:
        y_lower = torch.tensor(scattering_table.iloc[idxs_lower][unique_atom].values)
        y_higher = torch.tensor(scattering_table.iloc[idxs][unique_atom].values)
        scattering_factors = linear_interpolation(s, s_lower, s_higher, y_lower, y_higher)
        atom_dict[unique_atom] = scattering_factors.reshape(-1, 1)
    return atom_dict

def get_scattering_factors(scattering_dict,elements):
    return torch.concatenate([scattering_dict[element] for element in elements],axis=1)

def linear_interpolation(x, x0, x1, y0, y1):
    return y0 + (x - x0) * (y1 - y0) / (x1 - x0)

def get_scattering_itc92(df,s):
    import gemmi
    all_atoms = df.element.values
    atoms = torch.unique(df.element)
    s_squared = ((s/4)**2).reshape(-1,1)
    elements = {}
    for element in atoms:
        SF = gemmi.Element(element).it92
        A = torch.tensor(SF.a).reshape(1,-1)
        B = torch.tensor(SF.b).reshape(1,-1)
        C = torch.tensor(SF.c).reshape(1,-1)
        f = torch.sum(A * np.exp(-B * s_squared),axis=1).reshape(-1,1)
        f += C
        elements[element] = f
    return torch.concatenate([elements[element] for element in all_atoms],axis=1)

def calc_scattering_factors_paramtetrization(parametrization, s, atom_list):
    scattering_factors = []
    for atom in atom_list:
        A,B,C = parametrization[atom]
        f = torch.sum(A * torch.exp(-B * s.reshape(-1,1)), axis=1).reshape(-1, 1)
        f += C
        scattering_factors.append(f)
    return torch.concatenate(scattering_factors, axis=1)

def get_parameterization(df):
    import gemmi
    charge_elements = []
    for i,df in df.groupby(['element','charge']):
        charge_elements.append(i)
    print('charge_elements', charge_elements)
    atoms_dict = {}
    for atom,charge in charge_elements:
        SF = gemmi.IT92_get_exact(gemmi.Element(atom), charge)
        A = torch.tensor(SF.a).reshape(1,-1)
        B = torch.tensor(SF.b).reshape(1,-1)
        C = torch.tensor(SF.c).reshape(1,-1)
        parametrization = [A,B,C]
        atoms_dict[atom] = parametrization
    return atoms_dict