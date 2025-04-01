import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def get_scattering_factors(atoms, s):
    scattering_table = pd.read_feather('/das/work/p17/p17490/Peter/manual_refinement/Scattering_table_as_Excel_corrected.feather')
    PSE = list(scattering_table)
    ionized_equivalents= atoms.element
    ionized_nr_equivalents = np.array([PSE.index(ion) for ion in ionized_equivalents])
    ionized_nr_equivalents -= atoms.charge.values
    ionized_element_equivalents = np.array([PSE[ion] for ion in ionized_nr_equivalents])
    unique_atoms = np.unique(ionized_element_equivalents)
    scattering_angle = scattering_table.index.values
    idxs = np.digitize(s, scattering_angle)
    idxs_lower = idxs - 1
    s_lower = scattering_angle[idxs_lower]
    s_higher = scattering_angle[idxs]
    atom_dict = {}
    for unique_atom in unique_atoms:
        y_lower = scattering_table.iloc[idxs_lower][unique_atom].values
        y_higher = scattering_table.iloc[idxs][unique_atom].values
        scattering_factors = linear_interpolation(s, s_lower, s_higher, y_lower, y_higher)
        atom_dict[unique_atom] = scattering_factors.reshape(-1, 1)
    scattering_values = np.concatenate([atom_dict[atom] for atom in ionized_element_equivalents],axis=1)
    return scattering_values

def get_scattering_factors_unique(atoms, s):
    scattering_table = pd.read_feather('/das/work/p17/p17490/Peter/manual_refinement/Scattering_table_as_Excel_corrected.feather')
    PSE = list(scattering_table)
    ionized_equivalents= atoms.element
    ionized_nr_equivalents = np.array([PSE.index(ion) for ion in ionized_equivalents])
    ionized_nr_equivalents -= atoms.charge.values
    ionized_element_equivalents = np.array([PSE[ion] for ion in ionized_nr_equivalents])
    unique_atoms = np.unique(ionized_element_equivalents)
    scattering_angle = scattering_table.index.values
    idxs = np.digitize(s, scattering_angle)
    idxs_lower = idxs - 1
    s_lower = scattering_angle[idxs_lower]
    s_higher = scattering_angle[idxs]
    atom_dict = {}
    for unique_atom in unique_atoms:
        y_lower = scattering_table.iloc[idxs_lower][unique_atom].values
        y_higher = scattering_table.iloc[idxs][unique_atom].values
        scattering_factors = linear_interpolation(s, s_lower, s_higher, y_lower, y_higher)
        atom_dict[unique_atom] = scattering_factors.reshape(-1, 1)
    return atom_dict

def linear_interpolation(x, x0, x1, y0, y1):
    return y0 + (x - x0) * (y1 - y0) / (x1 - x0)

def get_scattering_itc92(df,s):
    import gemmi
    all_atoms = df.element.values
    atoms = np.unique(df.element)
    s_squared = ((s/4)**2).reshape(-1,1)
    elements = {}
    for element in atoms:
        SF = gemmi.Element(element).it92
        A = np.array(SF.a).reshape(1,-1)
        B = np.array(SF.b).reshape(1,-1)
        C = np.array(SF.c).reshape(1,-1)
        f = np.sum(A * np.exp(-B * s_squared),axis=1).reshape(-1,1)
        f += C
        elements[element] = f
    return np.concatenate([elements[element] for element in all_atoms],axis=1)  

