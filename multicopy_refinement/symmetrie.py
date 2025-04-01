import torch as np

def apply_space_group(fractional_coords,space_group):
    space_group = space_group.replace(' ','')
    if space_group == 'P1':
        return P1(fractional_coords)
    elif space_group == 'P-1':
        return P_minus1(fractional_coords)
    elif space_group == 'P1211':
        return P1211(fractional_coords)
    raise ValueError(f'space group, {space_group} not implemented')

def P1(fractional_coords):
    return fractional_coords.reshape(3,-1,1)

def P_minus1(fractional_coords):
    mirrored = fractional_coords * -1
    mirrored = mirrored.reshape(3,-1,1)
    fractional_coords = fractional_coords.reshape(3,-1,1)
    return np.concatenate([fractional_coords,mirrored],axis=2)  

def P1211(fractional_coords):
    fractional_coords = fractional_coords.reshape(3,-1,1)
    sym_frac = fractional_coords * np.array([-1,1,-1]).reshape(3,1,1) + np.array((0,0.5,0)).reshape(3,1,1)
    return np.concatenate([fractional_coords,sym_frac],axis=2)