from crystfel_tools.handling.fast_math import calculate_scattering_factor_cctbx,get_resolution
import reciprocalspaceship as rs
import numpy as np

pdb = '/das/work/p17/p17490/Peter/Library/multicopy_refinement/test_FT/dark.pdb'

F,hkl = calculate_scattering_factor_cctbx(pdb,d_min=0.8)

cell = [74.530, 92.580, 83.990, 90.00, 96.71, 90.00]
spacegroup = 'P21'

dataset = rs.DataSet({'F-model':np.abs(F),'SIGF':np.abs(F)*0.1,'PHIF-model':np.rad2deg(np.angle(F)),'H':hkl[:,0],'K':hkl[:,1],'L':hkl[:,2]}).set_index(['H','K','L'])
dataset.infer_mtz_dtypes(inplace=True)
dataset.cell = cell
dataset.spacegroup = spacegroup
dataset.write_mtz('/das/work/p17/p17490/Peter/Library/multicopy_refinement/test_FT/dark_fcalc.mtz')