from crystfel_tools.handling.fast_math import calculate_scattering_factor_cctbx,get_resolution
import reciprocalspaceship as rs
import numpy as np

pdb = '/das/work/p17/p17490/Peter/Library/multicopy_refinement/test_data/test_onecopy.pdb'

F,hkl = calculate_scattering_factor_cctbx(pdb,d_min=0.8)

cell = [14.97,18.85,18.89,89.37,84.94,67.84]
spacegroup = 'P1'

dataset = rs.DataSet({'F-model':np.abs(F),'SIGF':np.abs(F)*0.1,'PHIF-model':np.rad2deg(np.angle(F)),'H':hkl[:,0],'K':hkl[:,1],'L':hkl[:,2]}).set_index(['H','K','L'])
dataset.infer_mtz_dtypes(inplace=True)
dataset.cell = cell
dataset.spacegroup = spacegroup
dataset.write_mtz('/das/work/p17/p17490/Peter/Library/multicopy_refinement/test_data/test_onecopy.mtz')

pdb = '/das/work/p17/p17490/Peter/Library/multicopy_refinement/test_data/test_multicopy.pdb'

F,hkl = calculate_scattering_factor_cctbx(pdb,d_min=0.8)

cell = [14.97,18.85,18.89,89.37,84.94,67.84]
spacegroup = 'P1'

dataset = rs.DataSet({'F-model':np.abs(F),'SIGF':np.abs(F)*0.1,'PHIF-model':np.rad2deg(np.angle(F)),'H':hkl[:,0],'K':hkl[:,1],'L':hkl[:,2]}).set_index(['H','K','L'])
dataset.infer_mtz_dtypes(inplace=True)
dataset.cell = cell
dataset.spacegroup = spacegroup
dataset.write_mtz('/das/work/p17/p17490/Peter/Library/multicopy_refinement/test_data/test_multicopy.mtz')