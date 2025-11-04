import pdb_tools 
import pandas as pd


import numpy as np


df = pdb_tools.load_pdb_as_pd('/das/work/p17/p17490/Peter/Library/multicopy_refinement/test_data/test_all.pdb')

df.loc[:,['x','y','z']] += np.random.normal(0,0.2,(df.shape[0],3)) 

pdb_tools.write_file(df, '/das/work/p17/p17490/Peter/Library/multicopy_refinement/test_data/test_all_perturbed.pdb')
