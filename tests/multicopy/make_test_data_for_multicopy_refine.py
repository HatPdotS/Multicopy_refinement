import pdb_tools 
import pandas as pd


import numpy as np


df = pdb_tools.load_pdb_as_pd('/das/work/p17/p17490/Peter/Library/multicopy_refinement/test_data/test_all.pdb')
df.tempfactor = 20
df.anisou_flag = False

df = df.iloc[:1]
df.element = 'C'    

df.loc[:,['x', 'y', 'z']] = np.array([0,0,0])
pdb_tools.write_file(df, '/das/work/p17/p17490/Peter/Library/multicopy_refinement/test_data/test_onecopy.pdb')
print(df.keys())


df.altloc =  'A' 
df.occupancy = 0.5
df2 = df.copy()
df2.altloc = 'B' 
df2.loc[0,['x', 'y', 'z']] += np.array([0.5,1,0.2])




df = pd.concat([df, df2], ignore_index=True)


pdb_tools.write_file(df, '/das/work/p17/p17490/Peter/Library/multicopy_refinement/test_data/test_multicopy.pdb')

