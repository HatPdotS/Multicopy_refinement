import pdb_tools 
import pandas as pd





df = pdb_tools.load_pdb_as_pd('/das/work/p17/p17490/Peter/Library/multicopy_refinement/test_FT/dark.pdb')

df = df.loc[df.element != 'H']  # Remove hydrogens

print(df.keys())
pdb_tools.write_file(df, '/das/work/p17/p17490/Peter/Library/multicopy_refinement/test_FT/dark_no_H.pdb')