import pdb_tools 
import pandas as pd





df = pdb_tools.load_pdb_as_pd('/das/work/p17/p17490/Peter/Library/multicopy_refinement/test_data/Alvra_BT_01-2025_refine_61.pdb')


print(df.keys())
pdb_tools.write_file(df, '/das/work/p17/p17490/Peter/Library/multicopy_refinement/test_data/test_all.pdb')