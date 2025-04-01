import multicopy_refinement.Model as Model
import pandas as pd
from pdb_tools import write_file

model = Model.model()

model.load_pdb_from_file('/das/work/p17/p17490/Peter/Library/multicopy_refinement/test_data/Alvra_BT_01-2025_refine_100.pdb')
# model.pdb = model.pdb.loc[~model.pdb['name'].isin(['C16','C18','C7','C13'])]


selections = [(None,102,None,"A"),(None,102,None,"B"),(None,102,None,"E"),(None,103,None,'A'),(None,103,None,'B'),(None,103,None,'E')]





models = model.align_models(selections)
print(len(models))

model = pd.concat(models)

write_file(model,'/das/work/p17/p17490/Peter/Library/multicopy_refinement/test_data/test_aligned_copies_1.pdb')



selections = [(None,102,None,"C"),(None,102,None,"D"),(None,103,None,'C'),(None,103,None,'D')]



model = Model.model()

model.load_pdb_from_file('/das/work/p17/p17490/Peter/Library/multicopy_refinement/test_data/Alvra_BT_01-2025_refine_100.pdb')
# model.pdb = model.pdb.loc[~model.pdb['name'].isin(['C16','C18','C7','C13'])]

models = model.align_models(selections)
print(len(models))

model = pd.concat(models)

write_file(model,'/das/work/p17/p17490/Peter/Library/multicopy_refinement/test_data/test_aligned_copies_2.pdb')


selections = [(None,102,None,"C"),(None,102,None,"D"),(None,103,None,'C'),(None,103,None,'D')] + [(None,102,None,"A"),(None,102,None,"B"),(None,103,None,'A'),(None,103,None,'B')]



model = Model.model()

model.load_pdb_from_file('/das/work/p17/p17490/Peter/Library/multicopy_refinement/test_data/Alvra_BT_01-2025_refine_100.pdb')
# model.pdb = model.pdb.loc[~model.pdb['name'].isin(['C16','C18','C7','C13'])]

models = model.align_models(selections)
print(len(models))

model = pd.concat(models)

write_file(model,'/das/work/p17/p17490/Peter/Library/multicopy_refinement/test_data/test_aligned_copies_both.pdb')