from multicopy_refinement.model import model


test = model()

test.load_pdb_from_file('/das/work/p17/p17490/Peter/Library/multicopy_refinement/test_data/dark.pdb')

print(test.pdb)
