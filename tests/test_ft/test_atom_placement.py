from multicopy_refinement.model_ft import ModelFT



M = ModelFT()   
M.load_pdb_from_file('/das/work/p17/p17490/Peter/Library/multicopy_refinement/test_data/test_onecopy.pdb')

print(M.pdb)

M.setup_grids()
for residue in M.residues.values():
    M.build_atom_full(residue)
print(M.parametrization)
M.save_map('/das/work/p17/p17490/Peter/Library/multicopy_refinement/tests/test_ft/test.ccp4')