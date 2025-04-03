from multicopy_refinement import restraints_torch



cif = '/das/work/p17/p17490/Peter/Library/multicopy_refinement/test_data/Merged_restraints_all_opened.cif'



restraints = restraints_torch.read_cif(cif)
nested_keys = list(set([key for i in restraints.values() for key in i.keys()]))
print(restraints['4PZ'].keys())
torsion_key = [key for key in nested_keys if 'tor' in key]
print(torsion_key)