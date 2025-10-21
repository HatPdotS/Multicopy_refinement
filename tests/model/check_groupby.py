import pdb_tools

pdb = pdb_tools.load_pdb_as_pd('/das/work/p17/p17490/Peter/Library/multicopy_refinement/test_data/dark.pdb')

# Replicate the logic from register_alternative_conformations
pdb_with_altlocs = pdb[pdb['altloc'] != '']

print("Total atoms with altlocs:", len(pdb_with_altlocs))

grouped = pdb_with_altlocs.groupby(['resname', 'resseq', 'chainid', 'name'])

print("\nFirst 3 groups:")
for i, ((resname, resseq, chainid, name), group) in enumerate(grouped):
    if i >= 3:
        break
    
    print(f"\nGroup {i}: {resname}-{resseq}-{chainid}-{name}")
    print(f"  Group size: {len(group)}")
    print(f"  Indices (from 'index' column): {group['index'].tolist()}")
    print(f"  Altlocs: {group['altloc'].tolist()}")
    print(f"  DataFrame indices: {group.index.tolist()}")
    
    # Sort by altloc
    sorted_pairs = sorted(zip(group['altloc'].tolist(), group['index'].tolist()))
    sorted_indices = tuple(idx for _, idx in sorted_pairs)
    print(f"  Sorted indices tuple: {sorted_indices}")
