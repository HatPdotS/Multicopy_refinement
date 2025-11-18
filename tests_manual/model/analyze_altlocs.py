import pdb_tools
import pandas as pd

# Load a test PDB file
pdb = pdb_tools.load_pdb_as_pd('/das/work/p17/p17490/Peter/Library/multicopy_refinement/test_data/dark.pdb')

# Get all atoms with alternative conformations
altloc_df = pdb[pdb['altloc'] != '']

print("Total atoms with altlocs:", len(altloc_df))
print("\nSample of altloc atoms:")
print(altloc_df[['serial', 'name', 'altloc', 'resname', 'chainid', 'resseq', 'index']].head(20))

# Group by residue to find pairs
print("\n\nGrouping by residue (resname, resseq, chainid):")
grouped = altloc_df.groupby(['resname', 'resseq', 'chainid'])

for (resname, resseq, chainid), group in list(grouped)[:3]:
    print(f"\n{resname}-{resseq} Chain {chainid}:")
    print(f"  Unique altlocs: {sorted(group['altloc'].unique())}")
    print(f"  Atom names: {sorted(group['name'].unique())}")
    
    # For each atom name, show the indices
    for atom_name in sorted(group['name'].unique()):
        atom_group = group[group['name'] == atom_name]
        indices = atom_group['index'].tolist()
        altlocs = atom_group['altloc'].tolist()
        print(f"    {atom_name}: altlocs={altlocs}, indices={indices}")
