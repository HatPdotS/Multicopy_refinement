#!/usr/bin/env python3
"""Debug script to see what's being parsed"""

def split_respecting_quotes(line):
    line_new = ''
    in_quotes = False
    for character in line:
        if character == "'" or character == '"':
            in_quotes = not in_quotes
        if in_quotes and character == ' ':
            continue
        line_new += character
    return line_new.split()

with open('/das/work/p17/p17490/Peter/Library/multicopy_refinement/external_monomer_library/m/MET.cif') as f:
    lines = f.readlines()

lines_iter = iter(lines)
for line in lines_iter:
    if line.strip() == 'data_comp_list':
        print("Found data_comp_list")
        for line in lines_iter:
            if line.strip() == 'loop_':
                print("Found loop_")
                comp_list = []
                values = []
                for line in lines_iter:
                    print(f"Processing: {line.strip()}")
                    if line.startswith('#'):
                        print(f"Breaking on comment: {line.strip()}")
                        break
                    if line.startswith('_chem_comp'):
                        col_name = line.split('.')[1].strip()
                        comp_list.append(col_name)
                        print(f"  Added column: {col_name}")
                    else:
                        split_items = split_respecting_quotes(line)
                        if split_items:  # Only add non-empty lines
                            values.append(split_items)
                            print(f"  Added values: {split_items}")
                
                print(f"\nFinal comp_list ({len(comp_list)} columns): {comp_list}")
                print(f"Final values ({len(values)} rows):")
                for i, v in enumerate(values):
                    print(f"  Row {i} ({len(v)} items): {v}")
                break
