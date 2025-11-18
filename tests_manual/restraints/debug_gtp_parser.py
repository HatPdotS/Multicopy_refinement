#!/usr/bin/env python3
"""Debug script to see what's being parsed in GTP"""

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

with open('/das/work/p17/p17490/Peter/Library/multicopy_refinement/external_monomer_library/g/GTP.cif') as f:
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
                in_data_section = False
                for i, line in enumerate(lines_iter):
                    print(f"Line {i}: '{line.rstrip()}'")
                    
                    # Stop on comment line
                    if line.startswith('#'):
                        print(f"  -> Breaking on comment")
                        break
                    # Stop on blank line if we've already seen data
                    if in_data_section and line.strip() == '':
                        print(f"  -> Breaking on blank line (in_data_section={in_data_section})")
                        break
                    # Stop if we hit a new section marker
                    if line.strip().startswith('data_') or line.strip().startswith('loop_'):
                        print(f"  -> Breaking on new section")
                        break
                    # Collect column names
                    if line.startswith('_chem_comp'):
                        col_name = line.split('.')[1].strip()
                        comp_list.append(col_name)
                        print(f"  -> Added column: {col_name}")
                    else:
                        # Only process non-empty lines as data
                        split_items = split_respecting_quotes(line.strip())
                        if split_items:
                            values.append(split_items)
                            in_data_section = True
                            print(f"  -> Added data: {split_items}")
                
                print(f"\nFinal comp_list ({len(comp_list)} columns): {comp_list}")
                print(f"Final values ({len(values)} rows):")
                for i, v in enumerate(values):
                    print(f"  Row {i} ({len(v)} items): {v}")
                break
