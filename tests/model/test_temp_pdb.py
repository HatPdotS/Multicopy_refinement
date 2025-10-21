import pdb_tools
import tempfile
import os

pdb_content = """ATOM      1  N   ALA A   1      10.000  10.000  10.000  1.00 20.00           N
ATOM      2  CA  ALA A   1      11.000  11.000  11.000  1.00 20.00           C
ATOM      3  C   ALA A   1      12.000  12.000  12.000  1.00 20.00           C
ATOM      4  O   ALA A   1      13.000  13.000  13.000  1.00 20.00           O
ATOM      5  CB  ALA A   1      14.000  14.000  14.000  1.00 20.00           C
END
"""

with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as f:
    f.write(pdb_content)
    temp_pdb = f.name

print(f"Created temp file: {temp_pdb}")

try:
    pdb = pdb_tools.load_pdb_as_pd(temp_pdb)
    print("Success!")
    print(pdb.head())
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
finally:
    os.unlink(temp_pdb)
