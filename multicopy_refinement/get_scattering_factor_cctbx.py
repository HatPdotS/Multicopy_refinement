from cctbx import crystal, xray
from iotbx import pdb
import numpy as np
import gemmi

def calculate_scattering_factor(pdb_file,hkls = None):
    pdb_input = pdb.input(file_name=pdb_file)
    xray_structure = pdb_input.xray_structure_simple()
    f_calc = xray_structure.structure_factors(d_min=1.0).f_calc()
    idx = np.array(f_calc.indices())
    f_calc = np.array(f_calc.data())
    f_calc = f_calc
    if hkls is not None:
        hkls = set([tuple(hkl) for hkl in hkls])
        f_calc_new = []
        idx_new = []
        for hkl,val in zip(idx,f_calc):
            if tuple(hkl) in hkls:
                f_calc_new.append(val)
                idx_new.append(hkl)
        f_calc = np.array(f_calc_new)
        idx = np.array(idx_new)
    return f_calc, idx


def calculate_scattering_factor_direct_gemmi(pdb_file,hkls):
    st = gemmi.read_structure(pdb_file)
    calc_e = gemmi.StructureFactorCalculatorE(st.cell)
    fs = []
    for hkl in hkls:
        f = calc_e.calculate_sf_from_model(st[0], hkl)
        fs.append(f)
    return np.abs(np.array(fs))

if __name__ == "__main__":
    pdb_file = "/das/work/p17/p17490/Peter/manual_refinement/test_data/2024-02-09_ESRF_SFX_refine_1.pdb"  # Replace with your PDB file path
    f_calc,idx = calculate_scattering_factor(pdb_file)
    print(idx)
