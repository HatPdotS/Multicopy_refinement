import torch
import numpy as np
import pdb_tools
import reciprocalspaceship as rs
from crystfel_tools import crystfel_tools as cft
from source.math_numpy import get_grids,put_hkl_on_grid
import pandas as pd


class refinement:
    def __init__(self,start_model,reflections,cell,restraints=None):
        self.start_model = pdb_tools.load_pdb_as_pd(start_model)
        self.cell = cell
        if isinstance(reflections,pd.DataFrame):
            self.reflections = reflections
        elif reflections[-4:](".mtz"):
            self.reflections = rs.read_mtz(reflections).reset_index().rename(columns={"H":"h","K":"k","L":"l"}) 
        elif reflections[-4:](".hkl"):
            self.reflections = cft.read_partialator_hkl(reflections)
        else:
            raise ValueError("Unknown reflection file format")
        self.restraints = restraints
        
    def french_wilson(self,I_obs):
        return rs.utils.french_wilson(I_obs)
        
    def find_intensity(self):
        for key in self.reflections:
            if key.upper() in ["I","I(+)", "I(-)"]:
                return key

    def setup_grid(self):
        self.reflections = cft.get_max_resolution(self.reflections)
        max_res = self.reflections["resolution"].min()
        self.recgrid, self.real_space_grid = get_grids(self.cell,max_res)
    
    def setup_recgrid(self):
        try:
            self.real_space_grid
        except AttributeError:
            self.setup_grid()
        self.recgrid = put_hkl_on_grid(self.real_space_grid,self.reflections["F"],self.reflections[["h","k","l"]].values)

