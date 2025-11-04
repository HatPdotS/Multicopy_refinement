from multicopy_refinement.Data import ReflectionData
from multicopy_refinement.model_ft import ModelFT
from torch.nn import Module as nnModule
from multicopy_refinement.kinetics import KineticModel
import torch

class kinetic_refinement(nnModule):
    def __init__(self, mtz_files, time, consecutive_pdb_files, kinetic: KineticModel, verbose=1):
        self.time = time
        self.verbose = verbose
        self.nstates = len(consecutive_pdb_files)
        self.mtzs = [ReflectionData(verbose=verbose).load_from_mtz(mtz) for mtz in mtz_files]
        self.models = [ModelFT().load_pdb_from_file(pdb) for pdb in consecutive_pdb_files]
        self.kinetic = kinetic

    def parameters(self):
        params = []
        for model in self.models:
            params += list(model.parameters())
        params += list(self.kinetic.parameters())
        return params

    def setup_optimiser(self, lr=1e-3):
        import torch.optim as optim
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def setup_hkls(self):
        hkl_start = self.mtzs[0].get_hkls()
        clipped = []
        valid_hkls = []
        for data in self.mtzs:
            present_data, valid_hkl = data.validate_hkl(hkl_start)
            clipped.append(present_data)
            valid_hkls.append(valid_hkl)
        valid_hkls = torch.logical_and(torch.stack(valid_hkls, dim=0), dim=0)
        self.hkl = hkl_start[valid_hkls]
        final_clipped = []
        for data in clipped:
            data = data.__select__(valid_hkls)
            final_clipped.append(data)
        self.mtzs = final_clipped
        self.F, self.I, self.sigma, self.rfree_flags = zip(*[data() for data in self.mtzs])
        self.F = torch.stack(self.F, dim=0)
        self.I = torch.stack(self.I, dim=0)
        self.sigma = torch.stack(self.sigma, dim=0)
        self.rfree_flags = torch.stack(self.rfree_flags, dim=0)
        return valid_hkls
    

    
    