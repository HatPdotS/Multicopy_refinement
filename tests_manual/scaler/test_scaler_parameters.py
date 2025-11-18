from multicopy_refinement.model_ft import ModelFT
from multicopy_refinement.scaler import Scaler
from multicopy_refinement.Data import ReflectionData
import torch
import matplotlib.pyplot as plt
import os

data = ReflectionData(verbose=2).load_mtz('/das/work/p17/p17490/Peter/Library/multicopy_refinement/tests_manual/scaler/dark.mtz')

M = ModelFT(verbose=0,max_res=1.7).load_pdb('/das/work/p17/p17490/Peter/Library/multicopy_refinement/tests_manual/scaler/dark.pdb')
data.filter_by_resolution(d_min=1.7)

S = Scaler(M, data, nbins=20,verbose=0)

S.initialize()

for param in S.parameters():
    print(param.shape)

print('---solvent---')

for param in S.solvent.parameters():
    print(param.shape)