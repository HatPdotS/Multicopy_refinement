#!/das/work/p17/p17490/CONDA/muticopy_refinement/bin/python
#SBATCH -c 16
#SBATCH -o /das/work/p17/p17490/Peter/Library/multicopy_refinement/tests/validation_on_different_samples/Br/test_scaler_integrated_fitting.out

from multicopy_refinement.model_ft import ModelFT
from multicopy_refinement.scaler import Scaler
from multicopy_refinement.Data import ReflectionData
import torch
import matplotlib.pyplot as plt

data = ReflectionData(verbose=2).load_from_mtz('/das/work/p17/p17490/Peter/Library/multicopy_refinement/tests/validation_on_different_samples/Br/BR_LCLS_refine_8.mtz')

M = ModelFT(verbose=0,max_res=1.5).load_pdb_from_file('/das/work/p17/p17490/Peter/Library/multicopy_refinement/tests/validation_on_different_samples/Br/BR_LCLS_refine_8.pdb')
data.filter_by_resolution(d_min=1.5)

S = Scaler(M, data, nbins=20,verbose=2)
S.initialize()

outdir = '/das/work/p17/p17490/Peter/Library/multicopy_refinement/tests/validation_on_different_samples/Br'

S.screen_solvent_params()

data.find_outliers(M, S, z_threshold=4.0)

rwork, rfree = S.rfactor()
print(f"Initial R-work: {rwork:.4f}, R-free: {rfree:.4f}")

fcalc = S(M(data.get_hkl()))
_, fobs, _, _ = data()

plt.scatter(torch.abs(fobs).cpu().numpy(), torch.abs(fcalc).detach().cpu().numpy(), alpha=0.5)
plt.plot([0,500],[0,500], color='red')
plt.xlim(0,500)
plt.ylim(0,500)
plt.xlabel('Observed |F|')
plt.ylabel('Calculated |F| before Scaling Fit')
plt.title('Observed vs Calculated Structure Factors before Scaling Fit')
plt.savefig(f'{outdir}/observed_vs_calculated_before_fit.png')
plt.close()

S.fit_all_scales()

rwork, rfree = S.rfactor()
print(f"Post-fitting R-work: {rwork:.4f}, R-free: {rfree:.4f}")

data.find_outliers(M, S,z_threshold=4.0)

rwork, rfree = S.rfactor()
print(f"Post-outlier R-work: {rwork:.4f}, R-free: {rfree:.4f}")
max_res, rwork, rfree = S.bin_wise_rfactor()


plt.plot(max_res.cpu().numpy(), rwork.cpu().numpy(), label='R-work')
plt.plot(max_res.cpu().numpy(), rfree.cpu().numpy(), label='R-free')
plt.xlabel('Resolution (Å)')
plt.ylabel('R-factor')
plt.title('Bin-wise R-factors after Scaling Fit')
plt.legend()
plt.savefig(f'{outdir}/bin_wise_rfactors.png')

plt.close()


fcalc = S(M(data.get_hkl()))
_, fobs, _, _ = data()

plt.scatter(torch.abs(fobs).cpu().numpy(), torch.abs(fcalc).detach().cpu().numpy(), alpha=0.5)
plt.plot([0,500],[0,500], color='red')
plt.xlim(0,500)
plt.ylim(0,500)
plt.xlabel('Observed |F|')
plt.ylabel('Calculated |F| after Scaling')
plt.title('Observed vs Calculated Structure Factors after Scaling Fit')
plt.savefig(f'{outdir}/observed_vs_calculated.png')
plt.close()

fobsmean, fcalcmean, res = S.get_binwise_mean_intensity()
plt.plot(res.detach().cpu().numpy(), fobsmean.detach().cpu().numpy(), label='Observed Mean |F|')
plt.plot(res.detach().cpu().numpy(), fcalcmean.detach().cpu().numpy(), label='Calculated Mean |F|')
plt.xlabel('Resolution (Å)')
plt.ylabel('Mean |F|')
plt.title('Bin-wise Mean Structure Factor Amplitudes after Scaling Fit')
plt.legend()
plt.savefig(f'{outdir}/mean_structure_factors.png')
plt.close()

res, work, rfree = S.bin_wise_rfactor()

plt.plot(res.detach().cpu().numpy(), work.detach().cpu().numpy(), label='R-work')
plt.plot(res.detach().cpu().numpy(), rfree.detach().cpu().numpy(), label='R-free')
plt.xlabel('Resolution (Å)')
plt.ylabel('R-factor')
plt.title('Bin-wise R-factors after Scaling Fit')
plt.legend()
plt.savefig(f'{outdir}/bin_wise_rfactors_post_fit.png')
plt.close()

