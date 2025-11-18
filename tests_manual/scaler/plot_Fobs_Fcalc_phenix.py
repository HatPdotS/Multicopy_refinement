import reciprocalspaceship as rs
from matplotlib import pyplot as plt    


mtz = '/das/work/p17/p17490/Peter/Library/multicopy_refinement/tests_manual/scaler/dark.mtz'


data = rs.read_mtz(mtz)
data.dropna(inplace=True)
print(data)

f = data['F-model'].values.astype(float)
fobs = data['F-obs-filtered'].values.astype(float)


def rfactor(fobs, fcalc):
    """Calculate R-factor between observed and calculated structure factors."""
    return abs(fobs - fcalc).sum() / abs(fobs).sum()


print(f"R-factor (Fobs vs Fcalc): {rfactor(fobs, f):.4f}")

plt.plot(fobs, f, 'o', alpha=0.5)

plt.xlabel('Fobs')
plt.ylabel('Fcalc')
plt.title('Fobs vs Fcalc')
plt.savefig('/das/work/p17/p17490/Peter/Library/multicopy_refinement/tests_manual/scaler/Fobs_vs_Fcalc.png', dpi=300)