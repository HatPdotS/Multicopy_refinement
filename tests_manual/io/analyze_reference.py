import reciprocalspaceship as rs
import numpy as np

mtzin = rs.read_mtz('/das/work/p17/p17490/Peter/Library/multicopy_refinement/tests/refinement/dark.mtz')
mtzin.dropna(inplace=True)

I_obs = mtzin['I-obs'].values
F_ref = mtzin['F-obs-filtered'].values

# Check relationship between F_ref and I_obs
# Theory: F_ref might be something like F = k * sqrt(|I| + correction)

# For negative I
neg_mask = I_obs < 0
if neg_mask.sum() > 0:
    print(f"Number of negative intensities: {neg_mask.sum()}")
    print("\nFor negative I, check if F^2 correlates with |I|:")
    F2_neg = F_ref[neg_mask]**2
    Ineg = np.abs(I_obs[neg_mask])
    
    # Check correlation
    print(f"Correlation between F^2 and |I|: {np.corrcoef(F2_neg, Ineg)[0,1]:.4f}")
    
    # Check if F^2 ~ c * |I| for some constant
    ratio = F2_neg / (Ineg + 1e-6)
    print(f"Mean F^2 / |I|: {np.mean(ratio):.2f} +/- {np.std(ratio):.2f}")
    
    # Try F^2 = a + b*|I|
    from scipy import stats
    slope, intercept, r_value, p_value, std_err = stats.linregress(Ineg, F2_neg)
    print(f"\nLinear fit: F^2 = {intercept:.2f} + {slope:.2f} * |I|")
    print(f"R^2 = {r_value**2:.4f}")
    
print("\n" + "="*50)
print("For positive weak intensities (0 < I < 100):")
pos_weak = (I_obs > 0) & (I_obs < 100)
if pos_weak.sum() > 0:
    I_pos = I_obs[pos_weak]
    F_pos = F_ref[pos_weak]
    
    # Check F vs sqrt(I)
    sqrt_I = np.sqrt(I_pos)
    F2 = F_pos**2
    
    print(f"Correlation between F and sqrt(I): {np.corrcoef(F_pos, sqrt_I)[0,1]:.4f}")
    print(f"Correlation between F^2 and I: {np.corrcoef(F2, I_pos)[0,1]:.4f}")
    
    # Check ratio
    ratio = F_pos / sqrt_I
    print(f"Mean F / sqrt(I): {np.mean(ratio):.2f} +/- {np.std(ratio):.2f}")
    
    # Try F^2 = a + b*I
    slope, intercept, r_value, p_value, std_err = stats.linregress(I_pos, F2)
    print(f"\nLinear fit: F^2 = {intercept:.2f} + {slope:.2f} * I")
    print(f"R^2 = {r_value**2:.4f}")
