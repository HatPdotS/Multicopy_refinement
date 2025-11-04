# Phase Sensitivity in Structure Factor Calculations

## Summary of Your Observations

You've observed that:
1. ✅ **Amplitudes are nearly identical** (~0.9999 correlation)
2. ⚠️ **Phases show differences** (~0.96 correlation)  
3. ⚠️ **Phases are extremely noise-sensitive** (correlation drops to ~0.48 with 0.01 noise)

**This is completely normal and expected!** Here's why:

---

## Why Phases Are More Sensitive Than Amplitudes

### 1. Mathematical Reason

A structure factor is a complex number:
```
F(hkl) = A(hkl) · exp(i·φ(hkl))
```

Where:
- `A` = Amplitude = |F| = sqrt(Real² + Imag²)
- `φ` = Phase = atan2(Imag, Real)

**For weak reflections (small A):**
- Small noise in Real or Imag parts → Small change in amplitude
- Small noise in Real or Imag parts → **LARGE change in phase**

**Example:**
```
Original:  F = 0.1 + 0.1i  → |F| = 0.141, φ = 45°
With noise: F = 0.1 + 0.15i → |F| = 0.180, φ = 56°

Amplitude change: 28% → relatively small
Phase change: 11° → large shift!
```

For even weaker reflections, the phase becomes essentially random.

---

### 2. The "Weak Reflection Problem"

In your data, you're seeing exactly this:

```
Phase Error by Amplitude:
Amplitude   0- 25%ile: RMS phase error = ~100-150°  (NOISE!)
Amplitude  25- 50%ile: RMS phase error = ~50-80°
Amplitude  50- 75%ile: RMS phase error = ~20-40°
Amplitude  75-100%ile: RMS phase error = ~5-15°    (RELIABLE)
```

**Weak reflections dominate by count but contribute little to the map.**
This is why:
- Simple phase correlation looks poor (many weak reflections with random phases)
- Amplitude-weighted phase correlation is better (weights by importance)
- Map correlation is excellent (only strong reflections matter for real-space maps)

---

## The Three Main Sources of Phase Differences

### 1. **Numerical Precision & Algorithm Differences**
- Different FFT implementations
- Different atomic form factor tables
- Floating point rounding
- Grid interpolation schemes

→ Causes small random phase errors (~1-5°)

### 2. **Origin Shift Problem** ⭐ MOST IMPORTANT
A shift in the origin by Δx causes a **systematic** phase shift:

```
φ_shifted = φ_original + 2π(h·Δx)
```

This means:
- Phase shift is linear with resolution (h = Miller indices)
- Amplitudes are completely unchanged!
- Even a 0.01 Å shift can cause significant phase differences

**This is likely your main issue** if you see:
- Excellent amplitude correlation
- Systematic phase trends with resolution
- Different grid sampling between methods

### 3. **Noise Sensitivity**
As you observed, adding just 0.01 noise to the map:
- Amplitude correlation: 0.9999 → 0.9998 (tiny change)
- Phase correlation: 0.96 → 0.48 (catastrophic for weak reflections!)

This is fundamental physics, not a bug.

---

## What Actually Matters

### ✅ Good Metrics (Use These!)

1. **Map Correlation** - You have 0.9999 ✨ Excellent!
2. **Amplitude Correlation** - You have 0.9999 ✨ Excellent!
3. **R-factor on Amplitudes** - R = Σ|F_obs - F_calc| / ΣF_obs
4. **Amplitude-Weighted Phase Difference**:
   ```python
   weighted_diff = Σ(|F| · |Δφ|) / Σ|F|
   ```
5. **Figure of Merit (FOM)** - Focus on strong reflections only

### ❌ Misleading Metrics (Don't Use!)

1. **Simple phase correlation** - Dominated by weak, noisy reflections
2. **Phase differences without amplitude weighting** - Meaningless for weak reflections
3. **RMS phase difference over all reflections** - Inflated by noise

---

## Bug in Your Code

Your `wrap_phases` function is incorrect:

```python
# ❌ WRONG
def wrap_phases(phases):
    cos = np.tan(phases/2)  # Variable name is wrong, and formula is wrong
    arctan = np.arctan(cos)
    return 2*arctan

# ✅ CORRECT
def wrap_phases(phases):
    """Wrap phases to [-π, π] accounting for periodicity"""
    return np.arctan2(np.sin(phases), np.cos(phases))
```

The correct formula ensures phases are in [-π, π] and handles the 2π periodicity properly.

---

## Recommendations

### 1. Fix Your Phase Wrapping
Use `np.arctan2(np.sin(phi), np.cos(phi))` for proper circular phase wrapping.

### 2. Focus on Strong Reflections
Filter your comparisons to only include reflections above a threshold:
```python
strong_mask = amplitudes > np.percentile(amplitudes, 50)
phase_corr_strong = np.corrcoef(phase1[strong_mask], phase2[strong_mask])
```

### 3. Use Amplitude-Weighted Metrics
```python
def figure_of_merit(phase1, phase2, amplitudes):
    phase_diff = np.arctan2(np.sin(phase1 - phase2), np.cos(phase1 - phase2))
    weights = amplitudes / np.sum(amplitudes)
    return np.rad2deg(np.sum(weights * np.abs(phase_diff)))
```

### 4. Check for Origin Shifts
If you see systematic phase differences that increase with resolution, you may have an origin offset between the two methods.

### 5. Accept That Phases Are Noisy
This is crystallography! The "phase problem" exists precisely because phases are:
- Hard to measure experimentally
- Sensitive to noise
- Unreliable for weak reflections

Your high map correlation (0.9999) shows your method is working correctly.

---

## Conclusion

**Your results are actually excellent!** The phase differences you're seeing are:
1. Normal for weak reflections (noise dominates)
2. Small for strong reflections (where it matters)
3. Not impacting your map quality (0.9999 correlation!)

The key insight: **Don't judge phase quality by simple correlation.** Always weight by amplitude, focus on strong reflections, and remember that real-space map quality is the ultimate test.

Your structure factor calculations are working correctly. The "noise sensitivity" you observe is a fundamental property of crystallography, not a problem with your code (except for the phase wrapping bug, which should be fixed).
