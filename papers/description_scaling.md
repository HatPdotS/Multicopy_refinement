# Bulk-Solvent and Overall Isotropic Scaling (Analytical Method)

This repository implements the **analytical bulk-solvent and isotropic scaling procedure**
from **Afonine et al. (2013)**  
[*Acta Cryst. D69, 625–634*](https://doi.org/10.1107/S0907444913000462),  
corrected by the 2023 addendum (Afonine et al., *Acta Cryst. D79, 666–667*).

The method provides **closed-form analytical expressions** for determining:
- the **bulk-solvent scale factor** `k_mask(s)`  
- the **overall isotropic scale parameter** \( K = k_\text{isotropic}^{-2}(s) \)

per resolution shell, without iterative numerical optimization.  
It is up to **100× faster** than traditional minimization-based approaches and avoids local minima.

---

## 📘 Background

The total model structure factor for each reflection \( s \) is defined as:

\[
F_\text{model}(s) = k_\text{total}(s) \left[ F_\text{calc}(s) + k_\text{mask}(s) F_\text{mask}(s) \right]
\]

where:
- \( F_\text{calc} \) – structure factor from the atomic model  
- \( F_\text{mask} \) – structure factor from the bulk-solvent mask  
- \( k_\text{mask} \) – bulk-solvent scaling coefficient  
- \( k_\text{total}(s) = k_\text{overall} \, k_\text{isotropic}(s) \, k_\text{anisotropic}(s) \)

The goal is to determine \( k_\text{mask} \) and \( k_\text{isotropic} \)
analytically from observed data \( F_\text{obs} \).

---

## 🧮 Analytical Formulation

### 1️⃣ Least-squares target
Define the residual (for each resolution shell):

\[
L_S(K, k_\text{mask}) =
\sum_s \left( |F_\text{calc} + k_\text{mask} F_\text{mask}|^2 - K I_s \right)^2
\]

where
\[
I_s = \left[k_\text{overall} \, k_\text{anisotropic}(s)\right]^2 F_\text{obs}^2(s), \quad
K = k_\text{isotropic}^{-2}.
\]

Each shell is typically **overdetermined** (many reflections, two parameters).

---

### 2️⃣ Simplify notation

For each reflection \( s \):

\[
u_s = |F_\text{calc}|^2, \quad
v_s = \operatorname{Re}(F_\text{calc} F_\text{mask}^*), \quad
w_s = |F_\text{mask}|^2.
\]

The target becomes:

\[
L_S(K, k_\text{mask}) = \sum_s \left[ (k_\text{mask}^2 w_s + 2k_\text{mask}v_s + u_s) - K I_s \right]^2.
\]

---

### 3️⃣ Analytical minimization

The optimal values satisfy:

\[
\frac{\partial L_S}{\partial K} = 0, \quad
\frac{\partial L_S}{\partial k_\text{mask}} = 0.
\]

Eliminating \( K \) gives a **cubic equation** in \( k_\text{mask} \):

\[
k_\text{mask}^3 + a k_\text{mask}^2 + b k_\text{mask} + c = 0,
\]

where the coefficients \( a, b, c \) are sums over reflection quantities.
This cubic can be solved **analytically** (standard cubic formula) or numerically (e.g. `numpy.roots`).

Once \( k_\text{mask} \) is found, \( K \) is obtained from:

\[
K = \frac{k_\text{mask}^2 C_2 + k_\text{mask} B_2 + A_2}{Y_2}.
\]

---

### 4️⃣ Root selection

- Only **real and positive roots** of the cubic are considered.
- If no positive roots exist, set \( k_\text{mask} = 0 \) (no solvent contribution).
- If several positive roots exist, select the one giving the **lowest** \( L_S(K, k_\text{mask}) \).

---

### 5️⃣ Resolution dependence (optional fit)

After solving per-bin values of \( k_\text{mask} \), they can be fit analytically to the
traditional exponential form:

\[
k_\text{mask}(s) = k_\text{sol} \exp\left(-B_\text{sol} s^2 / 4\right)
\]

by minimizing:
\[
\sum_s \left[k_\text{mask}(s) - k_\text{sol} \exp(-B_\text{sol} s^2 / 4)\right]^2.
\]

This fit also has a closed-form solution (see Appendix A in the paper).

---

## ⚙️ Implementation Overview

### `analytical_kmask_K()`

Analytically computes `kmask_per_ref` and `K_per_ref` using the equations above.

**Inputs:**
- `fcalc`: complex structure factors from atomic model
- `fmask`: complex structure factors from solvent mask
- `Fobs`: observed structure factor amplitudes
- `scattering_vecs`: reciprocal-space vectors per reflection (for resolution binning)
- `hkl`: Miller indices
- optional parameters: number of resolution bins, `overall_factor`, `anisotropic_factor`

**Outputs:**
- `kmask_per_ref` — real tensor of per-reflection bulk-solvent scale factors  
- `K_per_ref` — real tensor of per-reflection isotropic inverse-squared scale  
- `bin_info` — dictionary with per-bin statistics

**Core algorithm:**
1. Group reflections into resolution bins (uniform in log(d)).
2. Compute required reflection sums (A₂, B₂, C₂, D₃, Y₂, Y₃).
3. Form cubic equation in `k_mask` and solve analytically.
4. Select best positive root and compute `K`.
5. Optionally smooth or fit the resulting curves.

---

### `apply_bulk_solvent_and_isotropic_scale()`

Applies the calculated scaling parameters to obtain the final scaled model structure factors:

\[
F_\text{model}(s) = k_\text{total}(s) \left[F_\text{calc}(s) + k_\text{mask}(s) F_\text{mask}(s)\right]
\]

with
\[
k_\text{total}(s) = k_\text{overall} \, k_\text{anisotropic}(s) \, \frac{1}{\sqrt{K(s)}}.
\]

**Inputs:**
- `fcalc`, `fmask`: complex tensors
- `K_per_ref`, `kmask_per_ref`: scaling parameters from previous step
- optional: `k_overall`, `kanisotropic_per_ref`

**Outputs:**
- `F_model`: complex scaled structure factors
- `I_model = |F_model|^2`: real model intensities

---

## 🧩 Example Usage

```python
# Step 1: Analytical optimization per resolution shell
kmask_per_ref, K_per_ref, bin_info = analytical_kmask_K(
    fcalc=fcalc,
    fmask=fmask,
    hkl=hkl,
    scattering_vecs=scat_vecs,
    Fobs=Fobs,
    n_bins=20
)

# Step 2: Apply scaling
F_model, I_model = apply_bulk_solvent_and_isotropic_scale(
    fcalc=fcalc,
    fmask=fmask,
    K_per_ref=K_per_ref,
    kmask_per_ref=kmask_per_ref,
    k_overall=1.0
)