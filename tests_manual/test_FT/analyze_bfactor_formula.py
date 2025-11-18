"""
Test different B-factor conventions to match CCTBX.
"""
import numpy as np
import torch
import gemmi

print("=" * 70)
print("B-FACTOR CONVENTION ANALYSIS")
print("=" * 70)

# Standard crystallographic conventions:
print("\n1. ITC92 in reciprocal space:")
print("   f(s) = Σ aᵢ exp(-bᵢ s²) + c")
print("   where s = sin(θ)/λ = |h|/(2d) = |h|·d*/2")
print("")
print("2. B-factor in reciprocal space:")
print("   f(h) = f₀(h) · exp(-B·s²)")
print("   where B = 8π²⟨u²⟩ (mean square displacement)")
print("")
print("3. Fourier transform to real space:")
print("   For exp(-b·s²) in reciprocal space")
print("   FT gives: (π/b)^(3/2) · exp(-π²r²/b) in real space")
print("")
print("4. Therefore for ITC92 + B-factor:")
print("   f(s) = Σ aᵢ exp(-(bᵢ + B)s²)")
print("   FT gives: Σ aᵢ·(π/(bᵢ+B))^(3/2) · exp(-π²r²/(bᵢ+B))")

print("\n" + "=" * 70)
print("Let's verify with a simple example:")
print("=" * 70)

# Carbon atom
element = "C"
B_atom = 20.0  # Ų

fe = gemmi.Element(element).it92
a = np.array(fe.a)
b = np.array(fe.b)
c = fe.c

print(f"\nCarbon atom with B = {B_atom} Ų")
print(f"ITC92 a coefficients: {a}")
print(f"ITC92 b coefficients: {b}")

# Calculate density at origin (r=0)
print("\n" + "-" * 70)
print("Density at origin (r=0):")
print("-" * 70)

# Method 1: Using our current formula
print("\nMethod 1 (CURRENT CODE):")
print("  sigma² = B/(8π²) + b/(4π²)")
print("  ρ(0) = Σ aᵢ · exp(0) / (2π·sigma²)^(3/2)")

total_density_method1 = 0.0
for i in range(4):
    sigma_sq_itc = b[i] / (4 * np.pi**2)
    sigma_sq_bfac = B_atom / (8 * np.pi**2)
    total_sigma_sq = sigma_sq_itc + sigma_sq_bfac
    
    # Normalization for Gaussian exp(-r²/(2σ²)) is 1/(2π·σ²)^(3/2)
    norm = 1.0 / (2 * np.pi * total_sigma_sq) ** 1.5
    total_density_method1 += a[i] * norm

print(f"  ρ(0) = {total_density_method1:.6f} e/Ų")

# Method 2: Correct formula from Fourier transform
print("\nMethod 2 (CORRECT FORMULA):")
print("  B_total = B + bᵢ")
print("  ρ(r) = Σ aᵢ · (π/B_total)^(3/2) · exp(-π²r²/B_total)")

total_density_method2 = 0.0
for i in range(4):
    B_total = b[i] + B_atom
    norm = (np.pi / B_total) ** 1.5
    total_density_method2 += a[i] * norm

print(f"  ρ(0) = {total_density_method2:.6f} e/Ų")

# Method 3: What if B-factor should be divided by 2?
print("\nMethod 3 (B-FACTOR / 2):")
print("  sigma² = B/(4π²) + b/(4π²)")  # Factor of 2 difference!
print("  ρ(0) = Σ aᵢ · exp(0) / (2π·sigma²)^(3/2)")

total_density_method3 = 0.0
for i in range(4):
    sigma_sq_itc = b[i] / (4 * np.pi**2)
    sigma_sq_bfac = B_atom / (4 * np.pi**2)  # Changed from 8π² to 4π²
    total_sigma_sq = sigma_sq_itc + sigma_sq_bfac
    
    norm = 1.0 / (2 * np.pi * total_sigma_sq) ** 1.5
    total_density_method3 += a[i] * norm

print(f"  ρ(0) = {total_density_method3:.6f} e/Ų")

print("\n" + "=" * 70)
print("COMPARISON:")
print("=" * 70)
print(f"Method 1 / Method 2: {total_density_method1 / total_density_method2:.4f}")
print(f"Method 3 / Method 2: {total_density_method3 / total_density_method2:.4f}")
print("")
print("If CCTBX matches Method 2, we need to change our formula!")

print("\n" + "=" * 70)
print("THE ISSUE:")
print("=" * 70)
print("Our code uses:")
print("  exp(-r²/(2·σ²)) with σ² = b/(4π²) + B/(8π²)")
print("")
print("This gives exp(-r²/(2·(b+B/2)/(4π²)))")
print("         = exp(-2π²r²/(b+B/2))")
print("")
print("But correct formula should be:")
print("  exp(-π²r²/(b+B))")
print("")
print("Notice the factor of 2 difference in the exponent!")
print("And the B/2 in our formula means B is effectively halved.")

print("\n" + "=" * 70)
print("SOLUTION:")
print("=" * 70)
print("Change line 459 from:")
print("  sigma_squared_bfactor = b / (8 * np.pi**2)")
print("To:")
print("  sigma_squared_bfactor = b / (4 * np.pi**2)")
print("")
print("AND change line 463 from:")
print("  exp(-diff_coords_squared / (2 * total_variance))")
print("To:")
print("  exp(-diff_coords_squared * np.pi**2 / total_variance)")
