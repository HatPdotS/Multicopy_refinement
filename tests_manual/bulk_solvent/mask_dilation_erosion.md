# Mask Dilation and Erosion in Phenix

## Overview

The bulk solvent mask calculation in Phenix uses a **two-stage process** involving both **dilation** (expansion) and **erosion** (shrinking) operations to create smooth, physically meaningful mask boundaries.

## The Two-Radius System

Phenix uses **two independent radii** for mask calculation:

1. **`solvent_radius`** (default: 1.1 Å) - Controls the initial DILATION
2. **`shrink_truncation_radius`** (default: 0.9 Å) - Controls the final EROSION

These work together to create the final mask.

## Algorithm: Three-Step Process

### Step 1: Initial Dilation (Expansion)

**Cutoff radius** = `R_vdW(atom) + solvent_radius`

For each atom, create a sphere with radius = atomic vdW radius + solvent_radius.

```
Example for Carbon:
R_cutoff = 1.775 Å (C vdW) + 1.1 Å (solvent_radius) = 2.875 Å
```

**This creates three regions on the grid:**

```cpp
if (distance < R_vdW):
    grid_point = 0    // Inside the atom (PROTEIN)
elif (distance < R_cutoff):
    grid_point = -1   // In the "accessible surface" region (BOUNDARY)
else:
    grid_point = 1    // Bulk solvent region (SOLVENT)
```

After this step:
- **0**: Protein core (inside atomic radii)
- **-1**: Accessible surface layer (between R_vdW and R_cutoff)
- **1**: Bulk solvent

### Step 2: Erosion (Shrinking) Operation

Now apply the `shrink_truncation_radius` to smooth the mask.

For each grid point marked as **-1** (boundary), check if any neighboring grid points within `shrink_truncation_radius` are marked as **1** (solvent).

```cpp
for each grid_point with value == -1:
    for each neighbor within shrink_truncation_radius:
        if neighbor == 1:  // Found solvent nearby
            grid_point = 1  // Convert this boundary point to solvent
            break
        else:
            grid_point = 0  // Convert to protein (eroded away)
```

**Effect:**
- Boundary points (**-1**) near solvent → become solvent (**1**)
- Boundary points (**-1**) far from solvent → become protein (**0**)
- This creates a **smoothed transition** at the protein-solvent interface

### Step 3: Final Binary Mask

After erosion, the mask is simplified to binary:
- **0** = Protein region (no solvent)
- **1** = Solvent region (bulk solvent)

The **-1** values are all resolved to either 0 or 1.

## Mathematical Details

### Dilation Operation

The initial expansion uses a **distance test**:

```python
for each grid point (x, y, z):
    d_min = infinity
    for each atom i:
        # Compute distance in fractional coordinates
        d = sqrt(M · Δr)  # M = metrical matrix
        if d < R_vdW(i):
            mask[x,y,z] = 0  # Protein core
            break
        elif d < R_vdW(i) + solvent_radius:
            d_min = min(d_min, d - R_vdW(i))
            mask[x,y,z] = -1  # Accessible surface
    
    if mask[x,y,z] not set:
        mask[x,y,z] = 1  # Bulk solvent
```

### Erosion Operation

The shrinking uses a **neighborhood search**:

```python
for each point where mask[x,y,z] == -1:
    found_solvent = False
    
    # Search within shrink_truncation_radius
    for dx in range(-n_x, n_x+1):
        for dy in range(-n_y, n_y+1):
            for dz in range(-n_z, n_z+1):
                neighbor = (x+dx, y+dy, z+dz) % grid_size
                
                # Check if neighbor is within radius
                distance = unit_cell.distance(point, neighbor)
                if distance < shrink_truncation_radius:
                    if mask[neighbor] == 1:  # Found solvent
                        found_solvent = True
                        break
    
    if found_solvent:
        mask[x,y,z] = 1  # Keep as solvent
    else:
        mask[x,y,z] = 0  # Erode to protein
```

## Physical Interpretation

### The Accessible Surface vs. Contact Surface

1. **Accessible Surface** (after dilation, before erosion):
   - Defined by rolling a sphere of radius `solvent_radius` over protein
   - The **center** of the rolling sphere traces the accessible surface
   - This is what you get with `shrink_truncation_radius = 0`

2. **Contact Surface** (after erosion):
   - The **contact points** between rolling sphere and protein
   - Created by eroding the accessible surface
   - This is the final mask used in bulk solvent calculations

### Why Two Radii?

The two-radius approach provides:

1. **`solvent_radius`** (larger, ~1.1 Å):
   - Represents size of a water molecule
   - Ensures solvent can't get too close to protein
   - Accounts for first hydration shell

2. **`shrink_truncation_radius`** (smaller, ~0.9 Å):
   - Creates smooth transition at boundary
   - Reduces Fourier artifacts from sharp edges
   - Typical choice: slightly less than solvent_radius

## Effect of Parameters

### Varying solvent_radius

| solvent_radius | Effect on Mask | Solvent Content |
|----------------|----------------|-----------------|
| 0.8 Å | Smaller expansion | Higher (more solvent) |
| 1.1 Å (default) | Standard | Typical (~50%) |
| 1.4 Å | Larger expansion | Lower (less solvent) |

**Larger solvent_radius** → Thicker protein layer → Less bulk solvent

### Varying shrink_truncation_radius

| shrink_truncation_radius | Effect on Boundary | Smoothness |
|--------------------------|-------------------|------------|
| 0.0 Å | No erosion | Sharp edges (more artifacts) |
| 0.9 Å (default) | Moderate smoothing | Good balance |
| 1.5 Å | Heavy smoothing | Very smooth (may be over-smoothed) |

**Larger shrink_truncation_radius** → Smoother boundary → Better behaved Fourier transform

## Code Implementation

### From `around_atoms.h` (C++)

The key functions are:

1. **`compute_accessible_surface()`**:
   ```cpp
   // For each grid point
   if (dist < radsq) 
       dr = 0;   // Inside atom
   else if (dr != 0)    
       dr = -1;  // Accessible surface
   // else dr = 1  (stays solvent)
   ```

2. **`compute_contact_surface()`**:
   ```cpp
   // For each point marked -1
   for each neighbor within shrink_truncation_radius:
       if neighbor == 1:  // Found solvent
           point = 1      // Keep as solvent
           goto next_point
   point = 0  // Erode to protein
   ```

### Python Interface

```python
from mmtbx import masks

mask = masks.bulk_solvent(
    xray_structure = xray_structure,
    ignore_zero_occupancy_atoms = True,
    solvent_radius = 1.1,              # Dilation radius
    shrink_truncation_radius = 0.9,    # Erosion radius
    ignore_hydrogen_atoms = True,
    grid_step = 0.6)

# Results stored in mask.data (0 or 1)
print("Accessible surface fraction:", mask.accessible_surface_fraction)
print("Contact surface fraction:", mask.contact_surface_fraction)
```

## Practical Examples

### Example 1: Standard Protein (Default Parameters)

```python
solvent_radius = 1.1 Å
shrink_truncation_radius = 0.9 Å

# For a Carbon atom (R_vdW = 1.775 Å):
R_expand = 1.775 + 1.1 = 2.875 Å    # Initial dilation
R_erode = 0.9 Å                      # Erosion distance

# Net effect:
# Points between 1.775 and 2.875 Å become boundary
# Boundary points within 0.9 Å of solvent → stay solvent
# Other boundary points → become protein
```

**Result**: Smooth mask with ~50% solvent content for typical protein crystal.

### Example 2: No Erosion (Sharp Mask)

```python
solvent_radius = 1.1 Å
shrink_truncation_radius = 0.0 Å  # No erosion!

# Result: Sharp transitions at protein-solvent boundary
# May cause Fourier artifacts in F_mask calculation
```

**Use case**: Quick calculations where smoothness is not critical.

### Example 3: Heavy Smoothing

```python
solvent_radius = 1.1 Å
shrink_truncation_radius = 1.5 Å  # Heavy erosion

# Result: Very smooth boundary
# May over-erode small pockets and crevices
```

**Use case**: Very high solvent content crystals or low-resolution data.

### Example 4: Tight Mask (Low Solvent Content)

```python
solvent_radius = 0.8 Å             # Smaller expansion
shrink_truncation_radius = 0.7 Å   # Moderate erosion

# Result: Tighter mask, higher estimated solvent content
```

**Use case**: Crystals with known high solvent content.

## Computational Efficiency

### Neighbor Search Optimization

The erosion step uses a **pre-computed neighbor table** to avoid checking all grid points:

```cpp
shrink_neighbors(unit_cell, gridding, shrink_truncation_radius):
    # Pre-compute which grid offsets are within shrink_truncation_radius
    for dx, dy, dz in grid_range:
        distance = unit_cell.distance(dx, dy, dz)
        if distance < shrink_truncation_radius:
            neighbor_table.add(dx, dy, dz)
    
    return neighbor_table
```

This reduces complexity from O(N × M³) to O(N × K) where:
- N = number of boundary points
- M = grid size
- K = average number of neighbors (typically ~100)

## Relationship to Morphological Operations

This is analogous to standard image processing operations:

| Phenix Operation | Image Processing Equivalent |
|------------------|----------------------------|
| Initial dilation (solvent_radius) | **Morphological dilation** |
| Erosion (shrink_truncation_radius) | **Morphological erosion** |
| Combined operation | **Morphological closing** |

**Morphological closing** = Dilation followed by Erosion

This is exactly what Phenix does! The operation:
1. Fills small gaps (dilation)
2. Smooths boundaries (erosion)
3. Preserves overall shape

## Common Issues and Solutions

### Issue 1: Mask Too Aggressive (Low Solvent Content)

**Symptom**: Estimated solvent content much lower than expected.

**Solution**: 
```python
solvent_radius = 1.0  # Reduce from 1.1
# OR
shrink_truncation_radius = 0.7  # Reduce erosion
```

### Issue 2: Fourier Artifacts in F_mask

**Symptom**: Ripples or oscillations in F_mask, high R-factors.

**Solution**:
```python
shrink_truncation_radius = 1.1  # Increase smoothing from 0.9
# OR
grid_step = 0.5  # Finer grid (from 0.6)
```

### Issue 3: Small Pockets Lost

**Symptom**: Internal cavities filled incorrectly.

**Solution**:
```python
shrink_truncation_radius = 0.7  # Reduce from 0.9
# Prevents over-erosion of narrow channels
```

### Issue 4: Mask Calculation Too Slow

**Symptom**: Mask calculation takes > 5 seconds.

**Solution**:
```python
grid_step = 0.7  # Coarser grid (from 0.6)
# OR
ignore_hydrogen_atoms = True  # Exclude H if not already
```

## Recommendations

### For Most Proteins

Use defaults - they work well:
```python
solvent_radius = 1.1 Å
shrink_truncation_radius = 0.9 Å
grid_step = 0.6 Å
```

### For High Solvent Content (> 60%)

```python
solvent_radius = 1.0 Å             # Slightly tighter
shrink_truncation_radius = 0.8 Å   # Less erosion
```

### For Low Solvent Content (< 40%)

```python
solvent_radius = 1.2 Å             # More expansion
shrink_truncation_radius = 1.0 Å   # More erosion
```

### For Very Low Resolution (> 3.5 Å)

```python
solvent_radius = 1.2 Å             
shrink_truncation_radius = 1.2 Å   # Heavy smoothing
grid_step = 0.8 Å                   # Coarser grid OK
```

## Validation

Check your mask quality:

```python
mask = masks.bulk_solvent(...)

mask.show_summary()
# Outputs:
# solvent radius:            1.10 A
# shrink truncation radius:  0.90 A
# number of atoms: 1234
# gridding:   (80, 85, 95)
# grid steps: (0.60, 0.59, 0.61) A
# estimated solvent content: 52.3 %

# Check if solvent content is reasonable
assert 0.3 < mask.contact_surface_fraction < 0.8
```

Expected solvent content:
- Proteins: 30-70% (typical 45-55%)
- Nucleic acids: 40-80%
- Protein-nucleic acid complexes: 40-65%

## Summary

**Dilation and Erosion work together:**

1. **Dilation** (via `solvent_radius`):
   - Expand protein by solvent_radius
   - Creates accessible surface
   - Ensures water molecules can't penetrate

2. **Erosion** (via `shrink_truncation_radius`):
   - Smooth the boundary
   - Remove sharp edges
   - Create contact surface

3. **Net Effect**:
   - Smooth, physically meaningful protein-solvent boundary
   - Reduced Fourier artifacts
   - Accurate bulk solvent modeling

The relationship is **NOT simply subtractive**. It's:
- First: expand by solvent_radius (creates boundary layer)
- Then: selectively erode boundary based on proximity to solvent
- Result: smooth transition optimized for crystallographic calculations

This two-stage process is more sophisticated than simple dilation/erosion and produces better masks for bulk solvent correction.

---

**See also:**
- [03_bulk_solvent_mask.md](03_bulk_solvent_mask.md) - Basic mask calculation
- [atomic_radii_reference.md](atomic_radii_reference.md) - Atomic radii tables
- [06_summary.md](06_summary.md) - Parameter reference
