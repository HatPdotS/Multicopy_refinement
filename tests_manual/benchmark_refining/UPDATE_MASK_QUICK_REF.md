# update_refinable_mask() Quick Reference

## When to Use

### Use `update_refinable_mask()` when:
- ✅ You know the exact final refinement pattern upfront
- ✅ Setting complex patterns from scratch (e.g., mainchain only)
- ✅ Implementing custom selection logic
- ✅ Working directly with group indices (advanced)
- ✅ Need maximum efficiency (one operation vs multiple)

### Use `freeze()`/`unfreeze()` when:
- ✅ Modifying existing refinement state incrementally
- ✅ Interactive or step-by-step freezing/unfreezing
- ✅ More intuitive for sequential operations
- ✅ Building up complex patterns step by step

## Basic Usage

```python
# Full atom space (default, most common)
mask = torch.zeros(n_atoms, dtype=torch.bool)
mask[indices] = True
occ.update_refinable_mask(mask, in_compressed_space=False)

# Compressed space (advanced, for performance)
group_mask = torch.zeros(n_groups, dtype=torch.bool)
group_mask[group_indices] = True
occ.update_refinable_mask(group_mask, in_compressed_space=True)
```

## Common Patterns

### Pattern 1: Mainchain Only
```python
# Assuming you have atom info
mainchain = df[df['atom_name'].isin(['N', 'CA', 'C', 'O'])]['atom_idx'].values
mask = torch.zeros(n_atoms, dtype=torch.bool)
mask[mainchain] = True
occ.update_refinable_mask(mask)
```

### Pattern 2: Sidechain Only
```python
sidechain = df[~df['atom_name'].isin(['N', 'CA', 'C', 'O'])]['atom_idx'].values
mask = torch.zeros(n_atoms, dtype=torch.bool)
mask[sidechain] = True
occ.update_refinable_mask(mask)
```

### Pattern 3: Active Site
```python
active_site = df[(df['resseq'] >= 100) & (df['resseq'] <= 120)]['atom_idx'].values
mask = torch.zeros(n_atoms, dtype=torch.bool)
mask[active_site] = True
occ.update_refinable_mask(mask)
```

### Pattern 4: High B-factor Atoms
```python
current_b = model.b()
high_b_mask = current_b > 50.0
occ.update_refinable_mask(high_b_mask)
```

### Pattern 5: Exclude Waters
```python
non_water = df[df['resname'] != 'HOH']['atom_idx'].values
mask = torch.zeros(n_atoms, dtype=torch.bool)
mask[non_water] = True
occ.update_refinable_mask(mask)
```

### Pattern 6: Alternating Groups (Advanced)
```python
# Work in compressed space for efficiency
n_groups = occ._collapsed_shape
alternating = torch.tensor([i % 2 == 0 for i in range(n_groups)])
occ.update_refinable_mask(alternating, in_compressed_space=True)
```

### Pattern 7: Complex Boolean Logic
```python
# Combine multiple conditions
mainchain = df['atom_name'].isin(['N', 'CA', 'C', 'O'])
in_active_site = (df['resseq'] >= 100) & (df['resseq'] <= 120)
not_water = df['resname'] != 'HOH'

combined_condition = mainchain & in_active_site & not_water
mask = torch.zeros(n_atoms, dtype=torch.bool)
mask[df[combined_condition]['atom_idx'].values] = True
occ.update_refinable_mask(mask)
```

## Comparison with freeze/unfreeze

```python
# Scenario: Refine mainchain in active site

# Approach 1: Multiple operations with freeze/unfreeze
occ.freeze_all()  # Start with all frozen
occ.unfreeze(mainchain_mask)  # Unfreeze mainchain
occ.freeze(non_active_site_mask)  # Freeze outside active site

# Approach 2: Single operation with update_refinable_mask
final_mask = mainchain_mask & active_site_mask
occ.update_refinable_mask(final_mask)  # One operation

# Approach 2 is:
# - More efficient (one operation vs three)
# - Clearer intent (explicit final state)
# - Less error-prone (no state accumulation)
```

## Performance Notes

### Full Atom Space (default)
- **Complexity**: O(n_atoms) for mask collapse
- **Use when**: Working with atom selections (most common)
- **Best for**: Intuitive, user-friendly code

### Compressed Space (advanced)
- **Complexity**: O(1) for direct assignment
- **Use when**: Working directly with sharing groups
- **Best for**: Maximum performance, advanced users

```python
# Full space (easy, slightly slower)
occ.update_refinable_mask(atom_mask, in_compressed_space=False)

# Compressed space (harder, faster)
occ.update_refinable_mask(group_mask, in_compressed_space=True)
```

## Error Handling

```python
# Wrong shape in full atom space
try:
    wrong_mask = torch.zeros(999, dtype=torch.bool)  # Wrong size!
    occ.update_refinable_mask(wrong_mask, in_compressed_space=False)
except ValueError as e:
    print(f"Error: {e}")
    # Output: "Mask in full atom space must have shape (n_atoms,), got shape (999,)"

# Wrong shape in compressed space
try:
    wrong_mask = torch.zeros(50, dtype=torch.bool)  # Wrong size!
    occ.update_refinable_mask(wrong_mask, in_compressed_space=True)
except ValueError as e:
    print(f"Error: {e}")
    # Output: "Mask in compressed space must have shape (n_groups,), got shape (50,)"
```

## Integration with Optimization

```python
# Set refinement pattern
occ.update_refinable_mask(my_selection_mask)

# Create optimizer with current refinable params
optimizer = torch.optim.Adam([occ.refinable_params], lr=0.01)

# Run optimization
for i in range(100):
    optimizer.zero_grad()
    loss = compute_loss(occ())
    loss.backward()
    optimizer.step()

# Change pattern and continue
new_mask = compute_new_selection()
occ.update_refinable_mask(new_mask)

# IMPORTANT: Create new optimizer after changing mask!
optimizer = torch.optim.Adam([occ.refinable_params], lr=0.01)
```

## Tips and Tricks

### Tip 1: Visualize Before Applying
```python
# Check which atoms will be affected
test_mask = your_complex_mask_logic()
n_affected = test_mask.sum().item()
print(f"Will make {n_affected} atoms refinable")

# If satisfied, apply
occ.update_refinable_mask(test_mask)
```

### Tip 2: Save and Restore Patterns
```python
# Save current pattern
saved_refinable = occ.get_refinable_atoms()

# Try different pattern
occ.update_refinable_mask(experimental_mask)
# ... do some work ...

# Restore original pattern
occ.update_refinable_mask(saved_refinable)
```

### Tip 3: Combine with Pandas for Complex Selections
```python
import pandas as pd

# Load atom info
df = pd.read_csv('atoms.csv')

# Complex query
selection = df.query("""
    (resname == 'LYS' and atom_name == 'NZ') or
    (resname == 'ARG' and atom_name.isin(['NH1', 'NH2'])) or
    (resname == 'HIS' and atom_name.isin(['ND1', 'NE2']))
""")['atom_idx'].values

# Apply selection
mask = torch.zeros(n_atoms, dtype=torch.bool)
mask[selection] = True
occ.update_refinable_mask(mask)
```

### Tip 4: Progressive Expansion
```python
# Start with core
core_mask = torch.zeros(n_atoms, dtype=torch.bool)
core_mask[core_indices] = True
occ.update_refinable_mask(core_mask)
refine(steps=50)

# Expand to shell
shell_mask = core_mask.clone()
shell_mask[shell_indices] = True
occ.update_refinable_mask(shell_mask)
refine(steps=50)

# Expand to full
full_mask = torch.ones(n_atoms, dtype=torch.bool)
occ.update_refinable_mask(full_mask)
refine(steps=50)
```

## Summary

`update_refinable_mask()` is ideal for:
- ✅ Setting explicit refinement patterns
- ✅ Complex selection logic
- ✅ Performance-critical code
- ✅ Clear, declarative refinement strategies

Use it when you know exactly what you want to refine!
