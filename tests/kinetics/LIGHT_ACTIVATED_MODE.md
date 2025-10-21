# Light-Activated Reaction Mode

## Problem

In light-activated photochemical reactions (like bacteriorhodopsin, rhodopsin, etc.), photoexcitation can only happen **once per molecule**. If the photocycle returns to the ground state (e.g., `O → A`), those molecules should **not** be able to undergo photoactivation again.

Without this constraint, the model allows molecules to cycle indefinitely:
```
A --light--> K → L → M → N → O → A --light--> K → ...
```

This causes **unphysical amplification** where populations can grow beyond 1, especially with:
- Long time ranges (ps to ms)
- Reactions that return to the initial state
- Logarithmic time scales

## Solution: `light_activated=True`

Enable light-activated mode to ensure photoexcitation happens only once:

```python
model = KineticModel(
    flow_chart='A->K,K->L,L->M,M->N,N->O,O->A',
    timepoints=t,
    light_activated=True  # KEY PARAMETER
)
```

### What It Does

1. **Creates an inactive state** `A*` (ground state that has already reacted)
2. **Redirects back-reactions**: Any transition returning to `A` from other states now goes to `A*` instead
3. **Prevents re-photoactivation**: `A*` cannot undergo the initial photoexcitation (`A*` → K blocked)
4. **Combines for display**: Plots show `A` as the sum of `A + A*` (total ground state population)

### Example

**Flow chart:** `A->K,K->L,L->M,M->N,N->O,O->A`

**Without `light_activated`:**
```python
States: ['A', 'K', 'L', 'M', 'N', 'O']
Transitions: [('A', 'K'), ('K', 'L'), ('L', 'M'), ('M', 'N'), ('N', 'O'), ('O', 'A')]
# Problem: O→A allows cycling back to photoactivatable A
```

**With `light_activated=True`:**
```python
States: ['A', 'K', 'L', 'M', 'N', 'O', 'A*']
Transitions: [('A', 'K'), ('K', 'L'), ('L', 'M'), ('M', 'N'), ('N', 'O'), ('O', 'A*')]
# Solution: O→A* prevents re-photoactivation
```

### Physical Interpretation

- **A (active)**: Ground state molecules that can be photoexcited
- **A\* (inactive)**: Ground state molecules that have already been through the photocycle and cannot be re-excited
- **Total ground state**: A + A*

At equilibrium:
- Without light_activated: Continuous cycling leads to amplification
- With light_activated: Molecules accumulate in A* (reacted ground state)

## Usage Examples

### Bacteriorhodopsin

```python
import torch
from multicopy_refinement.kinetics import KineticModel

# Logarithmic timepoints (ps to ms)
t = torch.logspace(-1, 6, 1000)  # 0.1 ps to 1 ms

model = KineticModel(
    flow_chart='BR->K,K->L,L->M,M->N,N->O,O->BR',
    timepoints=t,
    rate_constants={
        'BR->K': 1/3,        # τ = 3 ps
        'K->L': 1/10000,     # τ = 10 μs  
        'L->M': 1/2e6,       # τ = 2 ms
        # ...
    },
    light_activated=True  # Essential!
)

# Plot will show BR as sum of BR + BR*
model.plot_occupancies('br_photocycle.png', log=True)
```

### Rhodopsin

```python
model = KineticModel(
    flow_chart='Rh->Batho,Batho->Lumi,Lumi->Meta,Meta->Rh',
    timepoints=torch.logspace(-9, 0, 1000),  # fs to ns
    light_activated=True
)
```

### When NOT to Use

- **Thermal reactions**: If the reaction is thermally driven (not light-activated), use `light_activated=False` (default)
- **No back-reaction**: If the photocycle doesn't return to the initial state (e.g., `A->B->C` with no cycle), the flag doesn't matter
- **Multiple photoexcitations allowed**: Some systems can be re-excited (use with caution)

## Implementation Details

### State Creation
```python
if light_activated:
    inactive_state = initial_state + '*'  # e.g., 'A' → 'A*'
    states.append(inactive_state)
```

### Transition Redirection
```python
for from_state, to_state in transitions:
    if to_state == initial_state and from_state != initial_state:
        # Redirect: X→A becomes X→A*
        modified_transitions.append((from_state, inactive_state))
```

### Combined Plotting
The `plot_occupancies()` method automatically combines:
```python
combined_A = population(A) + population(A*)
# Displayed as "State A" in plots
```

## Testing

See test files:
- `test_light_activated.py`: Comparison with/without light-activated mode
- `tests_BR_like.py`: Bacteriorhodopsin-like photocycle
- `test_log_timepoints.py`: Logarithmic time scales

## Summary

| Mode | Behavior | Use Case |
|------|----------|----------|
| `light_activated=False` (default) | Allows cycling: A→...→A→... | Thermal reactions, enzyme catalysis |
| `light_activated=True` | Single photoexcitation: A→...→A* | Light-activated photocycles (BR, rhodopsin, etc.) |

**Key Point**: For photochemical reactions spanning large time scales (ps to ms), always use `light_activated=True` to prevent unphysical amplification!
