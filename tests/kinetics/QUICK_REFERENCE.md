# Kinetic Model - Quick Reference

## Create Model

```python
from multicopy_refinement.kinetics import KineticModel

model = KineticModel(
    flow_chart="A->B,B->C,C->D",           # Comma-separated transitions
    timepoints=t,                           # Time array
    rate_constants={"A->B": 2.0, ...},     # Optional: k values (dict or list)
    efficiencies={"A->B": 0.9, ...},       # Optional: η values 0-1 (dict or list)
    instrument_function='gaussian',         # 'gaussian' or 'none'
    instrument_width=0.2,                   # σ for Gaussian IRF
    initial_state='A',                      # Starting state (default: first)
    verbose=1                               # 0=quiet, 1=verbose
)
```

## Get Parameters

```python
rates = model.get_rate_constants()     # Dict: {"A->B": k, ...}
effs = model.get_efficiencies()        # Dict: {"A->B": η, ...}
eff_rates = model.get_effective_rates()  # Dict: {"A->B": k*η, ...}
times = model.get_time_constants()     # Dict: {"A->B": 1/(k*η), ...}
tensors = model.get_all_tensors()      # List: [log_k, logit_η, log_σ]
```

## Print Summary

```python
model.print_parameters()  # Shows k, η, k*η, τ, and σ
```

## Compute & Visualize

```python
populations = model()  # Forward pass: (n_timepoints, n_states)

model.plot_occupancies('output.png', log=False)  # Linear scale
model.plot_occupancies('output.png', log=True)   # Log scale
model.visualize('output.png')                    # Alias
```

## Fit to Data

```python
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(1000):
    optimizer.zero_grad()
    predicted = model()
    loss = torch.mean((predicted - experimental_data) ** 2)
    loss.backward()
    optimizer.step()
```

## Examples

### Sequential
```python
flow_chart="A->B,B->C"
```

### With Back Reaction  
```python
flow_chart="A->B,B->A,B->C"
```

### Cyclic
```python
flow_chart="A->B,B->C,C->A"
```

### Parallel Pathways
```python
flow_chart="A->B,A->C,B->D,C->D"
```

## Parameter Access

| What | Method | Returns |
|------|--------|---------|
| k values | `get_rate_constants()` | `{"A->B": 2.0, ...}` |
| η values | `get_efficiencies()` | `{"A->B": 0.9, ...}` |
| k*η | `get_effective_rates()` | `{"A->B": 1.8, ...}` |
| τ = 1/(k*η) | `get_time_constants()` | `{"A->B": 0.556, ...}` |
| All params | `get_all_tensors()` | `[tensor, tensor, tensor]` |

## Tips

- Use **dict** for clarity: `{"A->B": 2.0, "B->C": 0.5}`
- Use **list** for speed: `[2.0, 0.5]` (matches flow_chart order)
- Set `efficiencies=None` to default all to 1.0 (100% efficient)
- Use `log=True` for long timescales
- Access raw parameters: `model.log_rate_constants`, `model.logit_efficiencies`, `model.log_instrument_width`
