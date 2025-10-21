# Kinetic Model Module

A flexible, configurable PyTorch module for modeling and fitting arbitrary kinetic schemes in photochemical reactions.

## Features

- **Flexible Configuration**: Define any kinetic scheme using intuitive flow chart strings
- **PyTorch Integration**: Fully differentiable, optimizable with standard PyTorch optimizers
- **Instrument Response**: Built-in instrument function to account for time resolution
- **Photoabsorption**: Handles initial quasi-instant conversion around t=0
- **Conservation**: Automatically ensures population conservation
- **Easy Parameter Access**: Simple methods to retrieve rate and time constants

## Installation

The module is part of the `multicopy_refinement` package. If you haven't installed it yet:

```bash
pip install -e .
```

## Quick Start

```python
import torch
import numpy as np
from multicopy_refinement.kinetics import KineticModel

# Define time points
t = np.linspace(-1, 10, 200)

# Create a simple A -> B -> C model
model = KineticModel(
    flow_chart="A->B->C",
    timepoints=t,
    instrument_function='gaussian',
    instrument_width=0.2,
    initial_state='A'
)

# Set rate constants (optional, can also be fitted)
with torch.no_grad():
    model.log_rate_constants[0] = np.log(2.0)  # A->B: k=2.0
    model.log_rate_constants[1] = np.log(0.5)  # B->C: k=0.5

# Compute populations at all time points
populations = model()

# Access parameters
print(model.get_rate_constants())
print(model.get_time_constants())
```

## Flow Chart Syntax

The `flow_chart` parameter uses a simple string syntax:

- `->` indicates a transition between states
- `&` separates multiple transitions
- State names can be any alphanumeric string

### Examples

**Sequential kinetics:**
```python
"A->B->C"  # A converts to B, then B to C
```

**With back reactions:**
```python
"A->B->A&B->C"  # A ⇄ B → C
"A->B->C->A"   # Cyclic: A → B → C → A
```

**Parallel pathways:**
```python
"A->B&A->C"              # A branches to both B and C
"A->B&A->C&B->D&C->D"    # A → B/C → D (two pathways to D)
```

**Complex schemes:**
```python
"A->B->C&B->A&C->D->B"   # Multiple connections
```

## Parameters

### `__init__` Parameters

- **`flow_chart`** (str): Kinetic scheme definition (see syntax above)
- **`timepoints`** (array-like or torch.Tensor): Time points for evaluation
- **`instrument_function`** (str, optional): Type of instrument response
  - `'gaussian'`: Gaussian IRF (default)
  - `'none'`: No instrument response
- **`instrument_width`** (float, optional): Width parameter (e.g., σ for Gaussian). Default: 0.1
- **`initial_state`** (str, optional): Which state has initial population=1. Default: first state
- **`verbose`** (int, optional): Verbosity level. Default: 1

## Methods

### `forward()`
Compute state populations at all timepoints.

**Returns:** `torch.Tensor` of shape `(n_timepoints, n_states)`

### `get_rate_constants()`
Get current rate constants as a dictionary.

**Returns:** `Dict[str, float]` mapping transition strings (e.g., "A->B") to rate constants

### `get_time_constants()`
Get time constants (1/rate) for each transition.

**Returns:** `Dict[str, float]` mapping transition strings to time constants

### `print_parameters()`
Print a formatted summary of current parameters.

## Fitting Data

The model is fully differentiable and can be optimized using PyTorch optimizers:

```python
import torch
import torch.optim as optim

# Create model
model = KineticModel(flow_chart="A->B->C", timepoints=t)

# Your experimental data
experimental_data = torch.tensor(...)  # shape: (n_timepoints, n_states)

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop
for epoch in range(1000):
    optimizer.zero_grad()
    
    # Forward pass
    predicted_populations = model()
    
    # Loss (e.g., MSE)
    loss = torch.mean((predicted_populations - experimental_data) ** 2)
    
    # Backward pass
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch+1}, Loss: {loss.item():.6f}")

# Get fitted parameters
print(model.get_rate_constants())
```

## Mathematical Background

### Kinetic Equations

The model solves the first-order kinetic differential equations:

$$\frac{dP}{dt} = K \cdot P$$

where:
- $P(t)$ is the population vector at time $t$
- $K$ is the rate matrix

### Rate Matrix

The rate matrix $K$ is constructed such that:
- $K_{ij}$ (for $i \neq j$) is the rate constant from state $j$ to state $i$
- $K_{ii} = -\sum_{j \neq i} K_{ji}$ ensures population conservation
- Each column sums to zero: $\sum_i K_{ij} = 0$

### Solution

The analytical solution is:

$$P(t) = e^{Kt} \cdot P(0)$$

The model computes the matrix exponential using PyTorch's `torch.matrix_exp`.

### Instrument Response

The Gaussian instrument response function convolves the ideal populations with:

$$G(t) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{t^2}{2\sigma^2}\right)$$

This accounts for the finite time resolution of the measurement.

## Examples

See `examples/kinetics_example.py` for comprehensive examples including:

1. Simple sequential kinetics (A → B → C)
2. Kinetics with back reactions (A ⇄ B → C)
3. Cyclic kinetics (A → B → C → A)
4. Parallel pathways (A → B/C → D)
5. Fitting synthetic data with gradient descent

Run the examples:

```bash
python examples/kinetics_example.py
```

## Testing

Run unit tests:

```bash
pytest tests/test_kinetics.py -v
```

## Tips and Best Practices

### 1. Rate Constant Initialization
The model initializes rate constants randomly. For better convergence, you can initialize them:

```python
with torch.no_grad():
    # Set initial guess for A->B rate
    model.log_rate_constants[0] = np.log(1.0)  # k ≈ 1.0
```

### 2. Learning Rate Selection
- Start with `lr=0.01` for Adam optimizer
- Reduce if training is unstable
- Use learning rate schedulers for better convergence

### 3. Instrument Function
- Use `'gaussian'` for real experimental data
- Use `'none'` for testing or when time resolution is not a concern
- Typical widths: 0.05-0.5 depending on your instrument

### 4. Time Points
- Include negative time points to see instrument response
- Use enough points (100-500) for smooth curves
- Extend range to capture full decay

### 5. Handling Multiple Datasets
To fit multiple datasets simultaneously, create separate models or batch your data:

```python
# Option 1: Multiple models
models = [KineticModel(flow_chart, t) for _ in range(n_datasets)]

# Option 2: Shared parameters (custom implementation needed)
# Share log_rate_constants across models
```

## Physical Interpretation

### Rate Constants vs. Time Constants
- **Rate constant** $k$ (units: 1/time): How fast a reaction occurs
- **Time constant** $\tau = 1/k$ (units: time): Characteristic timescale
- Larger $k$ → faster reaction → smaller $\tau$

### Population Dynamics
- At $t=0$: All population in initial state (e.g., ground state A)
- Photoabsorption creates instant population transfer (with instrument broadening)
- Populations evolve according to rate equations
- Total population is always conserved: $\sum_i P_i(t) = 1$

## Advanced Usage

### Custom Loss Functions
You can use any differentiable loss function:

```python
# Weighted MSE
weights = torch.tensor([1.0, 2.0, 1.0])  # Weight each state differently
loss = torch.mean(weights * (predicted - experimental) ** 2)

# Logarithmic loss (emphasizes small values)
loss = torch.mean((torch.log(predicted + 1e-8) - torch.log(experimental + 1e-8)) ** 2)
```

### Constraints
Add constraints using penalties or bounded optimization:

```python
# Example: Penalize very fast rates (k > 100)
penalty = torch.sum(torch.relu(torch.exp(model.log_rate_constants) - 100.0))
loss = mse_loss + 0.1 * penalty
```

### Multiple Optimizers
Optimize different parameters with different learning rates:

```python
optimizer = optim.Adam([
    {'params': [model.log_rate_constants], 'lr': 0.01},
    {'params': [model.log_instrument_width], 'lr': 0.001}
])
```

## Troubleshooting

### NaN or Inf values
- Check that rate constants are reasonable (not too large)
- Reduce learning rate
- Check for numerical instability in time points

### Poor convergence
- Try different initialization
- Adjust learning rate
- Increase number of epochs
- Check that flow chart matches your physical system

### Negative populations
- Should not occur if implementation is correct
- Check for bugs in rate matrix construction
- Verify that instrument function preserves normalization

## Citation

If you use this module in your research, please cite:

```
[Your citation here]
```

## License

[Your license here]

## Contact

[Your contact information]
