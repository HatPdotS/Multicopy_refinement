# Kinetics Module - Tests and Documentation

This directory contains all tests, demonstrations, and documentation for the `KineticModel` module.

## 📁 Contents

### Test Files
- **`test_kinetics_quick.py`** - Quick verification tests for all features
  - Run: `/das/work/p17/p17490/CONDA/muticopy_refinement/bin/python test_kinetics_quick.py`

- **`demo_new_features.py`** - Comprehensive demonstration of all features
  - Run: `/das/work/p17/p17490/CONDA/muticopy_refinement/bin/python demo_new_features.py`

### Documentation Files

#### Quick References
- **`QUICK_REFERENCE.md`** - One-page quick reference card
- **`KINETICS_QUICK_REFERENCE.md`** - Original quick reference (legacy)

#### Comprehensive Documentation
- **`KINETICS_README.md`** - Complete user guide with examples
- **`UPDATE_SUMMARY.md`** - Summary of all recent changes and new features
- **`KINETICS_IMPLEMENTATION.md`** - Technical implementation details

## 🚀 Quick Start

### Run Tests
```bash
cd /das/work/p17/p17490/Peter/Library/multicopy_refinement/tests/kinetics
/das/work/p17/p17490/CONDA/muticopy_refinement/bin/python test_kinetics_quick.py
```

### Run Demo
```bash
cd /das/work/p17/p17490/Peter/Library/multicopy_refinement/tests/kinetics
/das/work/p17/p17490/CONDA/muticopy_refinement/bin/python demo_new_features.py
```

## 📚 Documentation Overview

### For Quick Help
→ Read **`QUICK_REFERENCE.md`**

### For Complete Guide
→ Read **`KINETICS_README.md`**

### For Recent Changes
→ Read **`UPDATE_SUMMARY.md`**

### For Implementation Details
→ Read **`KINETICS_IMPLEMENTATION.md`**

## ✨ Key Features

- **Comma-based syntax**: `"A->B,B->C,C->D"`
- **Two parameters per transition**: Rate constant (k) and efficiency (η)
- **Flexible initialization**: Dict or list format
- **Refinable instrument function**: Width is a learnable parameter
- **Easy parameter access**: `get_all_tensors()`, `get_rate_constants()`, etc.
- **Built-in visualization**: `plot_occupancies()` with log scale support

## 🎯 Example

```python
from multicopy_refinement.kinetics import KineticModel

model = KineticModel(
    flow_chart="A->B,B->C",
    timepoints=np.linspace(-1, 10, 200),
    rate_constants={"A->B": 2.0, "B->C": 0.5},
    efficiencies={"A->B": 0.9, "B->C": 0.8},
    instrument_width=0.2
)

populations = model()
model.plot_occupancies('output.png')
model.print_parameters()
```

## 📊 Test Results

All tests pass:
- ✓ Basic Functionality
- ✓ Flow Chart Parsing (Comma Syntax)
- ✓ Gradient Computation (k and η)
- ✓ Parameter Access & Efficiencies
- ✓ Instrument Function & Visualization
