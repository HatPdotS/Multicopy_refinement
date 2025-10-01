
# Multicopy Refinement

A Python package for crystallographic refinement of low occupancy states while assuming a known high occupancy state against structural data. This tool is particularly useful for studying conformational states, ligand binding, or other phenomena where multiple structural states coexist in crystallographic data.

## Overview

Multicopy refinement addresses the challenge of refining crystal structures that contain multiple conformational states or occupancy levels. Traditional refinement methods often struggle with low-occupancy states, but this package leverages knowledge of the high-occupancy state to improve refinement of the alternative conformations.

## Features

- **Direct summation structure factor calculations** - Currently the main working method
- Working on **FT** based methods but not available yet, main issus hkl extraction from reciporcal grid. If you have any ideas let me know.
- **PyTorch-based optimization** - Leverages automatic differentiation for efficient parameter optimization
- **Restraints handling** - Support for geometric and other crystallographic restraints
- **Absorption and extinction corrections** - Built-in corrections for experimental effects
- **Flexible model parameterization** - Support for various refinement strategies
- **Integration with crystallographic libraries** - Uses gemmi, reciprocalspaceship, and optionally cctbx

## Installation

### Requirements
- Python ≥ 3.8
- NumPy ≥ 1.20.0
- PyTorch ≥ 1.9.0
- gemmi ≥ 0.5.0
- reciprocalspaceship ≥ 0.9.0

### Install from source
```bash
git clone https://github.com/HatPdotS/Multicopy_refinement.git
cd Multicopy_refinement
pip install -e .
```

### Optional dependencies
For additional crystallographic functionality:
```bash
pip install -e .[crystallography]
```

For development:
```bash
pip install -e .[dev]
```

## Quick Start

```python
import multicopy_refinement as mcr
from multicopy_refinement.Model import model

# Load your structural model
my_model = model(model="your_structure.pdb")

# Set up corrections
my_model.setup_absorption()
my_model.setup_extinction()

# Configure refinement parameters
# ... refinement setup code ...

# Run refinement
# ... refinement execution code ...
```

## Core Components

### Model Class
The main `Model` class handles:
- Structure factor calculations
- Parameter optimization via PyTorch
- Absorption and extinction corrections
- Multiple conformational states

### Direct Summation
Currently the primary method for structure factor calculation:
- Handles multiple copies/conformations
- Efficient implementation using numba
- Support for various space groups (limited)

### Restraints System
Comprehensive restraints handling:
- Geometric restraints (bonds, angles, dihedrals)
- B-factor restraints
- Occupancy constraints
- Custom restraint definitions

## Current Limitations

- **Space group support**: Severe limitations in available space groups
- **Fourier Transform methods**: FT-based methods are in development but not fully functional
- **Documentation**: Limited API documentation (work in progress)

## Development Status

This package is under active development. The direct summation approach is stable and working, while Fourier Transform-based methods require additional work to be fully functional.

## Future Development

- Expand space group support
- Complete Fourier Transform implementation
- Improve computational efficiency
- Add comprehensive documentation and tutorials
- Implement additional refinement strategies

## Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

## Contact

For questions or support, send me an email hans.seidel@psi.ch
