"""
I/O module for reading crystallographic data files.

This module provides readers for various crystallographic file formats:
- MTZ files (structure factors)
- PDB files (structure models)
- CIF/mmCIF files (structure factors, models, or restraints)

The DataRouter class automatically detects file types and selects
the appropriate reader.
"""

from .cif_readers import (
    CIFReader,
    ReflectionCIFReader,
    ModelCIFReader,
    RestraintCIFReader,
)

from .legacy_format_readers import (
    MTZ,
    PDB,
)

from .data_router import (
    DataRouter,
    DataRouterError,
)

__all__ = [
    # CIF readers
    'CIFReader',
    'ReflectionCIFReader',
    'ModelCIFReader',
    'RestraintCIFReader',
    # Legacy readers
    'MTZ',
    'PDB',
    # Router
    'DataRouter',
    'DataRouterError',
]
