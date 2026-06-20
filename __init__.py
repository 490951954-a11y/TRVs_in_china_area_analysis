"""
Typhoon Tracking Library
=======================
A comprehensive library for tropical cyclone tracking using streamfunction 
vorticity data from ERA5 reanalysis.

Modules:
- tracker: Core typhoon tracking using streamfunction minima
- converter: Convert tracking results to TRV format with vorticity/wind data
- extratropical: Detect and flag extratropical transition events
- readers: Parse BST and TRV data files
- utils: Utility functions for calculations and visualization
"""

__version__ = "2.0.0"
__author__ = "YE"

from .tracker import StreamfunctionTyphoonTracker
from .converter import CSVToTRVConverter
from .extratropical import ExtratropicalTRVMatcher
from .readers import TCRLParser, CMABSTDataParserFixed
from .utils import calculate_spherical_distance, find_minima_in_contour

__all__ = [
    'StreamfunctionTyphoonTracker',
    'CSVToTRVConverter', 
    'ExtratropicalTRVMatcher',
    'TCRLParser',
    'CMABSTDataParserFixed',
    'calculate_spherical_distance',
    'find_minima_in_contour'
]