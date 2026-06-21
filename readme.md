# Typhoon Tracking Library

A comprehensive Python library for tropical cyclone tracking using streamfunction and vorticity data from ERA5 reanalysis.

## Features

- **Vortex Tracking**: Track tropical cyclones using streamfunction minima detection
- **Vorticity Calculation**: Compute streamfunction from vorticity fields
- **Data Conversion**: Convert tracking results to standard TRV format with vorticity and wind speed
- **Extratropical Transition Detection**: Identify and flag extratropical transition events
- **Multiple Data Formats**: Support for ERA5 NetCDF, BST, and TRV CSV formats
- **Comprehensive Output**: Generate detailed tracking summaries and unified datasets

## Installation

```bash
pip install -e .



dependencies = [
    "numpy>=1.20.0",
    "pandas>=1.3.0",
    "xarray>=0.20.0",
    "netCDF4>=1.5.0",
    "scipy>=1.7.0",
    "metpy>=1.0.0",
    "matplotlib>=3.4.0",
    "cartopy>=0.20.0",
    "geographiclib>=1.52",
    "pyshp>=2.1.0",
    "shapely>=1.8.0",
]


typhoon_tracking/
├── __init__.py          # Package initialization
├── tracker.py           # Core tracking algorithm
├── converter.py         # CSV to TRV format conversion
├── extratropical.py     # Extratropical transition detection
├── readers.py           # BST and TRV data parsers
├── utils.py             # Utility functions
└── cli.py               # Command line interface



Quick Start
1. Run Typhoon Tracking

from typhoon_tracking import StreamfunctionTyphoonTracker

# Initialize tracker
tracker = StreamfunctionTyphoonTracker(
    max_track_hours=120,
    psi_min_threshold=-1e5,
    max_hourly_distance=2.0,
    enable_vis=True
)

# Load end positions
tracker.load_end_positions("TRVStartPositions.csv")

# Process all typhoons (file discovery and tracking logic)
# See cli.py for complete workflow
2. Convert Results to TRV Format

from typhoon_tracking import CSVToTRVConverter

converter = CSVToTRVConverter(
    target_level=850,
    level_index=0,
    search_radius_km=200
)

converter.parse_input_csv("track_results/all_typhoons_tracking.csv")
converter.convert_to_trv_format(
    "TRV_test.csv",
    vor_file="vor_19800829.nc",
    uv_file="uv_19800829.nc"
)
3. Detect Extratropical Transitions

from typhoon_tracking import ExtratropicalTRVMatcher

matcher = ExtratropicalTRVMatcher()
matcher.match_and_add_flag(
    bst_directory="D:/CMABSTdata",
    trv_file_path="TRV_test.csv",
    output_dir="."
)
Command Line Usage

Tracking

python -m typhoon_tracking.cli track TRVStartPositions.csv E:/era5_vorticity_data \
    --output-dir track_results \
    --shp-path ./bou1_4m/bou1_4p.shp
Conversion

python -m typhoon_tracking.cli convert all_typhoons_tracking.csv TRV_test.csv \
    --vor-file vor_19800829.nc \
    --uv-file uv_19800829.nc

Extratropical Detection

python -m typhoon_tracking.cli extratropical D:/CMABSTdata TRV_test.csv --output-dir .


Data Formats
Input CSV (End Positions)
Required columns:

Typhoon Name: Name of the typhoon

End Time: End time in format YYYYMMDDHH

End Latitude: End latitude in degrees

End Longitude: End longitude in degrees

International ID: International typhoon ID

Tropical Cyclone Serial: TC serial number

China ID: Chinese typhoon ID

Source File: Source data file

TRV Format
Header (8 fields):
66666,INTL_ID,RECORD_COUNT,SEQ_NO,CHINA_ID,STOP_REASON,NAME,START_DATE


Track (6 fields):
YYYYMMDDHH,LAT,LON,STREAM_FUNC,VORTICITY,VELOCITY


Track with transition flag (7 fields):
YYYYMMDDHH,LAT,LON,STREAM_FUNC,VORTICITY,VELOCITY,TRANSITION_FLAG


%if this Python library have some problems，try files in ./example 