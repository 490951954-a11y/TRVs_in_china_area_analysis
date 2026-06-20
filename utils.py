"""
Utility Functions
=================
Common utility functions for typhoon tracking.
"""

import numpy as np
from scipy import ndimage
from typing import List, Tuple, Optional
from geographiclib.geodesic import Geodesic


def calculate_spherical_distance(lat1: float, lon1: float, 
                                 lat2: float, lon2: float) -> float:
    """
    Calculate spherical distance between two points in degrees.
    
    Args:
        lat1, lon1: First point coordinates
        lat2, lon2: Second point coordinates
        
    Returns:
        Distance in degrees
    """
    lon1 = (lon1 + 180) % 360 - 180
    lon2 = (lon2 + 180) % 360 - 180
    geod = Geodesic.WGS84
    result = geod.Inverse(lat1, lon1, lat2, lon2)
    return result['s12'] / 111319.9  # 1 degree ≈ 111319.9 meters


def find_minima_in_contour(psi_field: np.ndarray, 
                           contour_mask: np.ndarray) -> List[Tuple[int, int, float]]:
    """
    Find all local minima within a contour.
    
    Args:
        psi_field: Streamfunction field
        contour_mask: Boolean mask of contour region
        
    Returns:
        List of (row, col, psi_value) tuples sorted by value ascending
    """
    neighborhood = np.ones((3, 3), dtype=bool)
    neighborhood[1, 1] = False
    local_min = ndimage.minimum_filter(psi_field, footprint=neighborhood, mode='nearest')
    min_mask = psi_field < local_min
    
    contour_min_mask = min_mask & contour_mask
    minima_rows, minima_cols = np.where(contour_min_mask)
    
    minima_list = []
    for row, col in zip(minima_rows, minima_cols):
        psi_val = psi_field[row, col]
        minima_list.append((row, col, psi_val))
    
    minima_list.sort(key=lambda x: x[2])
    return minima_list


def intensity_code_to_string(code: int) -> str:
    """Convert intensity code to description."""
    intensity_map = {
        0: "弱于热带低压或未知",
        1: "热带低压(TD)",
        2: "热带风暴(TS)",
        3: "强热带风暴(STS)",
        4: "台风(TY)",
        5: "强台风(STY)",
        6: "超强台风(SuperTY)",
        9: "变性"
    }
    return intensity_map.get(code, "未知")


def find_nearest_grid(lat: float, lon: float, lat_array: np.ndarray, 
                      lon_array: np.ndarray) -> Tuple[int, int]:
    """
    Find the nearest grid indices for given latitude/longitude.
    
    Args:
        lat, lon: Target coordinates
        lat_array, lon_array: Grid arrays
        
    Returns:
        (lat_idx, lon_idx) tuple
    """
    if lon < 0:
        lon = lon + 360
    lat_idx = np.argmin(np.abs(lat_array - lat))
    lon_idx = np.argmin(np.abs(lon_array - lon))
    return lat_idx, lon_idx


def normalize_longitude(lon: float) -> float:
    """Normalize longitude to 0-360 range."""
    return (lon + 360) % 360


def calculate_grid_resolution(lon_array: np.ndarray) -> float:
    """Calculate grid resolution from longitude array."""
    if len(lon_array) > 1 and not np.isnan(np.diff(lon_array)).any():
        return np.mean(np.diff(lon_array))
    return 0.25  # Default resolution