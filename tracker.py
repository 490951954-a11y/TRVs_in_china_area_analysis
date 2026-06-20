"""
Typhoon Tracking Module
=======================
Core tracking algorithm using streamfunction minima detection from vorticity data.
"""

import os
import gc
import numpy as np
import pandas as pd
import xarray as xr
import metpy.calc as mpcalc
from metpy.units import units
from scipy import ndimage
from datetime import datetime, timedelta
import warnings
from typing import Optional, List, Dict, Tuple
from geographiclib.geodesic import Geodesic
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.ticker as mticker
import shapefile

from .utils import calculate_spherical_distance, find_minima_in_contour

warnings.filterwarnings('ignore')

# Configure Chinese font display
plt.rcParams['font.sans-serif'] = ['SimHei', 'WenQuanYi Micro Hei', 'Heiti TC']
plt.rcParams['axes.unicode_minus'] = False


def solve_poisson_iterative(vorticity, dx, dy, max_iter=1500, tol=1e4):
    """Solve Poisson equation iteratively for streamfunction."""
    ny, nx = vorticity.shape
    psi = np.zeros_like(vorticity)
    alpha = 1.6
    beta = dx**2 * dy**2 / (2 * (dx**2 + dy**2))
    
    for iteration in range(max_iter):
        max_diff = 0.0
        for i in range(1, ny-1):
            for j in range(1, nx-1):
                new_val = beta * (
                    (psi[i+1, j] + psi[i-1, j]) / dy**2 +
                    (psi[i, j+1] + psi[i, j-1]) / dx**2 -
                    (vorticity[i, j])
                )
                diff = alpha * (new_val - psi[i, j])
                psi[i, j] += diff
                max_diff = max(max_diff, abs(diff))
        if max_diff < tol:
            break
    return psi


def solve_poisson_equation_fast(vorticity, dx, dy):
    """Fast Poisson solver wrapper."""
    vorticity_values = vorticity.values if hasattr(vorticity, 'values') else vorticity
    dx_val = abs(dx.magnitude) if hasattr(dx, 'magnitude') else abs(dx)
    dy_val = abs(dy.magnitude) if hasattr(dy, 'magnitude') else abs(dy)
    return solve_poisson_iterative(vorticity_values, dx_val, dy_val)


def calculate_streamfunction(vorticity_da, lat_dim='latitude', lon_dim='longitude'):
    """Calculate streamfunction from vorticity data."""
    if len(vorticity_da.dims) != 2:
        raise ValueError(
            f"Streamfunction calculation requires 2D vorticity data ({lat_dim}, {lon_dim}), "
            f"but input has {len(vorticity_da.dims)} dimensions ({vorticity_da.dims})"
        )
    
    vorticity_da = vorticity_da.where(np.isfinite(vorticity_da), 0)
    vorticity_da = vorticity_da.clip(min=-1e-3, max=1e-3)
    
    lat = vorticity_da[lat_dim].values
    if lat[0] < lat[-1]:
        lat = lat[::-1]
        vorticity_da = vorticity_da.isel({lat_dim: slice(None, None, -1)})
    
    lon = vorticity_da[lon_dim].values
    dx, dy = mpcalc.lat_lon_grid_deltas(lon, lat)
    
    dx_mean = abs(np.mean(dx.magnitude))
    dy_mean = abs(np.mean(dy.magnitude))
    
    psi_values = solve_poisson_equation_fast(vorticity_da, dx_mean, dy_mean)
    psi_da = xr.DataArray(
        psi_values,
        coords={lat_dim: vorticity_da[lat_dim].values, lon_dim: vorticity_da[lon_dim].values},
        dims=[lat_dim, lon_dim],
        attrs={'long_name': 'stream_function', 'units': 'm²/s'}
    )
    return psi_da


def read_typhoon_end_positions(csv_path: str) -> pd.DataFrame:
    """Read end position CSV file and parse time format."""
    try:
        df = pd.read_csv(csv_path, encoding='utf-8')
        
        required_cols = ['Typhoon Name', 'End Time', 'End Latitude', 'End Longitude', 'Source File',
                         'International ID', 'Tropical Cyclone Serial', 'China ID']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"End position CSV missing required columns: {missing_cols}")
        
        df['End Time_str'] = df['End Time'].astype(str)
        df['End Time_dt'] = pd.to_datetime(
            df['End Time_str'],
            format='%Y%m%d%H',
            errors='coerce'
        )
        
        invalid_mask = df['End Time_dt'].isna()
        invalid_count = invalid_mask.sum()
        if invalid_count > 0:
            invalid_names = df.loc[invalid_mask, 'Typhoon Name'].tolist()
            print(f"⚠️  Skipping invalid time records: {invalid_names} (total {invalid_count} records)")
        
        return df[~invalid_mask].reset_index(drop=True)
    
    except Exception as e:
        print(f"❌ Failed to read end position CSV: {str(e)}")
        raise


def check_vortex_merged(psi_field: np.ndarray, contour_mask: np.ndarray, 
                        target_min_idx: Tuple[int, int]) -> Tuple[bool, List[Tuple[int, int, float]], float]:
    """Check if vortex has merged with another."""
    all_minima = find_minima_in_contour(psi_field, contour_mask)
    if not all_minima:
        return False, [], np.nan
    
    target_row, target_col = target_min_idx
    target_psi = psi_field[target_row, target_col]
    
    min_psi_in_contour = all_minima[0][2]
    if min_psi_in_contour < target_psi * 1.1:
        print(f"⚠️  Vortex merger detected: smaller minimum exists within contour!")
        print(f"  Target minimum (current typhoon): position({target_row},{target_col}), streamfunction value {target_psi:.2e} m²/s")
        print(f"  Minimum minimum within contour (dominant vortex): position({all_minima[0][0]},{all_minima[0][1]}), streamfunction value {min_psi_in_contour:.2e} m²/s")
        return True, all_minima, target_psi
    else:
        print(f"✅ Vortex independence check passed: no smaller minimum within contour")
        print(f"  Target minimum (current typhoon): streamfunction value {target_psi:.2e} m²/s")
        print(f"  Minimum minimum within contour: streamfunction value {min_psi_in_contour:.2e} m²/s")
        return False, all_minima, target_psi


class StreamfunctionTyphoonTracker:
    """Main typhoon tracker using streamfunction minima detection."""
    
    def __init__(self, delta_psi=2.0e6, min_lifetime=1, detection_threshold=1.0e6,
                 max_track_hours=120, level_idx: int = 0, boundary_buffer: float = 0.25,
                 psi_min_threshold: float = 0.0, delta_psi_contour: float = 8e4,
                 search_radius: int = 8, global_search_threshold: int = 10,
                 max_contour_steps: int = 30, max_hourly_distance: float = 1.0,
                 time_match_tolerance: int = 7200, enable_vis: bool = True,
                 local_shp_path: str = None):
        """
        Initialize the tracker.
        
        Args:
            delta_psi: Streamfunction contour increment
            min_lifetime: Minimum lifetime in hours
            detection_threshold: Threshold for vortex detection
            max_track_hours: Maximum tracking duration in hours
            level_idx: Vertical level index
            boundary_buffer: Buffer distance from data boundary
            psi_min_threshold: Minimum streamfunction threshold
            delta_psi_contour: Contour increment for inner contour
            search_radius: Search radius for local minimum
            global_search_threshold: Number of candidates in global search
            max_contour_steps: Maximum contour steps
            max_hourly_distance: Maximum hourly movement distance (degrees)
            time_match_tolerance: Time matching tolerance in seconds
            enable_vis: Enable visualization
            local_shp_path: Path to local SHP file for boundary checking
        """
        self.delta_psi = delta_psi
        self.min_lifetime = min_lifetime
        self.detection_threshold = detection_threshold
        self.max_track_hours = max_track_hours
        self.level_idx = level_idx
        self.boundary_buffer = boundary_buffer
        self.psi_min_threshold = psi_min_threshold
        self.delta_psi_contour = delta_psi_contour
        self.search_radius = search_radius
        self.global_search_threshold = global_search_threshold
        self.max_contour_steps = max_contour_steps
        self.max_hourly_distance = max_hourly_distance
        self.time_match_tolerance = time_match_tolerance
        self.enable_vis = enable_vis
        self.local_shp_path = local_shp_path
        self.track_records = {}
        self.end_positions_df = None
        self.all_tracks_df = None
        
    def load_end_positions(self, csv_path: str) -> None:
        """Load end position CSV file."""
        self.end_positions_df = read_typhoon_end_positions(csv_path)

    def get_typhoon_end_info(self, typhoon_name: str, end_time: Optional[datetime] = None) -> Optional[Dict]:
        """Get end position information by typhoon name and end time."""
        if self.end_positions_df is None:
            raise ValueError("Please call load_end_positions first to load end position data")
    
        info = self.end_positions_df[self.end_positions_df['Typhoon Name'] == typhoon_name]
        if len(info) == 0:
            print(f"⚠️  No end position record found for typhoon {typhoon_name}")
            return None
    
        if end_time is not None:
            time_tolerance = timedelta(minutes=30)
            time_mask = (info['End Time_dt'] >= end_time - time_tolerance) & \
                        (info['End Time_dt'] <= end_time + time_tolerance)
            info = info[time_mask]
            if len(info) == 0:
                print(f"⚠️  No end position record found for typhoon {typhoon_name} near {end_time}")
                return None
    
        if len(info) > 1:
            raise ValueError(
                f"Typhoon name {typhoon_name} has {len(info)} matching records, "
                f"please specify precisely using end_time parameter.\n"
                f"Conflicting record times: {[t.strftime('%Y-%m-%d %H:%M') for t in info['End Time_dt']]}"
            )
    
        record = info.iloc[0].to_dict()
        return {
            'End Time': record['End Time_dt'],
            'End Latitude': record['End Latitude'],
            'End Longitude': record['End Longitude'],
            'Source File': record['Source File'],
            'International ID': record['International ID'],
            'Tropical Cyclone Serial': record['Tropical Cyclone Serial'],
            'China ID': record['China ID']
        }

    def save_track_results(self, track_results: List[Dict], typhoon_name: str) -> None:
        """Append tracking results to unified DataFrame."""
        if not track_results:
            print(f"⚠️  Typhoon {typhoon_name} has no tracking results to save")
            return
        
        df = pd.DataFrame(track_results)
        
        columns_order = [
            'International ID', 'Tropical Cyclone Serial', 'China ID', 'Typhoon Name',
            'Tracking Time', 'Center Latitude', 'Center Longitude', 
            'Center Streamfunction Value', 'Center Vorticity Value',
            'Hourly Movement Distance', 'Tracking Status', 'Is Vortex Merged',
            'Is Boundary Reached', 'Is No Vortex', 'Is Abnormal Movement',
            'Total Minima Within Contour', 'Other Minima Count', 'Source NC File'
        ]
        for col in columns_order:
            if col not in df.columns:
                df[col] = None
        df = df[columns_order]
        
        if self.all_tracks_df is None:
            self.all_tracks_df = df
        else:
            self.all_tracks_df = pd.concat([self.all_tracks_df, df], ignore_index=True)
        
        if typhoon_name not in self.track_records:
            self.track_records[typhoon_name] = []
        self.track_records[typhoon_name].extend(track_results)
        
        print(f"✅ Typhoon {typhoon_name} tracking results appended to unified dataset "
              f"(total {len(track_results)} records)")

    def save_all_track_summary(self, output_dir: str = "track_results") -> None:
        """Generate summary file for all typhoon tracking results."""
        if not self.track_records:
            print("⚠️  No typhoon tracking records available to generate summary")
            return
            
        os.makedirs(output_dir, exist_ok=True)
        
        if self.all_tracks_df is not None and not self.all_tracks_df.empty:
            unified_path = os.path.join(output_dir, f"all_typhoons_tracking.csv")
            self.all_tracks_df.to_csv(unified_path, index=False, encoding='utf-8-sig')
            print(f"✅ All typhoons unified tracking data saved to: {os.path.abspath(unified_path)}")
            print(f"   Total records: {len(self.all_tracks_df)}")
        
        summary = []
        for typhoon_name, records in self.track_records.items():
            if not records:
                continue
                
            first_record = records[0]
            last_record = records[-1]
            total_hours = len(records)
            start_time = first_record['Tracking Time']
            end_time = last_record['Tracking Time']
            start_lat = first_record['Center Latitude']
            start_lon = first_record['Center Longitude']
            end_lat = last_record['Center Latitude']
            end_lon = last_record['Center Longitude']
            max_psi = min(record['Center Streamfunction Value'] for record in records 
                         if record['Center Streamfunction Value'] is not None)
            merged = any(record['Is Vortex Merged'] for record in records 
                        if record['Is Vortex Merged'] is not None)
            
            summary.append({
                'Typhoon Name': typhoon_name,
                'Tracking Duration (hours)': total_hours,
                'Start Time': start_time,
                'End Time': end_time,
                'Start Latitude': start_lat,
                'Start Longitude': start_lon,
                'End Latitude': end_lat,
                'End Longitude': end_lon,
                'Maximum Streamfunction Value': max_psi,
                'Is Merged': 'Yes' if merged else 'No',
                'Number of Data Files': len(set(record['Source NC File'] for record in records 
                                               if record['Source NC File'] is not None))
            })
        
        if summary:
            summary_df = pd.DataFrame(summary)
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            summary_path = os.path.join(output_dir, f"all_typhoons_summary_{timestamp}.csv")
            summary_df.to_csv(summary_path, index=False, encoding='utf-8-sig')
            print(f"✅ All typhoons summary saved to: {os.path.abspath(summary_path)}")

    def identify_typhoon_centers(self, psi_field: np.ndarray) -> List[Tuple[int, int]]:
        """Identify local minimum points of streamfunction."""
        neighborhood = np.ones((3, 3), dtype=bool)
        neighborhood[1, 1] = False
        local_min = ndimage.minimum_filter(psi_field, footprint=neighborhood, mode='nearest')
        min_mask = psi_field < local_min
        background_mean = np.mean(psi_field)
        threshold_mask = psi_field < (background_mean - self.detection_threshold * 0.8)
        valid_centers = np.where(min_mask & threshold_mask)
        return list(zip(valid_centers[0], valid_centers[1]))

    def find_inner_closed_contour(self, psi_field: np.ndarray, 
                                   min_point: Tuple[int, int]) -> Optional[np.ndarray]:
        """Find the innermost closed contour enclosing the target minimum."""
        min_psi_value = psi_field[min_point]
        ny, nx = psi_field.shape
        
        for step in range(self.max_contour_steps):
            contour_value = min_psi_value + (step + 1) * self.delta_psi_contour
            contour_mask = psi_field <= contour_value
            
            labeled, num_features = ndimage.label(contour_mask, structure=np.ones((3, 3)))
            min_label = labeled[min_point]
            if min_label == 0:
                continue
            
            region_mask = (labeled == min_label)
            if np.sum(region_mask) < 5:
                continue
            
            return region_mask
        
        return None

    def calculate_contour_center(self, contour_mask: np.ndarray, lat_dim: np.ndarray,
                                 lon_dim: np.ndarray, weight_by_psi: bool = True,
                                 psi_field: Optional[np.ndarray] = None) -> Tuple[float, float]:
        """Calculate contour center using weighted or arithmetic mean."""
        lat_idxs, lon_idxs = np.where(contour_mask)
        if len(lat_idxs) == 0:
            raise ValueError("No valid grid points in contour mask")
        
        latitudes = lat_dim[lat_idxs]
        longitudes = lon_dim[lon_idxs]
        
        if not weight_by_psi:
            return np.mean(latitudes), np.mean(longitudes)
        else:
            if psi_field is None:
                raise ValueError("Weighted mean requires psi_field parameter")
            psi_values = psi_field[lat_idxs, lon_idxs]
            psi_range = np.max(psi_values) - np.min(psi_values)
            if psi_range < 1e-10:
                return np.mean(latitudes), np.mean(longitudes)
            weights = np.exp(-(psi_values - np.min(psi_values)) / psi_range)
            weights /= np.sum(weights)
            return np.sum(latitudes * weights), np.sum(longitudes * weights)

    def find_best_center(self, psi_field: np.ndarray, last_center_lat: float, 
                         last_center_lon: float, lat_dim: np.ndarray, lon_dim: np.ndarray,
                         is_first_track: bool = False) -> Tuple[Tuple[float, float], float, 
                                                                 Tuple[int, int], Optional[np.ndarray]]:
        """Find target typhoon center using contour and minimum detection."""
        if is_first_track:
            print(f"ℹ️  First tracking: global search for target minimum point "
                  f"(reference position: {last_center_lat:.2f}°N, {last_center_lon:.2f}°E)")
            candidate_mins = self.identify_typhoon_centers(psi_field)
            if not candidate_mins:
                raise ValueError("Global search found no local minimum points (no vortex)")
            
            ref_lat_idx = np.argmin(np.abs(lat_dim - last_center_lat))
            ref_lon_idx = np.argmin(np.abs(lon_dim - last_center_lon))
            
            candidate_mins_sorted = sorted(
                candidate_mins,
                key=lambda x: np.sqrt((x[0] - ref_lat_idx)**2 + (x[1] - ref_lon_idx)**2)
            )[:self.global_search_threshold]
            
            min_dist = float('inf')
            target_min_idx = None
            for (lat_idx, lon_idx) in candidate_mins_sorted:
                dist = np.sqrt((lat_idx - ref_lat_idx)**2 + (lon_idx - ref_lon_idx)**2)
                if dist < min_dist:
                    min_dist = dist
                    target_min_idx = (lat_idx, lon_idx)
            if target_min_idx is None:
                raise ValueError("Global search found no minimum point near the reference position")
        
        else:
            last_lat_idx = np.argmin(np.abs(lat_dim - last_center_lat))
            last_lon_idx = np.argmin(np.abs(lon_dim - last_center_lon))
            
            lat_search_min = max(0, last_lat_idx - self.search_radius)
            lat_search_max = min(len(lat_dim), last_lat_idx + self.search_radius + 1)
            lon_search_min = max(0, last_lon_idx - self.search_radius)
            lon_search_max = min(len(lon_dim), last_lon_idx + self.search_radius + 1)
            local_psi = psi_field[lat_search_min:lat_search_max, lon_search_min:lon_search_max]
            
            neighborhood = np.ones((3, 3), dtype=bool)
            neighborhood[1, 1] = False
            local_min = ndimage.minimum_filter(local_psi, footprint=neighborhood, mode='nearest')
            min_mask = local_psi < local_min
            local_min_idxs = np.where(min_mask)
            
            if not local_min_idxs[0].size:
                print(f"⚠️  Local search (radius {self.search_radius}) failed, triggering global search")
                candidate_mins = self.identify_typhoon_centers(psi_field)
                if not candidate_mins:
                    raise ValueError("Global search still found no minimum points (no vortex)")
                
                candidate_mins_sorted = sorted(
                    candidate_mins,
                    key=lambda x: np.sqrt((x[0] - last_lat_idx)**2 + (x[1] - last_lon_idx)**2)
                )[:self.global_search_threshold]
                
                min_dist = float('inf')
                target_min_idx = None
                for (lat_idx, lon_idx) in candidate_mins_sorted:
                    dist = np.sqrt((lat_idx - last_lat_idx)**2 + (lon_idx - last_lon_idx)**2)
                    if dist < min_dist:
                        min_dist = dist
                        target_min_idx = (lat_idx, lon_idx)
                if target_min_idx is None:
                    raise ValueError("Global search found no minimum point near the previous time step center")
            
            else:
                min_dist = float('inf')
                target_min_idx = None
                for i, j in zip(local_min_idxs[0], local_min_idxs[1]):
                    global_i = lat_search_min + i
                    global_j = lon_search_min + j
                    dist = np.sqrt((global_i - last_lat_idx)**2 + (global_j - last_lon_idx)**2)
                    if dist < min_dist:
                        min_dist = dist
                        target_min_idx = (global_i, global_j)
        
        target_row, target_col = target_min_idx
        target_psi = psi_field[target_row, target_col]
        if target_psi > self.psi_min_threshold:
            raise ValueError(
                f"Target minimum invalid: streamfunction value {target_psi:.2e} m²/s "
                f"> threshold {self.psi_min_threshold:.2e} m²/s (no vortex特征)"
            )
        print(f"✅ Target minimum found:")
        print(f"  Grid index: ({target_row}, {target_col}) → "
              f"Latitude/Longitude: ({lat_dim[target_row]:.2f}°N, {lon_dim[target_col]:.2f}°E)")
        print(f"  Streamfunction value: {target_psi:.2e} m²/s")
        
        inner_contour = self.find_inner_closed_contour(psi_field, target_min_idx)
        if inner_contour is None:
            print(f"⚠️  No closed contour found enclosing the target minimum, "
                  f"falling back to minimum point as center")
            avg_lat, avg_lon = lat_dim[target_row], lon_dim[target_col]
        else:
            print(f"✅ Innermost closed contour found: contains {np.sum(inner_contour)} grid points")
            try:
                avg_lat, avg_lon = self.calculate_contour_center(
                    contour_mask=inner_contour,
                    lat_dim=lat_dim,
                    lon_dim=lon_dim,
                    weight_by_psi=True,
                    psi_field=psi_field
                )
            except Exception as e:
                avg_lat, avg_lon = self.calculate_contour_center(
                    contour_mask=inner_contour,
                    lat_dim=lat_dim,
                    lon_dim=lon_dim,
                    weight_by_psi=False
                )
                print(f"⚠️  Weighted mean failed, falling back to arithmetic mean: {str(e)}")
        
        print(f"✅ Final typhoon center: ({avg_lat:.2f}°N, {avg_lon:.2f}°E)")
        return (avg_lat, avg_lon), target_psi, target_min_idx, inner_contour

    def check_boundary_condition(self, center_lat: float, center_lon: float,
                                 lat_min: float, lat_max: float, 
                                 lon_min: float, lon_max: float) -> bool:
        """
        Check if the center has reached the data boundary or is outside China.
        
        Returns:
            True if tracking should stop, False to continue
        """
        # Data boundary check
        lat_dist_min = center_lat - (lat_min + self.boundary_buffer)
        lat_dist_max = (lat_max - self.boundary_buffer) - center_lat
        lon_dist_min = center_lon - (lon_min + self.boundary_buffer)
        lon_dist_max = (lon_max - self.boundary_buffer) - center_lon

        if any(dist <= 0 for dist in [lat_dist_min, lat_dist_max, lon_dist_min, lon_dist_max]):
            print(f"⚠️  Reached data boundary:")
            print(f"  Center position: ({center_lat:.2f}°N, {center_lon:.2f}°E)")
            print(f"  Buffered boundary: latitude [{lat_min+self.boundary_buffer:.2f}, "
                  f"{lat_max-self.boundary_buffer:.2f}]°N, "
                  f"longitude [{lon_min+self.boundary_buffer:.2f}, {lon_max-self.boundary_buffer:.2f}]°E")
            return True

        # China boundary check using local SHP
        if self.local_shp_path is None:
            return False

        china_geoms = None
        min_dist_km = float('inf')
        is_inside = False
        
        if not os.path.exists(self.local_shp_path):
            print(f"❌ Local SHP file does not exist: {self.local_shp_path}")
            return False

        try:
            sf = shapefile.Reader(self.local_shp_path)
            china_geoms = sf.shapes()
            print(f"✅ Successfully loaded local SHP file: {os.path.basename(self.local_shp_path)} "
                  f"(total {len(china_geoms)} boundary shapes)")
        except Exception as e:
            print(f"❌ Failed to read local SHP: {str(e)}")
            return False

        if china_geoms:
            from shapely.geometry import Point, Polygon
            
            typhoon_point = Point(center_lon, center_lat)
            
            for geom in china_geoms:
                if geom.shapeType == 5:
                    polygon = Polygon(geom.points)
                    if polygon.contains(typhoon_point):
                        is_inside = True
                        break

            if is_inside:
                print(f"✅ Typhoon is within China's national boundary, continuing tracking")
                return False

            geod = Geodesic.WGS84
            for geom in china_geoms:
                if geom.shapeType == 5:
                    vertices = geom.points
                    for (lon, lat) in vertices:
                        dist_m = geod.Inverse(center_lat, center_lon, lat, lon)['s12']
                        dist_km = dist_m / 1000
                        if dist_km < min_dist_km:
                            min_dist_km = dist_km

            if min_dist_km > 200:
                print(f"⚠️  Typhoon is outside China's national boundary and distance exceeds 200km, stopping tracking:")
                print(f"  Center position: ({center_lat:.2f}°N, {center_lon:.2f}°E)")
                print(f"  Shortest distance to China's national boundary: {min_dist_km:.1f}km > threshold 200km")
                return True
            else:
                print(f"✅ Typhoon is outside China's national boundary but distance ≤200km, continuing tracking")

        return False

    def check_psi_min_condition(self, center_psi: float) -> bool:
        """Check if there is no vortex feature."""
        if center_psi > self.psi_min_threshold:
            print(f"⚠️  No vortex feature: center streamfunction value {center_psi:.2e} "
                  f"> threshold {self.psi_min_threshold:.2e} m²/s")
            return True
        return False

    def check_hourly_distance(self, last_lat: float, last_lon: float, 
                              current_lat: float, current_lon: float) -> Tuple[bool, float]:
        """Check if movement is abnormal (> threshold degrees/hour)."""
        distance = calculate_spherical_distance(last_lat, last_lon, current_lat, current_lon)
        if distance > self.max_hourly_distance:
            print(f"⚠️  Hourly movement distance abnormal!")
            print(f"  Previous time step position: ({last_lat:.2f}°N, {last_lon:.2f}°E)")
            print(f"  Current time step position: ({current_lat:.2f}°N, {current_lon:.2f}°E)")
            print(f"  Movement distance: {distance:.2f}° > threshold {self.max_hourly_distance}° "
                  f"(judged as abnormal movement)")
            return True, distance
        else:
            print(f"✅ Hourly movement distance normal: {distance:.2f}° "
                  f"(≤ threshold {self.max_hourly_distance}°)")
            return False, distance

    def _validate_tracking_consistency(self, continue_tracking: bool, 
                                       boundary_reached: bool, track_failed: bool,
                                       psi_exceeded: bool, distance_exceeded: bool,
                                       vortex_merged: bool) -> Tuple[bool, str]:
        """Validate tracking state consistency."""
        termination_conditions = {
            boundary_reached: "Reached data boundary",
            track_failed: "Tracking process failed",
            psi_exceeded: "Streamfunction value exceeded threshold",
            distance_exceeded: "Movement distance abnormal",
            vortex_merged: "Vortex merged"
        }
    
        if any(termination_conditions.keys()) and continue_tracking:
            reasons = [msg for cond, msg in termination_conditions.items() if cond]
            return False, f"Termination conditions exist [{','.join(reasons)}] but continue_tracking is still True"
    
        if not any(termination_conditions.keys()) and not continue_tracking:
            return False, "No termination conditions but continue_tracking is False"
    
        return True, "State consistent"
    
    def get_center_vorticity(self, ds, center_lat, center_lon, lat_dim, lon_dim, 
                             time_idx, level_dim='pressure_level', level_idx=None):
        """Extract vorticity value at the center point."""
        try:
            if level_idx is None:
                level_idx = self.level_idx
                if level_idx is None:
                    raise ValueError("Vertical level index level_idx not initialized")
            
            lat_vals = ds[lat_dim].values
            lon_vals = ds[lon_dim].values
            lat_idx = np.argmin(np.abs(lat_vals - center_lat))
            lon_idx = np.argmin(np.abs(lon_vals - center_lon))
            
            if not (0 <= time_idx < ds.sizes['valid_time']):
                raise IndexError(f"Time index out of range: {time_idx} "
                                 f"(valid range 0-{ds.sizes['valid_time']-1})")
            if not (0 <= lat_idx < len(lat_vals)):
                raise IndexError(f"Latitude index out of range: {lat_idx} "
                                 f"(valid range 0-{len(lat_vals)-1})")
            if not (0 <= lon_idx < len(lon_vals)):
                raise IndexError(f"Longitude index out of range: {lon_idx} "
                                 f"(valid range 0-{len(lon_vals)-1})")
            if not (0 <= level_idx < ds.sizes[level_dim]):
                raise IndexError(f"Vertical level index out of range: {level_idx} "
                                 f"(valid range 0-{ds.sizes[level_dim]-1})")
            
            vort_array = ds['vo'].isel(
                {
                    'valid_time': time_idx,
                    lat_dim: lat_idx,
                    lon_dim: lon_idx,
                    level_dim: level_idx
                }
            ).values
            
            if vort_array.size != 1:
                raise ValueError(f"Extracted vorticity value is not a single scalar "
                                 f"(actual size: {vort_array.size})")
            
            return float(vort_array.item())

        except Exception as e:
            print(f"❌ Failed to extract center vorticity value: {str(e)}")
            print(f"  Center position: ({center_lat:.2f}°N, {center_lon:.2f}°E)")
            print(f"  Current time step index: {time_idx}")
            return np.nan
    
    def track_single_file(self, nc_file_path: str, start_time: datetime, ftime: datetime,
                          last_center_lat: float, last_center_lon: float, 
                          typhoon_name: str, time_dim: str = 'valid_time',
                          lat_dim: str = 'latitude', lon_dim: str = 'longitude',
                          is_first_file: bool = True) -> Tuple[List[Dict], bool, datetime, 
                                                                bool, bool, bool, bool, bool]:
        """Track typhoon path in a single NC file."""
        track_results = []
        continue_tracking = True
        last_succ_time = None
        boundary_reached = False
        track_failed = False
        psi_exceeded = False
        distance_exceeded = False
        vortex_merged = False
        prev_center_lat = last_center_lat
        prev_center_lon = last_center_lon
        
        try:
            with xr.open_dataset(nc_file_path) as ds:
                required_dims = [time_dim, lat_dim, lon_dim]
                required_vars = ['vo']
                for dim in required_dims:
                    if dim not in ds.dims:
                        raise KeyError(f"NC file missing required dimension: {dim}")
                for var in required_vars:
                    if var not in ds.variables:
                        raise KeyError(f"NC file missing vorticity variable: {var}")
                
                ds[time_dim] = xr.decode_cf(ds)[time_dim]
                time_series = [pd.to_datetime(t).to_pydatetime() for t in ds[time_dim].values]
                lat_vals = ds[lat_dim].values
                lon_vals = ds[lon_dim].values
                lat_min, lat_max = np.min(lat_vals), np.max(lat_vals)
                lon_min, lon_max = np.min(lon_vals), np.max(lon_vals)
                print(f"📊 Data range: latitude [{lat_min:.2f},{lat_max:.2f}]°N, "
                      f"longitude [{lon_min:.2f},{lon_max:.2f}]°E")
                
                time_diffs = [abs((dt - start_time).total_seconds()) for dt in time_series]
                min_diff = min(time_diffs) if time_diffs else float('inf')
                start_idx = time_diffs.index(min_diff) if time_diffs else None
                
                if start_idx is None or min_diff > self.time_match_tolerance:
                    print(f"⏱️  Expected time step: {start_time.strftime('%Y-%m-%d %H:%M')}")
                    print(f"⏱️  Time steps in file: {[dt.strftime('%Y-%m-%d %H:%M') for dt in time_series[:5]]}...")
                    raise ValueError(
                        f"No time step close to {start_time.strftime('%Y-%m-%d %H:%M')} found "
                        f"(allow ±{self.time_match_tolerance/3600} hour deviation)"
                    )
                print(f"✅ Matching time step found: {time_series[start_idx].strftime('%Y-%m-%d %H:%M')}, "
                      f"time difference {min_diff/3600:.1f} hours")
                
                vo_dims = ds['vo'].dims
                extra_dims = [d for d in vo_dims if d not in [time_dim, lat_dim, lon_dim]]
                if extra_dims:
                    print(f"🔍 vo extra dimensions: {extra_dims}, extracting by index {self.level_idx}")
                    if extra_dims[0] not in ds.dims:
                        raise KeyError(f"Extra dimension {extra_dims[0]} not in NC file")
                    if self.level_idx >= ds.sizes[extra_dims[0]]:
                        raise IndexError(f"{extra_dims[0]} index {self.level_idx} out of range "
                                         f"(0~{ds.sizes[extra_dims[0]]-1})")
                
                for hour_idx in range(start_idx, len(time_series)):
                    current_dt = time_series[hour_idx]
                    print(f"\n{'='*50}")
                    print(f"⏱️  Processing time step: {current_dt.strftime('%Y-%m-%d %H:%M:%S')}")
                    print(f"{'='*50}")
                    
                    if last_succ_time is not None:
                        tracked_hours = (current_dt - last_succ_time).total_seconds() / 3600
                        if tracked_hours > self.max_track_hours:
                            print(f"⏰ Continuous tracking {tracked_hours:.1f} hours > threshold {self.max_track_hours}, stopping")
                            continue_tracking = False
                            break
                    if last_succ_time is None:
                        last_succ_time = current_dt
                    
                    try:
                        vo_time = ds['vo'].isel({time_dim: hour_idx})
                        if extra_dims:
                            vo_time = vo_time.isel({extra_dims[0]: self.level_idx})
                        vo_time = vo_time.where(np.isfinite(vo_time), 0)
                        vorticity_da = vo_time
                        print(f"✅ Extracted vo data: dimensions {list(vorticity_da.dims)}, "
                              f"shape {vorticity_da.shape}")
                    except Exception as e:
                        print(f"❌ Failed to extract vo data: {str(e)}")
                        track_failed = True
                        continue_tracking = False
                        break
                    
                    try:
                        psi_da = calculate_streamfunction(vorticity_da, lat_dim=lat_dim, lon_dim=lon_dim)
                        psi_field = psi_da.values
                        print(f"✅ Calculated streamfunction: shape {psi_field.shape}")
                    except ValueError as e:
                        print(f"❌ Streamfunction calculation failed: {str(e)}")
                        track_failed = True
                        continue_tracking = False
                        break
                    
                    try:
                        is_first_track = is_first_file and (hour_idx == start_idx)
                        (center_lat, center_lon), center_psi, target_min_idx, inner_contour = self.find_best_center(
                            psi_field=psi_field,
                            last_center_lat=last_center_lat if not is_first_track else last_center_lat,
                            last_center_lon=last_center_lon if not is_first_track else last_center_lon,
                            lat_dim=lat_vals,
                            lon_dim=lon_vals,
                            is_first_track=is_first_track
                        )
                    except ValueError as e:
                        print(f"❌ Center calculation failed: {str(e)}")
                        track_failed = True
                        continue_tracking = False
                        break
                    
                    all_minima = []
                    if inner_contour is not None:
                        vortex_merged, all_minima, target_psi = check_vortex_merged(
                            psi_field=psi_field,
                            contour_mask=inner_contour,
                            target_min_idx=target_min_idx
                        )
                    else:
                        print(f"ℹ️  No closed contour, skipping vortex merger detection")
                    
                    if vortex_merged:
                        continue_tracking = False
                    
                    boundary_reached = self.check_boundary_condition(
                        center_lat, center_lon, lat_min, lat_max, lon_min, lon_max
                    )
                    if boundary_reached:
                        continue_tracking = False
                    
                    psi_exceeded = self.check_psi_min_condition(center_psi)
                    if psi_exceeded:
                        continue_tracking = False
                    
                    distance_exceeded, hourly_distance = self.check_hourly_distance(
                        prev_center_lat, prev_center_lon, center_lat, center_lon
                    )
                    if distance_exceeded:
                        continue_tracking = False
                    
                    prev_center_lat, prev_center_lon = center_lat, center_lon
                    
                    typhoon_info = self.get_typhoon_end_info(typhoon_name, ftime)
                    if not typhoon_info:
                        raise ValueError(f"Unable to get basic information for typhoon {typhoon_name}")
                        
                    center_vort = self.get_center_vorticity(
                        ds=ds,
                        center_lat=center_lat,
                        center_lon=center_lon,
                        lat_dim=lat_dim,
                        lon_dim=lon_dim,
                        time_idx=hour_idx,
                        level_idx=self.level_idx
                    )
            
                    track_record = {
                        'International ID': typhoon_info['International ID'],
                        'Tropical Cyclone Serial': typhoon_info['Tropical Cyclone Serial'],
                        'China ID': typhoon_info['China ID'],
                        'Typhoon Name': typhoon_name,
                        'Tracking Time': current_dt,
                        'Center Latitude': center_lat,
                        'Center Longitude': center_lon,
                        'Center Streamfunction Value': target_psi,
                        'Center Vorticity Value': center_vort,
                        'Hourly Movement Distance': round(hourly_distance, 4) if hourly_distance else 0,
                        'Tracking Status': 'Normal' if continue_tracking else 'Terminated',
                        'Is Vortex Merged': 'Yes' if vortex_merged else 'No',
                        'Is Boundary Reached': 'Yes' if boundary_reached else 'No',
                        'Is No Vortex': 'Yes' if psi_exceeded else 'No',
                        'Is Abnormal Movement': 'Yes' if distance_exceeded else 'No',
                        'Total Minima Within Contour': len(all_minima) if all_minima else 0,
                        'Other Minima Count': len(all_minima) - 1 if all_minima else 0,
                        'Source NC File': os.path.basename(nc_file_path)
                    }
                    track_results.append(track_record)
                    print(f"✅ Tracking result recorded: {track_record['Tracking Time']}")
                    
                    del psi_field, vorticity_da, vo_time
                    gc.collect()
                    
                    prev_center_lat, prev_center_lon = center_lat, center_lon
                    last_center_lat, last_center_lon = center_lat, center_lon
                    
                    if not continue_tracking:
                        consistent, reason = self._validate_tracking_consistency(
                            continue_tracking, boundary_reached, track_failed,
                            psi_exceeded, distance_exceeded, vortex_merged
                        )
                        if not consistent:
                            print(f"⚠️  Tracking state inconsistent: {reason}")
                        break
                
                self.save_track_results(track_results, typhoon_name)
                
                return (track_results, continue_tracking, current_dt, 
                        boundary_reached, track_failed, psi_exceeded, 
                        distance_exceeded, vortex_merged)
                
        except Exception as e:
            print(f"❌ Tracking process error: {str(e)}")
            track_failed = True
            continue_tracking = False
            return ([], continue_tracking, datetime.now(), 
                    boundary_reached, track_failed, psi_exceeded, 
                    distance_exceeded, vortex_merged)