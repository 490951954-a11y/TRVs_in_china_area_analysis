import os
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
from matplotlib.colors import LinearSegmentedColormap
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.ticker as mticker
from geographiclib.geodesic import Geodesic
import os
import shapefile

# Configure Chinese font display
plt.rcParams['font.sans-serif'] = ['SimHei', 'WenQuanYi Micro Hei', 'Heiti TC']
plt.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore')


# -------------------------- Utility Functions --------------------------
def calculate_spherical_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate spherical distance between two points (degrees)"""
    lon1 = (lon1 + 180) % 360 - 180
    lon2 = (lon2 + 180) % 360 - 180
    geod = Geodesic.WGS84
    result = geod.Inverse(lat1, lon1, lat2, lon2)
    return result['s12'] / 111319.9  # 1 degree ≈ 111319.9 meters


def find_minima_in_contour(psi_field: np.ndarray, contour_mask: np.ndarray) -> List[Tuple[int, int, float]]:
    """Find all local minima within contour (including coordinates and streamfunction values), sorted by value ascending"""
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


def check_vortex_merged(psi_field: np.ndarray, contour_mask: np.ndarray, target_min_idx: Tuple[int, int]) -> Tuple[bool, List[Tuple[int, int, float]], float]:
    """Vortex merger detection: smaller minimum within contour → merged"""
    all_minima = find_minima_in_contour(psi_field, contour_mask)
    if not all_minima:
        return False, [], np.nan
    
    target_row, target_col = target_min_idx
    target_psi = psi_field[target_row, target_col]
    
    min_psi_in_contour = all_minima[0][2]
    if min_psi_in_contour < target_psi * 1.1:  # Allow computational error
        print(f"⚠️  Vortex merger detected: smaller minimum exists within contour!")
        print(f"  Target minimum (current typhoon): position({target_row},{target_col}), streamfunction value {target_psi:.2e} m²/s")
        print(f"  Minimum minimum within contour (dominant vortex): position({all_minima[0][0]},{all_minima[0][1]}), streamfunction value {min_psi_in_contour:.2e} m²/s")
        return True, all_minima, target_psi
    else:
        print(f"✅ Vortex independence check passed: no smaller minimum within contour")
        print(f"  Target minimum (current typhoon): streamfunction value {target_psi:.2e} m²/s")
        print(f"  Minimum minimum within contour: streamfunction value {min_psi_in_contour:.2e} m²/s")
        return False, all_minima, target_psi


# -------------------------- CSV Reading and Time Parsing Functions --------------------------
# Modified read_typhoon_end_positions function to add new field reading
def read_typhoon_end_positions(csv_path: str) -> pd.DataFrame:
    """Read end position CSV file and parse time format (integer to datetime)"""
    try:
        df = pd.read_csv(csv_path, encoding='utf-8')
        
        # Check required columns (added new ID fields as required)
        required_cols = ['Typhoon Name', 'End Time', 'End Latitude', 'End Longitude', 'Source File',
                         'International ID', 'Tropical Cyclone Serial', 'China ID']  # Three new ID fields
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"End position CSV missing required columns: {missing_cols}")
        
        # Parse integer time (1980052418 → datetime)
        df['End Time_str'] = df['End Time'].astype(str)
        df['End Time_dt'] = pd.to_datetime(
            df['End Time_str'],
            format='%Y%m%d%H',
            errors='coerce'
        )
        
        # Filter invalid time records
        invalid_mask = df['End Time_dt'].isna()
        invalid_count = invalid_mask.sum()
        if invalid_count > 0:
            invalid_names = df.loc[invalid_mask, 'Typhoon Name'].tolist()
            print(f"⚠️  Skipping invalid time records: {invalid_names} (total {invalid_count} records)")
        
        return df[~invalid_mask].reset_index(drop=True)
    
    except Exception as e:
        print(f"❌ Failed to read end position CSV: {str(e)}")
        raise





# -------------------------- Streamfunction Calculation Functions --------------------------
def solve_poisson_iterative(vorticity, dx, dy, max_iter=1500, tol=1e4):
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
    vorticity_values = vorticity.values if hasattr(vorticity, 'values') else vorticity
    dx_val = abs(dx.magnitude) if hasattr(dx, 'magnitude') else abs(dx)
    dy_val = abs(dy.magnitude) if hasattr(dy, 'magnitude') else abs(dy)
    return solve_poisson_iterative(vorticity_values, dx_val, dy_val)


def calculate_streamfunction(vorticity_da, lat_dim='latitude', lon_dim='longitude'):
    if len(vorticity_da.dims) != 2:
        raise ValueError(
            f"Streamfunction calculation requires 2D vorticity data ({lat_dim}, {lon_dim}), but input has {len(vorticity_da.dims)} dimensions ({vorticity_da.dims})"
        )
    
    vorticity_da = vorticity_da.where(np.isfinite(vorticity_da), 0)
    vorticity_da = vorticity_da.clip(min=-1e-3, max=1e-3)
    
    # Ensure latitude decreases from north to south
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


# -------------------------- Typhoon Tracking Core Class --------------------------
class StreamfunctionTyphoonTracker:
    def __init__(self, delta_psi=2.0e6, min_lifetime=1, detection_threshold=1.0e6,
                 max_track_hours=120, level_idx: int = 0, boundary_buffer: float = 0.25,
                 psi_min_threshold: float = 0.0, delta_psi_contour: float = 8e4,
                 search_radius: int = 8, global_search_threshold: int = 10,
                 max_contour_steps: int = 30, max_hourly_distance: float = 1.0,
                 time_match_tolerance: int = 7200,  # 2-hour tolerance
                 enable_vis: bool = True):  # New: visualization switch (can be disabled when memory is insufficient)
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
        self.enable_vis = enable_vis  # Controls whether to generate visualizations (core optimization)
        self.track_records = {}
        self.end_positions_df = None
        self.all_tracks_df = None  # New: store all tracking data in one DataFrame
        
    def load_end_positions(self, csv_path: str) -> None:
        """Load and parse end position CSV file"""
        self.end_positions_df = read_typhoon_end_positions(csv_path)

    def get_typhoon_end_info(self, typhoon_name: str, end_time: Optional[datetime] = None) -> Optional[Dict]:
        """Get end position information by typhoon name and end time (supports distinguishing typhoons with same name)"""
        if self.end_positions_df is None:
            raise ValueError("Please call load_end_positions first to load end position data")
    
        # First filter by name
        info = self.end_positions_df[self.end_positions_df['Typhoon Name'] == typhoon_name]
        if len(info) == 0:
            print(f"⚠️  No end position record found for typhoon {typhoon_name}")
            return None
    
        # If end time is provided, further filter by time (allow ±30 minutes error)
        if end_time is not None:
            time_tolerance = timedelta(minutes=30)
            time_mask = (info['End Time_dt'] >= end_time - time_tolerance) & \
                        (info['End Time_dt'] <= end_time + time_tolerance)
            info = info[time_mask]
            if len(info) == 0:
                print(f"⚠️  No end position record found for typhoon {typhoon_name} near {end_time}")
                return None
    
        # Handle cases where multiple records still exist
        if len(info) > 1:
            raise ValueError(
                f"Typhoon name {typhoon_name} has {len(info)} matching records, please specify precisely using end_time parameter.\n"
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

    # Modified save_track_results function: append to unified DataFrame instead of saving separate files
    def save_track_results(self, track_results: List[Dict], typhoon_name: str) -> None:
        """Append tracking results to unified DataFrame (not saving individual files)"""
        if not track_results:
            print(f"⚠️  Typhoon {typhoon_name} has no tracking results to save")
            return
        
        # Convert to DataFrame
        df = pd.DataFrame(track_results)
        
        # Ensure all required columns exist
        columns_order = [
            'International ID', 'Tropical Cyclone Serial', 'China ID', 'Typhoon Name',
            'Tracking Time', 'Center Latitude', 'Center Longitude', 'Center Streamfunction Value', 'Center Vorticity Value',
            'Hourly Movement Distance', 'Tracking Status', 'Is Vortex Merged',
            'Is Boundary Reached', 'Is No Vortex', 'Is Abnormal Movement',
            'Total Minima Within Contour', 'Other Minima Count', 'Source NC File'
        ]
        for col in columns_order:
            if col not in df.columns:
                df[col] = None
        df = df[columns_order]
        
        # Append to unified DataFrame
        if self.all_tracks_df is None:
            self.all_tracks_df = df
        else:
            self.all_tracks_df = pd.concat([self.all_tracks_df, df], ignore_index=True)
        
        # Also keep individual records for summary purposes
        if typhoon_name not in self.track_records:
            self.track_records[typhoon_name] = []
        self.track_records[typhoon_name].extend(track_results)
        
        print(f"✅ Typhoon {typhoon_name} tracking results appended to unified dataset (total {len(track_results)} records)")

    def save_all_track_summary(self, output_dir: str = "track_results") -> None:
        """Generate summary file for all typhoon tracking results"""
        if not self.track_records:
            print("⚠️  No typhoon tracking records available to generate summary")
            return
            
        os.makedirs(output_dir, exist_ok=True)
        
        # Save unified tracking data (all time steps for all typhoons)
        if self.all_tracks_df is not None and not self.all_tracks_df.empty:
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            unified_path = os.path.join(output_dir, f"all_typhoons_tracking_test.csv")
            self.all_tracks_df.to_csv(unified_path, index=False, encoding='utf-8-sig')
            print(f"✅ All typhoons unified tracking data saved to: {os.path.abspath(unified_path)}")
            print(f"   Total records: {len(self.all_tracks_df)}")
        
        # Generate summary statistics (one row per typhoon)
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
            max_psi = min(record['Center Streamfunction Value'] for record in records if record['Center Streamfunction Value'] is not None)
            merged = any(record['Is Vortex Merged'] for record in records if record['Is Vortex Merged'] is not None)
            
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
                'Number of Data Files': len(set(record['Source NC File'] for record in records if record['Source NC File'] is not None))
            })
        
        if summary:
            summary_df = pd.DataFrame(summary)
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            summary_path = os.path.join(output_dir, f"all_typhoons_summary_{timestamp}.csv")
            summary_df.to_csv(summary_path, index=False, encoding='utf-8-sig')
            print(f"✅ All typhoons summary saved to: {os.path.abspath(summary_path)}")

    def identify_typhoon_centers(self, psi_field: np.ndarray) -> List[Tuple[int, int]]:
        """Identify local minimum points of streamfunction (for initial positioning)"""
        neighborhood = np.ones((3, 3), dtype=bool)
        neighborhood[1, 1] = False
        local_min = ndimage.minimum_filter(psi_field, footprint=neighborhood, mode='nearest')
        min_mask = psi_field < local_min
        background_mean = np.mean(psi_field)
        threshold_mask = psi_field < (background_mean - self.detection_threshold * 0.8)
        valid_centers = np.where(min_mask & threshold_mask)
        return list(zip(valid_centers[0], valid_centers[1]))

    def find_inner_closed_contour(self, psi_field: np.ndarray, min_point: Tuple[int, int]) -> Optional[np.ndarray]:
        """Find the innermost closed contour enclosing the target minimum"""
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
        """Calculate contour center (weighted/arithmetic mean)"""
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

    def find_best_center(self, psi_field: np.ndarray, last_center_lat: float, last_center_lon: float,
                         lat_dim: np.ndarray, lon_dim: np.ndarray, is_first_track: bool = False) -> Tuple[Tuple[float, float], float, Tuple[int, int], Optional[np.ndarray]]:
        """Find target typhoon center (contour + minimum)"""
        if is_first_track:
            print(f"ℹ️  First tracking: global search for target minimum point (reference position: {last_center_lat:.2f}°N, {last_center_lon:.2f}°E)")
            candidate_mins = self.identify_typhoon_centers(psi_field)
            if not candidate_mins:
                raise ValueError("Global search found no local minimum points (no vortex)")
            
            ref_lat_idx = np.argmin(np.abs(lat_dim - last_center_lat))
            ref_lon_idx = np.argmin(np.abs(lon_dim - last_center_lon))
            
            candidate_mins_sorted = sorted(
                candidate_mins,
                key=lambda x: np.sqrt((x[0] - ref_lat_idx)**2 + (x[1] - ref_lon_idx)** 2)
            )[:self.global_search_threshold]
            
            min_dist = float('inf')
            target_min_idx = None
            for (lat_idx, lon_idx) in candidate_mins_sorted:
                dist = np.sqrt((lat_idx - ref_lat_idx)**2 + (lon_idx - ref_lon_idx)** 2)
                if dist < min_dist:
                    min_dist = dist
                    target_min_idx = (lat_idx, lon_idx)
            if target_min_idx is None:
                raise ValueError(f"Global search found no minimum point near the reference position")
        
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
                    key=lambda x: np.sqrt((x[0] - last_lat_idx)**2 + (x[1] - last_lon_idx)** 2)
                )[:self.global_search_threshold]
                
                min_dist = float('inf')
                target_min_idx = None
                for (lat_idx, lon_idx) in candidate_mins_sorted:
                    dist = np.sqrt((lat_idx - last_lat_idx)**2 + (lon_idx - last_lon_idx)** 2)
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
                    dist = np.sqrt((global_i - last_lat_idx)**2 + (global_j - last_lon_idx)** 2)
                    if dist < min_dist:
                        min_dist = dist
                        target_min_idx = (global_i, global_j)
        
        target_row, target_col = target_min_idx
        target_psi = psi_field[target_row, target_col]
        if target_psi > self.psi_min_threshold:
            raise ValueError(
                f"Target minimum invalid: streamfunction value {target_psi:.2e} m²/s > threshold {self.psi_min_threshold:.2e} m²/s (no vortex特征)"
            )
        print(f"✅ Target minimum found:")
        print(f"  Grid index: ({target_row}, {target_col}) → Latitude/Longitude: ({lat_dim[target_row]:.2f}°N, {lon_dim[target_col]:.2f}°E)")
        print(f"  Streamfunction value: {target_psi:.2e} m²/s")
        
        inner_contour = self.find_inner_closed_contour(psi_field, target_min_idx)
        if inner_contour is None:
            print(f"⚠️  No closed contour found enclosing the target minimum, falling back to minimum point as center")
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
                                 lat_min: float, lat_max: float, lon_min: float, lon_max: float) -> bool:
        """
        Check whether the center has reached the data boundary, or is outside China's national boundary and >100km away (using local SHP file only)
        Returns: True (need to stop tracking), False (continue tracking normally)
        """
        # -------------------------- 1. Original data boundary judgment (保留) --------------------------
        lat_dist_min = center_lat - (lat_min + self.boundary_buffer)
        lat_dist_max = (lat_max - self.boundary_buffer) - center_lat
        lon_dist_min = center_lon - (lon_min + self.boundary_buffer)
        lon_dist_max = (lon_max - self.boundary_buffer) - center_lon

        # Data boundary judgment: stop tracking if exceeded
        if any(dist <= 0 for dist in [lat_dist_min, lat_dist_max, lon_dist_min, lon_dist_max]):
            print(f"⚠️  Reached data boundary:")
            print(f"  Center position: ({center_lat:.2f}°N, {center_lon:.2f}°E)")
            print(f"  Buffered boundary: latitude [{lat_min+self.boundary_buffer:.2f}, {lat_max-self.boundary_buffer:.2f}]°N, longitude [{lon_min+self.boundary_buffer:.2f}, {lon_max-self.boundary_buffer:.2f}]°E")
            return True

        # -------------------------- 2. Use local SHP only to calculate China boundary relationship --------------------------
        LOCAL_SHP_PATH = r".\bou1_4m\bou1_4p.shp"
        china_geoms = None
        min_dist_km = float('inf')
        is_inside = False  # New: flag for whether inside national boundary
        
        if not os.path.exists(LOCAL_SHP_PATH):
            print(f"❌ Local SHP file does not exist: {LOCAL_SHP_PATH}, skipping China boundary judgment")
            return False

        try:
            sf = shapefile.Reader(LOCAL_SHP_PATH)
            china_geoms = sf.shapes()
            print(f"✅ Successfully loaded local SHP file: {os.path.basename(LOCAL_SHP_PATH)} (total {len(china_geoms)} boundary shapes)")
        except Exception as e:
            print(f"❌ Failed to read local SHP: {str(e)}, skipping China boundary judgment")
            return False

        # 2.1 First determine whether inside national boundary (point-in-polygon detection)
        if china_geoms:
            from shapely.geometry import Point, Polygon  # Lazy import shapely library
            
            # Convert typhoon center to point object
            typhoon_point = Point(center_lon, center_lat)
            
            # Check if inside any China polygon
            for geom in china_geoms:
                if geom.shapeType == 5:  # Polygon type
                    # Convert SHP vertices to shapely polygon
                    polygon = Polygon(geom.points)
                    if polygon.contains(typhoon_point):
                        is_inside = True
                        break  # Inside one polygon is enough to determine as domestic

            # 2.2 If domestic, directly judge as normal tracking
            if is_inside:
                print(f"✅ Typhoon is within China's national boundary, continuing tracking")
                return False

            # 2.3 If outside, calculate shortest distance to national boundary
            geod = Geodesic.WGS84
            for geom in china_geoms:
                if geom.shapeType == 5:
                    vertices = geom.points
                    for (lon, lat) in vertices:
                        dist_m = geod.Inverse(center_lat, center_lon, lat, lon)['s12']
                        dist_km = dist_m / 1000
                        if dist_km < min_dist_km:
                            min_dist_km = dist_km

            # 2.4 Stop tracking only if outside and distance exceeds 100km
            if min_dist_km > 200:
                print(f"⚠️  Typhoon is outside China's national boundary and distance exceeds 200km, stopping tracking:")
                print(f"  Center position: ({center_lat:.2f}°N, {center_lon:.2f}°E)")
                print(f"  Shortest distance to China's national boundary: {min_dist_km:.1f}km > threshold 200km")
                return True
            else:
                print(f"✅ Typhoon is outside China's national boundary but distance ≤200km, continuing tracking")

        # -------------------------- 3. All judgments passed, continue tracking --------------------------
        return False
            

    def check_psi_min_condition(self, center_psi: float) -> bool:
        """Check whether there is no vortex feature"""
        if center_psi > self.psi_min_threshold:
            print(f"⚠️  No vortex feature: center streamfunction value {center_psi:.2e} > threshold {self.psi_min_threshold:.2e} m²/s")
            return True
        return False

    def check_hourly_distance(self, last_lat: float, last_lon: float, current_lat: float, current_lon: float) -> Tuple[bool, float]:
        """Check whether movement is abnormal (>1 degree/hour)"""
        distance = calculate_spherical_distance(last_lat, last_lon, current_lat, current_lon)
        if distance > self.max_hourly_distance:
            print(f"⚠️  Hourly movement distance abnormal!")
            print(f"  Previous time step position: ({last_lat:.2f}°N, {last_lon:.2f}°E)")
            print(f"  Current time step position: ({current_lat:.2f}°N, {current_lon:.2f}°E)")
            print(f"  Movement distance: {distance:.2f}° > threshold {self.max_hourly_distance}° (judged as abnormal movement)")
            return True, distance
        else:
            print(f"✅ Hourly movement distance normal: {distance:.2f}° (≤ threshold {self.max_hourly_distance}°)")
            return False, distance

    def _validate_tracking_consistency(self, continue_tracking: bool, 
                                  boundary_reached: bool,
                                  track_failed: bool,
                                  psi_exceeded: bool,
                                  distance_exceeded: bool,
                                  vortex_merged: bool) -> Tuple[bool, str]:
        """Validate tracking state consistency"""
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
        """
        Fix 1: Added time_idx parameter to explicitly specify current time step index
        Ensure extraction of vorticity value for a single time step and single grid point only
        """
        try:
            # 1. Validate and set vertical level index
            if level_idx is None:
                level_idx = self.level_idx
                if level_idx is None:
                    raise ValueError("Vertical level index level_idx not initialized (need to set in __init__)")
            
            # 2. Locate latitude/longitude indices of center point
            lat_vals = ds[lat_dim].values
            lon_vals = ds[lon_dim].values
            lat_idx = np.argmin(np.abs(lat_vals - center_lat))
            lon_idx = np.argmin(np.abs(lon_vals - center_lon))
            
            # 3. Validate all indices (time, latitude/longitude, vertical level)
            if not (0 <= time_idx < ds.sizes['valid_time']):  # Assume time dimension is 'valid_time'
                raise IndexError(f"Time index out of range: {time_idx} (valid range 0-{ds.sizes['valid_time']-1})")
            if not (0 <= lat_idx < len(lat_vals)):
                raise IndexError(f"Latitude index out of range: {lat_idx} (valid range 0-{len(lat_vals)-1})")
            if not (0 <= lon_idx < len(lon_vals)):
                raise IndexError(f"Longitude index out of range: {lon_idx} (valid range 0-{len(lon_vals)-1})")
            if not (0 <= level_idx < ds.sizes[level_dim]):
                raise IndexError(f"Vertical level index out of range: {level_idx} (valid range 0-{ds.sizes[level_dim]-1})")
            
            # 4. Extract vorticity value: explicitly specify [time + latitude/longitude + vertical level], ensure scalar return
            vort_array = ds['vo'].isel(
                {
                    'valid_time': time_idx,  # Key: specify current time step index
                    lat_dim: lat_idx,
                    lon_dim: lon_idx,
                    level_dim: level_idx
                }
            ).values
            
            # 5. Verify result is a single scalar
            if vort_array.size != 1:
                raise ValueError(f"Extracted vorticity value is not a single scalar (actual size: {vort_array.size}, may not have time dimension specified)")
            
            return float(vort_array.item())  # Convert to Python float to avoid array type

        except Exception as e:
            # Fix 2: Remove reference to current_time (out-of-scope variable), use time index description instead
            print(f"❌ Failed to extract center vorticity value: {str(e)}")
            print(f"  Center position: ({center_lat:.2f}°N, {center_lon:.2f}°E)")
            print(f"  Current time step index: {time_idx} (corresponding to time step {time_idx+1} in NC file)")
            return np.nan  # Return NaN on error, do not interrupt tracking
    
    def track_single_file(self, nc_file_path: str, start_time: datetime, ftime: datetime, last_center_lat: float,
                          last_center_lon: float, typhoon_name: str, time_dim: str = 'valid_time',
                          lat_dim: str = 'latitude', lon_dim: str = 'longitude',
                          is_first_file: bool = True) -> Tuple[List[Dict], bool, datetime, bool, bool, bool, bool, bool]:
            """Track typhoon path in single NC file (memory-optimized version)"""
            track_results = []
            continue_tracking = True
            last_succ_time = None
            boundary_reached = False
            track_failed = False
            psi_exceeded = False
            distance_exceeded = False
            vortex_merged = False
            # Initialize previous time step position as tracking start position (for calculating first time step movement distance)
            prev_center_lat = last_center_lat  # Modified: initialize to tracking start latitude
            prev_center_lon = last_center_lon  # Modified: initialize to tracking start longitude
            
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
                    print(f"📊 Data range: latitude [{lat_min:.2f},{lat_max:.2f}]°N, longitude [{lon_min:.2f},{lon_max:.2f}]°E")
                    
                    # Optimized time matching logic
                    time_diffs = [abs((dt - start_time).total_seconds()) for dt in time_series]
                    min_diff = min(time_diffs) if time_diffs else float('inf')
                    start_idx = time_diffs.index(min_diff) if time_diffs else None
                    
                    if start_idx is None or min_diff > self.time_match_tolerance:
                        print(f"⏱️  Expected time step: {start_time.strftime('%Y-%m-%d %H:%M')}")
                        print(f"⏱️  Time steps in file: {[dt.strftime('%Y-%m-%d %H:%M') for dt in time_series[:5]]}...")  # Show only first 5 time steps
                        raise ValueError(
                            f"No time step close to {start_time.strftime('%Y-%m-%d %H:%M')} found (allow ±{self.time_match_tolerance/3600} hour deviation)"
                        )
                    print(f"✅ Matching time step found: {time_series[start_idx].strftime('%Y-%m-%d %H:%M')}, time difference {min_diff/3600:.1f} hours")
                    
                    vo_dims = ds['vo'].dims
                    extra_dims = [d for d in vo_dims if d not in [time_dim, lat_dim, lon_dim]]
                    if extra_dims:
                        print(f"🔍 vo extra dimensions: {extra_dims}, extracting by index {self.level_idx}")
                        if extra_dims[0] not in ds.dims:
                            raise KeyError(f"Extra dimension {extra_dims[0]} not in NC file")
                        if self.level_idx >= ds.sizes[extra_dims[0]]:
                            raise IndexError(f"{extra_dims[0]} index {self.level_idx} out of range (0~{ds.sizes[extra_dims[0]]-1})")
                    
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
                        
                        # -------------------------- Memory optimization 5: Timely release of large arrays --------------------------
                        try:
                            vo_time = ds['vo'].isel({time_dim: hour_idx})
                            if extra_dims:
                                vo_time = vo_time.isel({extra_dims[0]: self.level_idx})
                            vo_time = vo_time.where(np.isfinite(vo_time), 0)
                            vorticity_da = vo_time
                            print(f"✅ Extracted vo data: dimensions {list(vorticity_da.dims)}, shape {vorticity_da.shape}")
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
                        
                        # Modified: calculate hourly movement distance (first time step uses initial position as reference)
                        distance_exceeded = False
                        hourly_distance = 0.0
                        # Even for first time step, calculate movement distance (using prev_center_lat/lon, initialized to tracking start position)
                        distance_exceeded, hourly_distance = self.check_hourly_distance(
                            prev_center_lat, prev_center_lon, center_lat, center_lon
                        )
                        if distance_exceeded:
                            continue_tracking = False
                        
                        # Update previous time step position to current time step position (preparing for next time step)
                        prev_center_lat, prev_center_lon = center_lat, center_lon
                        
                        # Get typhoon ID information
                        typhoon_info = self.get_typhoon_end_info(typhoon_name, ftime)
                        if not typhoon_info:
                            raise ValueError(f"Unable to get basic information for typhoon {typhoon_name}")
                            
                        center_vort = self.get_center_vorticity(
                            ds=ds,
                            center_lat=center_lat,
                            center_lon=center_lon,
                            lat_dim=lat_dim,
                            lon_dim=lon_dim,
                            time_idx=hour_idx,  # Key: pass current time step index
                            level_idx=self.level_idx
                        )
            
                        # 4. Construct tracking record (only necessary fields)
                        track_record = {
                            'International ID': typhoon_info['International ID'],
                            'Tropical Cyclone Serial': typhoon_info['Tropical Cyclone Serial'],
                            'China ID': typhoon_info['China ID'],
                            'Typhoon Name': typhoon_name,
                            'Tracking Time': current_dt,
                            'Center Latitude': center_lat,
                            'Center Longitude': center_lon,
                            'Center Streamfunction Value': target_psi,
                            'Center Vorticity Value': center_vort,  # Only keep center vorticity value
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
                        
                        # Clear large arrays, release memory
                        del psi_field, vorticity_da, vo_time
                        import gc
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
                    
                    # Append results to unified dataset (not saving individual files)
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


# -------------------------- Main Program --------------------------
def main():
    # Configure paths (adapted to "year/month" level NC file storage)
    CSV_PATH = r".\TRVStartPositions.csv"          # Typhoon end position CSV
    ERA5_ROOT_DIR = r"E:\era5_vorticity_data"  # ERA5 root directory (contains subdirectories like "1980/01")
    
    # -------------------------- Memory optimization 7: Disable/limit visualization on initialization --------------------------
    tracker = StreamfunctionTyphoonTracker(
        delta_psi=2.0e6,
        min_lifetime=6,
        detection_threshold=1.0e6,
        max_track_hours=120,
        psi_min_threshold=-1e5,
        max_hourly_distance=2,
        time_match_tolerance=7200,  # 2-hour time tolerance
        enable_vis=False  # Set to False when memory is insufficient, only output data without generating images
    )
    
    def get_era5_files_by_year_month(start_time: datetime) -> List[str]:
        """Adapt to "year/month" level NC file search"""
        file_list = []
        # Forward tracking: find all NC files within 5 days after start time (covers cross-month cases)
        for i in range(5):
            target_date = start_time + timedelta(days=i)
            year = target_date.strftime("%Y")
            month = target_date.strftime("%m")  # Ensure month is two digits (01~12)
            
            # NC file naming (modify according to actual filename)
            nc_filename = f"vor_{target_date.strftime('%Y%m%d')}.nc"
            # Full path: root/year/month/filename
            nc_filepath = os.path.join(ERA5_ROOT_DIR, year, month, nc_filename)
            
            # Also support single-digit month folders (e.g., both "05" and "5" can be recognized)
            if not os.path.exists(nc_filepath):
                month_single = str(int(month))  # Remove leading zero (05→5)
                nc_filepath_single = os.path.join(ERA5_ROOT_DIR, year, month_single, nc_filename)
                if os.path.exists(nc_filepath_single):
                    nc_filepath = nc_filepath_single
                else:
                    print(f"⚠️  NC file not found: {os.path.basename(nc_filepath)}")
                    continue
            
            file_list.append(nc_filepath)
        
        # Sort by time ascending (ensuring correct forward tracking order)
        file_list.sort()
        return file_list
    
    # Load typhoon end position data
    try:
        tracker.load_end_positions(CSV_PATH)
    except Exception as e:
        print(f"❌ Failed to load end position data, program exiting: {str(e)}")
        return
    
    # -------------------------- Memory optimization 8: Limit concurrency/timely cleanup during batch processing --------------------------
    for idx, (_, row) in enumerate(tracker.end_positions_df.iterrows()):
        # Force memory cleanup every 3 typhoons (avoid accumulation)
        if idx % 3 == 0 and idx != 0:
            print(f"\n📌 Processed {idx} typhoons, forcing memory cleanup...")
            import gc
            gc.collect()
        
        typhoon_name = row['Typhoon Name']
        try:
            end_time = row['End Time_dt']
            start_time = end_time  # Forward tracking 5 days
            start_lat = float(row['End Latitude'])
            start_lon = float(row['End Longitude'])
        except Exception as e:
            print(f"⚠️  Skipping invalid record {typhoon_name}: {str(e)}")
            continue
    
        print(f"\n{'#'*60}")
        print(f"Processing typhoon {idx+1}/{len(tracker.end_positions_df)}: {typhoon_name}")
        print(f"Forward tracking start point: {start_lat:.2f}°N, {start_lon:.2f}°E, start time: {start_time.strftime('%Y-%m-%d %H:%M')}")
        print(f"Expected end time: {end_time.strftime('%Y-%m-%d %H:%M')}")
        print(f"{'#'*60}")
    
        # Find corresponding "year/month" level NC files
        era5_files = get_era5_files_by_year_month(start_time)
        if not era5_files:
            print(f"⚠️  No ERA5 data files found for {typhoon_name}, skipping")
            continue
        print(f"📂 Found {len(era5_files)} related ERA5 files")
    
        # Forward tracking initialization
        current_lat = start_lat
        current_lon = start_lon
        current_time = start_time
        ftime=start_time
        all_track_results = []
        is_first_file = True
    
        for file in era5_files:
            print(f"\nProcessing file: {os.path.basename(file)}")
            try:
                track_results, cont_flag, last_t, boundary, failed, psi_exceed, dist_exceed, merged = tracker.track_single_file(
                    nc_file_path=file,
                    start_time=current_time,
                    ftime=ftime,
                    last_center_lat=current_lat,
                    last_center_lon=current_lon,
                    typhoon_name=typhoon_name,
                    is_first_file=is_first_file
                )
            except Exception as e:
                print(f"❌ Failed to process file {file}: {str(e)}")
                break
    
            if track_results:
                all_track_results.extend(track_results)
                current_lat = track_results[-1]['Center Latitude']
                current_lon = track_results[-1]['Center Longitude']
                current_time = last_t
    
            if not cont_flag:
                print(f"🛑 Tracking terminated, reason: {'Boundary reached' if boundary else 'Tracking failed' if failed else 'Vortex merged' if merged else 'Other reason'}")
                break
    
            is_first_file = False
    
        if all_track_results:
            print(f"📝 Typhoon {typhoon_name} tracking completed, total {len(all_track_results)} time steps")
        else:
            print(f"⚠️  Typhoon {typhoon_name} has no valid tracking results")
    
    # Generate summary for all typhoons (now saves unified tracking data + summary)
    tracker.save_all_track_summary()
    print("\nAll typhoons processing completed")


if __name__ == "__main__":
    main()