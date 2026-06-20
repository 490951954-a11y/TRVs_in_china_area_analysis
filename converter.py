"""
CSV to TRV Format Converter
===========================
Convert tracking results to standard TRV format with vorticity and wind speed data.
"""

import csv
import os
from datetime import datetime
import netCDF4 as nc
import numpy as np
import re
from scipy.ndimage import uniform_filter
from typing import Optional, List, Dict, Any


class CSVToTRVConverter:
    """Convert tracking CSV to TRV format with ERA5 data."""
    
    def __init__(self, target_level: int = 850, level_index: int = 0,
                 lon_range: tuple = (70.0, 140.0), lat_range: tuple = (0.0, 60.0),
                 grid_res: float = 0.25, search_radius_km: float = 200):
        """
        Initialize converter.
        
        Args:
            target_level: Target pressure level in hPa
            level_index: Index of target level in NC file
            lon_range: Valid longitude range
            lat_range: Valid latitude range
            grid_res: Default grid resolution in degrees
            search_radius_km: Search radius for wind speed calculation in km
        """
        self.target_level = target_level
        self.level_index = level_index
        self.lon_range = lon_range
        self.lat_range = lat_range
        self.grid_res = grid_res
        self.search_radius_km = search_radius_km
        
        self.input_file = None
        self.output_file = None
        self.data = []
        
        # ERA5 data
        self.vor_data = None
        self.vor_lat = None
        self.vor_lon = None
        self.u_data = None
        self.v_data = None
        self.uv_lat = None
        self.uv_lon = None
    
    def parse_datetime(self, time_str: str) -> datetime:
        """Parse time string to datetime object."""
        formats = [
            '%Y/%m/%d %H:%M',
            '%Y/%m/%d %H:%M:%S',
            '%Y-%m-%d %H:%M',
            '%Y-%m-%d %H:%M:%S',
            '%Y/%m/%d',
            '%Y-%m-%d',
            '%Y%m%d',
            '%Y%m%d%H',
            '%Y%m%d%H%M',
            '%Y%m%d%H%M%S',
        ]
        
        for fmt in formats:
            try:
                return datetime.strptime(time_str.strip(), fmt)
            except ValueError:
                continue
        
        date_match = re.search(r'(\d{4})[/-](\d{1,2})[/-](\d{1,2})', time_str)
        if date_match:
            year, month, day = date_match.groups()
            time_match = re.search(r'(\d{1,2}):(\d{2})(?::(\d{2}))?', time_str)
            if time_match:
                hour = int(time_match.group(1))
                minute = int(time_match.group(2))
                second = int(time_match.group(3)) if time_match.group(3) else 0
                return datetime(int(year), int(month), int(day), hour, minute, second)
            else:
                return datetime(int(year), int(month), int(day))
        
        raise ValueError(f"Cannot parse time string: {time_str}")
    
    def load_era5_data(self, vor_file_path: str, uv_file_path: str) -> None:
        """Load ERA5 vorticity and wind field data."""
        # Load vorticity
        if os.path.exists(vor_file_path):
            try:
                ds_vor = nc.Dataset(vor_file_path)
                print(f"Loading vorticity file: {vor_file_path}")
                
                if 'vo' in ds_vor.variables:
                    self.vor_data = ds_vor.variables['vo'][:]
                    print(f"  Vorticity data shape: {self.vor_data.shape}")
                
                if 'latitude' in ds_vor.variables:
                    self.vor_lat = ds_vor.variables['latitude'][:]
                elif 'lat' in ds_vor.variables:
                    self.vor_lat = ds_vor.variables['lat'][:]
                
                if 'longitude' in ds_vor.variables:
                    self.vor_lon = ds_vor.variables['longitude'][:]
                elif 'lon' in ds_vor.variables:
                    self.vor_lon = ds_vor.variables['lon'][:]
                
                if self.vor_lon is not None and self.vor_lon.min() < 0:
                    self.vor_lon = (self.vor_lon + 360) % 360
                
                print(f"  Latitude range: {self.vor_lat.min():.2f} ~ {self.vor_lat.max():.2f}")
                print(f"  Longitude range: {self.vor_lon.min():.2f} ~ {self.vor_lon.max():.2f}")
                ds_vor.close()
            except Exception as e:
                print(f"Error loading vorticity data: {e}")
        else:
            print(f"Warning: Vorticity file does not exist - {vor_file_path}")
        
        # Load wind field
        if os.path.exists(uv_file_path):
            try:
                ds_uv = nc.Dataset(uv_file_path)
                print(f"Loading wind field file: {uv_file_path}")
                
                if 'u' in ds_uv.variables and 'v' in ds_uv.variables:
                    self.u_data = ds_uv.variables['u'][:]
                    self.v_data = ds_uv.variables['v'][:]
                    print(f"  Wind field data shape: {self.u_data.shape}")
                
                if 'latitude' in ds_uv.variables:
                    self.uv_lat = ds_uv.variables['latitude'][:]
                elif 'lat' in ds_uv.variables:
                    self.uv_lat = ds_uv.variables['lat'][:]
                
                if 'longitude' in ds_uv.variables:
                    self.uv_lon = ds_uv.variables['longitude'][:]
                elif 'lon' in ds_uv.variables:
                    self.uv_lon = ds_uv.variables['lon'][:]
                
                if self.uv_lon is not None and self.uv_lon.min() < 0:
                    self.uv_lon = (self.uv_lon + 360) % 360
                
                print(f"  Latitude range: {self.uv_lat.min():.2f} ~ {self.uv_lat.max():.2f}")
                print(f"  Longitude range: {self.uv_lon.min():.2f} ~ {self.uv_lon.max():.2f}")
                ds_uv.close()
            except Exception as e:
                print(f"Error loading wind field data: {e}")
    
    def get_vorticity(self, lat: float, lon: float, hour: int) -> Optional[int]:
        """Get vorticity value at specified location and time."""
        if self.vor_data is None:
            return None
        
        try:
            time_idx = hour % 24
            vor_field = self.vor_data[time_idx, self.level_index, :, :]
            
            if lon < 0:
                lon = lon + 360
            
            lat_idx = np.argmin(np.abs(self.vor_lat - lat))
            lon_idx = np.argmin(np.abs(self.vor_lon - lon))
            
            vorticity = float(vor_field[lat_idx, lon_idx])
            vorticity_scaled = int(round(vorticity * 1e5))
            
            print(f"  Vorticity: time={time_idx}, pressure_level={self.level_index}({self.target_level}hPa), "
                  f"value={vorticity:.6f} s^-1 -> {vorticity_scaled}")
            return vorticity_scaled
            
        except Exception as e:
            print(f"  Error getting vorticity value: {e}")
            return None
    
    def calc_cyclone_central_wind(self, u_field: np.ndarray, v_field: np.ndarray,
                                   center_lon: float, center_lat: float,
                                   lats: np.ndarray, lons: np.ndarray,
                                   grid_res: float = 0.25) -> Optional[float]:
        """Calculate maximum sustained wind near cyclone center."""
        try:
            search_radius_deg = self.search_radius_km / 111
            search_grid_num = int(np.ceil(search_radius_deg / grid_res))
            
            center_lon_idx = np.argmin(np.abs(lons - center_lon))
            center_lat_idx = np.argmin(np.abs(lats - center_lat))
            
            lon_start = max(0, center_lon_idx - search_grid_num)
            lon_end = min(u_field.shape[1], center_lon_idx + search_grid_num + 1)
            lat_start = max(0, center_lat_idx - search_grid_num)
            lat_end = min(u_field.shape[0], center_lat_idx + search_grid_num + 1)
            
            u_search = u_field[lat_start:lat_end, lon_start:lon_end]
            v_search = v_field[lat_start:lat_end, lon_start:lon_end]
            wind_speed = np.sqrt(u_search**2 + v_search**2)
            
            wind_speed_smoothed = uniform_filter(wind_speed, size=3, mode='constant')
            
            return round(np.max(wind_speed_smoothed), 2)
        except Exception as e:
            print(f"  Error calculating center wind speed: {str(e)}")
            return None
    
    def get_wind_speed(self, lat: float, lon: float, hour: int) -> Optional[int]:
        """Get maximum wind speed at specified location and time."""
        if self.u_data is None or self.v_data is None:
            return None
        
        try:
            time_idx = hour % 24
            u_field = self.u_data[time_idx, self.level_index, :, :]
            v_field = self.v_data[time_idx, self.level_index, :, :]
            
            if lon < 0:
                lon = lon + 360
            
            if len(self.uv_lon) > 1 and not np.isnan(np.diff(self.uv_lon)).any():
                grid_res = np.mean(np.diff(self.uv_lon))
            else:
                grid_res = self.grid_res
            
            wind_speed = self.calc_cyclone_central_wind(
                u_field, v_field, lon, lat, self.uv_lat, self.uv_lon, grid_res
            )
            
            if wind_speed is not None:
                wind_scaled = int(round(wind_speed * 10))
                print(f"  Wind speed: time={time_idx}, pressure_level={self.level_index}({self.target_level}hPa), "
                      f"max={wind_speed:.2f} m/s -> {wind_scaled}")
                return wind_scaled
            else:
                return None
            
        except Exception as e:
            print(f"  Error getting maximum wind speed: {e}")
            return None
    
    def determine_stop_reason(self, row: Dict) -> int:
        """Determine the stop reason code."""
        tracking_status = row.get('Tracking Status', '').strip()
        
        if tracking_status == 'Normal':
            return 0
        elif tracking_status == 'Terminated':
            if row.get('Is Vortex Merged', '').strip().lower() == 'yes':
                return 1
            if row.get('Is Abnormal Movement', '').strip().lower() == 'yes':
                return 2
            if row.get('Is Boundary Reached', '').strip().lower() == 'yes':
                return 3
            return 0
        else:
            return 0
    
    def parse_input_csv(self, input_file: str) -> bool:
        """Parse input CSV file."""
        self.input_file = input_file
        self.data = []
        
        try:
            encodings = ['utf-8-sig', 'utf-8', 'gbk', 'gb2312']
            for encoding in encodings:
                try:
                    with open(input_file, 'r', encoding=encoding) as f:
                        reader = csv.DictReader(f)
                        self.data = list(reader)
                    print(f"Successfully read {len(self.data)} rows (using encoding: {encoding})")
                    return True
                except UnicodeDecodeError:
                    continue
            raise ValueError("Cannot read file with any encoding")
        except Exception as e:
            print(f"Error reading CSV file: {e}")
            return False
    
    def convert_to_trv_format(self, output_file: str, vor_file: str = None, 
                              uv_file: str = None) -> bool:
        """Convert data to TRV format."""
        if not self.data:
            print("No data to convert, please parse input file first")
            return False
        
        if vor_file and uv_file:
            self.load_era5_data(vor_file, uv_file)
        
        output_lines = []
        first_row = self.data[0]
        last_row = self.data[-1]
        
        try:
            intl_id = int(float(first_row.get('International ID', '16')))
            china_id = first_row.get('China ID', '8012')
            tc_name = first_row.get('Typhoon Name', 'Unnamed')
            
            first_time = first_row.get('Tracking Time', '')
            dt = self.parse_datetime(first_time)
            start_date = dt.strftime('%Y%m%d')
            
            last_status = last_row.get('Tracking Status', '').strip()
            
            if last_status == 'Terminated':
                use_data = self.data[:-1]
                stop_reason = self.determine_stop_reason(last_row)
            else:
                use_data = self.data
                stop_reason = 0
            
            # Check vorticity and truncate if negative
            valid_data = []
            for i, row in enumerate(use_data):
                time_str = row.get('Tracking Time', '')
                dt_temp = self.parse_datetime(time_str)
                hour = dt_temp.hour
                lat = float(row.get('Center Latitude', 0))
                lon = float(row.get('Center Longitude', 0))
                
                if self.vor_data is not None:
                    vorticity = self.get_vorticity(lat, lon, hour)
                    if vorticity is not None and vorticity < 0:
                        print(f"\n⚠️ Detected vorticity < 0 at row {i+1} (value: {vorticity}), "
                              f"truncating subsequent records")
                        valid_data = use_data[:i]
                        break
                valid_data = use_data
            
            if valid_data != use_data:
                use_data = valid_data
            
            sequence_num = first_row.get('International ID', '0016')
            record_count = len(use_data)
            header = f"66666,{intl_id:04d},{record_count:03d},{int(sequence_num):04d},"
            header += f"{china_id},{stop_reason},{tc_name},{start_date}"
            output_lines.append(header)
            
            print(f"\nHeader record: {header}")
            print(f"Record count: {record_count}, Stop reason: {stop_reason}")
            print("-" * 70)
            
            for i, row in enumerate(use_data):
                time_str = row.get('Tracking Time', '')
                dt = self.parse_datetime(time_str)
                time_formatted = dt.strftime('%Y%m%d%H')
                hour = dt.hour
                
                lat = float(row.get('Center Latitude', 0))
                lon = float(row.get('Center Longitude', 0))
                lat_int = int(round(lat * 10))
                lon_int = int(round(lon * 10))
                
                stream_func_raw = float(row.get('Center Streamfunction Value', 0))
                stream_func = int(round(abs(stream_func_raw) / 10000))
                
                print(f"\nRecord {i+1}: time={time_str}, hour={hour}, "
                      f"position=({lat:.2f}, {lon:.2f})")
                
                if self.vor_data is not None:
                    vorticity = self.get_vorticity(lat, lon, hour)
                    if vorticity is None:
                        vort_raw = float(row.get('Center Vorticity Value', 0))
                        vorticity = int(round(vort_raw * 1e5))
                        print(f"  ⚠️ Using CSV vorticity value (fallback): {vorticity}")
                else:
                    vort_raw = float(row.get('Center Vorticity Value', 0))
                    vorticity = int(round(vort_raw * 1e5))
                    print(f"  Using CSV vorticity value: {vorticity}")
                
                if self.u_data is not None:
                    velocity = self.get_wind_speed(lat, lon, hour)
                    if velocity is None:
                        velocity = int(stream_func * 0.3)
                        if velocity < 10:
                            velocity = 150
                        print(f"  ⚠️ Using estimated wind speed (fallback): {velocity}")
                else:
                    velocity = int(stream_func * 0.3)
                    if velocity < 10:
                        velocity = 150
                    print(f"  Using estimated wind speed: {velocity}")
                
                track_line = f"{time_formatted},{lat_int:03d},{lon_int:04d},"
                track_line += f"{stream_func},{vorticity},{velocity}"
                output_lines.append(track_line)
                print(f"  Generated: {track_line}")
            
            print("-" * 70)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                for line in output_lines:
                    f.write(line + '\n')
            
            print(f"\nSuccessfully converted {len(use_data)} track records to {output_file}")
            return True
            
        except Exception as e:
            print(f"Error converting data: {e}")
            import traceback
            traceback.print_exc()
            return False