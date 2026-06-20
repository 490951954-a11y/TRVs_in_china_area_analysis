# -*- coding: utf-8 -*-
"""
Created on Sat Jun 20 18:27:53 2026

@author: YE
"""

# -*- coding: utf-8 -*-
"""
CSV to TRV Format Converter
Convert Norris_track_20260620163349.csv to TRVread.py readable format
Modified: Vorticity/wind speed extraction method aligned with vorticity_wind_temp2.py
"""

import csv
import os
from datetime import datetime
import netCDF4 as nc
import numpy as np
import re
from scipy.ndimage import uniform_filter  # New: for wind speed smoothing


class CSVToTRVConverter:
    """Converter from sample CSV to TRV format"""
    
    def __init__(self):
        self.input_file = None
        self.output_file = None
        self.data = []
        
        # Vorticity data
        self.vor_data = None
        self.vor_lat = None
        self.vor_lon = None
        
        # Wind field data
        self.u_data = None
        self.v_data = None
        self.uv_lat = None
        self.uv_lon = None
        
        # Constant configuration (consistent with vorticity_wind_temp2.py)
        self.TARGET_LEVEL = 850      # Target pressure level (hPa)
        self.LEVEL_INDEX = 0        # Index of 850hPa in NC file level dimension
        self.LON_RANGE = (70.0, 140.0)   # Valid longitude range
        self.LAT_RANGE = (0.0, 60.0)     # Valid latitude range
        self.GRID_RES_DEFAULT = 0.25     # Default grid resolution (degrees)
        self.SEARCH_RADIUS_KM = 200      # Search radius (km)
        
    def parse_datetime(self, time_str):
        """
        Parse time string, return datetime object
        """
        # Common time formats
        formats = [
            '%Y/%m/%d %H:%M',      # 1980/8/29 6:00
            '%Y/%m/%d %H:%M:%S',   # 1980/8/29 6:00:00
            '%Y-%m-%d %H:%M',      # 1980-08-29 06:00
            '%Y-%m-%d %H:%M:%S',   # 1980-08-29 06:00:00
            '%Y/%m/%d',            # 1980/8/29
            '%Y-%m-%d',            # 1980-08-29
            '%Y%m%d',              # 19800829
            '%Y%m%d%H',            # 1980082906
            '%Y%m%d%H%M',          # 198008290600
            '%Y%m%d%H%M%S',        # 19800829060000
        ]
        
        for fmt in formats:
            try:
                return datetime.strptime(time_str.strip(), fmt)
            except ValueError:
                continue
        
        # If direct match fails, try to extract date and time
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
    
    def load_era5_data(self, vor_file_path, uv_file_path):
        """
        Load ERA5 data files (aligned with read_nc_data logic in vorticity_wind_temp2.py)
        """
        # Load vorticity data
        if os.path.exists(vor_file_path):
            try:
                ds_vor = nc.Dataset(vor_file_path)
                print(f"Loading vorticity file: {vor_file_path}")
                
                if 'vo' in ds_vor.variables:
                    self.vor_data = ds_vor.variables['vo'][:]
                    print(f"  Vorticity data shape: {self.vor_data.shape}")
                    print(f"  Time steps: {self.vor_data.shape[0]}")
                    print(f"  Pressure levels: {self.vor_data.shape[1]}")
                
                # Get latitude/longitude (compatible with different field names)
                if 'latitude' in ds_vor.variables:
                    self.vor_lat = ds_vor.variables['latitude'][:]
                elif 'lat' in ds_vor.variables:
                    self.vor_lat = ds_vor.variables['lat'][:]
                
                if 'longitude' in ds_vor.variables:
                    self.vor_lon = ds_vor.variables['longitude'][:]
                elif 'lon' in ds_vor.variables:
                    self.vor_lon = ds_vor.variables['lon'][:]
                
                # Normalize longitude to 0-360 (consistent with vorticity_wind_temp2.py)
                if self.vor_lon is not None and self.vor_lon.min() < 0:
                    self.vor_lon = (self.vor_lon + 360) % 360
                
                print(f"  Latitude range: {self.vor_lat.min():.2f} ~ {self.vor_lat.max():.2f}")
                print(f"  Longitude range: {self.vor_lon.min():.2f} ~ {self.vor_lon.max():.2f}")
                ds_vor.close()
            except Exception as e:
                print(f"Error loading vorticity data: {e}")
        else:
            print(f"Warning: Vorticity file does not exist - {vor_file_path}")
        
        # Load wind field data
        if os.path.exists(uv_file_path):
            try:
                ds_uv = nc.Dataset(uv_file_path)
                print(f"Loading wind field file: {uv_file_path}")
                
                if 'u' in ds_uv.variables and 'v' in ds_uv.variables:
                    self.u_data = ds_uv.variables['u'][:]
                    self.v_data = ds_uv.variables['v'][:]
                    print(f"  Wind field data shape: {self.u_data.shape}")
                    print(f"  Time steps: {self.u_data.shape[0]}")
                    print(f"  Pressure levels: {self.u_data.shape[1]}")
                
                # Get latitude/longitude (compatible with different field names)
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
    
    def get_vorticity(self, lat, lon, hour):
        """
        Get vorticity value at specified location/time (aligned with get_meteorological_data logic in vorticity_wind_temp2.py)
        Use 850hPa level (LEVEL_INDEX=15), preserve original sign (no absolute value)
        """
        if self.vor_data is None:
            return None
        
        try:
            # Use hour directly as time index (consistent with vorticity_wind_temp2.py)
            time_idx = hour % 24
            
            # Key modification: Use LEVEL_INDEX=15 (850hPa), not layer 0
            vor_field = self.vor_data[time_idx, self.LEVEL_INDEX, :, :]
            
            # Handle longitude
            if lon < 0:
                lon = lon + 360
            
            # Find nearest grid point
            lat_idx = np.argmin(np.abs(self.vor_lat - lat))
            lon_idx = np.argmin(np.abs(self.vor_lon - lon))
            
            # Get vorticity value (preserve original sign, no absolute value)
            vorticity = float(vor_field[lat_idx, lon_idx])
            
            # Consistent with vorticity_wind_temp2.py: multiply by 1e5 and round to integer
            vorticity_scaled = int(round(vorticity * 1e5))
            
            print(f"  Vorticity: time={time_idx}, pressure_level={self.LEVEL_INDEX}(850hPa), value={vorticity:.6f} s^-1 -> {vorticity_scaled}")
            return vorticity_scaled
            
        except Exception as e:
            print(f"  Error getting vorticity value: {e}")
            return None
    
    def calc_cyclone_central_wind(self, u_field, v_field, center_lon, center_lat, 
                                   lats, lons, grid_res=0.25):
        """
        Calculate maximum sustained wind near cyclone center (identical to vorticity_wind_temp2.py)
        200km search radius + 9-point smoothing
        """
        try:
            # Calculate grid points corresponding to 200km search radius (1 deg ≈ 111 km)
            search_radius_deg = self.SEARCH_RADIUS_KM / 111
            search_grid_num = int(np.ceil(search_radius_deg / grid_res))
            
            # Locate cyclone center indices in arrays
            center_lon_idx = np.argmin(np.abs(lons - center_lon))
            center_lat_idx = np.argmin(np.abs(lats - center_lat))
            
            # Define search range (avoid array out-of-bounds)
            lon_start = max(0, center_lon_idx - search_grid_num)
            lon_end = min(u_field.shape[1], center_lon_idx + search_grid_num + 1)
            lat_start = max(0, center_lat_idx - search_grid_num)
            lat_end = min(u_field.shape[0], center_lat_idx + search_grid_num + 1)
            
            # Extract wind field within search range, calculate composite wind speed
            u_search = u_field[lat_start:lat_end, lon_start:lon_end]
            v_search = v_field[lat_start:lat_end, lon_start:lon_end]
            wind_speed = np.sqrt(u_search**2 + v_search**2)
            
            # 9-point smoothing (3x3 window) to reduce grid noise
            wind_speed_smoothed = uniform_filter(wind_speed, size=3, mode='constant')
            
            # Return maximum wind speed rounded to 2 decimal places
            return round(np.max(wind_speed_smoothed), 2)
        except Exception as e:
            print(f"  Error calculating center wind speed: {str(e)}")
            return None
    
    def get_wind_speed(self, lat, lon, hour):
        """
        Get center wind speed at specified location/time (aligned with get_meteorological_data logic in vorticity_wind_temp2.py)
        Use 850hPa level (LEVEL_INDEX=15), 200km search radius, 9-point smoothing
        """
        if self.u_data is None or self.v_data is None:
            return None
        
        try:
            # Use hour directly as time index
            time_idx = hour % 24
            
            # Key modification: Use LEVEL_INDEX=15 (850hPa)
            u_field = self.u_data[time_idx, self.LEVEL_INDEX, :, :]
            v_field = self.v_data[time_idx, self.LEVEL_INDEX, :, :]
            
            # Handle longitude
            if lon < 0:
                lon = lon + 360
            
            # Calculate actual grid resolution
            if len(self.uv_lon) > 1 and not np.isnan(np.diff(self.uv_lon)).any():
                grid_res = np.mean(np.diff(self.uv_lon))
            else:
                grid_res = self.GRID_RES_DEFAULT
            
            # Call wind speed calculation function (200km search radius + 9-point smoothing)
            wind_speed = self.calc_cyclone_central_wind(
                u_field, v_field, lon, lat, self.uv_lat, self.uv_lon, grid_res
            )
            
            if wind_speed is not None:
                # Consistent with vorticity_wind_temp2.py: multiply by 10 and round to integer
                wind_scaled = int(round(wind_speed * 10))
                print(f"  Wind speed: time={time_idx}, pressure_level={self.LEVEL_INDEX}(850hPa), max={wind_speed:.2f} m/s -> {wind_scaled}")
                return wind_scaled
            else:
                return None
            
        except Exception as e:
            print(f"  Error getting maximum wind speed: {e}")
            return None
    
    def determine_stop_reason(self, row):
        """Determine the 6th digit in header record"""
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
    
    def parse_input_csv(self, input_file):
        """Parse input CSV file"""
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
    
    def convert_to_trv_format(self, output_file, vor_file=None, uv_file=None):
        """Convert data to TRV format"""
        if not self.data:
            print("No data to convert, please parse input file first")
            return False
        
        # Load ERA5 data
        if vor_file and uv_file:
            self.load_era5_data(vor_file, uv_file)
        
        output_lines = []
        first_row = self.data[0]
        last_row = self.data[-1]
        
        try:
            # Parse header information
            intl_id = int(float(first_row.get('International ID', '16')))
            china_id = first_row.get('China ID', '8012')
            tc_name = first_row.get('Typhoon Name', 'Norris')
            
            first_time = first_row.get('Tracking Time', '1980/8/29 6:00')
            dt = self.parse_datetime(first_time)
            start_date = dt.strftime('%Y%m%d')
            
            # Determine whether to keep the last row
            record_count = len(self.data)
            last_status = last_row.get('Tracking Status', '').strip()
            
            if last_status == 'Terminated':
                record_count -= 1
                use_data = self.data[:-1]
                stop_reason = self.determine_stop_reason(last_row)
            else:
                use_data = self.data
                stop_reason = 0
            
            # New: Iterate through data, check vorticity values, truncate if vorticity < 0 detected
            valid_data = []
            for i, row in enumerate(use_data):
                time_str = row.get('Tracking Time', '')
                dt_temp = self.parse_datetime(time_str)
                hour = dt_temp.hour
                lat = float(row.get('Center Latitude', 0))
                lon = float(row.get('Center Longitude', 0))
                
                # Get vorticity value for checking
                if self.vor_data is not None:
                    vorticity = self.get_vorticity(lat, lon, hour)
                    # If vorticity < 0, truncate, keep only rows before current row
                    if vorticity is not None and vorticity < 0:
                        print(f"\n⚠️ Detected vorticity < 0 at row {i+1} (value: {vorticity}), truncating subsequent records")
                        valid_data = use_data[:i]  # Keep only rows before current row
                        break
                valid_data = use_data  # If no negative vorticity detected, keep all data
            
            # Use valid_data instead of use_data
            if valid_data != use_data:
                use_data = valid_data
                record_count = len(use_data)
                print(f"Truncated, remaining {record_count} records")
            
            # Construct header record
            sequence_num = first_row.get('International ID', '0016')
            header = f"66666,{intl_id:04d},{record_count:03d},{int(sequence_num):04d},{china_id},{stop_reason},{tc_name},{start_date}"
            output_lines.append(header)
            
            print(f"\nHeader record: {header}")
            print(f"Record count: {record_count}, Stop reason: {stop_reason}")
            print("-" * 70)
            
            # Process each track record
            for i, row in enumerate(use_data):
                time_str = row.get('Tracking Time', '')
                dt = self.parse_datetime(time_str)
                time_formatted = dt.strftime('%Y%m%d%H')
                hour = dt.hour
                
                lat = float(row.get('Center Latitude', 0))
                lon = float(row.get('Center Longitude', 0))
                lat_int = int(round(lat * 10))
                lon_int = int(round(lon * 10))
                
                # Stream function processing (consistent with vorticity_wind_temp2.py)
                stream_func_raw = float(row.get('Center Streamfunction Value', 0))
                stream_func = int(round(abs(stream_func_raw) / 10000))
                
                print(f"\nRecord {i+1}: time={time_str}, hour={hour}, position=({lat:.2f}, {lon:.2f})")
                
                # Use new method to get vorticity value (850hPa, preserve sign)
                if self.vor_data is not None:
                    vorticity = self.get_vorticity(lat, lon, hour)
                    if vorticity is None:
                        print(f"  ⚠️ Error in vorticity")
                        vorticity = 0  # If error, set to default value
                
                # Use new method to get wind speed value (850hPa, 200km, 9-point smoothing)
                if self.u_data is not None:
                    velocity = self.get_wind_speed(lat, lon, hour)
                    if velocity is None:
                        velocity = 0  # If error, set to default value
                

                
                # Construct track record (added mutation_flag as last column)
                track_line = f"{time_formatted},{lat_int:03d},{lon_int:04d},{stream_func},{vorticity},{velocity}"
                output_lines.append(track_line)
                print(f"  Generated: {track_line}")
            
            print("-" * 70)
            
            # Write output file
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
        """Convert data to TRV format"""
        if not self.data:
            print("No data to convert, please parse input file first")
            return False
        
        # Load ERA5 data
        if vor_file and uv_file:
            self.load_era5_data(vor_file, uv_file)
        
        output_lines = []
        first_row = self.data[0]
        last_row = self.data[-1]
        
        try:
            # Parse header information
            intl_id = int(float(first_row.get('International ID', '16')))
            china_id = first_row.get('China ID', '8012')
            tc_name = first_row.get('Typhoon Name', 'Norris')
            
            first_time = first_row.get('Tracking Time', '1980/8/29 6:00')
            dt = self.parse_datetime(first_time)
            start_date = dt.strftime('%Y%m%d')
            
            # Determine whether to keep the last row
            record_count = len(self.data)
            last_status = last_row.get('Tracking Status', '').strip()
            
            if last_status == 'Terminated':
                record_count -= 1
                use_data = self.data[:-1]
                stop_reason = self.determine_stop_reason(last_row)
            else:
                use_data = self.data
                stop_reason = 0
            
            # Construct header record
            sequence_num = first_row.get('International ID', '0016')
            header = f"66666,{intl_id:04d},{record_count:03d},{int(sequence_num):04d},{china_id},{stop_reason},{tc_name},{start_date}"
            output_lines.append(header)
            
            print(f"\nHeader record: {header}")
            print(f"Record count: {record_count}, Stop reason: {stop_reason}")
            print("-" * 70)
            
            # Process each track record
            for i, row in enumerate(use_data):
                time_str = row.get('Tracking Time', '')
                dt = self.parse_datetime(time_str)
                time_formatted = dt.strftime('%Y%m%d%H')
                hour = dt.hour
                
                lat = float(row.get('Center Latitude', 0))
                lon = float(row.get('Center Longitude', 0))
                lat_int = int(round(lat * 10))
                lon_int = int(round(lon * 10))
                
                # Stream function processing (consistent with vorticity_wind_temp2.py)
                stream_func_raw = float(row.get('Center Streamfunction Value', 0))
                stream_func = int(round(abs(stream_func_raw) / 10000))
                
                print(f"\nRecord {i+1}: time={time_str}, hour={hour}, position=({lat:.2f}, {lon:.2f})")
                
                # Use new method to get vorticity value (850hPa, preserve sign)
                if self.vor_data is not None:
                    vorticity = self.get_vorticity(lat, lon, hour)
                    if vorticity is None:
                        # If retrieval fails, use CSV value as fallback
                        vort_raw = float(row.get('Center Vorticity Value', 0))
                        vorticity = int(round(vort_raw * 1e5))
                        print(f"  ⚠️ Using CSV vorticity value (fallback): {vorticity}")
                else:
                    vort_raw = float(row.get('Center Vorticity Value', 0))
                    vorticity = int(round(vort_raw * 1e5))
                    print(f"  Using CSV vorticity value: {vorticity}")
                
                # Use new method to get wind speed value (850hPa, 200km, 9-point smoothing)
                if self.u_data is not None:
                    velocity = self.get_wind_speed(lat, lon, hour)
                    if velocity is None:
                        # If retrieval fails, use estimated value
                        velocity = int(stream_func * 0.3)
                        if velocity < 10:
                            velocity = 150
                        print(f"  ⚠️ Using estimated wind speed (fallback): {velocity}")
                else:
                    velocity = int(stream_func * 0.3)
                    if velocity < 10:
                        velocity = 150
                    print(f"  Using estimated wind speed: {velocity}")
                
                # Construct track record
                track_line = f"{time_formatted},{lat_int:03d},{lon_int:04d},{stream_func},{vorticity},{velocity}"
                output_lines.append(track_line)
                print(f"  Generated: {track_line}")
            
            print("-" * 70)
            
            # Write output file
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


def main():
    """Main function"""
    input_file = "all_typhoons_tracking_test.csv"
    output_file = "TRV_test.csv"
    
    vor_file = "vor_19800829.nc"
    uv_file = "uv_19800829.nc"
    
    converter = CSVToTRVConverter()
    
    if not converter.parse_input_csv(input_file):
        print("Failed to parse input file")
        return
    
    if os.path.exists(vor_file) and os.path.exists(uv_file):
        success = converter.convert_to_trv_format(output_file, vor_file, uv_file)
    else:
        print("Warning: ERA5 files do not exist, using CSV values for conversion")
        success = converter.convert_to_trv_format(output_file)
    
    if success:
        print(f"\nConversion complete! Output file: {output_file}")
        print("\nFinal output content:")
        with open(output_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                print(f"{i+1}: {line.strip()}")
    else:
        print("Conversion failed")


if __name__ == "__main__":
    main()