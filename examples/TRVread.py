

# -*- coding: utf-8 -*-
"""
Created on Tue Jan 20 21:21:19 2026

@author: Lenovo
"""

"""
TRV (Tropical Residual V) CSV File Parser
Description: Parse CSV files containing tropical cyclone residual low tracking data
Data format: Head data + track data for each residual low
Time range: 1980-2024
Time resolution: 1 hour
Spatial range: 15°N-55°N, 95°E-140°E, within 200km of Chinese mainland
"""

import csv
from collections import namedtuple
import os
import json
from typing import List, Dict, Any

# Data structure for header information
HeaderData = namedtuple('HeaderData', [
    'flag',           # '66666' - indicates best track data
    'intl_code',      # International ID (last 2 digits of year + 2-digit number)
    'record_count',   # Number of track records
    'sequence_num',   # Original TC sequence number
    'china_code',     # Chinese TC number
    'stop_reason',    # Reason for stopping tracking (0-3)
    'name',           # English name of the TC
    'start_date'      # Start tracking date (YYYYMMDD)
])

# Data structure for track information
TrackData = namedtuple('TrackData', [
    'time',           # Timestamp (YYYYMMDDHH)
    'lat',            # Latitude (0.1°N)
    'lon',            # Longitude (0.1°E)
    'stream_func',    # 850hPa stream function (10⁴m²/s)
    'vorticity',      # 850hPa vorticity (10⁻⁵s⁻¹)
    'velocity'        # 850hPa velocity (0.1m/s)
])

class TCRLParser:
    """Parser for TCRL CSV files"""
    
    def __init__(self):
        """Initialize the parser"""
        self.data_blocks = []  # List of parsed data blocks
        
    def parse_file(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Parse TCRL format CSV file
        
        Args:
            file_path: Path to the CSV file
            
        Returns:
            List of data blocks, each containing header and track data
        """
        self.data_blocks = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        except FileNotFoundError:
            print(f"Error: File not found: {file_path}")
            return []
        except Exception as e:
            print(f"Error reading file: {e}")
            return []
        
        i = 0
        line_count = len(lines)
        
        while i < line_count:
            line = lines[i].strip()
            
            # Check if this is a header line (starts with '66666')
            if line.startswith('66666'):
                # Parse header data
                header_fields = line.split(',')
                
                if len(header_fields) != 8:
                    print(f"Warning: Line {i+1} has incorrect number of header fields: {len(header_fields)}")
                    i += 1
                    continue
                
                try:
                    # Create HeaderData object
                    header = HeaderData(
                        flag=header_fields[0],
                        intl_code=header_fields[1],
                        record_count=int(header_fields[2]),
                        sequence_num=header_fields[3],
                        china_code=header_fields[4],
                        stop_reason=int(header_fields[5]),
                        name=header_fields[6],
                        start_date=header_fields[7]
                    )
                except ValueError as e:
                    print(f"Warning: Error parsing header at line {i+1}: {e}")
                    i += 1
                    continue
                
                i += 1  # Move to next line
                
                # Parse track data for this TC
                track_data = []
                expected_records = header.record_count
                actual_records = 0
                
                # Read track records
                while actual_records < expected_records and i < line_count:
                    # Check if next line is a new header
                    if lines[i].strip().startswith('66666'):
                        break
                    
                    track_fields = lines[i].strip().split(',')
                    
                    if len(track_fields) != 6:
                        print(f"Warning: Line {i+1} has incorrect number of track fields: {len(track_fields)}")
                        i += 1
                        continue
                    
                    try:
                        # Create TrackData object
                        track = TrackData(
                            time=track_fields[0],
                            lat=int(track_fields[1]),
                            lon=int(track_fields[2]),
                            stream_func=int(track_fields[3]),
                            vorticity=int(track_fields[4]),
                            velocity=int(track_fields[5])
                        )
                        track_data.append(track)
                        actual_records += 1
                    except ValueError as e:
                        print(f"Warning: Error parsing track data at line {i+1}: {e}")
                    
                    i += 1
                
                # Check if we got all expected records
                if actual_records != expected_records:
                    print(f"Warning: {header.name} ({header.start_date}) "
                          f"expected {expected_records} records, got {actual_records}")
                
                # Store the complete data block
                self.data_blocks.append({
                    'header': header,
                    'track': track_data
                })
            else:
                # Skip lines that don't start with '66666' (shouldn't happen in valid files)
                i += 1
        
        print(f"Successfully parsed {len(self.data_blocks)} TC residual low blocks")
        return self.data_blocks
    
    
    def get_tc_by_name(self, name: str) -> List[Dict[str, Any]]:
        """
        Get all TC residual lows with a specific name
        
        Args:
            name: TC name to search for
            
        Returns:
            List of matching TC data blocks
        """
        return [block for block in self.data_blocks if block['header'].name.lower() == name.lower()]
    
    def get_tc_by_year(self, year: str) -> List[Dict[str, Any]]:
        """
        Get all TC residual lows from a specific year
        
        Args:
            year: Year to search for (YYYY format)
            
        Returns:
            List of matching TC data blocks
        """
        return [block for block in self.data_blocks if block['header'].start_date.startswith(year)]
    
    
    def export_to_json(self, output_file: str) -> bool:
        """
        Export parsed data to JSON format
        
        Args:
            output_file: Path to output JSON file
            
        Returns:
            True if successful, False otherwise
        """
        if not self.data_blocks:
            print("No data to export. Please parse a file first.")
            return False
        
        try:
            # Convert to serializable format
            serializable_data = []
            for block in self.data_blocks:
                block_dict = {
                    'header': block['header']._asdict(),
                    'track': [t._asdict() for t in block['track']]
                }
                serializable_data.append(block_dict)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(serializable_data, f, ensure_ascii=False, indent=2)
            
            print(f"Data exported to {output_file}")
            return True
            
        except Exception as e:
            print(f"Error exporting to JSON: {e}")
            return False
    
    def export_to_csv(self, output_file: str) -> bool:
        """
        Export parsed data to simplified CSV format
        
        Args:
            output_file: Path to output CSV file
            
        Returns:
            True if successful, False otherwise
        """
        if not self.data_blocks:
            print("No data to export. Please parse a file first.")
            return False
        
        try:
            with open(output_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                
                # Write header
                writer.writerow([
                    'TC_Name', 'Start_Date', 'China_Code', 'Stop_Reason',
                    'Time', 'Latitude', 'Longitude', 
                    'Stream_Function', 'Vorticity', 'Velocity'
                ])
                
                # Write data
                for block in self.data_blocks:
                    header = block['header']
                    for track in block['track']:
                        writer.writerow([
                            header.name,
                            header.start_date,
                            header.china_code,
                            header.stop_reason,
                            track.time,
                            track.lat / 10.0,  # Convert to degrees
                            track.lon / 10.0,  # Convert to degrees
                            track.stream_func,
                            track.vorticity,
                            track.velocity / 10.0  # Convert to m/s
                        ])
            
            print(f"Data exported to {output_file}")
            return True
            
        except Exception as e:
            print(f"Error exporting to CSV: {e}")
            return False
        
        
def main():
    """
    Main function to demonstrate the parser
    """
    # File path - update this to your actual file path
    file_path = "TRV_1980_2024.csv"
    
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        print("Please update the file_path variable to point to your TCRL CSV file")
        return
    
    # Create parser instance
    parser = TCRLParser()
    
    # Parse the file
    print("Parsing TCRL CSV file...")
    data_blocks = parser.parse_file(file_path)
    
    if not data_blocks:
        print("No data was parsed. Exiting.")
        return
    
if __name__ == "__main__":
    main()