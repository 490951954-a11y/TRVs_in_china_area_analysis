# -*- coding: utf-8 -*-
"""
Created on Tue Jan 20 21:21:19 2026

@author: Lenovo
"""

"""
TRV (Tropical Residual V) CSV File Parser
Description: Parse CSV files containing tropical cyclone residual low tracking data
Data format: Head data (8 fields) + track data (7 fields, including transition flag)
Time range: 1980-2024
Time resolution: 1 hour
"""

import csv
from collections import namedtuple
import os
import json
from typing import List, Dict, Any

# Header: 8 fields (unchanged)
HeaderData = namedtuple('HeaderData', [
    'flag',           # '66666'
    'intl_code',      # International ID
    'record_count',   # Number of track records
    'sequence_num',   # Original TC sequence number
    'china_code',     # Chinese TC number
    'stop_reason',    # Reason for stopping tracking (0-3)
    'name',           # English name of the TC
    'start_date'      # Start tracking date (YYYYMMDD)
])

# Track: 7 fields (added transition_flag at the end)
TrackData = namedtuple('TrackData', [
    'time',           # Timestamp (YYYYMMDDHH)
    'lat',            # Latitude (0.1°N)
    'lon',            # Longitude (0.1°E)
    'stream_func',    # 850hPa stream function (10⁴m²/s)
    'vorticity',      # 850hPa vorticity (10⁻⁵s⁻¹)
    'velocity',       # 850hPa velocity (0.1m/s)
    'transition_flag' # 1 = extratropical transition, 0 = not transitioned
])

class TCRLParser:
    """Parser for TCRL CSV files with transition flag in track data"""
    
    def __init__(self):
        self.data_blocks = []
        
    def parse_file(self, file_path: str) -> List[Dict[str, Any]]:
        """Parse TCRL format CSV file"""
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
            
            # Check header line (starts with '66666')
            if line.startswith('66666'):
                # Parse header (8 fields)
                header_fields = line.split(',')
                
                if len(header_fields) != 8:
                    print(f"Warning: Line {i+1} has {len(header_fields)} fields, expected 8")
                    i += 1
                    continue
                
                try:
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
                
                i += 1
                
                # Parse track records (7 fields each)
                track_data = []
                expected_records = header.record_count
                actual_records = 0
                
                while actual_records < expected_records and i < line_count:
                    # Stop if we hit a new header
                    if lines[i].strip().startswith('66666'):
                        break
                    
                    track_fields = lines[i].strip().split(',')
                    
                    # Track should have 7 fields
                    if len(track_fields) != 7:
                        print(f"Warning: Line {i+1} has {len(track_fields)} fields, expected 7")
                        i += 1
                        continue
                    
                    try:
                        track = TrackData(
                            time=track_fields[0],
                            lat=int(track_fields[1]),
                            lon=int(track_fields[2]),
                            stream_func=int(track_fields[3]),
                            vorticity=int(track_fields[4]),
                            velocity=int(track_fields[5]),
                            transition_flag=int(track_fields[6])  # 变性标志
                        )
                        track_data.append(track)
                        actual_records += 1
                    except ValueError as e:
                        print(f"Warning: Error parsing track at line {i+1}: {e}")
                    
                    i += 1
                
                if actual_records != expected_records:
                    print(f"Warning: {header.name} expected {expected_records} records, got {actual_records}")
                
                self.data_blocks.append({
                    'header': header,
                    'track': track_data
                })
            else:
                i += 1
        
        print(f"Parsed {len(self.data_blocks)} TC residual low blocks")
        return self.data_blocks
    
    def get_tc_by_name(self, name: str) -> List[Dict[str, Any]]:
        return [block for block in self.data_blocks if block['header'].name.lower() == name.lower()]
    
    def get_tc_by_year(self, year: str) -> List[Dict[str, Any]]:
        return [block for block in self.data_blocks if block['header'].start_date.startswith(year)]
    
    def get_transition_stats(self) -> Dict[str, Any]:
        """Get statistics about transition flags"""
        if not self.data_blocks:
            return {}
        
        total_tracks = 0
        trans_count = 0
        
        for block in self.data_blocks:
            for track in block['track']:
                total_tracks += 1
                if track.transition_flag == 1:
                    trans_count += 1
        
        return {
            'total_tracks': total_tracks,
            'transitioned': trans_count,
            'non_transitioned': total_tracks - trans_count,
            'rate': trans_count / total_tracks if total_tracks > 0 else 0
        }
    
    def export_to_json(self, output_file: str) -> bool:
        if not self.data_blocks:
            print("No data to export")
            return False
        
        try:
            serializable_data = []
            for block in self.data_blocks:
                block_dict = {
                    'header': block['header']._asdict(),
                    'track': [t._asdict() for t in block['track']]
                }
                serializable_data.append(block_dict)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(serializable_data, f, ensure_ascii=False, indent=2)
            
            print(f"Exported to {output_file}")
            return True
        except Exception as e:
            print(f"Error exporting: {e}")
            return False
    
    def export_to_csv(self, output_file: str) -> bool:
        if not self.data_blocks:
            print("No data to export")
            return False
        
        try:
            with open(output_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'Name', 'Start_Date', 'China_Code', 'Stop_Reason',
                    'Time', 'Lat', 'Lon', 'Stream_Func', 'Vorticity', 
                    'Velocity', 'Transition_Flag'
                ])
                
                for block in self.data_blocks:
                    header = block['header']
                    for track in block['track']:
                        writer.writerow([
                            header.name,
                            header.start_date,
                            header.china_code,
                            header.stop_reason,
                            track.time,
                            track.lat / 10.0,
                            track.lon / 10.0,
                            track.stream_func,
                            track.vorticity,
                            track.velocity / 10.0,
                            track.transition_flag
                        ])
            
            print(f"Exported to {output_file}")
            return True
        except Exception as e:
            print(f"Error exporting: {e}")
            return False


def main():
    file_path = "TRV_1980_2024_r1.csv"
    
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return
    
    parser = TCRLParser()
    data_blocks = parser.parse_file(file_path)
    
    if not data_blocks:
        print("No data parsed")
        return
    
    # Show first TC
    first = data_blocks[0]
    print(f"\nFirst TC: {first['header'].name} ({first['header'].start_date})")
    print(f"Records: {len(first['track'])}")
    print(f"First track: {first['track'][0]}")
    print(f"Last track: {first['track'][-1]}")
    
    # Statistics
    stats = parser.get_transition_stats()
    print(f"\n=== Transition Statistics ===")
    print(f"Total track records: {stats['total_tracks']}")
    print(f"Transitioned (flag=1): {stats['transitioned']}")
    print(f"Non-transitioned (flag=0): {stats['non_transitioned']}")
    print(f"Transition rate: {stats['rate']*100:.2f}%")

if __name__ == "__main__":
    main()