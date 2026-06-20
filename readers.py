"""
Data Readers
============
Parsers for BST and TRV data formats.
"""

import csv
import os
import json
import glob
from collections import namedtuple
from typing import List, Dict, Any, Optional
from datetime import datetime

# TRV data structures (without transition flag)
HeaderData = namedtuple('HeaderData', [
    'flag', 'intl_code', 'record_count', 'sequence_num',
    'china_code', 'stop_reason', 'name', 'start_date'
])

TrackData = namedtuple('TrackData', [
    'time', 'lat', 'lon', 'stream_func', 'vorticity', 'velocity'
])

# TRV data structures (with transition flag)
HeaderDataR1 = namedtuple('HeaderDataR1', [
    'flag', 'intl_code', 'record_count', 'sequence_num',
    'china_code', 'stop_reason', 'name', 'start_date'
])

TrackDataR1 = namedtuple('TrackDataR1', [
    'time', 'lat', 'lon', 'stream_func', 'vorticity', 'velocity', 'transition_flag'
])


class TCRLParser:
    """Parser for TCRL CSV files (without transition flag)."""
    
    def __init__(self):
        self.data_blocks = []
    
    def parse_file(self, file_path: str) -> List[Dict[str, Any]]:
        """Parse TCRL format CSV file."""
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
            
            if line.startswith('66666'):
                header_fields = line.split(',')
                
                if len(header_fields) != 8:
                    print(f"Warning: Line {i+1} has incorrect number of header fields: {len(header_fields)}")
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
                
                track_data = []
                expected_records = header.record_count
                actual_records = 0
                
                while actual_records < expected_records and i < line_count:
                    if lines[i].strip().startswith('66666'):
                        break
                    
                    track_fields = lines[i].strip().split(',')
                    
                    if len(track_fields) != 6:
                        print(f"Warning: Line {i+1} has incorrect number of track fields: {len(track_fields)}")
                        i += 1
                        continue
                    
                    try:
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
                
                if actual_records != expected_records:
                    print(f"Warning: {header.name} ({header.start_date}) "
                          f"expected {expected_records} records, got {actual_records}")
                
                self.data_blocks.append({
                    'header': header,
                    'track': track_data
                })
            else:
                i += 1
        
        print(f"Successfully parsed {len(self.data_blocks)} TC residual low blocks")
        return self.data_blocks
    
    def get_tc_by_name(self, name: str) -> List[Dict[str, Any]]:
        return [block for block in self.data_blocks if block['header'].name.lower() == name.lower()]
    
    def get_tc_by_year(self, year: str) -> List[Dict[str, Any]]:
        return [block for block in self.data_blocks if block['header'].start_date.startswith(year)]
    
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


class TCRLParserR1(TCRLParser):
    """Parser for TCRL CSV files with transition flag."""
    
    def parse_file(self, file_path: str) -> List[Dict[str, Any]]:
        """Parse TCRL format CSV file with transition flag."""
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
            
            if line.startswith('66666'):
                header_fields = line.split(',')
                
                if len(header_fields) != 8:
                    print(f"Warning: Line {i+1} has incorrect number of header fields: {len(header_fields)}")
                    i += 1
                    continue
                
                try:
                    header = HeaderDataR1(
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
                
                track_data = []
                expected_records = header.record_count
                actual_records = 0
                
                while actual_records < expected_records and i < line_count:
                    if lines[i].strip().startswith('66666'):
                        break
                    
                    track_fields = lines[i].strip().split(',')
                    
                    if len(track_fields) != 7:
                        print(f"Warning: Line {i+1} has incorrect number of track fields: {len(track_fields)}")
                        i += 1
                        continue
                    
                    try:
                        track = TrackDataR1(
                            time=track_fields[0],
                            lat=int(track_fields[1]),
                            lon=int(track_fields[2]),
                            stream_func=int(track_fields[3]),
                            vorticity=int(track_fields[4]),
                            velocity=int(track_fields[5]),
                            transition_flag=int(track_fields[6])
                        )
                        track_data.append(track)
                        actual_records += 1
                    except ValueError as e:
                        print(f"Warning: Error parsing track data at line {i+1}: {e}")
                    
                    i += 1
                
                if actual_records != expected_records:
                    print(f"Warning: {header.name} ({header.start_date}) "
                          f"expected {expected_records} records, got {actual_records}")
                
                self.data_blocks.append({
                    'header': header,
                    'track': track_data
                })
            else:
                i += 1
        
        print(f"Successfully parsed {len(self.data_blocks)} TC residual low blocks")
        return self.data_blocks
    
    def get_transition_stats(self) -> Dict[str, Any]:
        """Get statistics about transition flags."""
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


class CMABSTDataParserFixed:
    """Parser for CMA BST data files."""
    
    def __init__(self):
        self.header_columns = [
            'classification_flag', 'international_id', 'record_count',
            'cyclone_sequence', 'china_id', 'end_status', 'time_interval',
            'name', 'data_generation_date'
        ]
        self.data_columns = [
            'timestamp', 'intensity', 'latitude', 'longitude',
            'pressure', 'max_wind_speed'
        ]
    
    def parse_header_line(self, line: str) -> Optional[Dict[str, Any]]:
        """Parse header record line."""
        try:
            parts = line.split()
            
            if len(parts) < 9:
                print(f"字段不足: {len(parts)}")
                return None
            
            header = {
                'classification_flag': parts[0],
                'international_id': parts[1],
                'record_count': int(parts[2]),
                'cyclone_sequence': parts[3],
                'china_id': parts[4],
                'end_status': parts[5],
                'time_interval': int(parts[6]),
                'name': parts[7] if parts[7] != '(nameless)' else '',
                'data_generation_date': parts[8] if len(parts) > 8 else ''
            }
            
            if len(parts) > 9:
                header['extra_info'] = ' '.join(parts[9:])
            
            return header
        except Exception as e:
            print(f"解析头记录错误: {e}")
            print(f"行内容: {line}")
            return None
    
    def parse_data_line(self, line: str) -> Optional[Dict[str, Any]]:
        """Parse data record line."""
        try:
            data_record = {
                'timestamp': line[0:10].strip(),
                'intensity': int(line[10:12].strip()) if line[10:12].strip() else -1,
                'latitude': float(line[12:17].strip()) / 10.0 if line[12:17].strip() else 0,
                'longitude': float(line[17:22].strip()) / 10.0 if line[17:22].strip() else 0,
                'pressure': int(line[22:26].strip()) if line[22:26].strip() else 0,
                'max_wind_speed': int(line[26:30].strip()) if line[26:30].strip() else 0
            }
            
            if len(line) > 30:
                additional_str = line[30:35].strip() if len(line) > 35 else line[30:].strip()
                if additional_str:
                    try:
                        parts = additional_str.split()
                        if len(parts) >= 1:
                            data_record['additional_wind'] = int(parts[0]) if parts[0].isdigit() else 0
                        if len(parts) >= 2:
                            data_record['additional_wind2'] = int(parts[1]) if parts[1].isdigit() else 0
                    except:
                        data_record['additional_info'] = additional_str
            
            return data_record
        except Exception as e:
            print(f"解析数据记录错误: {e}, 行内容: {line}")
            return None
    
    def parse_file(self, file_path: str) -> List[Dict[str, Any]]:
        """Parse a single file."""
        typhoons_data = []
        current_header = None
        data_lines = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                lines = file.readlines()
            
            for line in lines:
                line = line.rstrip('\n')
                if not line.strip():
                    continue
                
                if line.startswith('66666'):
                    if current_header and data_lines:
                        typhoon_data = self._process_typhoon(current_header, data_lines, file_path)
                        if typhoon_data:
                            typhoons_data.append(typhoon_data)
                    
                    current_header = self.parse_header_line(line)
                    data_lines = []
                else:
                    data_lines.append(line)
            
            if current_header and data_lines:
                typhoon_data = self._process_typhoon(current_header, data_lines, file_path)
                if typhoon_data:
                    typhoons_data.append(typhoon_data)
                    
        except Exception as e:
            print(f"解析文件 {file_path} 时出错: {e}")
        
        return typhoons_data
    
    def _process_typhoon(self, header: Dict[str, Any], data_lines: List[str], 
                         file_path: str) -> Optional[Dict[str, Any]]:
        """Process a single typhoon data."""
        if not header:
            return None
            
        typhoon_data = {
            'file_name': os.path.basename(file_path),
            'header': header,
            'track_data': []
        }
        
        for line in data_lines:
            data_record = self.parse_data_line(line)
            if data_record:
                typhoon_data['track_data'].append(data_record)
        
        return typhoon_data
    
    def read_all_files(self, directory_path: str, pattern: str = "CH*BST.txt") -> Dict[str, List[Dict[str, Any]]]:
        """Read all data files in directory."""
        all_typhoons = {}
        
        search_pattern = os.path.join(directory_path, pattern)
        files = glob.glob(search_pattern)
        
        print(f"找到 {len(files)} 个数据文件")
        
        for file_path in files:
            print(f"正在解析文件: {os.path.basename(file_path)}")
            typhoons = self.parse_file(file_path)
            all_typhoons[os.path.basename(file_path)] = typhoons
        
        return all_typhoons