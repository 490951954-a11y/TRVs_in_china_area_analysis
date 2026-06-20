# -*- coding: utf-8 -*-
"""
Find extratropical transition TC (Tropical Cyclone Remnant) data corresponding to TRV
Add extratropical transition flag column at the end of original TRV track data
"""

import os
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
import re

# Import existing parsers
from BST_dis1 import CMABSTDataParserFixed
from TRVread import TCRLParser

class ExtratropicalTRVMatcher:
    """Extratropical typhoon and TRV data matcher"""
    
    def __init__(self):
        self.bst_parser = CMABSTDataParserFixed()
        self.trv_parser = TCRLParser()
        self.trv_data = None
        self.original_trv_lines = []
    
    def load_trv_data(self, trv_file_path: str) -> bool:
        """Load TRV data"""
        if not os.path.exists(trv_file_path):
            print(f"TRV file does not exist: {trv_file_path}")
            return False
        
        print(f"Loading TRV data: {trv_file_path}")
        self.trv_data = self.trv_parser.parse_file(trv_file_path)
        print(f"Load complete, total {len(self.trv_data)} TRV records")
        
        try:
            with open(trv_file_path, 'r', encoding='utf-8') as f:
                self.original_trv_lines = f.readlines()
            print(f"Original TRV file loaded successfully, total {len(self.original_trv_lines)} lines")
            return True
        except Exception as e:
            print(f"Failed to load original TRV file: {e}")
            return True
    
    def load_bst_data(self, bst_directory: str) -> Dict[str, List[Dict[str, Any]]]:
        """Load BST data"""
        print(f"Loading BST data: {bst_directory}")
        all_typhoons = self.bst_parser.read_all_files(bst_directory, pattern="CH*BST.txt")
        return all_typhoons
    
    def find_extratropical_typhoons(self, all_typhoons: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """Find all extratropical transition typhoons"""
        extratropical_typhoons = []
        
        for filename, typhoons in all_typhoons.items():
            for typhoon in typhoons:
                header = typhoon['header']
                track_data = typhoon['track_data']
                
                has_intensity_9 = any(point.get('intensity') == 9 for point in track_data)
                
                if has_intensity_9:
                    first_extratropical_idx = None
                    first_extratropical_point = None
                    for i, point in enumerate(track_data):
                        if point.get('intensity') == 9:
                            first_extratropical_idx = i
                            first_extratropical_point = point
                            break
                    
                    pre_extratropical_point = None
                    if first_extratropical_idx and first_extratropical_idx > 0:
                        pre_extratropical_point = track_data[first_extratropical_idx - 1]
                    
                    extratropical_typhoons.append({
                        'typhoon_data': typhoon,
                        'first_extratropical_point': first_extratropical_point,
                        'first_extratropical_index': first_extratropical_idx,
                        'pre_extratropical_point': pre_extratropical_point
                    })
        
        return extratropical_typhoons
    
    def find_matching_trv(self, extratropical_typhoon: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Find matching TRV data for extratropical typhoon"""
        typhoon_data = extratropical_typhoon['typhoon_data']
        header = typhoon_data['header']
        first_point = extratropical_typhoon['first_extratropical_point']
        
        if not first_point or not self.trv_data:
            return []
        
        extratropical_time_str = first_point['timestamp']
        typhoon_name = header.get('name', '').strip()
        international_id = header.get('international_id', '')
        china_id = header.get('china_id', '')
        
        perfect_matches = []
        
        for trv_block in self.trv_data:
            trv_header = trv_block['header']
            trv_track = trv_block['track']
            
            trv_name = trv_header.name.strip()
            
            # Check name match
            name_match = False
            if typhoon_name and trv_name:
                if typhoon_name.lower() == trv_name.lower():
                    name_match = True
                elif typhoon_name.lower() in trv_name.lower() or trv_name.lower() in typhoon_name.lower():
                    name_match = True
            
            # Check ID match
            id_match = False
            if china_id and trv_header.china_code:
                if china_id == trv_header.china_code:
                    id_match = True
            if international_id and trv_header.intl_code:
                if international_id == trv_header.intl_code:
                    id_match = True
            
            is_candidate = name_match or id_match
            
            if is_candidate:
                extratropical_index_in_trv = None
                
                if trv_track and extratropical_time_str:
                    for idx, track_point in enumerate(trv_track):
                        if track_point.time == extratropical_time_str:
                            extratropical_index_in_trv = idx
                            break
                
                if extratropical_index_in_trv is not None:
                    perfect_matches.append({
                        'trv_block': trv_block,
                        'extratropical_index_in_trv': extratropical_index_in_trv,
                        'extratropical_time_str': extratropical_time_str,
                    })
        
        return perfect_matches
    
    def add_extratropical_flag_to_trv(self, match_results: pd.DataFrame, output_file: str):
        """Add extratropical transition flag column at the end of each line in original TRV file"""
        if not self.original_trv_lines:
            print("Original TRV data not loaded")
            return
        
        print("\nAdding extratropical transition flag to TRV file...")
        
        # Collect all perfect match TRV information
        trv_extratropical_info = {}
        for _, row in match_results.iterrows():
            if row['has_perfect_match'] and row['perfect_matches']:
                for match in row['perfect_matches']:
                    trv_block = match['trv_block']
                    trv_header = trv_block['header']
                    extratropical_index = match.get('extratropical_index_in_trv')
                    
                    key = (trv_header.name, trv_header.start_date[:8])
                    if key not in trv_extratropical_info:
                        trv_extratropical_info[key] = extratropical_index
                        print(f"  Found matching TRV: {trv_header.name} (start: {trv_header.start_date[:8]}), extratropical point index: {extratropical_index}")
        
        # Process each line
        output_lines = []
        current_trv_key = None
        current_trv_path_index = -1
        extratropical_index = None
        
        for line in self.original_trv_lines:
            line = line.rstrip('\n\r')
            
            if not line.strip():
                output_lines.append(line)
                continue
            
            parts = line.split(',')
            is_header = False
            
            # Determine if it's a header line
            if len(parts) >= 7 and re.search(r'[A-Za-z]', parts[6]):
                is_header = True
                trv_name = parts[6].strip()
                trv_start = parts[7].strip()[:8] if len(parts) >= 8 else None
                
                current_trv_key = (trv_name, trv_start)
                current_trv_path_index = -1
                extratropical_index = trv_extratropical_info.get(current_trv_key)
                
                # Header lines do not get the flag
                output_lines.append(line)
                continue
            
            # Track data lines
            if not is_header:
                current_trv_path_index += 1
                is_extratropical = 0
                
                if extratropical_index is not None and current_trv_path_index >= extratropical_index:
                    is_extratropical = 1
                
                output_lines.append(line + ',' + str(is_extratropical))
        
        # Write output file
        with open(output_file, 'w', encoding='utf-8') as f:
            for line in output_lines:
                f.write(line + '\n')
        
        print(f"\nEnhanced TRV data saved to: {output_file}")
    
    def match_and_add_flag(self, bst_directory: str, trv_file_path: str, output_dir: str):
        """Match extratropical typhoons and add flag"""
        
        # Load data
        if not self.load_trv_data(trv_file_path):
            return
        
        all_typhoons = self.load_bst_data(bst_directory)
        extratropical_typhoons = self.find_extratropical_typhoons(all_typhoons)
        
        print(f"\nFound {len(extratropical_typhoons)} extratropical transition typhoons")
        
        # Match results
        match_results = []
        
        for i, et in enumerate(extratropical_typhoons, 1):
            typhoon_data = et['typhoon_data']
            header = typhoon_data['header']
            first_point = et['first_extratropical_point']
            
            typhoon_name = header.get('name', 'Unnamed')
            print(f"\nMatching {i}/{len(extratropical_typhoons)}: {typhoon_name}")
            
            perfect_matches = self.find_matching_trv(et)
            perfect_count = len(perfect_matches)
            
            result = {
                'typhoon_name': typhoon_name,
                'international_id': header.get('international_id', ''),
                'china_id': header.get('china_id', ''),
                'extratropical_time': first_point['timestamp'] if first_point else None,
                'has_perfect_match': perfect_count > 0,
                'perfect_match_count': perfect_count,
                'perfect_matches': perfect_matches,
            }
            
            match_results.append(result)
            
            if perfect_matches:
                print(f"  ✓ Found {perfect_count} perfect matches")
                for m in perfect_matches:
                    trv = m['trv_block']['header']
                    idx = m.get('extratropical_index_in_trv')
                    print(f"    - {trv.name} ({trv.start_date}) - extratropical point index: {idx}")
            else:
                print(f"  ✗ No matching TRV record found")
        
        results_df = pd.DataFrame(match_results)
        
        # Add flag
        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(trv_file_path))[0]
        output_file = os.path.join(output_dir, f"{base_name}_r1.csv")
        
        self.add_extratropical_flag_to_trv(results_df, output_file)
        
        # Statistics
        total_marked = 0
        with open(output_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.rstrip('\n\r').endswith(',1'):
                    total_marked += 1
        
        print(f"\nTotal {total_marked} track records marked as extratropical transition status")
        print("\nProgram execution complete!")

def main():
    """Main function"""
    
    # Set file paths
    bst_directory = r"D:\CMABSTdata"
    trv_file_path = r"TRV_test.csv"
    output_directory = r"."
    
    if not os.path.exists(bst_directory):
        print(f"BST directory does not exist: {bst_directory}")
        return
    
    if not os.path.exists(trv_file_path):
        print(f"TRV file does not exist: {trv_file_path}")
        return
    
    matcher = ExtratropicalTRVMatcher()
    matcher.match_and_add_flag(bst_directory, trv_file_path, output_directory)

if __name__ == "__main__":
    main()