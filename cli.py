"""
Command Line Interface
======================
Main entry point for typhoon tracking workflows.
"""

import os
import argparse
from datetime import datetime, timedelta

from .tracker import StreamfunctionTyphoonTracker
from .converter import CSVToTRVConverter
from .extratropical import ExtratropicalTRVMatcher


def run_tracking(args):
    """Run the typhoon tracking workflow."""
    tracker = StreamfunctionTyphoonTracker(
        delta_psi=args.delta_psi,
        min_lifetime=args.min_lifetime,
        detection_threshold=args.detection_threshold,
        max_track_hours=args.max_track_hours,
        psi_min_threshold=args.psi_min_threshold,
        max_hourly_distance=args.max_hourly_distance,
        time_match_tolerance=args.time_match_tolerance,
        enable_vis=not args.no_vis,
        local_shp_path=args.shp_path
    )
    
    tracker.load_end_positions(args.csv_path)
    
    # Get files for each typhoon
    for idx, (_, row) in enumerate(tracker.end_positions_df.iterrows()):
        typhoon_name = row['Typhoon Name']
        try:
            end_time = row['End Time_dt']
            start_time = end_time
            start_lat = float(row['End Latitude'])
            start_lon = float(row['End Longitude'])
        except Exception as e:
            print(f"⚠️  Skipping invalid record {typhoon_name}: {str(e)}")
            continue
        
        print(f"\n{'#'*60}")
        print(f"Processing typhoon {idx+1}: {typhoon_name}")
        print(f"{'#'*60}")
        
        # Find NC files
        file_list = []
        for i in range(5):
            target_date = start_time + timedelta(days=i)
            year = target_date.strftime("%Y")
            month = target_date.strftime("%m")
            nc_filename = f"vor_{target_date.strftime('%Y%m%d')}.nc"
            nc_filepath = os.path.join(args.era5_dir, year, month, nc_filename)
            
            if not os.path.exists(nc_filepath):
                month_single = str(int(month))
                nc_filepath_single = os.path.join(args.era5_dir, year, month_single, nc_filename)
                if os.path.exists(nc_filepath_single):
                    nc_filepath = nc_filepath_single
                else:
                    continue
            
            file_list.append(nc_filepath)
        
        if not file_list:
            print(f"⚠️  No ERA5 data files found for {typhoon_name}, skipping")
            continue
        
        print(f"📂 Found {len(file_list)} related ERA5 files")
        
        current_lat = start_lat
        current_lon = start_lon
        current_time = start_time
        all_track_results = []
        is_first_file = True
        
        for file in file_list:
            print(f"\nProcessing file: {os.path.basename(file)}")
            try:
                track_results, cont_flag, last_t, boundary, failed, psi_exceed, dist_exceed, merged = tracker.track_single_file(
                    nc_file_path=file,
                    start_time=current_time,
                    ftime=start_time,
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
                print(f"🛑 Tracking terminated")
                break
            
            is_first_file = False
        
        if all_track_results:
            print(f"📝 Typhoon {typhoon_name} tracking completed, total {len(all_track_results)} time steps")
        else:
            print(f"⚠️  Typhoon {typhoon_name} has no valid tracking results")
    
    tracker.save_all_track_summary(args.output_dir)
    print("\nAll typhoons processing completed")


def run_conversion(args):
    """Run the CSV to TRV conversion workflow."""
    converter = CSVToTRVConverter(
        target_level=args.target_level,
        level_index=args.level_index,
        search_radius_km=args.search_radius
    )
    
    if not converter.parse_input_csv(args.input_csv):
        print("Failed to parse input file")
        return
    
    if args.vor_file and args.uv_file:
        success = converter.convert_to_trv_format(args.output_csv, args.vor_file, args.uv_file)
    else:
        print("Warning: ERA5 files not provided, using CSV values for conversion")
        success = converter.convert_to_trv_format(args.output_csv)
    
    if success:
        print(f"\nConversion complete! Output file: {args.output_csv}")
    else:
        print("Conversion failed")


def run_extratropical(args):
    """Run the extratropical transition detection workflow."""
    matcher = ExtratropicalTRVMatcher()
    matcher.match_and_add_flag(
        bst_directory=args.bst_dir,
        trv_file_path=args.trv_file,
        output_dir=args.output_dir
    )


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Typhoon Tracking Library - Track tropical cyclones from ERA5 data"
    )
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Tracking command
    track_parser = subparsers.add_parser('track', help='Run typhoon tracking')
    track_parser.add_argument('csv_path', help='Path to end positions CSV')
    track_parser.add_argument('era5_dir', help='Root directory of ERA5 data')
    track_parser.add_argument('--output-dir', '-o', default='track_results', 
                             help='Output directory for results')
    track_parser.add_argument('--shp-path', help='Path to SHP file for boundary checking')
    track_parser.add_argument('--no-vis', action='store_true', 
                             help='Disable visualization')
    track_parser.add_argument('--delta-psi', type=float, default=2.0e6,
                             help='Streamfunction delta')
    track_parser.add_argument('--min-lifetime', type=int, default=1,
                             help='Minimum lifetime in hours')
    track_parser.add_argument('--detection-threshold', type=float, default=1.0e6,
                             help='Detection threshold')
    track_parser.add_argument('--max-track-hours', type=int, default=120,
                             help='Maximum tracking hours')
    track_parser.add_argument('--psi-min-threshold', type=float, default=-1e5,
                             help='Minimum streamfunction threshold')
    track_parser.add_argument('--max-hourly-distance', type=float, default=2.0,
                             help='Maximum hourly movement in degrees')
    track_parser.add_argument('--time-match-tolerance', type=int, default=7200,
                             help='Time match tolerance in seconds')
    
    # Conversion command
    convert_parser = subparsers.add_parser('convert', help='Convert tracking results to TRV format')
    convert_parser.add_argument('input_csv', help='Input tracking CSV file')
    convert_parser.add_argument('output_csv', help='Output TRV CSV file')
    convert_parser.add_argument('--vor-file', help='Vorticity NC file')
    convert_parser.add_argument('--uv-file', help='Wind field NC file')
    convert_parser.add_argument('--target-level', type=int, default=850,
                               help='Target pressure level in hPa')
    convert_parser.add_argument('--level-index', type=int, default=0,
                               help='Level index in NC file')
    convert_parser.add_argument('--search-radius', type=float, default=200,
                               help='Search radius in km')
    
    # Extratropical command
    extra_parser = subparsers.add_parser('extratropical', 
                                         help='Detect extratropical transitions')
    extra_parser.add_argument('bst_dir', help='BST data directory')
    extra_parser.add_argument('trv_file', help='TRV CSV file')
    extra_parser.add_argument('--output-dir', '-o', default='.',
                              help='Output directory')
    
    args = parser.parse_args()
    
    if args.command == 'track':
        run_tracking(args)
    elif args.command == 'convert':
        run_conversion(args)
    elif args.command == 'extratropical':
        run_extratropical(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()