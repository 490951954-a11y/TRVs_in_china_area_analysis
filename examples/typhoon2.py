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
from datetime import datetime, timedelta
import netCDF4 as nc
import numpy as np
import re
from scipy.ndimage import uniform_filter
from typing import List, Optional


class CSVToTRVConverter:
    """Converter from sample CSV to TRV format"""
    
    def __init__(self, vor_root_dir=None, uv_root_dir=None):
        """
        初始化转换器
        
        Parameters:
        -----------
        vor_root_dir : str, optional
            涡度数据根目录（E:/era5_vorticity_data）
        uv_root_dir : str, optional
            风速数据根目录（E:/era5_uv_data）
        """
        self.input_file = None
        self.output_file = None
        self.data = []
        
        # ERA5数据根目录
        self.VOR_ROOT_DIR = vor_root_dir or "./era5_vorticity_data"
        self.UV_ROOT_DIR = uv_root_dir or "./era5_uv_data"
        
        # 当前打开的文件句柄（避免重复打开）
        self.current_vor_file = None
        self.current_vor_ds = None
        self.current_uv_file = None
        self.current_uv_ds = None
        self.current_date = None
        
        # Vorticity data (当前文件)
        self.vor_data = None
        self.vor_lat = None
        self.vor_lon = None
        
        # Wind field data (当前文件)
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
    
    def get_nc_file_path(self, date: datetime, data_type: str = 'vor') -> Optional[str]:
        """
        根据日期获取NC文件路径
        
        Parameters:
        -----------
        date : datetime
            日期
        data_type : str
            数据类型：'vor' 或 'uv'
            
        Returns:
        --------
        Optional[str] : NC文件路径，如果不存在则返回None
        """
        # 选择对应的根目录
        root_dir = self.VOR_ROOT_DIR if data_type == 'vor' else self.UV_ROOT_DIR
        prefix = 'vor' if data_type == 'vor' else 'uv'
        
        year = date.strftime("%Y")
        month = date.strftime("%m")  # 确保月份为两位数（01~12）
        
        # NC文件命名
        nc_filename = f"{prefix}_{date.strftime('%Y%m%d')}.nc"
        # 完整路径：根目录/年/月/文件名
        nc_filepath = os.path.join(root_dir, year, month, nc_filename)
        
        # 兼容单数月份文件夹（如"05"和"5"都能识别）
        if not os.path.exists(nc_filepath):
            month_single = str(int(month))  # 去掉前导零（05→5）
            nc_filepath_single = os.path.join(root_dir, year, month_single, nc_filename)
            if os.path.exists(nc_filepath_single):
                return nc_filepath_single
            else:
                return None
        
        return nc_filepath
    
    def load_file_for_date(self, date: datetime):
        """
        加载指定日期的NC文件
        
        Parameters:
        -----------
        date : datetime
            要加载的日期
        """
        # 如果当前日期与要加载的日期相同，则不需要重新加载
        if self.current_date == date.date():
            return
        
        # 关闭之前打开的文件
        self.close_current_files()
        
        # 加载涡度文件
        vor_file = self.get_nc_file_path(date, 'vor')
        if vor_file and os.path.exists(vor_file):
            try:
                self.current_vor_ds = nc.Dataset(vor_file)
                self.current_vor_file = vor_file
                self.vor_data = self.current_vor_ds.variables['vo'][:]
                
                # 获取经纬度
                if 'latitude' in self.current_vor_ds.variables:
                    self.vor_lat = self.current_vor_ds.variables['latitude'][:]
                elif 'lat' in self.current_vor_ds.variables:
                    self.vor_lat = self.current_vor_ds.variables['lat'][:]
                
                if 'longitude' in self.current_vor_ds.variables:
                    self.vor_lon = self.current_vor_ds.variables['longitude'][:]
                elif 'lon' in self.current_vor_ds.variables:
                    self.vor_lon = self.current_vor_ds.variables['lon'][:]
                
                # 标准化经度到0-360
                if self.vor_lon is not None and self.vor_lon.min() < 0:
                    self.vor_lon = (self.vor_lon + 360) % 360
                
                print(f"  加载涡度文件: {os.path.basename(vor_file)}, 形状: {self.vor_data.shape}")
            except Exception as e:
                print(f"  ⚠️  加载涡度文件失败 {vor_file}: {e}")
                self.vor_data = None
        else:
            print(f"  ⚠️  涡度文件不存在: {vor_file}")
            self.vor_data = None
        
        # 加载风速文件
        uv_file = self.get_nc_file_path(date, 'uv')
        if uv_file and os.path.exists(uv_file):
            try:
                self.current_uv_ds = nc.Dataset(uv_file)
                self.current_uv_file = uv_file
                self.u_data = self.current_uv_ds.variables['u'][:]
                self.v_data = self.current_uv_ds.variables['v'][:]
                
                # 获取经纬度
                if 'latitude' in self.current_uv_ds.variables:
                    self.uv_lat = self.current_uv_ds.variables['latitude'][:]
                elif 'lat' in self.current_uv_ds.variables:
                    self.uv_lat = self.current_uv_ds.variables['lat'][:]
                
                if 'longitude' in self.current_uv_ds.variables:
                    self.uv_lon = self.current_uv_ds.variables['longitude'][:]
                elif 'lon' in self.current_uv_ds.variables:
                    self.uv_lon = self.current_uv_ds.variables['lon'][:]
                
                # 标准化经度到0-360
                if self.uv_lon is not None and self.uv_lon.min() < 0:
                    self.uv_lon = (self.uv_lon + 360) % 360
                
                print(f"  加载风速文件: {os.path.basename(uv_file)}, 形状: {self.u_data.shape}")
            except Exception as e:
                print(f"  ⚠️  加载风速文件失败 {uv_file}: {e}")
                self.u_data = None
                self.v_data = None
        else:
            print(f"  ⚠️  风速文件不存在: {uv_file}")
            self.u_data = None
            self.v_data = None
        
        # 更新当前日期
        self.current_date = date.date()
    
    def close_current_files(self):
        """关闭当前打开的文件"""
        if self.current_vor_ds is not None:
            self.current_vor_ds.close()
            self.current_vor_ds = None
            self.current_vor_file = None
        
        if self.current_uv_ds is not None:
            self.current_uv_ds.close()
            self.current_uv_ds = None
            self.current_uv_file = None
        
        self.vor_data = None
        self.u_data = None
        self.v_data = None
    
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
    
    def get_vorticity(self, lat, lon, hour):
        """
        获取指定位置/时间的涡度值（从当前加载的文件中读取）
        
        Parameters:
        -----------
        lat, lon : float
            经纬度
        hour : int
            小时（0-23）
        """
        if self.vor_data is None:
            return None
        
        try:
            # 直接使用小时作为时间索引（NC文件包含24小时数据）
            time_idx = hour
            
            if time_idx >= self.vor_data.shape[0]:
                print(f"  ⚠️  时间索引 {time_idx} 超出范围 ({self.vor_data.shape[0]})")
                return None
            
            # 使用 LEVEL_INDEX=15 (850hPa)
            vor_field = self.vor_data[time_idx, self.LEVEL_INDEX, :, :]
            
            # 处理经度
            if lon < 0:
                lon = lon + 360
            
            # 查找最近网格点
            lat_idx = np.argmin(np.abs(self.vor_lat - lat))
            lon_idx = np.argmin(np.abs(self.vor_lon - lon))
            
            # 获取涡度值
            vorticity = float(vor_field[lat_idx, lon_idx])
            
            # 乘以1e5并取整
            vorticity_scaled = int(round(vorticity * 1e5))
            
            print(f"  涡度：time_idx={time_idx}, 压力层={self.LEVEL_INDEX}(850hPa), 值={vorticity:.6f} s^-1 -> {vorticity_scaled}")
            return vorticity_scaled
            
        except Exception as e:
            print(f"  获取涡度值失败：{e}")
            return None
    
    def calc_cyclone_central_wind(self, u_field, v_field, center_lon, center_lat, 
                                   lats, lons, grid_res=0.25):
        """
        计算台风中心附近最大持续风速（200km搜索半径 + 9点平滑）
        """
        try:
            # 计算200km搜索半径对应的网格点数
            search_radius_deg = self.SEARCH_RADIUS_KM / 111
            search_grid_num = int(np.ceil(search_radius_deg / grid_res))
            
            # 定位台风中心索引
            center_lon_idx = np.argmin(np.abs(lons - center_lon))
            center_lat_idx = np.argmin(np.abs(lats - center_lat))
            
            # 定义搜索范围（避免数组越界）
            lon_start = max(0, center_lon_idx - search_grid_num)
            lon_end = min(u_field.shape[1], center_lon_idx + search_grid_num + 1)
            lat_start = max(0, center_lat_idx - search_grid_num)
            lat_end = min(u_field.shape[0], center_lat_idx + search_grid_num + 1)
            
            # 提取搜索范围内的风场，计算合成风速
            u_search = u_field[lat_start:lat_end, lon_start:lon_end]
            v_search = v_field[lat_start:lat_end, lon_start:lon_end]
            wind_speed = np.sqrt(u_search**2 + v_search**2)
            
            # 9点平滑（3x3窗口）减少网格噪声
            wind_speed_smoothed = uniform_filter(wind_speed, size=3, mode='constant')
            
            # 返回最大风速，保留2位小数
            return round(np.max(wind_speed_smoothed), 2)
        except Exception as e:
            print(f"  计算中心风速失败：{str(e)}")
            return None
    
    def get_wind_speed(self, lat, lon, hour):
        """
        获取指定位置/时间的风速（从当前加载的文件中读取）
        
        Parameters:
        -----------
        lat, lon : float
            经纬度
        hour : int
            小时（0-23）
        """
        if self.u_data is None or self.v_data is None:
            return None
        
        try:
            # 直接使用小时作为时间索引
            time_idx = hour
            
            if time_idx >= self.u_data.shape[0]:
                print(f"  ⚠️  时间索引 {time_idx} 超出范围 ({self.u_data.shape[0]})")
                return None
            
            # 使用 LEVEL_INDEX=15 (850hPa)
            u_field = self.u_data[time_idx, self.LEVEL_INDEX, :, :]
            v_field = self.v_data[time_idx, self.LEVEL_INDEX, :, :]
            
            # 处理经度
            if lon < 0:
                lon = lon + 360
            
            # 计算实际网格分辨率
            if len(self.uv_lon) > 1 and not np.isnan(np.diff(self.uv_lon)).any():
                grid_res = np.mean(np.diff(self.uv_lon))
            else:
                grid_res = self.GRID_RES_DEFAULT
            
            # 调用风速计算函数
            wind_speed = self.calc_cyclone_central_wind(
                u_field, v_field, lon, lat, self.uv_lat, self.uv_lon, grid_res
            )
            
            if wind_speed is not None:
                # 乘以10并取整
                wind_scaled = int(round(wind_speed * 10))
                print(f"  风速：time_idx={time_idx}, 压力层={self.LEVEL_INDEX}(850hPa), max={wind_speed:.2f} m/s -> {wind_scaled}")
                return wind_scaled
            else:
                return None
            
        except Exception as e:
            print(f"  获取最大风速失败：{e}")
            return None
    
    def determine_stop_reason(self, row):
        """确定头记录的第六位数字"""
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
        """解析输入CSV文件"""
        self.input_file = input_file
        self.data = []
        
        try:
            encodings = ['utf-8-sig', 'utf-8', 'gbk', 'gb2312']
            for encoding in encodings:
                try:
                    with open(input_file, 'r', encoding=encoding) as f:
                        reader = csv.DictReader(f)
                        self.data = list(reader)
                    print(f"成功读取 {len(self.data)} 行数据 (使用编码: {encoding})")
                    return True
                except UnicodeDecodeError:
                    continue
            raise ValueError("无法使用任何编码读取文件")
        except Exception as e:
            print(f"读取CSV文件失败：{e}")
            return False
    
    def convert_to_trv_format(self, output_file):
        """转换数据为TRV格式"""
        if not self.data:
            print("没有数据可转换，请先解析输入文件")
            return False
        
        output_lines = []
        first_row = self.data[0]
        last_row = self.data[-1]
        
        try:
            # 解析头信息
            intl_id = int(float(first_row.get('International ID', '16')))
            china_id = first_row.get('China ID', '8012')
            tc_name = first_row.get('Typhoon Name', 'Norris')
            
            first_time = first_row.get('Tracking Time', '1980/8/29 6:00')
            start_dt = self.parse_datetime(first_time)
            start_date = start_dt.strftime('%Y%m%d')
            
            print(f"\n开始时间: {start_dt.strftime('%Y-%m-%d %H:%M')}")
            print(f"涡度数据根目录: {self.VOR_ROOT_DIR}")
            print(f"风速数据根目录: {self.UV_ROOT_DIR}")
            print("-" * 70)
            
            # 判断是否保留最后一行
            record_count = len(self.data)
            last_status = last_row.get('Tracking Status', '').strip()
            
            if last_status == 'Terminated':
                record_count -= 1
                use_data = self.data[:-1]
                stop_reason = self.determine_stop_reason(last_row)
            else:
                use_data = self.data
                stop_reason = 0
            
            # 检查涡度值，如果检测到负值则截断
            valid_data = []
            previous_date = None
            
            for i, row in enumerate(use_data):
                time_str = row.get('Tracking Time', '')
                dt_temp = self.parse_datetime(time_str)
                hour = dt_temp.hour
                current_date = dt_temp.date()
                
                # 如果日期变化，加载对应的NC文件
                if current_date != previous_date:
                    print(f"\n加载日期: {current_date.strftime('%Y-%m-%d')}")
                    self.load_file_for_date(dt_temp)
                    previous_date = current_date
                
                lat = float(row.get('Center Latitude', 0))
                lon = float(row.get('Center Longitude', 0))
                
                if self.vor_data is not None:
                    vorticity = self.get_vorticity(lat, lon, hour)
                    if vorticity is not None and vorticity < 0:
                        print(f"\n⚠️  在第 {i+1} 行检测到负涡度 (值: {vorticity})，截断后续记录")
                        valid_data = use_data[:i]
                        break
                valid_data = use_data
            
            if valid_data != use_data:
                use_data = valid_data
                record_count = len(use_data)
                print(f"截断后剩余 {record_count} 条记录")
            
            # 构建头记录
            sequence_num = first_row.get('International ID', '0016')
            header = f"66666,{intl_id:04d},{record_count:03d},{int(sequence_num):04d},{china_id},{stop_reason},{tc_name},{start_date}"
            output_lines.append(header)
            
            print(f"\n头记录: {header}")
            print(f"记录数: {record_count}, 停止原因: {stop_reason}")
            print("-" * 70)
            
            # 处理每条轨迹记录
            previous_date = None
            
            for i, row in enumerate(use_data):
                time_str = row.get('Tracking Time', '')
                dt = self.parse_datetime(time_str)
                time_formatted = dt.strftime('%Y%m%d%H')
                hour = dt.hour
                current_date = dt.date()
                
                # 如果日期变化，加载对应的NC文件
                if current_date != previous_date:
                    print(f"\n加载日期: {current_date.strftime('%Y-%m-%d')}")
                    self.load_file_for_date(dt)
                    previous_date = current_date
                
                lat = float(row.get('Center Latitude', 0))
                lon = float(row.get('Center Longitude', 0))
                lat_int = int(round(lat * 10))
                lon_int = int(round(lon * 10))
                
                # 流函数处理
                stream_func_raw = float(row.get('Center Streamfunction Value', 0))
                stream_func = int(round(abs(stream_func_raw) / 10000))
                
                print(f"\n记录 {i+1}: time={time_str}, hour={hour}, 位置=({lat:.2f}, {lon:.2f})")
                
                # 获取涡度值
                if self.vor_data is not None:
                    vorticity = self.get_vorticity(lat, lon, hour)
                    if vorticity is None:
                        vort_raw = float(row.get('Center Vorticity Value', 0))
                        vorticity = int(round(vort_raw * 1e5))
                        print(f"  ⚠️  使用CSV涡度值（备用）: {vorticity}")
                else:
                    vort_raw = float(row.get('Center Vorticity Value', 0))
                    vorticity = int(round(vort_raw * 1e5))
                    print(f"  使用CSV涡度值: {vorticity}")
                
                # 获取风速值
                if self.u_data is not None:
                    velocity = self.get_wind_speed(lat, lon, hour)
                    if velocity is None:
                        velocity = int(stream_func * 0.3)
                        if velocity < 10:
                            velocity = 150
                        print(f"  ⚠️  使用估算风速（备用）: {velocity}")
                else:
                    velocity = int(stream_func * 0.3)
                    if velocity < 10:
                        velocity = 150
                    print(f"  使用估算风速: {velocity}")
                
                # 构建轨迹记录
                track_line = f"{time_formatted},{lat_int:03d},{lon_int:04d},{stream_func},{vorticity},{velocity}"
                output_lines.append(track_line)
                print(f"  生成: {track_line}")
            
            # 关闭所有打开的文件
            self.close_current_files()
            
            print("-" * 70)
            
            # 写入输出文件
            with open(output_file, 'w', encoding='utf-8') as f:
                for line in output_lines:
                    f.write(line + '\n')
            
            print(f"\n成功转换 {len(use_data)} 条轨迹记录到 {output_file}")
            return True
            
        except Exception as e:
            print(f"转换数据失败：{e}")
            import traceback
            traceback.print_exc()
            # 确保关闭文件
            self.close_current_files()
            return False


def main():
    """主函数"""
    input_file = "all_typhoons_tracking_x1.csv"
    output_file = "TRV_x1.csv"
    
    # 设置ERA5数据根目录（分别指定涡度和风速）
    vor_root_dir = r"E:\era5_vorticity_data"
    uv_root_dir = r"E:\era5_uv_data"
    
    converter = CSVToTRVConverter(vor_root_dir, uv_root_dir)
    
    if not converter.parse_input_csv(input_file):
        print("解析输入文件失败")
        return
    
    success = converter.convert_to_trv_format(output_file)
    
    if success:
        print(f"\n转换完成！输出文件：{output_file}")
        print("\n最终输出内容：")
        with open(output_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                print(f"{i+1}: {line.strip()}")
    else:
        print("转换失败")


if __name__ == "__main__":
    main()