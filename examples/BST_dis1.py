# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 15:59:53 2026

@author: YE
"""

# -*- coding: utf-8 -*-
"""
查找包含变性（强度代码9）的台风程序
"""

import os
import pandas as pd
import glob
from typing import List, Dict, Any
from datetime import datetime

class CMABSTDataParserFixed:
    """修复版中国气象局最佳路径数据解析器"""
    
    def __init__(self):
        self.header_columns = [
            'classification_flag', 'international_id', 'record_count', 
            'cyclone_sequence', 'china_id', 'end_status', 'time_interval',
            'name', 'data_generation_date'
        ]
        
        # 基础数据列
        self.data_columns = [
            'timestamp', 'intensity', 'latitude', 'longitude', 
            'pressure', 'max_wind_speed'
        ]
    
    def parse_header_line(self, line: str) -> Dict[str, Any]:
        """解析头记录行 - 使用空格分隔"""
        try:
            # 按空格分割，过滤空字符串
            parts = line.split()
            
            if len(parts) < 9:  # 最少需要的字段数
                print(f"字段不足: {len(parts)}")
                return None
            
            header = {
                'classification_flag': parts[0],      # "66666"
                'international_id': parts[1],         # "0000"
                'record_count': int(parts[2]),        # "10"
                'cyclone_sequence': parts[3],         # "0003"
                'china_id': parts[4],                 # "0000"
                'end_status': parts[5],               # "0"
                'time_interval': int(parts[6]),       # "6"
                'name': parts[7] if parts[7] != '(nameless)' else '',  # "(nameless)"
                'data_generation_date': parts[8] if len(parts) > 8 else ''  # "20110729"
            }
            
            # 处理可能的额外字段
            if len(parts) > 9:
                header['extra_info'] = ' '.join(parts[9:])
            
            return header
        except Exception as e:
            print(f"解析头记录错误: {e}")
            print(f"行内容: {line}")
            return None
    
    def parse_data_line(self, line: str) -> Dict[str, Any]:
        """解析数据记录行 - 修复版"""
        try:
            # 基本字段解析
            data_record = {
                'timestamp': line[0:10].strip(),
                'intensity': int(line[10:12].strip()) if line[10:12].strip() else -1,
                'latitude': float(line[12:17].strip()) / 10.0 if line[12:17].strip() else 0,
                'longitude': float(line[17:22].strip()) / 10.0 if line[17:22].strip() else 0,
                'pressure': int(line[22:26].strip()) if line[22:26].strip() else 0,
                'max_wind_speed': int(line[26:30].strip()) if line[26:30].strip() else 0
            }
            
            # 处理可能存在的额外字段（如风速、风向等）
            if len(line) > 30:
                additional_str = line[30:35].strip() if len(line) > 35 else line[30:].strip()
                if additional_str:
                    # 尝试解析额外字段
                    try:
                        # 有些行有两个额外数值
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
        """解析单个文件"""
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
                
                # 检查是否为头记录（以66666开头）
                if line.startswith('66666'):
                    # 保存前一个台风的数据
                    if current_header and data_lines:
                        typhoon_data = self._process_typhoon(current_header, data_lines, file_path)
                        if typhoon_data:
                            typhoons_data.append(typhoon_data)
                    
                    # 开始新的台风记录
                    current_header = self.parse_header_line(line)
                    data_lines = []
                else:
                    # 数据记录行
                    data_lines.append(line)
            
            # 处理最后一个台风
            if current_header and data_lines:
                typhoon_data = self._process_typhoon(current_header, data_lines, file_path)
                if typhoon_data:
                    typhoons_data.append(typhoon_data)
                    
        except Exception as e:
            print(f"解析文件 {file_path} 时出错: {e}")
        
        return typhoons_data
    
    def _process_typhoon(self, header: Dict[str, Any], data_lines: List[str], file_path: str) -> Dict[str, Any]:
        """处理单个台风数据"""
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
        """读取目录下的所有数据文件"""
        all_typhoons = {}
        
        # 查找所有匹配的文件
        search_pattern = os.path.join(directory_path, pattern)
        files = glob.glob(search_pattern)
        
        print(f"找到 {len(files)} 个数据文件")
        
        for file_path in files:
            print(f"正在解析文件: {os.path.basename(file_path)}")
            typhoons = self.parse_file(file_path)
            all_typhoons[os.path.basename(file_path)] = typhoons
        
        return all_typhoons

def analyze_intensity(intensity_code: int) -> str:
    """将强度代码转换为中文描述"""
    intensity_map = {
        0: "弱于热带低压或未知",
        1: "热带低压(TD)",
        2: "热带风暴(TS)",
        3: "强热带风暴(STS)",
        4: "台风(TY)",
        5: "强台风(STY)",
        6: "超强台风(SuperTY)",
        9: "变性"
    }
    return intensity_map.get(intensity_code, "未知")

def find_typhoons_with_intensity_9(all_typhoons: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """找出所有包含强度9（变性）的台风"""
    result_typhoons = []
    
    for filename, typhoons in all_typhoons.items():
        for typhoon in typhoons:
            header = typhoon['header']
            track_data = typhoon['track_data']
            
            # 检查轨迹中是否包含强度9
            has_intensity_9 = any(point.get('intensity') == 9 for point in track_data)
            
            if has_intensity_9:
                # 找到变性发生的位置
                extratropical_points = []
                for i, point in enumerate(track_data):
                    if point.get('intensity') == 9:
                        extratropical_points.append({
                            'index': i + 1,
                            'timestamp': point['timestamp'],
                            'latitude': point['latitude'],
                            'longitude': point['longitude'],
                            'pressure': point['pressure'],
                            'max_wind_speed': point['max_wind_speed']
                        })
                
                result_typhoons.append({
                    'filename': filename,
                    'header': header,
                    'track_data': track_data,
                    'total_points': len(track_data),
                    'extratropical_points': extratropical_points,
                    'first_extratropical_index': extratropical_points[0]['index'] if extratropical_points else None
                })
    
    return result_typhoons

def save_results_to_file(result_typhoons: List[Dict[str, Any]], output_file: str):
    """将结果保存到文件"""
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=" * 100 + "\n")
        f.write("包含变性（强度代码9）的台风列表\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 100 + "\n\n")
        
        f.write(f"共找到 {len(result_typhoons)} 个包含变性的台风\n\n")
        
        for i, typhoon in enumerate(result_typhoons, 1):
            header = typhoon['header']
            track_data = typhoon['track_data']
            extratropical_points = typhoon['extratropical_points']
            
            f.write(f"\n{'='*80}\n")
            f.write(f"台风 #{i}\n")
            f.write(f"{'='*80}\n")
            
            # 基本信息
            f.write(f"文件名: {typhoon['filename']}\n")
            f.write(f"台风名称: {header.get('name', '未命名')}\n")
            f.write(f"国际编号: {header.get('international_id', '')}\n")
            f.write(f"中国编号: {header.get('china_id', '')}\n")
            f.write(f"记录总数: {typhoon['total_points']}\n")
            f.write(f"变性点数量: {len(extratropical_points)}\n")
            f.write(f"首次变性位置: 第 {typhoon['first_extratropical_index']} 个记录点\n\n")
            
            # 变性详情
            f.write("变性发生位置详细信息:\n")
            f.write("-" * 80 + "\n")
            f.write(f"{'序号':<6} {'时间':<12} {'纬度(°N)':<10} {'经度(°E)':<10} {'气压(hPa)':<10} {'风速(m/s)':<10}\n")
            f.write("-" * 80 + "\n")
            
            for point in extratropical_points:
                f.write(f"{point['index']:<6} {point['timestamp']:<12} {point['latitude']:<10.1f} "
                       f"{point['longitude']:<10.1f} {point['pressure']:<10} {point['max_wind_speed']:<10}\n")
            
            # 变性前后的强度变化
            f.write("\n变性过程分析:\n")
            f.write("-" * 80 + "\n")
            
            first_extratropical_idx = typhoon['first_extratropical_index']
            if first_extratropical_idx and first_extratropical_idx > 1:
                before_point = track_data[first_extratropical_idx - 2]
                f.write(f"变性前 (第{first_extratropical_idx-1}点): ")
                f.write(f"强度={analyze_intensity(before_point['intensity'])}, ")
                f.write(f"位置=({before_point['latitude']:.1f}°N, {before_point['longitude']:.1f}°E)\n")
            
            first_point = extratropical_points[0]
            f.write(f"变性开始(第{first_point['index']}点): 强度=变性, ")
            f.write(f"位置=({first_point['latitude']:.1f}°N, {first_point['longitude']:.1f}°E)\n")
            
            if len(extratropical_points) > 1:
                last_point = extratropical_points[-1]
                f.write(f"最后变性点(第{last_point['index']}点): 强度=变性, ")
                f.write(f"位置=({last_point['latitude']:.1f}°N, {last_point['longitude']:.1f}°E)\n")
            
            # 完整轨迹摘要
            f.write(f"\n完整轨迹摘要（共{len(track_data)}个点）:\n")
            f.write("-" * 80 + "\n")
            f.write(f"{'时间':<12} {'强度':<15} {'纬度(°N)':<10} {'经度(°E)':<10} {'气压':<8} {'风速':<8}\n")
            f.write("-" * 80 + "\n")
            
            for point in track_data:
                intensity_str = analyze_intensity(point['intensity'])
                # 标记变性点
                if point['intensity'] == 9:
                    intensity_str = "【变性】"
                
                f.write(f"{point['timestamp']:<12} {intensity_str:<15} "
                       f"{point['latitude']:<10.1f} {point['longitude']:<10.1f} "
                       f"{point['pressure']:<8} {point['max_wind_speed']:<8}\n")
            
            f.write("\n")

def save_to_csv(result_typhoons: List[Dict[str, Any]], output_csv: str):
    """将结果保存为CSV格式"""
    rows = []
    
    for typhoon in result_typhoons:
        header = typhoon['header']
        track_data = typhoon['track_data']
        
        for point in track_data:
            row = {
                'filename': typhoon['filename'],
                'international_id': header.get('international_id', ''),
                'china_id': header.get('china_id', ''),
                'typhoon_name': header.get('name', ''),
                'total_points': typhoon['total_points'],
                'has_extratropical': '是',
                'extratropical_count': len(typhoon['extratropical_points']),
                'timestamp': point['timestamp'],
                'intensity': point['intensity'],
                'intensity_desc': analyze_intensity(point['intensity']),
                'is_extratropical': '是' if point['intensity'] == 9 else '否',
                'latitude': point['latitude'],
                'longitude': point['longitude'],
                'pressure': point['pressure'],
                'max_wind_speed': point['max_wind_speed']
            }
            rows.append(row)
    
    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    print(f"CSV文件已保存到: {output_csv}")

def main():
    """主函数"""
    # 设置数据目录路径（请根据实际情况修改）
    data_directory = r"D:\CMABSTdata"  # 修改为您的实际路径
    
    # 输出文件路径
    output_directory = data_directory
    output_txt = os.path.join(output_directory, "extratropical_typhoons.txt")
    output_csv = os.path.join(output_directory, "extratropical_typhoons.csv")
    
    # 检查目录是否存在
    if not os.path.exists(data_directory):
        print(f"目录 {data_directory} 不存在！")
        print("请修改 data_directory 变量为正确的路径")
        return
    
    # 创建解析器实例
    parser = CMABSTDataParserFixed()
    
    # 读取所有数据文件
    print("=" * 80)
    print("开始解析台风数据...")
    print("=" * 80)
    all_typhoons = parser.read_all_files(data_directory, pattern="CH*BST.txt")
    
    # 统计信息
    total_typhoons = sum(len(typhoons) for typhoons in all_typhoons.values())
    print(f"\n解析完成！")
    print(f"共处理 {len(all_typhoons)} 个文件")
    print(f"共找到 {total_typhoons} 个台风")
    
    # 查找包含变性的台风
    print("\n正在查找包含变性（强度代码9）的台风...")
    result_typhoons = find_typhoons_with_intensity_9(all_typhoons)
    
    # 输出结果
    print(f"\n{'='*80}")
    print(f"结果统计")
    print(f"{'='*80}")
    print(f"找到 {len(result_typhoons)} 个包含变性的台风")
    
    if result_typhoons:
        # 保存到文本文件
        save_results_to_file(result_typhoons, output_txt)
        print(f"详细结果已保存到: {output_txt}")
        
        # 保存到CSV文件
        save_to_csv(result_typhoons, output_csv)
        
        # 在控制台显示简要结果
        print("\n包含变性的台风列表:")
        print("-" * 80)
        for i, typhoon in enumerate(result_typhoons, 1):
            header = typhoon['header']
            name = header.get('name', '未命名')
            extratropical_count = len(typhoon['extratropical_points'])
            print(f"{i:2}. {name:15} (国际编号:{header.get('international_id', '')}) "
                  f"- 变性点数量:{extratropical_count}, 总记录数:{typhoon['total_points']}")
    else:
        print("\n未找到包含变性的台风")
    
    print("\n程序执行完成！")

if __name__ == "__main__":
    main()