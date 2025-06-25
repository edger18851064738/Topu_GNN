#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Excel坐标数据转换为矿山地图JSON格式的转换器
支持将xlsx文件中的x,y坐标转换为map_creater.py可读取的JSON格式
"""

import pandas as pd
import numpy as np
import json
import argparse
import os
from typing import List, Dict, Tuple, Optional

class XlsxToMapConverter:
    def __init__(self):
        self.grid_size = 1.0  # 网格单元大小（米）
        self.map_width = 500   # 地图宽度（网格单元数）
        self.map_height = 500  # 地图高度（网格单元数）
        
    def read_xlsx_file(self, file_path: str, sheet_name: Optional[str] = None) -> pd.DataFrame:
        """读取xlsx文件"""
        try:
            if sheet_name:
                df = pd.read_excel(file_path, sheet_name=sheet_name)
            else:
                df = pd.read_excel(file_path)
            print(f"成功读取文件 {file_path}，共 {len(df)} 行数据")
            return df
        except Exception as e:
            raise Exception(f"读取xlsx文件失败: {str(e)}")
    
    def extract_coordinates(self, df: pd.DataFrame, x_col: str = 'x', y_col: str = 'y') -> List[Tuple[float, float]]:
        """提取x,y坐标"""
        if x_col not in df.columns or y_col not in df.columns:
            available_cols = list(df.columns)
            raise ValueError(f"找不到坐标列 '{x_col}' 或 '{y_col}'。可用列: {available_cols}")
        
        # 提取坐标并过滤掉NaN值
        coords = []
        for _, row in df.iterrows():
            x, y = row[x_col], row[y_col]
            if pd.notna(x) and pd.notna(y):
                coords.append((float(x), float(y)))
        
        print(f"提取到 {len(coords)} 个有效坐标点")
        return coords
    
    def calculate_map_bounds(self, coords: List[Tuple[float, float]], padding: float = 10.0) -> Tuple[float, float, float, float]:
        """计算地图边界"""
        if not coords:
            return 0, 0, 100, 100
            
        x_coords = [coord[0] for coord in coords]
        y_coords = [coord[1] for coord in coords]
        
        min_x, max_x = min(x_coords), max(x_coords)
        min_y, max_y = min(y_coords), max(y_coords)
        
        # 添加边距
        min_x -= padding
        max_x += padding
        min_y -= padding
        max_y += padding
        
        print(f"坐标范围: X({min_x:.2f}, {max_x:.2f}), Y({min_y:.2f}, {max_y:.2f})")
        return min_x, min_y, max_x, max_y
    
    def coords_to_grid(self, coords: List[Tuple[float, float]], 
                      bounds: Tuple[float, float, float, float]) -> List[Tuple[int, int]]:
        """将实际坐标转换为网格坐标"""
        min_x, min_y, max_x, max_y = bounds
        
        # 计算所需的地图尺寸
        width_needed = int((max_x - min_x) / self.grid_size) + 1
        height_needed = int((max_y - min_y) / self.grid_size) + 1
        
        # 更新地图尺寸
        self.map_width = max(self.map_width, width_needed)
        self.map_height = max(self.map_height, height_needed)
        
        grid_coords = []
        for x, y in coords:
            # 转换为网格坐标
            grid_x = int((x - min_x) / self.grid_size)
            grid_y = int((y - min_y) / self.grid_size)
            
            # 确保在地图范围内
            grid_x = max(0, min(grid_x, self.map_width - 1))
            grid_y = max(0, min(grid_y, self.map_height - 1))
            
            grid_coords.append((grid_x, grid_y))
        
        print(f"转换后地图尺寸: {self.map_width} x {self.map_height}")
        return grid_coords
    
    def group_by_border_id(self, df: pd.DataFrame, coords: List[Tuple[int, int]], 
                          border_col: str = 'borderId') -> Dict[int, List[Tuple[int, int]]]:
        """按borderId分组坐标"""
        if border_col not in df.columns:
            # 如果没有borderId列，将所有点归为一组
            return {1: coords}
        
        groups = {}
        for i, (_, row) in enumerate(df.iterrows()):
            if i < len(coords):
                border_id = int(row[border_col]) if pd.notna(row[border_col]) else 1
                if border_id not in groups:
                    groups[border_id] = []
                groups[border_id].append(coords[i])
        
        print(f"按borderId分组: {len(groups)} 个组")
        for bid, points in groups.items():
            print(f"  组 {bid}: {len(points)} 个点")
        
        return groups
    
    def create_obstacles_from_coords(self, grid_coords: List[Tuple[int, int]]) -> List[Dict]:
        """将坐标点转换为障碍物"""
        obstacles = []
        for x, y in grid_coords:
            obstacles.append({
                "x": x,
                "y": y,
                "width": 1,
                "height": 1
            })
        return obstacles
    
    def create_loading_points_from_groups(self, coord_groups: Dict[int, List[Tuple[int, int]]]) -> List[List]:
        """将分组坐标转换为装载点（取每组的第一个点）"""
        loading_points = []
        for border_id, coords in coord_groups.items():
            if coords:
                x, y = coords[0]  # 取第一个点
                # 格式: [row, col, theta]
                loading_points.append([y, x, 0.0])
        return loading_points
    
    def create_path_obstacles(self, coord_groups: Dict[int, List[Tuple[int, int]]]) -> List[Dict]:
        """将分组坐标转换为路径障碍物"""
        obstacles = []
        for border_id, coords in coord_groups.items():
            for x, y in coords:
                obstacles.append({
                    "x": x,
                    "y": y,
                    "width": 1,
                    "height": 1
                })
        return obstacles
    
    def convert_to_map_format(self, df: pd.DataFrame, conversion_type: str = "obstacles",
                             x_col: str = 'x', y_col: str = 'y', border_col: str = 'borderId') -> Dict:
        """转换为地图格式"""
        # 提取坐标
        coords = self.extract_coordinates(df, x_col, y_col)
        if not coords:
            raise ValueError("没有找到有效的坐标数据")
        
        # 计算边界并转换为网格坐标
        bounds = self.calculate_map_bounds(coords)
        grid_coords = self.coords_to_grid(coords, bounds)
        
        # 创建基础地图数据
        map_data = {
            "dimensions": {
                "rows": self.map_height,
                "cols": self.map_width
            },
            "width": self.map_width,
            "height": self.map_height,
            "resolution": self.grid_size,
            "grid": [[0 for _ in range(self.map_width)] for _ in range(self.map_height)],
            "loading_points": [],
            "unloading_points": [],
            "parking_areas": [],
            "vehicle_positions": [],
            "obstacles": [],
            "vehicles_info": []
        }
        
        # 根据转换类型处理数据
        if conversion_type == "obstacles":
            # 转换为障碍物
            map_data["obstacles"] = self.create_obstacles_from_coords(grid_coords)
            # 同时更新网格
            for x, y in grid_coords:
                if 0 <= y < self.map_height and 0 <= x < self.map_width:
                    map_data["grid"][y][x] = 1
                    
        elif conversion_type == "loading_points":
            # 按borderId分组，每组的第一个点作为装载点
            coord_groups = self.group_by_border_id(df, grid_coords, border_col)
            map_data["loading_points"] = self.create_loading_points_from_groups(coord_groups)
            
        elif conversion_type == "unloading_points":
            # 按borderId分组，每组的第一个点作为卸载点
            coord_groups = self.group_by_border_id(df, grid_coords, border_col)
            unloading_points = []
            for border_id, coords in coord_groups.items():
                if coords:
                    x, y = coords[0]
                    unloading_points.append([y, x, 0.0])
            map_data["unloading_points"] = unloading_points
            
        elif conversion_type == "paths":
            # 将路径转换为障碍物（可用于绘制道路边界等）
            coord_groups = self.group_by_border_id(df, grid_coords, border_col)
            map_data["obstacles"] = self.create_path_obstacles(coord_groups)
            # 同时更新网格
            for x, y in grid_coords:
                if 0 <= y < self.map_height and 0 <= x < self.map_width:
                    map_data["grid"][y][x] = 1
        
        # 添加元数据
        map_data["metadata"] = {
            "source_file": "xlsx_converted",
            "conversion_type": conversion_type,
            "total_points": len(coords),
            "grid_size": self.grid_size,
            "bounds": bounds
        }
        
        return map_data
    
    def save_map_json(self, map_data: Dict, output_path: str):
        """保存地图JSON文件"""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(map_data, f, indent=2, ensure_ascii=False)
            print(f"地图文件已保存到: {output_path}")
        except Exception as e:
            raise Exception(f"保存文件失败: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description='将xlsx坐标数据转换为矿山地图JSON格式')
    parser.add_argument('input_file', help='输入的xlsx文件路径')
    parser.add_argument('-o', '--output', help='输出的JSON文件路径（默认为input_file_mine.json）')
    parser.add_argument('-t', '--type', choices=['obstacles', 'loading_points', 'unloading_points', 'paths'],
                       default='obstacles', help='转换类型（默认：obstacles）')
    parser.add_argument('-x', '--x_col', default='x', help='X坐标列名（默认：x）')
    parser.add_argument('-y', '--y_col', default='y', help='Y坐标列名（默认：y）')
    parser.add_argument('-b', '--border_col', default='borderId', help='边界ID列名（默认：borderId）')
    parser.add_argument('-s', '--sheet', help='Excel工作表名称（可选）')
    parser.add_argument('--grid_size', type=float, default=1.0, help='网格大小（默认：1.0米）')
    
    args = parser.parse_args()
    
    # 检查输入文件
    if not os.path.exists(args.input_file):
        print(f"错误: 输入文件 {args.input_file} 不存在")
        return
    
    # 确定输出文件路径
    if args.output:
        output_path = args.output
    else:
        base_name = os.path.splitext(args.input_file)[0]
        output_path = f"{base_name}_mine.json"
    
    try:
        # 创建转换器
        converter = XlsxToMapConverter()
        converter.grid_size = args.grid_size
        
        # 读取Excel文件
        print(f"正在读取文件: {args.input_file}")
        df = converter.read_xlsx_file(args.input_file, args.sheet)
        
        # 显示列信息
        print(f"数据列: {list(df.columns)}")
        
        # 转换为地图格式
        print(f"转换类型: {args.type}")
        map_data = converter.convert_to_map_format(
            df, args.type, args.x_col, args.y_col, args.border_col
        )
        
        # 保存文件
        converter.save_map_json(map_data, output_path)
        
        print("\n转换完成！")
        print(f"地图尺寸: {map_data['width']} x {map_data['height']}")
        print(f"网格大小: {converter.grid_size}米")
        print(f"总点数: {map_data['metadata']['total_points']}")
        
    except Exception as e:
        print(f"转换失败: {str(e)}")

if __name__ == "__main__":
    # 如果没有命令行参数，提供交互式模式
    import sys
    if len(sys.argv) == 1:
        print("Excel坐标转换器 - 交互模式")
        print("=" * 40)
        
        # 获取输入文件
        input_file = input("请输入xlsx文件路径: ").strip()
        if not os.path.exists(input_file):
            print("文件不存在！")
            sys.exit(1)
        
        # 获取转换类型
        print("\n转换类型:")
        print("1. obstacles - 障碍物")
        print("2. loading_points - 装载点")
        print("3. unloading_points - 卸载点")
        print("4. paths - 路径")
        
        type_choice = input("请选择转换类型 (1-4, 默认1): ").strip() or "1"
        type_map = {"1": "obstacles", "2": "loading_points", "3": "unloading_points", "4": "paths"}
        conversion_type = type_map.get(type_choice, "obstacles")
        
        # 获取列名
        x_col = input("X坐标列名 (默认 'x'): ").strip() or "x"
        y_col = input("Y坐标列名 (默认 'y'): ").strip() or "y"
        border_col = input("边界ID列名 (默认 'borderId'): ").strip() or "borderId"
        
        # 输出文件
        base_name = os.path.splitext(input_file)[0]
        output_file = f"{base_name}_mine.json"
        
        try:
            converter = XlsxToMapConverter()
            df = converter.read_xlsx_file(input_file)
            print(f"可用的列: {list(df.columns)}")
            
            map_data = converter.convert_to_map_format(df, conversion_type, x_col, y_col, border_col)
            converter.save_map_json(map_data, output_file)
            
            print(f"\n转换成功！输出文件: {output_file}")
            
        except Exception as e:
            print(f"转换失败: {str(e)}")
    else:
        main()