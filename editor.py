import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
import json
import os
from PIL import Image, ImageDraw, ImageTk

class MineMapCreator(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("露天矿地图创建工具")
        self.geometry("1000x700")

        # 地图数据
        self.rows = 100
        self.cols = 100
        self.cell_size = 6  # 单元格大小
        self.resolution = 1.0  # 分辨率

        # 点位数据
        self.loading_points = []
        self.unloading_points = []
        self.parking_areas = []
        self.vehicles = []
        
        # 网格 - 0为可通行，1为障碍物
        self.grid = np.zeros((self.rows, self.cols), dtype=np.int8)
        
        self.current_tool = "obstacle"  # 默认工具
        self.brush_size = 2  # 默认画笔大小
        self.last_painted_cell = None  # 避免重复绘制
        
        # 创建UI
        self.create_ui()

    def create_ui(self):
        # 主布局
        main_frame = tk.Frame(self)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # 左侧控制面板
        control_frame = tk.Frame(main_frame, width=200)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        
        # 右侧画布
        canvas_frame = tk.Frame(main_frame)
        canvas_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        # 创建控制面板
        self.create_controls(control_frame)
        
        # 创建画布
        self.create_canvas(canvas_frame)
        
        # 初始化地图
        self.init_map()

    def create_controls(self, parent):
        # 地图尺寸设置
        size_frame = tk.LabelFrame(parent, text="地图尺寸", padx=5, pady=5)
        size_frame.pack(fill=tk.X, padx=5, pady=5)

        tk.Label(size_frame, text="宽度:").grid(row=0, column=0, sticky=tk.W)
        self.col_entry = tk.Entry(size_frame, width=6)
        self.col_entry.grid(row=0, column=1, padx=5, pady=2)
        self.col_entry.insert(0, str(self.cols))

        tk.Label(size_frame, text="高度:").grid(row=1, column=0, sticky=tk.W)
        self.row_entry = tk.Entry(size_frame, width=6)
        self.row_entry.grid(row=1, column=1, padx=5, pady=2)
        self.row_entry.insert(0, str(self.rows))

        tk.Button(size_frame, text="更新尺寸", command=self.update_map_size).grid(row=2, column=0, columnspan=2, pady=5)

        # 工具选择
        tool_frame = tk.LabelFrame(parent, text="绘图工具", padx=5, pady=5)
        tool_frame.pack(fill=tk.X, padx=5, pady=5)

        self.tool_var = tk.StringVar(value="obstacle")
        tools = [
            ("障碍物", "obstacle", "black"),
            ("可通行", "passable", "white"),
            ("装载点", "loading", "green"),
            ("卸载点", "unloading", "red"),
            ("停车区", "parking", "blue"),
            ("车辆", "vehicle", "orange")
        ]

        for i, (text, value, color) in enumerate(tools):
            rb = tk.Radiobutton(tool_frame, text=text, value=value, variable=self.tool_var, 
                              command=self.update_current_tool)
            rb.pack(anchor=tk.W)

        # 画笔大小
        brush_frame = tk.Frame(tool_frame)
        brush_frame.pack(fill=tk.X, pady=5)
        
        tk.Label(brush_frame, text="画笔大小:").pack(side=tk.LEFT)
        self.brush_scale = tk.Scale(brush_frame, from_=1, to=10, orient=tk.HORIZONTAL,
                                   command=self.update_brush_size)
        self.brush_scale.set(self.brush_size)
        self.brush_scale.pack(side=tk.RIGHT, fill=tk.X, expand=True)

        # 属性设置
        attr_frame = tk.LabelFrame(parent, text="属性设置", padx=5, pady=5)
        attr_frame.pack(fill=tk.X, padx=5, pady=5)
        
        tk.Label(attr_frame, text="方向角度(度):").grid(row=0, column=0, sticky=tk.W)
        self.theta_entry = tk.Entry(attr_frame, width=6)
        self.theta_entry.grid(row=0, column=1, padx=5, pady=2)
        self.theta_entry.insert(0, "0")
        
        tk.Label(attr_frame, text="停车容量:").grid(row=1, column=0, sticky=tk.W)
        self.capacity_entry = tk.Entry(attr_frame, width=6)
        self.capacity_entry.grid(row=1, column=1, padx=5, pady=2)
        self.capacity_entry.insert(0, "5")
        
        tk.Label(attr_frame, text="车辆最大载重:").grid(row=2, column=0, sticky=tk.W)
        self.load_entry = tk.Entry(attr_frame, width=6)
        self.load_entry.grid(row=2, column=1, padx=5, pady=2)
        self.load_entry.insert(0, "100")

        # 文件操作
        file_frame = tk.LabelFrame(parent, text="文件操作", padx=5, pady=5)
        file_frame.pack(fill=tk.X, padx=5, pady=5)

        tk.Label(file_frame, text="地图名称:").pack(anchor=tk.W)
        self.name_entry = tk.Entry(file_frame)
        self.name_entry.pack(fill=tk.X, pady=2)
        self.name_entry.insert(0, "mine_map")

        save_btn = tk.Button(file_frame, text="保存地图", command=self.save_map)
        save_btn.pack(fill=tk.X, pady=2)

        load_btn = tk.Button(file_frame, text="加载地图", command=self.load_map)
        load_btn.pack(fill=tk.X, pady=2)

        clear_btn = tk.Button(file_frame, text="清空地图", command=self.clear_map)
        clear_btn.pack(fill=tk.X, pady=2)

        # 状态信息
        self.status_label = tk.Label(parent, text="就绪", bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X)

    def create_canvas(self, parent):
        # 创建带滚动条的画布
        self.canvas = tk.Canvas(parent, bg="white")
        h_scrollbar = ttk.Scrollbar(parent, orient=tk.HORIZONTAL, command=self.canvas.xview)
        v_scrollbar = ttk.Scrollbar(parent, orient=tk.VERTICAL, command=self.canvas.yview)

        h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.canvas.configure(xscrollcommand=h_scrollbar.set, yscrollcommand=v_scrollbar.set)

        # 绑定事件
        self.canvas.bind("<ButtonPress-1>", self.on_canvas_click)
        self.canvas.bind("<ButtonRelease-1>", self.on_canvas_release)
        self.canvas.bind("<B1-Motion>", self.on_canvas_drag)
        self.canvas.bind("<MouseWheel>", self.on_mousewheel)  # Windows
        self.canvas.bind("<Button-4>", self.on_mousewheel)  # Linux scroll up
        self.canvas.bind("<Button-5>", self.on_mousewheel)  # Linux scroll down

    def init_map(self):
        """初始化地图"""
        self.draw_grid()
        self.set_status("地图已初始化")

    def normalize_point_format(self, point):
        """标准化点位格式，确保返回字典格式"""
        if isinstance(point, list):
            # 列表格式 [row, col, theta] 或 [row, col, theta, capacity]
            if len(point) >= 2:
                result = {
                    "x": point[1],  # col -> x
                    "y": point[0],  # row -> y
                    "theta": point[2] if len(point) > 2 else 0.0
                }
                if len(point) > 3:
                    result["capacity"] = point[3]
                return result
        elif isinstance(point, dict):
            # 字典格式，直接返回副本
            return point.copy()
        else:
            # 其他格式，返回默认值
            return {"x": 0, "y": 0, "theta": 0.0}

    def normalize_vehicle_format(self, vehicle):
        """标准化车辆格式"""
        if isinstance(vehicle, list):
            # 列表格式 [row, col, theta]
            if len(vehicle) >= 2:
                return {
                    "id": len(self.vehicles) + 1,
                    "x": vehicle[1],  # col -> x
                    "y": vehicle[0],  # row -> y
                    "theta": vehicle[2] if len(vehicle) > 2 else 0.0,
                    "type": "dump_truck",
                    "max_load": 100.0
                }
        elif isinstance(vehicle, dict):
            # 字典格式，确保有必需字段
            result = vehicle.copy()
            if "id" not in result:
                result["id"] = len(self.vehicles) + 1
            if "type" not in result:
                result["type"] = "dump_truck"
            if "max_load" not in result:
                result["max_load"] = 100.0
            return result
        
        # 默认值
        return {
            "id": len(self.vehicles) + 1,
            "x": 0,
            "y": 0,
            "theta": 0.0,
            "type": "dump_truck",
            "max_load": 100.0
        }

    def draw_grid(self):
        """绘制网格和所有元素"""
        self.canvas.delete("all")
        
        # 计算画布大小并设置滚动区域
        canvas_width = self.cols * self.cell_size
        canvas_height = self.rows * self.cell_size
        self.canvas.configure(scrollregion=(0, 0, canvas_width, canvas_height))
        
        # 绘制网格线
        for i in range(0, canvas_width + 1, self.cell_size):
            self.canvas.create_line(i, 0, i, canvas_height, fill="lightgray", width=1)
        for i in range(0, canvas_height + 1, self.cell_size):
            self.canvas.create_line(0, i, canvas_width, i, fill="lightgray", width=1)
        
        # 绘制障碍物
        for row in range(self.rows):
            for col in range(self.cols):
                if self.grid[row, col] == 1:
                    x1 = col * self.cell_size
                    y1 = row * self.cell_size
                    x2 = x1 + self.cell_size
                    y2 = y1 + self.cell_size
                    self.canvas.create_rectangle(x1, y1, x2, y2, fill="black", outline="gray")
        
        # 绘制装载点
        for i, point in enumerate(self.loading_points):
            # 标准化点位格式
            point = self.normalize_point_format(point)
            x, y = point["x"], point["y"]
            x_center = x * self.cell_size + self.cell_size/2
            y_center = y * self.cell_size + self.cell_size/2
            theta = point.get("theta", 0)
            
            # 绘制圆形
            radius = self.cell_size * 0.8
            self.canvas.create_oval(
                x_center - radius, y_center - radius,
                x_center + radius, y_center + radius,
                fill="green", outline="darkgreen", tags=f"loading_{i}"
            )
            
            # 添加方向指示线
            line_length = radius * 1.5
            direction_x = x_center + line_length * np.cos(theta)
            direction_y = y_center + line_length * np.sin(theta)
            self.canvas.create_line(
                x_center, y_center, direction_x, direction_y,
                fill="white", width=2, arrow=tk.LAST
            )
        
        # 绘制卸载点
        for i, point in enumerate(self.unloading_points):
            # 标准化点位格式
            point = self.normalize_point_format(point)
            x, y = point["x"], point["y"]
            x_center = x * self.cell_size + self.cell_size/2
            y_center = y * self.cell_size + self.cell_size/2
            theta = point.get("theta", 0)
            
            # 绘制矩形
            radius = self.cell_size * 0.7
            self.canvas.create_rectangle(
                x_center - radius, y_center - radius,
                x_center + radius, y_center + radius,
                fill="red", outline="darkred", tags=f"unloading_{i}"
            )
            
            # 添加方向指示线
            line_length = radius * 1.5
            direction_x = x_center + line_length * np.cos(theta)
            direction_y = y_center + line_length * np.sin(theta)
            self.canvas.create_line(
                x_center, y_center, direction_x, direction_y,
                fill="white", width=2, arrow=tk.LAST
            )
        
        # 绘制停车区
        for i, point in enumerate(self.parking_areas):
            # 标准化点位格式
            point = self.normalize_point_format(point)
            x, y = point["x"], point["y"]
            x_center = x * self.cell_size + self.cell_size/2
            y_center = y * self.cell_size + self.cell_size/2
            theta = point.get("theta", 0)
            capacity = point.get("capacity", 5)
            
            # 绘制六边形
            radius = self.cell_size * 0.8
            points = []
            for i in range(6):
                angle = i * np.pi / 3
                px = x_center + radius * np.cos(angle)
                py = y_center + radius * np.sin(angle)
                points.extend([px, py])
                
            self.canvas.create_polygon(
                points, fill="blue", outline="darkblue"
            )
            
            # 添加标签
            self.canvas.create_text(
                x_center, y_center,
                text=f"P{capacity}",
                fill="white", font=("Arial", 7, "bold")
            )
        
        # 绘制车辆
        for i, vehicle in enumerate(self.vehicles):
            # 标准化车辆格式
            vehicle = self.normalize_vehicle_format(vehicle)
            x, y = vehicle["x"], vehicle["y"]
            x_center = x * self.cell_size + self.cell_size/2
            y_center = y * self.cell_size + self.cell_size/2
            theta = vehicle.get("theta", 0)
            
            # 绘制三角形表示车辆
            r = self.cell_size * 0.8
            x1 = x_center + r * np.cos(theta)
            y1 = y_center + r * np.sin(theta)
            x2 = x_center + r * np.cos(theta + 2.3)
            y2 = y_center + r * np.sin(theta + 2.3)
            x3 = x_center + r * np.cos(theta - 2.3)
            y3 = y_center + r * np.sin(theta - 2.3)
            
            self.canvas.create_polygon(
                x1, y1, x2, y2, x3, y3,
                fill="orange", outline="darkorange"
            )

    def update_current_tool(self):
        """更新当前工具"""
        self.current_tool = self.tool_var.get()
        self.set_status(f"当前工具: {self.current_tool}")

    def update_brush_size(self, value):
        """更新画笔大小"""
        self.brush_size = int(value)

    def on_canvas_click(self, event):
        """处理画布点击事件"""
        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)
        
        col = int(canvas_x // self.cell_size)
        row = int(canvas_y // self.cell_size)
        
        if 0 <= col < self.cols and 0 <= row < self.rows:
            self.last_painted_cell = (row, col)
            self.apply_tool(row, col)

    def on_canvas_release(self, event):
        """处理鼠标释放事件"""
        self.last_painted_cell = None

    def on_canvas_drag(self, event):
        """处理画布拖动事件"""
        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)
        
        col = int(canvas_x // self.cell_size)
        row = int(canvas_y // self.cell_size)
        
        if 0 <= col < self.cols and 0 <= row < self.rows and (row, col) != self.last_painted_cell:
            self.last_painted_cell = (row, col)
            self.apply_tool(row, col)

    def apply_tool(self, row, col):
        """应用当前工具到指定位置"""
        if self.current_tool == "obstacle":
            self.paint_obstacle(row, col)
        elif self.current_tool == "passable":
            self.clear_obstacle(row, col)
        else:
            self.place_special_point(row, col)

    def paint_obstacle(self, center_row, center_col):
        """绘制障碍物"""
        radius = self.brush_size
        
        # 将圆形区域内的点设为障碍物
        for r in range(max(0, center_row - radius), min(self.rows, center_row + radius + 1)):
            for c in range(max(0, center_col - radius), min(self.cols, center_col + radius + 1)):
                if ((r - center_row) ** 2 + (c - center_col) ** 2) <= radius ** 2:
                    self.grid[r, c] = 1
                    self.remove_special_at_position(c, r)
        
        self.draw_grid()

    def clear_obstacle(self, center_row, center_col):
        """清除障碍物"""
        radius = self.brush_size
        
        # 将圆形区域内的点设为可通行
        for r in range(max(0, center_row - radius), min(self.rows, center_row + radius + 1)):
            for c in range(max(0, center_col - radius), min(self.cols, center_col + radius + 1)):
                if ((r - center_row) ** 2 + (c - center_col) ** 2) <= radius ** 2:
                    self.grid[r, c] = 0
        
        self.draw_grid()

    def place_special_point(self, row, col):
        """放置特殊点"""
        # 移除该位置的任何特殊点
        self.remove_special_at_position(col, row)
        
        # 确保该位置可通行
        self.grid[row, col] = 0
        
        # 获取角度
        try:
            theta_deg = float(self.theta_entry.get())
            theta = theta_deg * np.pi / 180  # 转换为弧度
        except ValueError:
            theta = 0.0
        
        if self.current_tool == "loading":
            self.loading_points.append({
                "x": col,
                "y": row,
                "theta": theta
            })
        elif self.current_tool == "unloading":
            self.unloading_points.append({
                "x": col,
                "y": row,
                "theta": theta
            })
        elif self.current_tool == "parking":
            try:
                capacity = int(self.capacity_entry.get())
            except ValueError:
                capacity = 5
                
            self.parking_areas.append({
                "x": col,
                "y": row,
                "theta": theta,
                "capacity": capacity
            })
        elif self.current_tool == "vehicle":
            try:
                max_load = float(self.load_entry.get())
            except ValueError:
                max_load = 100.0
                
            vehicle_id = len(self.vehicles) + 1
            self.vehicles.append({
                "id": vehicle_id,
                "x": col,
                "y": row,
                "theta": theta,
                "type": "dump_truck",
                "max_load": max_load
            })
        
        self.draw_grid()

    def remove_special_at_position(self, x, y):
        """移除指定位置的特殊点"""
        self.loading_points = [p for p in self.loading_points 
                              if not (self.normalize_point_format(p)["x"] == x and 
                                     self.normalize_point_format(p)["y"] == y)]
        self.unloading_points = [p for p in self.unloading_points 
                                if not (self.normalize_point_format(p)["x"] == x and 
                                       self.normalize_point_format(p)["y"] == y)]
        self.parking_areas = [p for p in self.parking_areas 
                             if not (self.normalize_point_format(p)["x"] == x and 
                                    self.normalize_point_format(p)["y"] == y)]
        self.vehicles = [v for v in self.vehicles 
                        if not (self.normalize_vehicle_format(v)["x"] == x and 
                               self.normalize_vehicle_format(v)["y"] == y)]

    def update_map_size(self):
        """更新地图大小"""
        try:
            new_cols = int(self.col_entry.get())
            new_rows = int(self.row_entry.get())
            
            if new_rows <= 0 or new_cols <= 0:
                messagebox.showerror("错误", "宽度和高度必须大于0")
                return
            
            # 创建新网格
            new_grid = np.zeros((new_rows, new_cols), dtype=np.int8)
            
            # 复制现有网格数据
            min_rows = min(self.rows, new_rows)
            min_cols = min(self.cols, new_cols)
            new_grid[:min_rows, :min_cols] = self.grid[:min_rows, :min_cols]
            
            # 过滤超出新范围的特殊点
            self.loading_points = [p for p in self.loading_points 
                                  if (self.normalize_point_format(p)["x"] < new_cols and 
                                     self.normalize_point_format(p)["y"] < new_rows)]
            self.unloading_points = [p for p in self.unloading_points 
                                    if (self.normalize_point_format(p)["x"] < new_cols and 
                                       self.normalize_point_format(p)["y"] < new_rows)]
            self.parking_areas = [p for p in self.parking_areas 
                                 if (self.normalize_point_format(p)["x"] < new_cols and 
                                    self.normalize_point_format(p)["y"] < new_rows)]
            self.vehicles = [v for v in self.vehicles 
                            if (self.normalize_vehicle_format(v)["x"] < new_cols and 
                               self.normalize_vehicle_format(v)["y"] < new_rows)]
            
            # 更新网格
            self.grid = new_grid
            self.rows = new_rows
            self.cols = new_cols
            
            self.draw_grid()
            self.set_status(f"地图大小已更新为 {new_cols}x{new_rows}")
            
        except ValueError:
            messagebox.showerror("错误", "请输入有效的数值")

    def save_map(self):
        """保存地图"""
        map_name = self.name_entry.get()
        if not map_name:
            messagebox.showerror("错误", "请输入地图名称")
            return
        
        # 标准化所有点位数据
        normalized_loading = [self.normalize_point_format(p) for p in self.loading_points]
        normalized_unloading = [self.normalize_point_format(p) for p in self.unloading_points]
        normalized_parking = [self.normalize_point_format(p) for p in self.parking_areas]
        normalized_vehicles = [self.normalize_vehicle_format(v) for v in self.vehicles]
        
        # 创建符合系统格式的数据
        env_data = {
            "dimensions": {
                "rows": self.rows,
                "cols": self.cols
            },
            "width": self.cols,  # 保留这些直接字段以向后兼容
            "height": self.rows,
            "resolution": self.resolution,
            
            # 修复: 确保以正确方向存储网格，对应于GUI中的期望
            "grid": self.grid.tolist(),  # 存储为(rows, cols)格式
            
            # 修复: 确保坐标正确
            # 注意：保持[row, col, theta]格式，这是mine_loader期望的格式
            "loading_points": [[p["y"], p["x"], p.get("theta", 0.0)] for p in normalized_loading],
            "unloading_points": [[p["y"], p["x"], p.get("theta", 0.0)] for p in normalized_unloading],
            "parking_areas": [[p["y"], p["x"], p.get("theta", 0.0), p.get("capacity", 5)] for p in normalized_parking],
            "vehicle_positions": [[v["y"], v["x"], v.get("theta", 0.0)] for v in normalized_vehicles],
            
            # 修复: 添加直接的障碍物列表，确保GUI可以直接使用
            "obstacles": self.convert_grid_to_obstacles(),
            "vehicles_info": normalized_vehicles
        }
        
        # 保存标准格式文件 (mine_json格式)
        mine_filename = f"{map_name}_mine.json"
        with open(mine_filename, 'w') as f:
            json.dump(env_data, f, indent=2)
        
        # 保存内部格式文件
        internal_filename = f"{map_name}_internal.json"
        with open(internal_filename, 'w') as f:
            json.dump({
                "width": self.cols,
                "height": self.rows,
                "resolution": self.resolution,
                "grid": self.grid.tolist(),
                "loading_points": normalized_loading,
                "unloading_points": normalized_unloading,
                "parking_areas": normalized_parking,
                "vehicles": normalized_vehicles
            }, f, indent=2)
        
        # 保存场景文件用于任务设置
        scen_filename = f"{map_name}.scen"
        scen_data = {
            "scenarios": [
                {
                    "id": v["id"],
                    "initial_position": [v["y"], v["x"], v.get("theta", 0.0)],
                    "type": v.get("type", "dump_truck"),
                    "max_load": v.get("max_load", 100.0)
                }
                for v in normalized_vehicles
            ]
        }
        with open(scen_filename, 'w') as f:
            json.dump(scen_data, f, indent=2)
        
        messagebox.showinfo("保存成功", f"地图已保存为: {mine_filename}\n场景文件: {scen_filename}")
        self.set_status(f"地图已保存: {mine_filename}")
    
    def convert_grid_to_obstacles(self):
        """将网格转换为障碍物列表 - 分离成单点障碍物"""
        obstacles = []
        for row in range(self.rows):
            for col in range(self.cols):
                if self.grid[row, col] == 1:
                    obstacles.append({
                        "x": col,  # x对应列
                        "y": row,  # y对应行
                        "width": 1,
                        "height": 1
                    })
        return obstacles
    
    def convert_grid_to_rectangles(self):
        """将点阵障碍物转换为小矩形列表"""
        obstacles = []
        visited = np.zeros_like(self.grid, dtype=bool)
        
        for row in range(self.rows):
            for col in range(self.cols):
                if self.grid[row, col] == 1 and not visited[row, col]:
                    # 找到一个未访问的障碍物点
                    min_row, min_col = row, col
                    max_row, max_col = row, col
                    
                    # 使用BFS寻找连接的障碍物区域
                    queue = [(row, col)]
                    visited[row, col] = True
                    
                    while queue:
                        r, c = queue.pop(0)
                        
                        # 更新边界
                        min_row = min(min_row, r)
                        max_row = max(max_row, r)
                        min_col = min(min_col, c)
                        max_col = max(max_col, c)
                        
                        # 检查相邻点
                        for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                            nr, nc = r + dr, c + dc
                            if (0 <= nr < self.rows and 0 <= nc < self.cols and 
                                self.grid[nr, nc] == 1 and not visited[nr, nc]):
                                visited[nr, nc] = True
                                queue.append((nr, nc))
                    
                    # 矩形宽高
                    width = max_col - min_col + 1
                    height = max_row - min_row + 1
                    
                    # 添加矩形
                    obstacles.append({
                        "x": min_col,
                        "y": min_row,
                        "width": width,
                        "height": height
                    })
        
        return obstacles

    def validate_points(self, points_list, with_capacity=False):
        """验证点位数据格式"""
        validated = []
        for point in points_list:
            normalized = self.normalize_point_format(point)
            if with_capacity and "capacity" in point:
                normalized["capacity"] = int(point["capacity"])
            validated.append(normalized)
        return validated

    def validate_vehicles(self, vehicles_list):
        """验证车辆数据格式"""
        return [self.normalize_vehicle_format(v) for v in vehicles_list]

    def load_mine_map(self, data):
        """加载标准矿山格式的数据"""
        # 获取尺寸
        self.rows = data.get("dimensions", {}).get("rows", self.rows)
        self.cols = data.get("dimensions", {}).get("cols", self.cols)
        
        # 更新UI输入框
        self.col_entry.delete(0, tk.END)
        self.col_entry.insert(0, str(self.cols))
        self.row_entry.delete(0, tk.END)
        self.row_entry.insert(0, str(self.rows))
        
        # 创建新网格
        if "grid" in data:
            # 直接加载网格数据
            self.grid = np.array(data["grid"], dtype=np.int8)
        else:
            # 网格不存在，重置并从障碍物填充
            self.grid = np.zeros((self.rows, self.cols), dtype=np.int8)
            
            # 从矩形障碍物设置网格
            for obstacle in data.get("obstacles", []):
                x = int(obstacle["x"])
                y = int(obstacle["y"])
                width = int(obstacle.get("width", 1))
                height = int(obstacle.get("height", 1))
                
                # 填充网格障碍物
                for r in range(max(0, y), min(self.rows, y + height)):
                    for c in range(max(0, x), min(self.cols, x + width)):
                        self.grid[r, c] = 1
        
        # 清除现有特殊点
        self.loading_points = []
        self.unloading_points = []
        self.parking_areas = []
        self.vehicles = []
        
        # 加载并标准化所有点位数据
        for point in data.get("loading_points", []):
            self.loading_points.append(self.normalize_point_format(point))
        
        for point in data.get("unloading_points", []):
            self.unloading_points.append(self.normalize_point_format(point))
        
        for point in data.get("parking_areas", []):
            self.parking_areas.append(self.normalize_point_format(point))
        
        # 加载车辆
        for point in data.get("vehicle_positions", []):
            self.vehicles.append(self.normalize_vehicle_format(point))
        
        for vehicle in data.get("vehicles_info", []):
            # 避免重复添加
            if not any(v.get("id") == vehicle.get("id") for v in self.vehicles):
                self.vehicles.append(self.normalize_vehicle_format(vehicle))

    def load_map(self):
        """加载地图文件"""
        file_path = filedialog.askopenfilename(
            filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")]
        )
        
        if not file_path:
            return
            
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # 决定文件类型并标准化所有数据
            if "grid" in data and "loading_points" in data:
                # 内部格式 - 但仍需要标准化
                self.cols = data.get("width", 100)
                self.rows = data.get("height", 100)
                self.resolution = data.get("resolution", 1.0)
                
                # 更新UI输入框
                self.col_entry.delete(0, tk.END)
                self.col_entry.insert(0, str(self.cols))
                self.row_entry.delete(0, tk.END)
                self.row_entry.insert(0, str(self.rows))
                
                # 加载网格
                self.grid = np.array(data["grid"], dtype=np.int8)
                
                # 标准化并加载特殊点
                self.loading_points = [self.normalize_point_format(p) for p in data.get("loading_points", [])]
                self.unloading_points = [self.normalize_point_format(p) for p in data.get("unloading_points", [])]
                self.parking_areas = [self.normalize_point_format(p) for p in data.get("parking_areas", [])]
                self.vehicles = [self.normalize_vehicle_format(v) for v in data.get("vehicles", [])]
                
            elif "dimensions" in data:
                # 标准矿山格式
                self.load_mine_map(data)
                
            else:
                # 系统格式 - 需要完全标准化
                self.cols = data.get("width", 100)
                self.rows = data.get("height", 100)
                self.resolution = data.get("resolution", 1.0)
                
                # 更新UI输入框
                self.col_entry.delete(0, tk.END)
                self.col_entry.insert(0, str(self.cols))
                self.row_entry.delete(0, tk.END)
                self.row_entry.insert(0, str(self.rows))
                
                # 创建新网格
                self.grid = np.zeros((self.rows, self.cols), dtype=np.int8)
                
                # 从矩形障碍物设置网格
                for obstacle in data.get("obstacles", []):
                    x = int(obstacle["x"])
                    y = int(obstacle["y"])
                    width = int(obstacle.get("width", 1))
                    height = int(obstacle.get("height", 1))
                    
                    # 填充网格障碍物
                    for r in range(max(0, y), min(self.rows, y + height)):
                        for c in range(max(0, x), min(self.cols, x + width)):
                            self.grid[r, c] = 1
                
                # 标准化并加载特殊点
                self.loading_points = [self.normalize_point_format(p) for p in data.get("loading_points", [])]
                self.unloading_points = [self.normalize_point_format(p) for p in data.get("unloading_points", [])]
                self.parking_areas = [self.normalize_point_format(p) for p in data.get("parking_areas", [])]
                self.vehicles = [self.normalize_vehicle_format(v) for v in data.get("vehicles", [])]
            
            # 更新地图名称
            map_name = os.path.basename(file_path).split('.')[0]
            if map_name.endswith('_internal'):
                map_name = map_name[:-9]
            elif map_name.endswith('_mine'):
                map_name = map_name[:-5]
            self.name_entry.delete(0, tk.END)
            self.name_entry.insert(0, map_name)
            
            # 重绘地图
            self.draw_grid()
            
            self.set_status(f"已加载地图: {file_path}")
            
        except Exception as e:
            messagebox.showerror("错误", f"加载地图失败: {str(e)}")
            import traceback
            traceback.print_exc()

    def clear_map(self):
        """清空地图"""
        reply = messagebox.askyesno("确认", "确定要清空地图吗？")
        if reply:
            self.grid = np.zeros((self.rows, self.cols), dtype=np.int8)
            self.loading_points = []
            self.unloading_points = []
            self.parking_areas = []
            self.vehicles = []
            self.draw_grid()
            self.set_status("地图已清空")

    def on_mousewheel(self, event):
        """鼠标滚轮缩放"""
        if event.num == 4 or event.delta > 0:  # 向上滚动
            self.cell_size = min(20, self.cell_size + 1)
        elif event.num == 5 or event.delta < 0:  # 向下滚动
            self.cell_size = max(2, self.cell_size - 1)
        
        self.draw_grid()

    def set_status(self, message):
        """设置状态栏消息"""
        self.status_label.config(text=message)

if __name__ == "__main__":
    app = MineMapCreator()
    app.mainloop()