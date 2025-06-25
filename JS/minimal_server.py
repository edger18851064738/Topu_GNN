#!/usr/bin/env python3
"""
最小化服务器 - 如果主服务器有问题时使用
"""

import asyncio
import json
import time
from pathlib import Path

try:
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect, File, UploadFile
    from fastapi.staticfiles import StaticFiles
    from fastapi.responses import HTMLResponse, FileResponse
    import uvicorn
except ImportError as e:
    print(f"❌ Missing FastAPI dependencies: {e}")
    print("Please install: pip install fastapi uvicorn[standard] websockets python-multipart")
    exit(1)

try:
    from road_network import RoadNetwork
    from vehicle_manager import VehicleManager
except ImportError as e:
    print(f"❌ Missing project files: {e}")
    print("Please ensure road_network.py and vehicle_manager.py are in the current directory")
    exit(1)

# 创建应用
app = FastAPI(title="Enhanced GNN Vehicle Coordination System")

# 创建仿真引擎（简化版）
class SimpleSimulationEngine:
    def __init__(self):
        self.road_network = RoadNetwork()
        self.vehicle_manager = VehicleManager(self.road_network)
        self.current_time = 0.0
        self.is_running = False
        self.websocket_connections = set()
    
    async def add_websocket(self, websocket):
        self.websocket_connections.add(websocket)
        await self.send_state(websocket)
    
    async def remove_websocket(self, websocket):
        self.websocket_connections.discard(websocket)
    
    async def send_state(self, websocket):
        try:
            state = {
                'type': 'full_state',
                'data': {
                    'simulation': {
                        'current_time': self.current_time,
                        'is_running': self.is_running,
                        'vehicles': self.vehicle_manager.get_all_states(),
                        'statistics': self.vehicle_manager.get_statistics()
                    },
                    'network': self.road_network.get_network_state(),
                    'config': {'use_gnn': True, 'debug_mode': False}
                }
            }
            await websocket.send_text(json.dumps(state))
        except:
            pass
    
    async def broadcast_state(self):
        if not self.websocket_connections:
            return
        
        state = {
            'type': 'state_update',
            'data': {
                'current_time': self.current_time,
                'is_running': self.is_running,
                'vehicles': self.vehicle_manager.get_all_states(),
                'statistics': self.vehicle_manager.get_statistics()
            }
        }
        
        message = json.dumps(state)
        disconnected = set()
        
        for ws in self.websocket_connections:
            try:
                await ws.send_text(message)
            except:
                disconnected.add(ws)
        
        for ws in disconnected:
            self.websocket_connections.discard(ws)
    
    async def update_step(self):
        if self.is_running:
            dt = 0.033  # ~30 FPS
            self.current_time += dt
            self.road_network.update_time(self.current_time)
            self.vehicle_manager.update_all(self.current_time, dt)
            await self.broadcast_state()

# 创建仿真实例
simulation = SimpleSimulationEngine()

# 创建静态文件目录
static_dir = Path("static")
static_dir.mkdir(exist_ok=True)

# 简单的HTML内容
SIMPLE_HTML = """<!DOCTYPE html>
<html>
<head>
    <title>Enhanced GNN Vehicle Coordination System</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #1e3c72; color: white; }
        .container { max-width: 1200px; margin: 0 auto; }
        .controls { margin: 20px 0; }
        .btn { margin: 5px; padding: 10px 15px; cursor: pointer; background: #4CAF50; color: white; border: none; border-radius: 5px; }
        .btn:hover { background: #45a049; }
        .stats { background: rgba(0,0,0,0.5); padding: 15px; margin: 10px 0; border-radius: 5px; font-family: monospace; }
        #canvas { border: 2px solid #ccc; background: white; border-radius: 5px; }
        .file-upload { margin: 10px 0; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🚛 Enhanced GNN Vehicle Coordination System</h1>
        
        <div class="controls">
            <button class="btn" onclick="sendCommand('start')">▶️ Start</button>
            <button class="btn" onclick="sendCommand('pause')">⏸️ Pause</button>
            <button class="btn" onclick="sendCommand('reset')">🔄 Reset</button>
            <button class="btn" onclick="sendCommand('toggle_gnn')">🧠 Toggle GNN</button>
            <button class="btn" onclick="sendCommand('add_vehicle')">➕ Add Vehicle</button>
            <button class="btn" onclick="sendCommand('remove_vehicle')">➖ Remove Vehicle</button>
        </div>
        
        <div class="file-upload">
            <input type="file" id="topologyFile" accept=".json" onchange="uploadTopology()">
            <label for="topologyFile">📁 Load Topology</label>
        </div>
        
        <div class="stats">
            <div id="status">🟡 Connecting...</div>
            <div id="stats">Initializing...</div>
        </div>
        
        <canvas id="canvas" width="800" height="600"></canvas>
    </div>

    <script>
        const ws = new WebSocket('ws://localhost:8000/ws');
        const canvas = document.getElementById('canvas');
        const ctx = canvas.getContext('2d');
        
        let currentState = null;
        let networkData = null;
        
        ws.onopen = function() {
            document.getElementById('status').innerHTML = '🟢 Connected';
        };
        
        ws.onmessage = function(event) {
            const message = JSON.parse(event.data);
            
            if (message.type === 'full_state') {
                currentState = message.data.simulation;
                networkData = message.data.network;
                updateDisplay();
            } else if (message.type === 'state_update') {
                currentState = message.data;
                updateDisplay();
            }
        };
        
        ws.onclose = function() {
            document.getElementById('status').innerHTML = '🔴 Disconnected';
        };
        
        function sendCommand(command) {
            if (ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({command: command}));
            }
        }
        
        function updateDisplay() {
            if (!currentState || !networkData) return;
            
            const stats = currentState.statistics;
            document.getElementById('stats').innerHTML = 
                'Time: ' + currentState.current_time.toFixed(1) + 's | ' +
                'Vehicles: ' + stats.total_vehicles + ' | ' +
                'Cycles: ' + stats.total_cycles + ' | ' +
                'Running: ' + (currentState.is_running ? 'Yes' : 'No');
            
            drawNetwork();
        }
        
        function drawNetwork() {
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            
            if (!networkData) return;
            
            // Draw network nodes
            for (const [nodeId, nodeData] of Object.entries(networkData.nodes)) {
                const [x, y] = nodeData.position;
                const scaledX = x * 0.8 + 50;
                const scaledY = y * 0.8 + 50;
                
                ctx.fillStyle = '#lightblue';
                ctx.beginPath();
                ctx.arc(scaledX, scaledY, 6, 0, 2 * Math.PI);
                ctx.fill();
                
                // Draw edges
                for (const neighbor of nodeData.neighbors) {
                    const neighborData = networkData.nodes[neighbor];
                    if (neighborData) {
                        const [nx, ny] = neighborData.position;
                        const scaledNX = nx * 0.8 + 50;
                        const scaledNY = ny * 0.8 + 50;
                        
                        ctx.strokeStyle = '#gray';
                        ctx.lineWidth = 1;
                        ctx.beginPath();
                        ctx.moveTo(scaledX, scaledY);
                        ctx.lineTo(scaledNX, scaledNY);
                        ctx.stroke();
                    }
                }
            }
            
            // Draw special points
            for (const [type, points] of Object.entries(networkData.special_points)) {
                const color = type === 'loading' ? '#4CAF50' : 
                             type === 'unloading' ? '#2196F3' : '#9E9E9E';
                
                for (const [pointId, pointData] of Object.entries(points)) {
                    const [x, y] = pointData.position;
                    const scaledX = x * 0.8 + 50;
                    const scaledY = y * 0.8 + 50;
                    
                    ctx.fillStyle = color;
                    if (type === 'loading') {
                        ctx.fillRect(scaledX - 8, scaledY - 8, 16, 16);
                    } else if (type === 'unloading') {
                        ctx.beginPath();
                        ctx.moveTo(scaledX, scaledY - 8);
                        ctx.lineTo(scaledX - 8, scaledY + 8);
                        ctx.lineTo(scaledX + 8, scaledY + 8);
                        ctx.closePath();
                        ctx.fill();
                    } else {
                        ctx.beginPath();
                        ctx.arc(scaledX, scaledY, 8, 0, 2 * Math.PI);
                        ctx.fill();
                    }
                    
                    ctx.fillStyle = 'white';
                    ctx.font = '10px Arial';
                    ctx.textAlign = 'center';
                    ctx.fillText(pointId, scaledX, scaledY + 3);
                }
            }
            
            // Draw vehicles
            if (currentState && currentState.vehicles) {
                for (const vehicle of currentState.vehicles) {
                    const [x, y] = vehicle.position;
                    const scaledX = x * 0.8 + 50;
                    const scaledY = y * 0.8 + 50;
                    
                    ctx.fillStyle = vehicle.color;
                    ctx.beginPath();
                    ctx.arc(scaledX, scaledY, 10, 0, 2 * Math.PI);
                    ctx.fill();
                    
                    ctx.fillStyle = 'white';
                    ctx.font = 'bold 12px Arial';
                    ctx.textAlign = 'center';
                    ctx.fillText(vehicle.id.toString(), scaledX, scaledY + 4);
                }
            }
        }
        
        async function uploadTopology() {
            const fileInput = document.getElementById('topologyFile');
            const file = fileInput.files[0];
            
            if (!file) return;
            
            const formData = new FormData();
            formData.append('file', file);
            
            try {
                const response = await fetch('/upload_topology', {
                    method: 'POST',
                    body: formData
                });
                
                const result = await response.json();
                
                if (result.success) {
                    alert('Topology loaded successfully!');
                } else {
                    alert('Failed to load topology: ' + result.message);
                }
            } catch (error) {
                alert('Error uploading file: ' + error.message);
            }
        }
        
        // Keyboard shortcuts
        document.addEventListener('keydown', function(e) {
            switch(e.key) {
                case ' ':
                    e.preventDefault();
                    sendCommand(currentState?.is_running ? 'pause' : 'start');
                    break;
                case 'g':
                    sendCommand('toggle_gnn');
                    break;
                case '+':
                case '=':
                    sendCommand('add_vehicle');
                    break;
                case '-':
                    sendCommand('remove_vehicle');
                    break;
                case 'r':
                    sendCommand('reset');
                    break;
            }
        });
        
        // Auto-refresh
        setInterval(drawNetwork, 33);
    </script>
</body>
</html>"""

# 创建index.html文件
with open(static_dir / "index.html", 'w', encoding='utf-8') as f:
    f.write(SIMPLE_HTML)

# 静态文件服务
app.mount("/static", StaticFiles(directory="static"), name="static")

# 路由
@app.get("/")
async def get_index():
    return FileResponse("static/index.html")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    await simulation.add_websocket(websocket)
    
    try:
        while True:
            message = await websocket.receive_text()
            data = json.loads(message)
            command = data.get('command')
            
            if command == 'start':
                simulation.is_running = True
            elif command == 'pause':
                simulation.is_running = False
            elif command == 'reset':
                simulation.current_time = 0.0
                simulation.is_running = False
                simulation.vehicle_manager.reset_all()
            elif command == 'toggle_gnn':
                simulation.vehicle_manager.toggle_gnn_mode()
            elif command == 'add_vehicle':
                simulation.vehicle_manager.add_vehicle()
            elif command == 'remove_vehicle':
                simulation.vehicle_manager.remove_vehicle()
            
            await simulation.send_state(websocket)
            
    except WebSocketDisconnect:
        await simulation.remove_websocket(websocket)

@app.post("/upload_topology")
async def upload_topology(file: UploadFile = File(...)):
    try:
        content = await file.read()
        topology_data = json.loads(content.decode('utf-8'))
        success = simulation.road_network.load_topology_from_json(topology_data)
        
        if success:
            simulation.vehicle_manager.reset_all()
            return {'success': True, 'message': 'Topology loaded successfully', 'filename': file.filename}
        else:
            return {'success': False, 'message': 'Failed to load topology', 'filename': file.filename}
    except Exception as e:
        return {'success': False, 'message': f'Error: {str(e)}', 'filename': file.filename}

# 仿真循环
async def simulation_loop():
    while True:
        await simulation.update_step()
        await asyncio.sleep(0.033)

# 启动仿真
@app.on_event("startup")
async def startup():
    asyncio.create_task(simulation_loop())
    print("🚛 Minimal Enhanced GNN Vehicle Coordination System started")
    print("📡 WebSocket: ws://localhost:8000/ws")
    print("🌐 Interface: http://localhost:8000")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")