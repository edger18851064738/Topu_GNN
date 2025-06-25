import asyncio
import json
import time
from typing import Dict, Any, Optional
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, File, UploadFile, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
import uvicorn

from road_network import RoadNetwork
from vehicle_manager import VehicleManager


def create_static_files():
    """创建静态文件目录和基本HTML文件"""
    static_dir = Path("static")
    static_dir.mkdir(exist_ok=True)
    
    # 创建index.html
    index_file = static_dir / "index.html"
    index_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Enhanced GNN Multi-Vehicle Coordination System</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Monaco', 'Consolas', monospace;
            background: linear-gradient(135deg, #1e3c72, #2a5298);
            color: white;
            overflow: hidden;
            height: 100vh;
        }

        .container {
            display: flex;
            height: 100vh;
            gap: 10px;
            padding: 10px;
        }

        .main-panel {
            flex: 1;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 15px;
            padding: 15px;
            backdrop-filter: blur(10px);
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
            display: flex;
            flex-direction: column;
        }

        .side-panel {
            width: 380px;
            background: rgba(0, 0, 0, 0.3);
            border-radius: 15px;
            padding: 15px;
            backdrop-filter: blur(10px);
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
            overflow-y: auto;
            display: flex;
            flex-direction: column;
            gap: 15px;
        }

        .title {
            text-align: center;
            font-size: 18px;
            font-weight: bold;
            margin-bottom: 15px;
            color: #fff;
            text-shadow: 0 2px 4px rgba(0, 0, 0, 0.5);
        }

        .subtitle {
            font-size: 12px;
            margin-top: 5px;
            opacity: 0.8;
        }

        #canvas {
            flex: 1;
            border: 2px solid rgba(255, 255, 255, 0.3);
            border-radius: 10px;
            background: rgba(255, 255, 255, 0.95);
            cursor: crosshair;
            min-height: 400px;
        }

        .controls {
            display: flex;
            gap: 8px;
            margin-top: 10px;
            flex-wrap: wrap;
            justify-content: center;
        }

        .btn {
            padding: 8px 16px;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            font-family: inherit;
            font-size: 11px;
            font-weight: bold;
            transition: all 0.3s ease;
            background: linear-gradient(145deg, #667eea, #764ba2);
            color: white;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
            min-width: 80px;
        }

        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(0, 0, 0, 0.4);
        }

        .btn.active {
            background: linear-gradient(145deg, #f093fb, #f5576c);
        }

        .btn:disabled {
            opacity: 0.5;
            cursor: not-allowed;
            transform: none;
        }

        .file-input-wrapper {
            position: relative;
            display: flex;
            align-items: center;
            gap: 10px;
        }

        .file-input {
            width: 100%;
            padding: 12px;
            border: 2px dashed rgba(255, 255, 255, 0.3);
            border-radius: 8px;
            background: rgba(255, 255, 255, 0.1);
            color: white;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s ease;
            font-size: 12px;
        }

        .file-input:hover {
            border-color: rgba(255, 255, 255, 0.6);
            background: rgba(255, 255, 255, 0.2);
        }

        .file-input.success {
            border-color: #4caf50;
            background: rgba(76, 175, 80, 0.2);
        }

        .file-input.error {
            border-color: #f44336;
            background: rgba(244, 67, 54, 0.2);
        }

        #topologyFile {
            display: none;
        }

        .legend {
            background: rgba(0, 0, 0, 0.7);
            border-radius: 8px;
            padding: 12px;
            font-size: 10px;
        }

        .legend-title {
            font-weight: bold;
            margin-bottom: 8px;
            color: #fff;
            border-bottom: 1px solid rgba(255, 255, 255, 0.2);
            padding-bottom: 4px;
        }

        .legend-section {
            margin: 8px 0;
        }

        .legend-section-title {
            font-weight: bold;
            margin: 8px 0 4px 0;
            color: #ccc;
        }

        .legend-item {
            display: flex;
            align-items: center;
            margin: 3px 0;
            font-size: 9px;
        }

        .legend-color {
            width: 14px;
            height: 14px;
            border-radius: 3px;
            margin-right: 6px;
            border: 1px solid rgba(255, 255, 255, 0.3);
            flex-shrink: 0;
        }

        .stats {
            font-size: 9px;
            line-height: 1.2;
            background: rgba(0, 0, 0, 0.5);
            border-radius: 8px;
            padding: 12px;
            white-space: pre-line;
            font-family: 'Courier New', monospace;
            flex: 1;
            overflow-y: auto;
            min-height: 200px;
        }

        .status-indicator {
            display: inline-block;
            width: 8px;
            height: 8px;
            border-radius: 50%;
            margin-right: 5px;
            animation: pulse 2s infinite;
        }

        @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.5; }
            100% { opacity: 1; }
        }

        .status-connected { background: #4CAF50; }
        .status-disconnected { background: #F44336; }
        .status-connecting { background: #FF9800; }

        .connection-status {
            background: rgba(0, 0, 0, 0.7);
            border-radius: 8px;
            padding: 8px 12px;
            font-size: 11px;
            text-align: center;
        }

        .help-section {
            background: rgba(0, 0, 0, 0.5);
            border-radius: 8px;
            padding: 10px;
            font-size: 9px;
            line-height: 1.3;
        }

        .help-title {
            font-weight: bold;
            margin-bottom: 6px;
            color: #fff;
        }

        .keyboard-shortcuts {
            display: grid;
            grid-template-columns: auto 1fr;
            gap: 2px 8px;
            font-family: monospace;
        }

        .shortcut-key {
            background: rgba(255, 255, 255, 0.1);
            padding: 2px 4px;
            border-radius: 3px;
            font-weight: bold;
        }

        .features-list {
            list-style: none;
            padding: 0;
        }

        .features-list li {
            margin: 2px 0;
            padding-left: 12px;
            position: relative;
        }

        .features-list li::before {
            content: "•";
            position: absolute;
            left: 0;
            color: #4CAF50;
        }

        /* 响应式设计 */
        @media (max-width: 1200px) {
            .side-panel {
                width: 320px;
            }
            
            .btn {
                padding: 6px 12px;
                font-size: 10px;
            }
        }

        @media (max-width: 900px) {
            .container {
                flex-direction: column;
                gap: 5px;
                padding: 5px;
            }
            
            .side-panel {
                width: 100%;
                height: 300px;
                flex-direction: row;
                overflow-x: auto;
                overflow-y: hidden;
            }
            
            .side-panel > * {
                flex-shrink: 0;
                min-width: 250px;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="main-panel">
            <div class="title">
                🚛 Enhanced GNN Multi-Vehicle Coordination System
                <div class="subtitle">
                    Stage 2: Advanced Conflict Resolution & Time-Space Reservations
                </div>
            </div>
            
            <canvas id="canvas" width="800" height="600"></canvas>
            
            <div class="controls">
                <button class="btn" id="startBtn">▶️ Start</button>
                <button class="btn" id="pauseBtn">⏸️ Pause</button>
                <button class="btn" id="resetBtn">🔄 Reset</button>
                <button class="btn active" id="gnnBtn">🧠 GNN Mode</button>
                <button class="btn" id="addBtn">➕ Add Vehicle</button>
                <button class="btn" id="removeBtn">➖ Remove Vehicle</button>
                <button class="btn" id="speedBtn">⚡ Speed: 1x</button>
                <button class="btn" id="debugBtn">🐛 Debug: OFF</button>
            </div>
        </div>

        <div class="side-panel">
            <div class="connection-status">
                <div id="status">
                    <span class="status-indicator status-connecting"></span>
                    Connecting to server...
                </div>
            </div>

            <div class="file-input-wrapper">
                <div class="file-input" id="fileInputLabel" onclick="document.getElementById('topologyFile').click()">
                    📁 Load Stage 1 Topology
                </div>
                <input type="file" id="topologyFile" accept=".json">
            </div>
            <div id="uploadMessage"></div>

            <div class="legend">
                <div class="legend-title">📋 System Legend</div>
                
                <div class="legend-section">
                    <div class="legend-section-title">Special Points:</div>
                    <div class="legend-item">
                        <div class="legend-color" style="background: #4CAF50;"></div>
                        🟢 Loading Points (L0-L5)
                    </div>
                    <div class="legend-item">
                        <div class="legend-color" style="background: #2196F3;"></div>
                        🔵 Unloading Points (U0-U5)
                    </div>
                    <div class="legend-item">
                        <div class="legend-color" style="background: #9E9E9E;"></div>
                        🅿️ Parking Points (P0-P5)
                    </div>
                </div>

                <div class="legend-section">
                    <div class="legend-section-title">Vehicle States:</div>
                    <div class="legend-item">
                        <div class="legend-color" style="background: #9E9E9E;"></div>
                        IDLE (Waiting for task)
                    </div>
                    <div class="legend-item">
                        <div class="legend-color" style="background: #FF9800;"></div>
                        PLANNING (Path finding)
                    </div>
                    <div class="legend-item">
                        <div class="legend-color" style="background: #FFD700;"></div>
                        CONFIRMED (Path ready)
                    </div>
                    <div class="legend-item">
                        <div class="legend-color" style="background: #4CAF50;"></div>
                        MOVING (In transit)
                    </div>
                    <div class="legend-item">
                        <div class="legend-color" style="background: #8BC34A;"></div>
                        LOADING (At load point)
                    </div>
                    <div class="legend-item">
                        <div class="legend-color" style="background: #03A9F4;"></div>
                        UNLOADING (At unload point)
                    </div>
                </div>

                <div class="legend-section">
                    <div class="legend-section-title">Node States:</div>
                    <div class="legend-item">
                        <div class="legend-color" style="background: #E3F2FD;"></div>
                        FREE (Available)
                    </div>
                    <div class="legend-item">
                        <div class="legend-color" style="background: #FFE0B2;"></div>
                        RESERVED (Scheduled)
                    </div>
                    <div class="legend-item">
                        <div class="legend-color" style="background: #FFCDD2;"></div>
                        OCCUPIED (In use)
                    </div>
                </div>
            </div>

            <div class="help-section">
                <div class="help-title">⌨️ Keyboard Shortcuts</div>
                <div class="keyboard-shortcuts">
                    <span class="shortcut-key">Space</span>
                    <span>Start/Pause</span>
                    <span class="shortcut-key">G</span>
                    <span>Toggle GNN</span>
                    <span class="shortcut-key">+/-</span>
                    <span>Add/Remove Vehicle</span>
                    <span class="shortcut-key">R</span>
                    <span>Reset System</span>
                    <span class="shortcut-key">D</span>
                    <span>Debug Mode</span>
                </div>
            </div>

            <div class="help-section">
                <div class="help-title">🖱️ Mouse Controls</div>
                <ul class="features-list">
                    <li>Drag to pan the view</li>
                    <li>Scroll to zoom in/out</li>
                    <li>Hover vehicles for details</li>
                </ul>
            </div>

            <div class="help-section">
                <div class="help-title">🧠 GNN Features</div>
                <ul class="features-list">
                    <li>Node occupancy tracking</li>
                    <li>Edge congestion analysis</li>
                    <li>Time-space reservations</li>
                    <li>Safety buffer management</li>
                    <li>Conflict-free scheduling</li>
                    <li>Dynamic path replanning</li>
                </ul>
            </div>

            <div class="stats" id="stats">
                <span class="status-indicator status-connecting"></span>
                Initializing system...
            </div>
        </div>
    </div>

    <script>
        // 嵌入式前端代码（简化版）
        class VehicleCoordinationFrontend {
            constructor() {
                this.ws = null;
                this.canvas = null;
                this.ctx = null;
                this.currentState = null;
                this.networkData = null;
                this.config = null;
                this.scale = 0.8;
                this.offsetX = 50;
                this.offsetY = 50;
                this.init();
            }
            
            init() {
                this.setupCanvas();
                this.setupWebSocket();
                this.setupEventListeners();
                this.startAnimationLoop();
            }
            
            setupCanvas() {
                this.canvas = document.getElementById('canvas');
                this.ctx = this.canvas.getContext('2d');
                
                const rect = this.canvas.getBoundingClientRect();
                this.canvas.width = rect.width;
                this.canvas.height = rect.height;
            }
            
            setupWebSocket() {
                const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                const wsUrl = `${protocol}//${window.location.host}/ws`;
                
                this.ws = new WebSocket(wsUrl);
                
                this.ws.onopen = () => {
                    this.updateStatus('🟢 Connected');
                    this.sendCommand('get_full_state');
                };
                
                this.ws.onmessage = (event) => {
                    this.handleWebSocketMessage(event);
                };
                
                this.ws.onclose = () => {
                    this.updateStatus('🔴 Disconnected');
                    setTimeout(() => this.setupWebSocket(), 3000);
                };
            }
            
            handleWebSocketMessage(event) {
                try {
                    const message = JSON.parse(event.data);
                    
                    switch (message.type) {
                        case 'full_state':
                            this.currentState = message.data.simulation;
                            this.networkData = message.data.network;
                            this.config = message.data.config;
                            this.updateUI();
                            break;
                            
                        case 'state_update':
                            this.currentState = message.data;
                            this.updateUI();
                            break;
                    }
                } catch (error) {
                    console.error('Error parsing WebSocket message:', error);
                }
            }
            
            setupEventListeners() {
                document.getElementById('startBtn')?.addEventListener('click', () => this.sendCommand('start'));
                document.getElementById('pauseBtn')?.addEventListener('click', () => this.sendCommand('pause'));
                document.getElementById('resetBtn')?.addEventListener('click', () => this.sendCommand('reset'));
                document.getElementById('gnnBtn')?.addEventListener('click', () => this.sendCommand('toggle_gnn'));
                document.getElementById('addBtn')?.addEventListener('click', () => this.sendCommand('add_vehicle'));
                document.getElementById('removeBtn')?.addEventListener('click', () => this.sendCommand('remove_vehicle'));
                document.getElementById('debugBtn')?.addEventListener('click', () => this.sendCommand('toggle_debug'));
                
                // 文件上传
                document.getElementById('topologyFile')?.addEventListener('change', (e) => this.uploadTopology(e));
                
                // 键盘快捷键
                document.addEventListener('keydown', (e) => {
                    switch (e.key) {
                        case ' ':
                            e.preventDefault();
                            if (this.currentState?.is_running) {
                                this.sendCommand('pause');
                            } else {
                                this.sendCommand('start');
                            }
                            break;
                        case 'g':
                            this.sendCommand('toggle_gnn');
                            break;
                        case '+':
                        case '=':
                            this.sendCommand('add_vehicle');
                            break;
                        case '-':
                            this.sendCommand('remove_vehicle');
                            break;
                        case 'r':
                            this.sendCommand('reset');
                            break;
                        case 'd':
                            this.sendCommand('toggle_debug');
                            break;
                    }
                });
            }
            
            startAnimationLoop() {
                const animate = () => {
                    this.draw();
                    requestAnimationFrame(animate);
                };
                requestAnimationFrame(animate);
            }
            
            draw() {
                if (!this.networkData) return;
                
                this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
                this.drawNetwork();
                this.drawSpecialPoints();
                this.drawVehicles();
            }
            
            drawNetwork() {
                // 绘制边
                this.ctx.strokeStyle = '#e0e0e0';
                this.ctx.lineWidth = 1.5;
                
                for (const [nodeId, nodeData] of Object.entries(this.networkData.nodes)) {
                    const [x1, y1] = this.transformPoint(...nodeData.position);
                    
                    for (const neighborId of nodeData.neighbors) {
                        const neighborData = this.networkData.nodes[neighborId];
                        if (!neighborData) continue;
                        
                        const [x2, y2] = this.transformPoint(...neighborData.position);
                        
                        this.ctx.beginPath();
                        this.ctx.moveTo(x1, y1);
                        this.ctx.lineTo(x2, y2);
                        this.ctx.stroke();
                    }
                }
                
                // 绘制节点
                for (const [nodeId, nodeData] of Object.entries(this.networkData.nodes)) {
                    if (this.isSpecialPoint(nodeId)) continue;
                    
                    const [x, y] = this.transformPoint(...nodeData.position);
                    
                    this.ctx.fillStyle = '#e3f2fd';
                    this.ctx.strokeStyle = '#1976d2';
                    this.ctx.lineWidth = 1;
                    
                    this.ctx.beginPath();
                    this.ctx.arc(x, y, 6, 0, 2 * Math.PI);
                    this.ctx.fill();
                    this.ctx.stroke();
                }
            }
            
            drawSpecialPoints() {
                const types = [
                    { key: 'loading', symbol: 'L', color: '#4caf50' },
                    { key: 'unloading', symbol: 'U', color: '#2196f3' },
                    { key: 'parking', symbol: 'P', color: '#9e9e9e' }
                ];
                
                for (const typeInfo of types) {
                    const points = this.networkData.special_points[typeInfo.key];
                    
                    for (const [pointId, pointData] of Object.entries(points)) {
                        const [x, y] = this.transformPoint(...pointData.position);
                        
                        this.ctx.fillStyle = pointData.is_occupied ? 
                            this.darkenColor(typeInfo.color, 0.3) : typeInfo.color;
                        this.ctx.strokeStyle = '#333';
                        this.ctx.lineWidth = 2;
                        
                        if (typeInfo.key === 'loading') {
                            const size = 12;
                            this.ctx.fillRect(x - size/2, y - size/2, size, size);
                            this.ctx.strokeRect(x - size/2, y - size/2, size, size);
                        } else if (typeInfo.key === 'unloading') {
                            this.ctx.beginPath();
                            this.ctx.moveTo(x, y - 8);
                            this.ctx.lineTo(x - 6, y + 4);
                            this.ctx.lineTo(x + 6, y + 4);
                            this.ctx.closePath();
                            this.ctx.fill();
                            this.ctx.stroke();
                        } else {
                            this.ctx.beginPath();
                            this.ctx.arc(x, y, 8, 0, 2 * Math.PI);
                            this.ctx.fill();
                            this.ctx.stroke();
                        }
                        
                        this.ctx.fillStyle = 'white';
                        this.ctx.font = 'bold 10px monospace';
                        this.ctx.textAlign = 'center';
                        this.ctx.textBaseline = 'middle';
                        this.ctx.fillText(typeInfo.symbol, x, y);
                        
                        this.ctx.fillStyle = '#333';
                        this.ctx.font = 'bold 8px monospace';
                        this.ctx.fillText(pointId, x, y - 15);
                    }
                }
            }
            
            drawVehicles() {
                if (!this.currentState?.vehicles) return;
                
                for (const vehicle of this.currentState.vehicles) {
                    const [x, y] = this.transformPoint(...vehicle.position);
                    
                    this.ctx.fillStyle = vehicle.color;
                    this.ctx.strokeStyle = this.getVehicleStrokeColor(vehicle.state);
                    this.ctx.lineWidth = 2;
                    
                    this.ctx.beginPath();
                    this.ctx.arc(x, y, 10, 0, 2 * Math.PI);
                    this.ctx.fill();
                    this.ctx.stroke();
                    
                    this.ctx.fillStyle = 'white';
                    this.ctx.font = 'bold 10px monospace';
                    this.ctx.textAlign = 'center';
                    this.ctx.textBaseline = 'middle';
                    this.ctx.fillText(vehicle.id.toString(), x, y);
                }
            }
            
            transformPoint(x, y) {
                return [x * this.scale + this.offsetX, y * this.scale + this.offsetY];
            }
            
            isSpecialPoint(nodeId) {
                for (const points of Object.values(this.networkData.special_points)) {
                    for (const pointData of Object.values(points)) {
                        if (pointData.node_id === nodeId) return true;
                    }
                }
                return false;
            }
            
            getVehicleStrokeColor(state) {
                const colors = {
                    'loading': '#4caf50',
                    'unloading': '#2196f3',
                    'confirmed': '#ffd700',
                    'moving': 'white',
                    'planning': '#ff9800'
                };
                return colors[state] || '#333';
            }
            
            darkenColor(color, factor) {
                const hex = color.replace('#', '');
                const r = Math.round(parseInt(hex.substr(0, 2), 16) * (1 - factor));
                const g = Math.round(parseInt(hex.substr(2, 2), 16) * (1 - factor));
                const b = Math.round(parseInt(hex.substr(4, 2), 16) * (1 - factor));
                return `rgb(${r}, ${g}, ${b})`;
            }
            
            updateUI() {
                this.updateStats();
                this.updateControlStates();
            }
            
            updateStats() {
                if (!this.currentState) return;
                
                const stats = this.currentState.statistics;
                const statsText = `
╔═══ ENHANCED GNN SYSTEM ═══╗
║ Mode: ${this.config?.use_gnn ? 'GNN Enhanced' : 'Simple Mode  '}   ║
║ Time: ${this.currentState.current_time.toFixed(1).padStart(17)}s ║
║ Vehicles: ${stats.total_vehicles.toString().padStart(13)} ║
║ Status: ${this.currentState.is_running ? 'Running' : 'Paused'.padStart(15)} ║
╠═══ OPERATION STATS ════════╣
║ Completed Cycles: ${stats.total_cycles.toString().padStart(8)} ║
║ Total Distance: ${stats.total_distance.toFixed(1).padStart(10)} ║
║ Wait Time: ${stats.total_wait_time.toFixed(1).padStart(12)}s ║
║ Avg Cycle Time: ${stats.avg_cycle_time.toFixed(1).padStart(10)}s ║
╚════════════════════════════╝

Ready to use! Try the controls above.
                `.trim();
                
                const statsElement = document.getElementById('stats');
                if (statsElement) {
                    statsElement.textContent = statsText;
                }
            }
            
            updateControlStates() {
                const isRunning = this.currentState?.is_running || false;
                
                const startBtn = document.getElementById('startBtn');
                const pauseBtn = document.getElementById('pauseBtn');
                
                if (startBtn) startBtn.disabled = isRunning;
                if (pauseBtn) pauseBtn.disabled = !isRunning;
                
                const gnnBtn = document.getElementById('gnnBtn');
                if (gnnBtn) {
                    gnnBtn.textContent = this.config?.use_gnn ? '🧠 GNN Mode' : '🎯 Simple Mode';
                    gnnBtn.classList.toggle('active', this.config?.use_gnn);
                }
            }
            
            updateStatus(status) {
                const statusElement = document.getElementById('status');
                if (statusElement) {
                    statusElement.innerHTML = status;
                }
            }
            
            sendCommand(command, params = {}) {
                if (this.ws && this.ws.readyState === WebSocket.OPEN) {
                    this.ws.send(JSON.stringify({ command, ...params }));
                }
            }
            
            async uploadTopology(event) {
                const file = event.target.files[0];
                if (!file) return;
                
                const formData = new FormData();
                formData.append('file', file);
                
                const messageDiv = document.getElementById('uploadMessage');
                const fileLabel = document.getElementById('fileInputLabel');
                
                try {
                    fileLabel.className = 'file-input';
                    fileLabel.textContent = '📤 Uploading...';
                    
                    const response = await fetch('/upload_topology', {
                        method: 'POST',
                        body: formData
                    });
                    
                    const result = await response.json();
                    
                    if (result.success) {
                        fileLabel.className = 'file-input success';
                        fileLabel.textContent = '✅ ' + result.filename;
                        messageDiv.innerHTML = '<div class="success-message">Topology loaded successfully!</div>';
                        this.sendCommand('get_full_state');
                    } else {
                        fileLabel.className = 'file-input error';
                        fileLabel.textContent = '❌ Failed to load';
                        messageDiv.innerHTML = '<div class="error-message">' + result.message + '</div>';
                    }
                } catch (error) {
                    fileLabel.className = 'file-input error';
                    fileLabel.textContent = '❌ Upload error';
                    messageDiv.innerHTML = '<div class="error-message">Error: ' + error.message + '</div>';
                }
                
                // 重置文件输入
                event.target.value = '';
                
                // 5秒后恢复默认状态
                setTimeout(() => {
                    fileLabel.className = 'file-input';
                    fileLabel.textContent = '📁 Load Stage 1 Topology';
                    messageDiv.innerHTML = '';
                }, 5000);
            }
        }
        
        // 页面加载完成后初始化
        document.addEventListener('DOMContentLoaded', () => {
            window.frontend = new VehicleCoordinationFrontend();
        });
    </script>
</body>
</html>"""
    
    with open(index_file, 'w', encoding='utf-8') as f:
        f.write(index_content)
    
    print(f"✅ Created index.html: {index_file}")
    
    # 创建简化的frontend.js文件
    frontend_js = static_dir / "frontend.js"
    if not frontend_js.exists():
        with open(frontend_js, 'w', encoding='utf-8') as f:
            f.write('// Frontend JavaScript - loaded from index.html\nconsole.log("Frontend loaded");')
        print(f"✅ Created frontend.js: {frontend_js}")
    
    return True


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时
    print("📁 Ensuring static files exist...")
    create_static_files()
    
    asyncio.create_task(simulation_loop())
    print("🚛 Enhanced GNN Vehicle Coordination System started")
    print("📡 WebSocket endpoint: ws://localhost:8000/ws")
    print("🌐 Web interface: http://localhost:8000")
    
    yield
    
    # 关闭时
    simulation.pause_simulation()
    print("🛑 Simulation engine stopped")


class SimulationEngine:
    """仿真引擎，控制整个仿真过程"""
    
    def __init__(self):
        self.road_network = RoadNetwork()
        self.vehicle_manager = VehicleManager(self.road_network)
        
        # 仿真参数
        self.current_time = 0.0
        self.is_running = False
        self.speed_multiplier = 1.0
        self.use_gnn = True
        self.debug_mode = False
        
        # 时间控制
        self.last_update_time = time.time()
        self.target_fps = 30
        self.dt = 1.0 / self.target_fps
        
        # WebSocket连接管理
        self.websocket_connections: set[WebSocket] = set()
    
    async def add_websocket(self, websocket: WebSocket):
        """添加WebSocket连接"""
        self.websocket_connections.add(websocket)
        
        # 发送初始状态
        await self.send_full_state(websocket)
    
    async def remove_websocket(self, websocket: WebSocket):
        """移除WebSocket连接"""
        self.websocket_connections.discard(websocket)
    
    async def broadcast_state(self):
        """广播状态到所有连接的客户端"""
        if not self.websocket_connections:
            return
        
        state = self.get_simulation_state()
        message = json.dumps({
            'type': 'state_update',
            'data': state
        })
        
        # 广播到所有连接
        disconnected = set()
        for websocket in self.websocket_connections:
            try:
                await websocket.send_text(message)
            except Exception:
                disconnected.add(websocket)
        
        # 移除断开的连接
        for websocket in disconnected:
            self.websocket_connections.discard(websocket)
    
    async def send_full_state(self, websocket: WebSocket):
        """发送完整状态给单个客户端"""
        state = self.get_full_state()
        message = json.dumps({
            'type': 'full_state',
            'data': state
        })
        
        try:
            await websocket.send_text(message)
        except Exception:
            pass
    
    def start_simulation(self):
        """启动仿真"""
        self.is_running = True
        self.last_update_time = time.time()
    
    def pause_simulation(self):
        """暂停仿真"""
        self.is_running = False
    
    def reset_simulation(self):
        """重置仿真"""
        self.is_running = False
        self.current_time = 0.0
        self.vehicle_manager.reset_all()
        self.road_network.update_time(self.current_time)
    
    def set_speed(self, speed: float):
        """设置仿真速度"""
        self.speed_multiplier = max(0.1, min(10.0, speed))
    
    def toggle_gnn_mode(self):
        """切换GNN模式"""
        self.use_gnn = not self.use_gnn
        self.vehicle_manager.toggle_gnn_mode()
    
    def toggle_debug_mode(self):
        """切换调试模式"""
        self.debug_mode = not self.debug_mode
    
    def add_vehicle(self) -> bool:
        """添加车辆"""
        return self.vehicle_manager.add_vehicle()
    
    def remove_vehicle(self) -> bool:
        """移除车辆"""
        return self.vehicle_manager.remove_vehicle()
    
    def load_topology(self, topology_data: Dict[str, Any]) -> bool:
        """加载拓扑数据"""
        success = self.road_network.load_topology_from_json(topology_data)
        if success:
            # 重置仿真状态
            self.reset_simulation()
        return success
    
    async def update_step(self):
        """单步更新仿真"""
        if not self.is_running:
            return
        
        current_real_time = time.time()
        real_dt = current_real_time - self.last_update_time
        self.last_update_time = current_real_time
        
        # 计算仿真时间步长
        sim_dt = real_dt * self.speed_multiplier
        self.current_time += sim_dt
        
        # 更新网络时间
        self.road_network.update_time(self.current_time)
        
        # 更新所有车辆
        self.vehicle_manager.update_all(self.current_time, sim_dt)
        
        # 广播状态更新
        await self.broadcast_state()
    
    def get_simulation_state(self) -> Dict[str, Any]:
        """获取仿真状态（仅变化的数据）"""
        return {
            'current_time': self.current_time,
            'is_running': self.is_running,
            'vehicles': self.vehicle_manager.get_all_states(),
            'statistics': self.vehicle_manager.get_statistics(),
            'network_reservations': {
                'edge_reservations': {
                    edge_key: [
                        {
                            'vehicle_id': r.vehicle_id,
                            'start_time': r.start_time,
                            'end_time': r.end_time,
                            'direction': r.direction
                        }
                        for r in reservations
                    ]
                    for edge_key, reservations in self.road_network.edge_reservations.items()
                },
                'node_reservations': {
                    node_id: [
                        {
                            'vehicle_id': r.vehicle_id,
                            'start_time': r.start_time,
                            'end_time': r.end_time
                        }
                        for r in reservations
                    ]
                    for node_id, reservations in self.road_network.node_reservations.items()
                }
            }
        }
    
    def get_full_state(self) -> Dict[str, Any]:
        """获取完整状态（包括网络结构）"""
        return {
            'simulation': self.get_simulation_state(),
            'network': self.road_network.get_network_state(),
            'config': {
                'use_gnn': self.use_gnn,
                'debug_mode': self.debug_mode,
                'speed_multiplier': self.speed_multiplier
            }
        }


# 创建FastAPI应用
app = FastAPI(title="Enhanced GNN Vehicle Coordination System", lifespan=lifespan)

# 创建仿真引擎实例
simulation = SimulationEngine()

# 静态文件服务
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", response_class=HTMLResponse)
async def get_index():
    """返回主页面"""
    try:
        index_path = Path("static/index.html")
        if index_path.exists():
            return FileResponse(index_path)
        else:
            # 如果文件不存在，先创建
            create_static_files()
            return FileResponse(index_path)
    except Exception as e:
        return HTMLResponse(f"""
        <html>
            <body>
                <h1>🚛 Enhanced GNN Vehicle Coordination System</h1>
                <p>❌ Error loading interface: {str(e)}</p>
                <p>Please check if static files are properly created.</p>
                <script>
                    setTimeout(() => window.location.reload(), 3000);
                </script>
            </body>
        </html>
        """, status_code=500)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket端点，处理实时通信"""
    await websocket.accept()
    await simulation.add_websocket(websocket)
    
    try:
        while True:
            # 接收客户端消息
            message = await websocket.receive_text()
            data = json.loads(message)
            
            # 处理控制命令
            await handle_control_message(data, websocket)
            
    except WebSocketDisconnect:
        await simulation.remove_websocket(websocket)


async def handle_control_message(data: Dict[str, Any], websocket: WebSocket):
    """处理控制消息"""
    command = data.get('command')
    
    if command == 'start':
        simulation.start_simulation()
        
    elif command == 'pause':
        simulation.pause_simulation()
        
    elif command == 'reset':
        simulation.reset_simulation()
        await simulation.send_full_state(websocket)
        
    elif command == 'toggle_gnn':
        simulation.toggle_gnn_mode()
        
    elif command == 'toggle_debug':
        simulation.toggle_debug_mode()
        
    elif command == 'add_vehicle':
        success = simulation.add_vehicle()
        await websocket.send_text(json.dumps({
            'type': 'command_result',
            'command': 'add_vehicle',
            'success': success
        }))
        
    elif command == 'remove_vehicle':
        success = simulation.remove_vehicle()
        await websocket.send_text(json.dumps({
            'type': 'command_result',
            'command': 'remove_vehicle',
            'success': success
        }))
        
    elif command == 'set_speed':
        speed = data.get('speed', 1.0)
        simulation.set_speed(speed)
        
    elif command == 'get_full_state':
        await simulation.send_full_state(websocket)


@app.post("/upload_topology")
async def upload_topology(file: UploadFile = File(...)):
    """上传拓扑文件"""
    try:
        # 读取文件内容
        content = await file.read()
        topology_data = json.loads(content.decode('utf-8'))
        
        # 加载拓扑
        success = simulation.load_topology(topology_data)
        
        if success:
            return {
                'success': True,
                'message': 'Topology loaded successfully',
                'filename': file.filename
            }
        else:
            return {
                'success': False,
                'message': 'Failed to load topology',
                'filename': file.filename
            }
            
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON file")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")


@app.get("/api/state")
async def get_state():
    """获取当前仿真状态"""
    return simulation.get_simulation_state()


@app.get("/api/full_state")
async def get_full_state():
    """获取完整状态"""
    return simulation.get_full_state()


@app.post("/api/control/{command}")
async def control_simulation(command: str, params: Optional[Dict[str, Any]] = None):
    """控制仿真"""
    if command == 'start':
        simulation.start_simulation()
    elif command == 'pause':
        simulation.pause_simulation()
    elif command == 'reset':
        simulation.reset_simulation()
    elif command == 'toggle_gnn':
        simulation.toggle_gnn_mode()
    elif command == 'toggle_debug':
        simulation.toggle_debug_mode()
    elif command == 'add_vehicle':
        return {'success': simulation.add_vehicle()}
    elif command == 'remove_vehicle':
        return {'success': simulation.remove_vehicle()}
    elif command == 'set_speed':
        if params and 'speed' in params:
            simulation.set_speed(params['speed'])
    else:
        raise HTTPException(status_code=400, detail=f"Unknown command: {command}")
    
    return {'success': True}


async def simulation_loop():
    """仿真循环"""
    while True:
        await simulation.update_step()
        await asyncio.sleep(simulation.dt)


from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时
    print("📁 Ensuring static files exist...")
    create_static_files()
    
    asyncio.create_task(simulation_loop())
    print("🚛 Enhanced GNN Vehicle Coordination System started")
    print("📡 WebSocket endpoint: ws://localhost:8000/ws")
    print("🌐 Web interface: http://localhost:8000")
    
    yield
    
    # 关闭时
    simulation.pause_simulation()
    print("🛑 Simulation engine stopped")


def create_static_files():
    """创建静态文件目录和基本HTML文件"""
    static_dir = Path("static")
    static_dir.mkdir(exist_ok=True)
    
    # 如果index.html不存在，创建一个基本的
    index_file = static_dir / "index.html"
    if not index_file.exists():
        with open(index_file, 'w', encoding='utf-8') as f:
            f.write("""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Enhanced GNN Vehicle Coordination System</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .container { max-width: 1200px; margin: 0 auto; }
        .controls { margin: 20px 0; }
        .btn { margin: 5px; padding: 10px 15px; cursor: pointer; }
        .stats { background: #f5f5f5; padding: 15px; margin: 10px 0; }
        #canvas { border: 1px solid #ccc; }
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
        
        <div>
            <input type="file" id="topologyFile" accept=".json" onchange="uploadTopology()">
            <label for="topologyFile">📁 Load Topology</label>
        </div>
        
        <div class="stats">
            <div id="status">🟡 Connecting...</div>
            <div id="stats">No data</div>
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
            sendCommand('get_full_state');
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
        
        function sendCommand(command, params = {}) {
            if (ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({command, ...params}));
            }
        }
        
        function updateDisplay() {
            if (!currentState || !networkData) return;
            
            // Update stats
            const stats = currentState.statistics;
            document.getElementById('stats').innerHTML = `
                Time: ${currentState.current_time.toFixed(1)}s | 
                Vehicles: ${stats.total_vehicles} | 
                Cycles: ${stats.total_cycles} | 
                Running: ${currentState.is_running ? 'Yes' : 'No'}
            `;
            
            // Draw network and vehicles
            drawNetwork();
        }
        
        function drawNetwork() {
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            
            if (!networkData) return;
            
            // Draw nodes
            for (const [nodeId, nodeData] of Object.entries(networkData.nodes)) {
                const [x, y] = nodeData.position;
                const scaledX = x * 0.8 + 50;
                const scaledY = y * 0.8 + 50;
                
                ctx.fillStyle = '#lightblue';
                ctx.beginPath();
                ctx.arc(scaledX, scaledY, 8, 0, 2 * Math.PI);
                ctx.fill();
                
                // Draw edges
                for (const neighbor of nodeData.neighbors) {
                    const neighborData = networkData.nodes[neighbor];
                    if (neighborData) {
                        const [nx, ny] = neighborData.position;
                        const scaledNX = nx * 0.8 + 50;
                        const scaledNY = ny * 0.8 + 50;
                        
                        ctx.strokeStyle = '#gray';
                        ctx.beginPath();
                        ctx.moveTo(scaledX, scaledY);
                        ctx.lineTo(scaledNX, scaledNY);
                        ctx.stroke();
                    }
                }
            }
            
            // Draw special points
            for (const [type, points] of Object.entries(networkData.special_points)) {
                const color = type === 'loading' ? 'green' : 
                             type === 'unloading' ? 'blue' : 'gray';
                
                for (const [pointId, pointData] of Object.entries(points)) {
                    const [x, y] = pointData.position;
                    const scaledX = x * 0.8 + 50;
                    const scaledY = y * 0.8 + 50;
                    
                    ctx.fillStyle = color;
                    ctx.fillRect(scaledX - 10, scaledY - 10, 20, 20);
                    
                    ctx.fillStyle = 'white';
                    ctx.font = '12px Arial';
                    ctx.textAlign = 'center';
                    ctx.fillText(pointId, scaledX, scaledY + 4);
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
                    ctx.arc(scaledX, scaledY, 12, 0, 2 * Math.PI);
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
                    sendCommand('get_full_state');
                } else {
                    alert('Failed to load topology: ' + result.message);
                }
            } catch (error) {
                alert('Error uploading file: ' + error.message);
            }
        }
    </script>
</body>
</html>""")


if __name__ == "__main__":
    # 创建静态文件
    print("📁 Setting up static files...")
    create_static_files()
    
    # 启动服务器
    uvicorn.run(
        "simulation_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )