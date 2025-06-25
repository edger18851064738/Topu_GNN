/**
 * Enhanced GNN Vehicle Coordination Frontend
 * 提供可视化和用户交互功能
 */

class VehicleCoordinationFrontend {
    constructor() {
        this.ws = null;
        this.canvas = null;
        this.ctx = null;
        
        // 状态数据
        this.currentState = null;
        this.networkData = null;
        this.config = null;
        
        // 可视化参数
        this.scale = 1.0;
        this.offsetX = 0;
        this.offsetY = 0;
        this.devicePixelRatio = window.devicePixelRatio || 1;
        
        // 交互状态
        this.isDragging = false;
        this.lastMousePos = { x: 0, y: 0 };
        this.hoveredVehicle = null;
        
        // 动画
        this.animationFrame = null;
        this.lastFrameTime = 0;
        
        // UI状态
        this.showDebugInfo = false;
        this.showReservations = true;
        this.showPaths = true;
        
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
        
        // 设置高DPI支持
        const rect = this.canvas.getBoundingClientRect();
        this.canvas.width = rect.width * this.devicePixelRatio;
        this.canvas.height = rect.height * this.devicePixelRatio;
        this.ctx.scale(this.devicePixelRatio, this.devicePixelRatio);
        
        // 设置初始变换
        this.setupTransform();
    }
    
    setupTransform() {
        // 计算网络边界
        if (!this.networkData) return;
        
        let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
        
        for (const [nodeId, nodeData] of Object.entries(this.networkData.nodes)) {
            const [x, y] = nodeData.position;
            minX = Math.min(minX, x);
            minY = Math.min(minY, y);
            maxX = Math.max(maxX, x);
            maxY = Math.max(maxY, y);
        }
        
        const margin = 50;
        const canvasWidth = this.canvas.width / this.devicePixelRatio;
        const canvasHeight = this.canvas.height / this.devicePixelRatio;
        
        const networkWidth = maxX - minX;
        const networkHeight = maxY - minY;
        
        this.scale = Math.min(
            (canvasWidth - 2 * margin) / networkWidth,
            (canvasHeight - 2 * margin) / networkHeight
        );
        
        this.offsetX = margin - minX * this.scale + (canvasWidth - networkWidth * this.scale) / 2;
        this.offsetY = margin - minY * this.scale + (canvasHeight - networkHeight * this.scale) / 2;
    }
    
    transformPoint(x, y) {
        return [
            x * this.scale + this.offsetX,
            y * this.scale + this.offsetY
        ];
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
            // 尝试重连
            setTimeout(() => this.setupWebSocket(), 3000);
        };
        
        this.ws.onerror = (error) => {
            console.error('WebSocket error:', error);
            this.updateStatus('❌ Connection Error');
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
                    this.setupTransform();
                    this.updateUI();
                    break;
                    
                case 'state_update':
                    this.currentState = message.data;
                    this.updateUI();
                    break;
                    
                case 'command_result':
                    this.handleCommandResult(message);
                    break;
                    
                default:
                    console.log('Unknown message type:', message.type);
            }
        } catch (error) {
            console.error('Error parsing WebSocket message:', error);
        }
    }
    
    handleCommandResult(message) {
        const { command, success } = message;
        
        if (command === 'add_vehicle' && !success) {
            alert('Cannot add more vehicles. Maximum reached or no parking available.');
        } else if (command === 'remove_vehicle' && !success) {
            alert('Cannot remove vehicle. Minimum vehicle count required.');
        }
    }
    
    setupEventListeners() {
        // 控制按钮
        document.getElementById('startBtn')?.addEventListener('click', () => {
            this.sendCommand('start');
        });
        
        document.getElementById('pauseBtn')?.addEventListener('click', () => {
            this.sendCommand('pause');
        });
        
        document.getElementById('resetBtn')?.addEventListener('click', () => {
            this.sendCommand('reset');
        });
        
        document.getElementById('gnnBtn')?.addEventListener('click', () => {
            this.sendCommand('toggle_gnn');
        });
        
        document.getElementById('addBtn')?.addEventListener('click', () => {
            this.sendCommand('add_vehicle');
        });
        
        document.getElementById('removeBtn')?.addEventListener('click', () => {
            this.sendCommand('remove_vehicle');
        });
        
        document.getElementById('debugBtn')?.addEventListener('click', () => {
            this.sendCommand('toggle_debug');
        });
        
        // 速度控制
        document.getElementById('speedBtn')?.addEventListener('click', () => {
            this.cycleSpeed();
        });
        
        // 画布交互
        this.canvas.addEventListener('mousedown', (e) => this.handleMouseDown(e));
        this.canvas.addEventListener('mousemove', (e) => this.handleMouseMove(e));
        this.canvas.addEventListener('mouseup', (e) => this.handleMouseUp(e));
        this.canvas.addEventListener('wheel', (e) => this.handleWheel(e));
        
        // 窗口大小变化
        window.addEventListener('resize', () => {
            setTimeout(() => {
                this.setupCanvas();
                this.setupTransform();
            }, 100);
        });
        
        // 键盘快捷键
        document.addEventListener('keydown', (e) => this.handleKeyDown(e));
    }
    
    handleMouseDown(e) {
        this.isDragging = true;
        const rect = this.canvas.getBoundingClientRect();
        this.lastMousePos = {
            x: e.clientX - rect.left,
            y: e.clientY - rect.top
        };
    }
    
    handleMouseMove(e) {
        const rect = this.canvas.getBoundingClientRect();
        const currentPos = {
            x: e.clientX - rect.left,
            y: e.clientY - rect.top
        };
        
        if (this.isDragging) {
            // 拖拽平移
            const dx = currentPos.x - this.lastMousePos.x;
            const dy = currentPos.y - this.lastMousePos.y;
            
            this.offsetX += dx;
            this.offsetY += dy;
            
            this.lastMousePos = currentPos;
        } else {
            // 检查悬停的车辆
            this.checkHoveredVehicle(currentPos.x, currentPos.y);
        }
    }
    
    handleMouseUp(e) {
        this.isDragging = false;
    }
    
    handleWheel(e) {
        e.preventDefault();
        
        const rect = this.canvas.getBoundingClientRect();
        const mouseX = e.clientX - rect.left;
        const mouseY = e.clientY - rect.top;
        
        // 缩放因子
        const scaleFactor = e.deltaY > 0 ? 0.9 : 1.1;
        const newScale = this.scale * scaleFactor;
        
        // 限制缩放范围
        if (newScale < 0.1 || newScale > 5.0) return;
        
        // 以鼠标位置为中心缩放
        this.offsetX = mouseX - (mouseX - this.offsetX) * scaleFactor;
        this.offsetY = mouseY - (mouseY - this.offsetY) * scaleFactor;
        this.scale = newScale;
    }
    
    handleKeyDown(e) {
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
    }
    
    checkHoveredVehicle(mouseX, mouseY) {
        if (!this.currentState?.vehicles) {
            this.hoveredVehicle = null;
            return;
        }
        
        for (const vehicle of this.currentState.vehicles) {
            const [vx, vy] = this.transformPoint(vehicle.position[0], vehicle.position[1]);
            
            const dx = Math.abs(mouseX - vx);
            const dy = Math.abs(mouseY - vy);
            
            if (dx < 15 && dy < 15) {
                this.hoveredVehicle = vehicle;
                return;
            }
        }
        
        this.hoveredVehicle = null;
    }
    
    cycleSpeed() {
        const speeds = [0.5, 1, 2, 4];
        const currentSpeed = this.config?.speed_multiplier || 1;
        const currentIndex = speeds.indexOf(currentSpeed);
        const newSpeed = speeds[(currentIndex + 1) % speeds.length];
        
        this.sendCommand('set_speed', { speed: newSpeed });
        
        const speedBtn = document.getElementById('speedBtn');
        if (speedBtn) {
            speedBtn.textContent = `⚡ Speed: ${newSpeed}x`;
        }
    }
    
    sendCommand(command, params = {}) {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify({ command, ...params }));
        }
    }
    
    updateStatus(status) {
        const statusElement = document.getElementById('status');
        if (statusElement) {
            statusElement.innerHTML = status;
        }
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
╠═══ VEHICLE STATES ═════════╣
║ Idle: ${(stats.state_counts.idle || 0).toString().padStart(17)} ║
║ Planning: ${(stats.state_counts.planning || 0).toString().padStart(13)} ║
║ Moving: ${(stats.state_counts.moving || 0).toString().padStart(15)} ║
║ Loading: ${(stats.state_counts.loading || 0).toString().padStart(14)} ║
║ Unloading: ${(stats.state_counts.unloading || 0).toString().padStart(12)} ║
╚════════════════════════════╝

Controls:
Space: Start/Pause | G: Toggle GNN
+/-: Add/Remove Vehicle | R: Reset
D: Debug | Mouse: Pan/Zoom
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
        
        // 更新GNN按钮状态
        const gnnBtn = document.getElementById('gnnBtn');
        if (gnnBtn) {
            gnnBtn.textContent = this.config?.use_gnn ? '🧠 GNN Mode' : '🎯 Simple Mode';
            gnnBtn.classList.toggle('active', this.config?.use_gnn);
        }
        
        // 更新调试按钮状态
        const debugBtn = document.getElementById('debugBtn');
        if (debugBtn) {
            debugBtn.textContent = `🐛 Debug: ${this.config?.debug_mode ? 'ON' : 'OFF'}`;
            debugBtn.classList.toggle('active', this.config?.debug_mode);
        }
    }
    
    startAnimationLoop() {
        const animate = (currentTime) => {
            const deltaTime = currentTime - this.lastFrameTime;
            this.lastFrameTime = currentTime;
            
            this.draw();
            
            this.animationFrame = requestAnimationFrame(animate);
        };
        
        this.animationFrame = requestAnimationFrame(animate);
    }
    
    draw() {
        if (!this.networkData) return;
        
        this.ctx.clearRect(0, 0, this.canvas.width / this.devicePixelRatio, 
                          this.canvas.height / this.devicePixelRatio);
        
        this.drawNetwork();
        this.drawReservations();
        this.drawSpecialPoints();
        this.drawVehicles();
        this.drawHoverInfo();
    }
    
    drawNetwork() {
        // 绘制边
        this.ctx.strokeStyle = '#e0e0e0';
        this.ctx.lineWidth = 1.5;
        
        const drawnEdges = new Set();
        
        for (const [nodeId, nodeData] of Object.entries(this.networkData.nodes)) {
            const [x1, y1] = this.transformPoint(...nodeData.position);
            
            for (const neighborId of nodeData.neighbors) {
                const edgeKey = [nodeId, neighborId].sort().join('-');
                if (drawnEdges.has(edgeKey)) continue;
                drawnEdges.add(edgeKey);
                
                const neighborData = this.networkData.nodes[neighborId];
                if (!neighborData) continue;
                
                const [x2, y2] = this.transformPoint(...neighborData.position);
                
                this.ctx.beginPath();
                this.ctx.moveTo(x1, y1);
                this.ctx.lineTo(x2, y2);
                this.ctx.stroke();
            }
        }
        
        // 绘制普通节点
        for (const [nodeId, nodeData] of Object.entries(this.networkData.nodes)) {
            // 跳过特殊点
            const isSpecialPoint = this.isSpecialPoint(nodeId);
            if (isSpecialPoint) continue;
            
            const [x, y] = this.transformPoint(...nodeData.position);
            const degree = nodeData.neighbors.length;
            
            // 节点状态颜色
            const reservations = nodeData.reservations || [];
            const currentTime = this.currentState?.current_time || 0;
            
            let fillColor = '#e3f2fd'; // 默认蓝色 - 空闲
            let strokeColor = '#1976d2';
            let strokeWidth = 1;
            
            // 检查预订状态
            const hasActiveReservation = reservations.some(r =>
                r.start_time <= currentTime && r.end_time >= currentTime
            );
            const hasFutureReservation = reservations.some(r =>
                r.start_time > currentTime
            );
            
            if (hasActiveReservation) {
                fillColor = '#ffcdd2'; // 红色 - 被占用
                strokeColor = '#d32f2f';
                strokeWidth = 2;
            } else if (hasFutureReservation) {
                fillColor = '#ffe0b2'; // 橙色 - 已预订
                strokeColor = '#f57c00';
                strokeWidth = 2;
            } else if (nodeData.occupancy > 0) {
                fillColor = '#fce4ec'; // 粉色 - 有车辆
                strokeColor = '#c2185b';
                strokeWidth = 2;
            }
            
            const radius = Math.max(4, Math.min(10, 4 + degree * 1.5));
            
            this.ctx.fillStyle = fillColor;
            this.ctx.strokeStyle = strokeColor;
            this.ctx.lineWidth = strokeWidth;
            
            this.ctx.beginPath();
            this.ctx.arc(x, y, radius, 0, 2 * Math.PI);
            this.ctx.fill();
            this.ctx.stroke();
            
            // 节点标签
            if (this.showDebugInfo) {
                this.ctx.fillStyle = '#333';
                this.ctx.font = '10px monospace';
                this.ctx.textAlign = 'center';
                this.ctx.textBaseline = 'bottom';
                this.ctx.fillText(nodeId, x, y - radius - 3);
            }
        }
    }
    
    drawReservations() {
        if (!this.showReservations || !this.currentState?.network_reservations) return;
        
        const currentTime = this.currentState.current_time;
        
        // 绘制边预订
        for (const [edgeKey, reservations] of Object.entries(
            this.currentState.network_reservations.edge_reservations || {})) {
            
            const [node1, node2] = edgeKey.split('-');
            const node1Data = this.networkData.nodes[node1];
            const node2Data = this.networkData.nodes[node2];
            
            if (!node1Data || !node2Data) continue;
            
            const [x1, y1] = this.transformPoint(...node1Data.position);
            const [x2, y2] = this.transformPoint(...node2Data.position);
            
            // 活跃预订
            const activeReservations = reservations.filter(r => r.end_time > currentTime);
            
            for (let i = 0; i < activeReservations.length; i++) {
                const reservation = activeReservations[i];
                const vehicle = this.currentState.vehicles?.find(v => v.id === reservation.vehicle_id);
                if (!vehicle) continue;
                
                // 计算偏移
                const offsetFactor = (i - activeReservations.length / 2 + 0.5) * 0.05;
                const dx = node2Data.position[1] - node1Data.position[1];
                const dy = node1Data.position[0] - node2Data.position[0];
                const len = Math.sqrt(dx * dx + dy * dy);
                const offsetX = (dx / len) * offsetFactor * 50;
                const offsetY = (dy / len) * offsetFactor * 50;
                
                const ox1 = x1 + offsetX;
                const oy1 = y1 + offsetY;
                const ox2 = x2 + offsetX;
                const oy2 = y2 + offsetY;
                
                // 不同样式表示当前vs未来预订
                if (reservation.start_time <= currentTime) {
                    this.ctx.strokeStyle = vehicle.color;
                    this.ctx.lineWidth = 4;
                    this.ctx.globalAlpha = 0.8;
                } else {
                    this.ctx.strokeStyle = vehicle.color;
                    this.ctx.lineWidth = 2;
                    this.ctx.globalAlpha = 0.4;
                    this.ctx.setLineDash([5, 5]);
                }
                
                this.ctx.beginPath();
                this.ctx.moveTo(ox1, oy1);
                this.ctx.lineTo(ox2, oy2);
                this.ctx.stroke();
                
                this.ctx.setLineDash([]);
                this.ctx.globalAlpha = 1.0;
            }
        }
    }
    
    drawSpecialPoints() {
        const types = [
            { key: 'loading', symbol: 'L', color: '#4caf50', strokeColor: '#1b5e20' },
            { key: 'unloading', symbol: 'U', color: '#2196f3', strokeColor: '#0d47a1' },
            { key: 'parking', symbol: 'P', color: '#9e9e9e', strokeColor: '#212121' }
        ];
        
        for (const typeInfo of types) {
            const points = this.networkData.special_points[typeInfo.key];
            
            for (const [pointId, pointData] of Object.entries(points)) {
                const [x, y] = this.transformPoint(...pointData.position);
                
                // 根据状态确定颜色
                let fillColor = typeInfo.color;
                if (pointData.is_occupied) {
                    fillColor = this.darkenColor(typeInfo.color, 0.3);
                } else if (pointData.reserved_by !== null) {
                    fillColor = '#ff9800'; // 橙色表示已预订
                }
                
                this.ctx.fillStyle = fillColor;
                this.ctx.strokeStyle = typeInfo.strokeColor;
                this.ctx.lineWidth = 2;
                
                // 不同形状表示不同类型
                if (typeInfo.key === 'loading') {
                    // 正方形
                    const size = 14;
                    this.ctx.fillRect(x - size/2, y - size/2, size, size);
                    this.ctx.strokeRect(x - size/2, y - size/2, size, size);
                } else if (typeInfo.key === 'unloading') {
                    // 三角形
                    this.ctx.beginPath();
                    this.ctx.moveTo(x, y - 10);
                    this.ctx.lineTo(x - 8, y + 6);
                    this.ctx.lineTo(x + 8, y + 6);
                    this.ctx.closePath();
                    this.ctx.fill();
                    this.ctx.stroke();
                } else {
                    // 圆形
                    this.ctx.beginPath();
                    this.ctx.arc(x, y, 10, 0, 2 * Math.PI);
                    this.ctx.fill();
                    this.ctx.stroke();
                }
                
                // 符号
                this.ctx.fillStyle = 'white';
                this.ctx.font = 'bold 12px monospace';
                this.ctx.textAlign = 'center';
                this.ctx.textBaseline = 'middle';
                this.ctx.fillText(typeInfo.symbol, x, y);
                
                // 点ID
                this.ctx.fillStyle = '#333';
                this.ctx.font = 'bold 10px monospace';
                this.ctx.fillText(pointId, x, y - 18);
                
                // 状态指示
                if (pointData.is_occupied || pointData.reserved_by !== null) {
                    this.ctx.font = '8px monospace';
                    const status = pointData.is_occupied ? 
                        `V${pointData.reserved_by}` : `Rsv:V${pointData.reserved_by}`;
                    this.ctx.fillText(status, x, y + 18);
                }
            }
        }
    }
    
    drawVehicles() {
        if (!this.currentState?.vehicles) return;
        
        for (const vehicle of this.currentState.vehicles) {
            this.drawVehicle(vehicle);
        }
    }
    
    drawVehicle(vehicle) {
        const [x, y] = this.transformPoint(...vehicle.position);
        
        // 车辆大小基于模式
        let width = 16;
        let height = 10;
        
        if (vehicle.mode === 'loaded') {
            width = 20;
            height = 12;
        } else if (vehicle.mode === 'returning') {
            width = 18;
            height = 11;
        }
        
        // 状态相关样式
        let strokeColor = '#333';
        let strokeWidth = 2;
        let alpha = 1.0;
        
        switch (vehicle.state) {
            case 'loading':
                strokeColor = '#4caf50';
                strokeWidth = 4;
                break;
            case 'unloading':
                strokeColor = '#2196f3';
                strokeWidth = 4;
                break;
            case 'confirmed':
                strokeColor = '#ffd700';
                strokeWidth = 3;
                // 脉冲效果
                const pulse = Math.sin(Date.now() * 0.004) * 0.2 + 0.8;
                alpha = pulse;
                break;
            case 'moving':
                strokeColor = 'white';
                strokeWidth = 3;
                break;
            case 'waiting':
                alpha = 0.5;
                break;
            case 'planning':
                strokeColor = '#ff9800';
                strokeWidth = 2;
                break;
        }
        
        // 保存上下文用于旋转
        this.ctx.save();
        this.ctx.translate(x, y);
        this.ctx.rotate(vehicle.heading || 0);
        this.ctx.globalAlpha = alpha;
        
        // 绘制车辆主体
        this.ctx.fillStyle = vehicle.color;
        this.ctx.strokeStyle = strokeColor;
        this.ctx.lineWidth = strokeWidth;
        
        this.ctx.fillRect(-width/2, -height/2, width, height);
        this.ctx.strokeRect(-width/2, -height/2, width, height);
        
        // 绘制前部指示器
        this.ctx.fillStyle = 'white';
        this.ctx.fillRect(width/2 - 3, -height/4, 3, height/2);
        
        // 模式指示器
        if (vehicle.mode === 'loaded') {
            this.ctx.fillStyle = '#ffeb3b';
            this.ctx.fillRect(-width/2 + 2, -height/2 + 2, width - 4, height - 4);
            this.ctx.strokeStyle = '#f57f17';
            this.ctx.lineWidth = 1;
            this.ctx.strokeRect(-width/2 + 2, -height/2 + 2, width - 4, height - 4);
        }
        
        // 车辆ID（不旋转）
        this.ctx.rotate(-(vehicle.heading || 0));
        this.ctx.fillStyle = 'white';
        this.ctx.font = 'bold 10px monospace';
        this.ctx.textAlign = 'center';
        this.ctx.textBaseline = 'middle';
        this.ctx.fillText(vehicle.id.toString(), 0, 0);
        
        this.ctx.restore();
        this.ctx.globalAlpha = 1.0;
        
        // 状态和模式指示符（不旋转）
        const stateSymbol = this.getStateSymbol(vehicle.state);
        const modeSymbol = this.getModeSymbol(vehicle.mode);
        
        this.ctx.fillStyle = '#333';
        this.ctx.font = '9px monospace';
        this.ctx.textAlign = 'center';
        this.ctx.textBaseline = 'top';
        this.ctx.fillText(`${stateSymbol}${modeSymbol}`, x, y + height/2 + 3);
        
        // 绘制目标线
        if (this.showPaths && vehicle.target_node_id) {
            this.drawTargetLine(vehicle);
        }
        
        // 绘制确认路径
        if (this.showPaths && vehicle.path_confirmed && vehicle.path) {
            this.drawConfirmedPath(vehicle);
        }
        
        // 绘制进度条
        if (vehicle.state === 'loading' && vehicle.loading_progress > 0) {
            this.drawProgressBar(x, y - 25, 30, 5, vehicle.loading_progress, '#4caf50');
        } else if (vehicle.state === 'unloading' && vehicle.unloading_progress > 0) {
            this.drawProgressBar(x, y - 25, 30, 5, vehicle.unloading_progress, '#2196f3');
        }
    }
    
    drawTargetLine(vehicle) {
        const targetNodeData = this.networkData.nodes[vehicle.target_node_id];
        if (!targetNodeData) return;
        
        const [x1, y1] = this.transformPoint(...vehicle.position);
        const [x2, y2] = this.transformPoint(...targetNodeData.position);
        
        this.ctx.strokeStyle = vehicle.color;
        this.ctx.lineWidth = 2;
        this.ctx.globalAlpha = 0.5;
        
        if (vehicle.mode === 'empty') {
            this.ctx.strokeStyle = '#4caf50';
            this.ctx.setLineDash([5, 5]);
        } else if (vehicle.mode === 'loaded') {
            this.ctx.strokeStyle = '#2196f3';
            this.ctx.setLineDash([8, 3]);
        } else if (vehicle.mode === 'returning') {
            this.ctx.strokeStyle = '#9e9e9e';
            this.ctx.setLineDash([3, 3]);
        }
        
        this.ctx.beginPath();
        this.ctx.moveTo(x1, y1);
        this.ctx.lineTo(x2, y2);
        this.ctx.stroke();
        
        this.ctx.setLineDash([]);
        this.ctx.globalAlpha = 1.0;
    }
    
    drawConfirmedPath(vehicle) {
        if (!vehicle.path || vehicle.path.length < 2) return;
        
        this.ctx.strokeStyle = vehicle.color;
        this.ctx.lineWidth = 6;
        this.ctx.globalAlpha = 0.3;
        
        this.ctx.beginPath();
        for (let i = 0; i < vehicle.path.length; i++) {
            const nodeId = vehicle.path[i];
            const nodeData = this.networkData.nodes[nodeId];
            if (nodeData) {
                const [px, py] = this.transformPoint(...nodeData.position);
                if (i === 0) {
                    this.ctx.moveTo(px, py);
                } else {
                    this.ctx.lineTo(px, py);
                }
            }
        }
        this.ctx.stroke();
        
        // 路径节点
        this.ctx.globalAlpha = 0.8;
        for (let i = 0; i < vehicle.path.length; i++) {
            const nodeId = vehicle.path[i];
            const nodeData = this.networkData.nodes[nodeId];
            if (nodeData) {
                const [px, py] = this.transformPoint(...nodeData.position);
                
                this.ctx.fillStyle = vehicle.color;
                this.ctx.beginPath();
                this.ctx.arc(px, py, 8, 0, 2 * Math.PI);
                this.ctx.fill();
                
                this.ctx.fillStyle = 'white';
                this.ctx.font = 'bold 8px monospace';
                this.ctx.textAlign = 'center';
                this.ctx.textBaseline = 'middle';
                this.ctx.fillText(i.toString(), px, py);
            }
        }
        
        this.ctx.globalAlpha = 1.0;
    }
    
    drawProgressBar(x, y, width, height, progress, color) {
        // 背景
        this.ctx.fillStyle = 'rgba(0, 0, 0, 0.3)';
        this.ctx.fillRect(x - width/2, y, width, height);
        
        // 进度
        this.ctx.fillStyle = color;
        this.ctx.fillRect(x - width/2, y, width * progress, height);
        
        // 边框
        this.ctx.strokeStyle = 'white';
        this.ctx.lineWidth = 1;
        this.ctx.strokeRect(x - width/2, y, width, height);
    }
    
    drawHoverInfo() {
        if (!this.hoveredVehicle) return;
        
        const vehicle = this.hoveredVehicle;
        const [x, y] = this.transformPoint(...vehicle.position);
        
        // 工具提示背景
        const text = [
            `Vehicle ${vehicle.id}`,
            `State: ${vehicle.state}`,
            `Mode: ${vehicle.mode}`,
            `Heading: ${Math.round(vehicle.heading * 180 / Math.PI)}°`,
            `Cycles: ${vehicle.stats.completed_cycles}`,
            `Distance: ${vehicle.stats.total_distance.toFixed(1)}`
        ];
        
        const lineHeight = 14;
        const padding = 8;
        const maxWidth = Math.max(...text.map(t => this.ctx.measureText(t).width));
        const tooltipWidth = maxWidth + padding * 2;
        const tooltipHeight = text.length * lineHeight + padding * 2;
        
        // 调整位置避免超出画布
        let tooltipX = x + 15;
        let tooltipY = y - tooltipHeight - 10;
        
        const canvasWidth = this.canvas.width / this.devicePixelRatio;
        const canvasHeight = this.canvas.height / this.devicePixelRatio;
        
        if (tooltipX + tooltipWidth > canvasWidth) {
            tooltipX = x - tooltipWidth - 15;
        }
        if (tooltipY < 0) {
            tooltipY = y + 15;
        }
        
        // 绘制工具提示
        this.ctx.fillStyle = 'rgba(0, 0, 0, 0.9)';
        this.ctx.fillRect(tooltipX, tooltipY, tooltipWidth, tooltipHeight);
        
        this.ctx.strokeStyle = 'rgba(255, 255, 255, 0.3)';
        this.ctx.lineWidth = 1;
        this.ctx.strokeRect(tooltipX, tooltipY, tooltipWidth, tooltipHeight);
        
        this.ctx.fillStyle = 'white';
        this.ctx.font = '11px monospace';
        this.ctx.textAlign = 'left';
        this.ctx.textBaseline = 'top';
        
        for (let i = 0; i < text.length; i++) {
            this.ctx.fillText(text[i], tooltipX + padding, tooltipY + padding + i * lineHeight);
        }
    }
    
    getStateSymbol(state) {
        const symbols = {
            'idle': 'I',
            'planning': 'P',
            'waiting': 'W',
            'confirmed': 'C',
            'moving': 'M',
            'loading': 'L',
            'unloading': 'U',
            'blocked': 'B'
        };
        return symbols[state] || '?';
    }
    
    getModeSymbol(mode) {
        const symbols = {
            'parked': '🅿',
            'empty': '○',
            'loaded': '●',
            'returning': '◇'
        };
        return symbols[mode] || '?';
    }
    
    isSpecialPoint(nodeId) {
        for (const points of Object.values(this.networkData.special_points)) {
            for (const pointData of Object.values(points)) {
                if (pointData.node_id === nodeId) {
                    return true;
                }
            }
        }
        return false;
    }
    
    darkenColor(color, factor) {
        // 简单的颜色加深函数
        const hex = color.replace('#', '');
        const r = Math.round(parseInt(hex.substr(0, 2), 16) * (1 - factor));
        const g = Math.round(parseInt(hex.substr(2, 2), 16) * (1 - factor));
        const b = Math.round(parseInt(hex.substr(4, 2), 16) * (1 - factor));
        return `rgb(${r}, ${g}, ${b})`;
    }
}

// 文件上传处理
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

// 页面加载完成后初始化
document.addEventListener('DOMContentLoaded', () => {
    window.frontend = new VehicleCoordinationFrontend();
    
    // 绑定文件上传
    const fileInput = document.getElementById('topologyFile');
    if (fileInput) {
        fileInput.addEventListener('change', uploadTopology);
    }
});