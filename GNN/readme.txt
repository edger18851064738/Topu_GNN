# MAGEC使用指南

## 📋 概述

现在你有两个主要文件：
- `demo.py`: 训练MAGEC模型
- `visualize.py`: 加载训练好的模型并进行动画可视化

## 🚀 快速开始

### 1. 训练模型
```bash
# 完整训练和测试
python demo.py

# 只训练不测试
python demo.py --train_only

# 自定义参数训练
python demo.py --num_episodes 300 --num_agents 6 --graph_name milwaukee --train_only
```

### 2. 可视化已训练的模型
```bash
# 基本可视化
python visualize.py --show_live

# 指定模型和测试图
python visualize.py --model_path results/magec_model.pth --test_graph milwaukee --show_live

# 保存动画GIF
python visualize.py --test_graph ring --save_animation results/animation.gif --max_steps 50

# 在不同图上测试
python visualize.py --test_graph random --max_steps 200 --interval 300 --show_live
```

## 🎮 demo.py 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--num_episodes` | 200 | 训练轮数 |
| `--num_agents` | 4 | 智能体数量 |
| `--graph_name` | small_grid | 训练图类型 |
| `--hidden_size` | 128 | 网络隐藏层大小 |
| `--gnn_layers` | 3 | GNN层数 |
| `--lr` | 3e-4 | 学习率 |
| `--max_cycles` | 200 | 每个episode最大步数 |
| `--train_only` | False | 只训练不测试 |
| `--save_model` | results/magec_model.pth | 模型保存路径 |
| `--check_gpu` | False | 检查GPU信息 |

### 示例命令：
```bash
# 检查GPU
python demo.py --check_gpu

# 快速训练测试
python demo.py --num_episodes 50 --train_only

# 完整训练
python demo.py --num_episodes 500 --num_agents 6 --graph_name milwaukee --hidden_size 256
```

## 🎬 visualize.py 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model_path` | results/magec_model.pth | 模型文件路径 |
| `--test_graph` | milwaukee | 测试图类型 |
| `--max_steps` | 100 | 测试最大步数 |
| `--interval` | 500 | 动画帧间隔(毫秒) |
| `--save_animation` | None | 保存动画路径(.gif) |
| `--show_live` | False | 显示实时动画 |
| `--device` | cpu | 计算设备 |

### 示例命令：
```bash
# 实时动画显示
python visualize.py --show_live

# 测试不同图类型
python visualize.py --test_graph ring --show_live
python visualize.py --test_graph random --show_live

# 保存高质量动画
python visualize.py --test_graph milwaukee --max_steps 150 --interval 200 --save_animation results/milwaukee_demo.gif

# GPU加速测试
python visualize.py --device cuda --test_graph milwaukee --max_steps 200 --show_live
```

## 📊 可视化功能

### 动画包含4个子图：
1. **主图**: 实时显示智能体在图上的移动
   - 节点颜色表示闲置时间（红色越深闲置越久）
   - 彩色圆圈表示智能体
   - 淡化轨迹显示移动历史

2. **闲置时间柱状图**: 每个节点的实时闲置时间
   - 蓝色虚线表示平均闲置时间
   - 颜色深度对应闲置程度

3. **累积奖励曲线**: 展示训练效果
   - 实时更新的奖励累积

4. **统计信息面板**: 显示关键指标
   - 当前步数、平均闲置时间
   - 总奖励、活跃智能体数
   - 访问节点数、移动次数、效率

## 🎯 推荐工作流程

### 开发阶段：
```bash
# 1. 快速验证
python demo.py --num_episodes 50 --train_only

# 2. 测试可视化
python visualize.py --show_live

# 3. 调整参数后重新训练
python demo.py --num_episodes 100 --hidden_size 256 --train_only
```

### 最终训练：
```bash
# 1. 完整训练
python demo.py --num_episodes 500 --num_agents 6 --graph_name milwaukee --save_model results/final_model.pth

# 2. 多图测试
python visualize.py --model_path results/final_model.pth --test_graph milwaukee --save_animation results/milwaukee.gif
python visualize.py --model_path results/final_model.pth --test_graph ring --save_animation results/ring.gif
python visualize.py --model_path results/final_model.pth --test_graph random --save_animation results/random.gif
```

## 📁 输出文件

### demo.py 输出：
- `results/magec_model.pth`: 训练好的模型
- `results/training_curves.png`: 训练曲线
- `results/state_episode_*.png`: 训练过程状态图
- `results/test_step_*.png`: 测试过程状态图（如果不是train_only）

### visualize.py 输出：
- `*.gif`: 动画文件（如果指定保存）
- `results/test_report_*.json`: 测试报告

## 🔧 故障排除

### 常见问题：

1. **模型文件不存在**
   ```bash
   # 确保先训练模型
   python demo.py --train_only
   ```

2. **GPU内存不足**
   ```bash
   # 使用CPU或减少批大小
   python demo.py --no-cuda --batch_size 16
   ```

3. **动画太快/太慢**
   ```bash
   # 调整间隔时间
   python visualize.py --interval 1000 --show_live  # 慢一点
   python visualize.py --interval 200 --show_live   # 快一点
   ```

4. **想要更长的测试**
   ```bash
   # 增加测试步数
   python visualize.py --max_steps 300 --show_live
   ```

## 🎨 自定义扩展

你可以轻松扩展功能：
- 在`visualize.py`中添加新的测试图类型
- 修改可视化颜色和样式
- 添加更多统计指标
- 实现不同的动画效果

开始体验MAGEC的强大功能吧！🚀