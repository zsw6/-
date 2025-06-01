# 巡线小车强化学习仿真系统

这是一个使用深度强化学习（DQN）来训练虚拟巡线小车的仿真系统。该系统使用 PyGame 进行可视化，PyTorch 实现深度学习，通过强化学习使小车学会自主巡线。

## 功能特点

- 基于 DQN（深度 Q 网络）的强化学习算法
- 实时 2D 可视化仿真
- 多传感器模拟
- 实时速度显示
- 支持多种预定义路线
- 训练过程可视化
- 模型保存与加载
- 支持训练中断恢复

## 系统要求

- Python 3.6+
- CUDA（可选，用于 GPU 加速）

## 安装说明

1. 克隆仓库：
```bash
git clone [repository-url]
cd line-follower-rl
```

2. 安装依赖：
```bash
pip install -r requirements.txt
```

## 使用方法

### 训练模式

```bash
python main.py --mode train --episodes 1000 --render_interval 50 --batch_size 32
```

参数说明：
- `--episodes`: 训练回合数（默认：1000）
- `--render_interval`: 训练过程中渲染显示的间隔（默认：50）
- `--batch_size`: 训练批次大小（默认：32）

### 测试模式

```bash
python main.py --mode test --test_episodes 5
```

参数说明：
- `--test_episodes`: 测试回合数（默认：5）
- `--load_interrupted`: 使用此参数加载中断时保存的模型

## 控制说明

- ESC 键：终止程序
- 关闭窗口：终止程序

## 项目结构

```
line-follower-rl/
├── main.py          # 主程序入口
├── environment.py   # 仿真环境实现
├── agent.py         # DQN 智能体实现
├── utils.py         # 工具函数
├── requirements.txt # 项目依赖
└── README.md       # 项目文档
```

## 文件说明

- `main.py`: 程序入口，包含训练和测试的主要逻辑
- `environment.py`: 实现仿真环境，包括小车物理模型和传感器系统
- `agent.py`: 实现 DQN 智能体，包括神经网络结构和训练逻辑
- `utils.py`: 包含各种辅助函数，如路线生成和可视化工具

## 特性说明

1. 实时显示
   - 小车位置和方向
   - 传感器读数
   - 左右轮速度
   - 训练进度信息

2. 自动保存功能
   - 训练完成后保存：`line_follower_model.pth`
   - 中断时保存：`line_follower_model_interrupted.pth`

3. 训练可视化
   - 实时显示奖励值
   - 显示训练进度
   - 可视化传感器数据

## 注意事项

1. 性能考虑
   - 帧率限制在 30 FPS
   - 训练时按间隔渲染以提高效率
   - GPU 加速（如果可用）

2. 错误处理
   - 自动保存中断的训练进度
   - 提供清晰的错误提示
   - 优雅地处理异常情况

## 常见问题

1. 如果程序未响应：
   - 检查系统资源使用情况
   - 减小渲染间隔
   - 确保显卡驱动更新

2. 如果训练效果不理想：
   - 增加训练回合数
   - 调整学习率
   - 修改网络结构

## 贡献指南

欢迎提交 Issue 和 Pull Request 来帮助改进项目。请确保：
1. 代码符合 PEP 8 规范
2. 添加适当的注释
3. 更新相关文档

## 许可证

MIT License

## 联系方式

如有问题或建议，请通过以下方式联系：
- 提交 Issue
- 发送邮件至 576469387@qq.com

## 致谢

感谢所有为此项目做出贡献的开发者。
