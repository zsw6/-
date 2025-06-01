import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

class DQN(nn.Module):
    """深度Q网络模型"""
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 24)
        self.fc2 = nn.Linear(24, 24)
        self.fc3 = nn.Linear(24, action_size)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class DQNAgent:
    """DQN智能体"""
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        
        # DQN超参数
        self.gamma = 0.95    # 折扣因子
        self.epsilon = 1.0   # 探索率
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.update_target_frequency = 10  # 目标网络更新频率
        
        # 经验回放缓冲区
        self.memory = deque(maxlen=2000)
        
        # 创建Q网络和目标网络
        self.q_network = DQN(state_size, action_size)
        self.target_network = DQN(state_size, action_size)
        self.update_target_network()  # 初始化目标网络
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()
        
        # 离散动作空间
        self.actions = self._create_action_space()
        
    def _create_action_space(self):
        """创建离散的动作空间"""
        # 左右轮速度组合
        speeds = [-1.0, -0.5, 0.0, 0.5, 1.0]  # 归一化速度
        actions = []
        for left in speeds:
            for right in speeds:
                actions.append([left * 5, right * 5])  # 缩放到实际速度范围
        return actions
        
    def update_target_network(self):
        """更新目标网络"""
        self.target_network.load_state_dict(self.q_network.state_dict())
        
    def remember(self, state, action_idx, reward, next_state, done):
        """存储经验"""
        self.memory.append((state, action_idx, reward, next_state, done))
        
    def act(self, state, training=True):
        """选择动作"""
        if training and np.random.rand() <= self.epsilon:
            # 探索：随机选择动作
            action_idx = random.randrange(self.action_size)
        else:
            # 利用：选择Q值最大的动作
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
            action_idx = torch.argmax(q_values).item()
            
        return action_idx, self.actions[action_idx]
        
    def replay(self, batch_size):
        """从经验回放缓冲区中学习"""
        if len(self.memory) < batch_size:
            return
            
        # 随机采样
        minibatch = random.sample(self.memory, batch_size)
        
        for state, action_idx, reward, next_state, done in minibatch:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
            
            # 计算目标Q值
            with torch.no_grad():
                if done:
                    target = reward
                else:
                    next_q_values = self.target_network(next_state_tensor)
                    target = reward + self.gamma * torch.max(next_q_values).item()
            
            # 计算当前Q值
            current_q_values = self.q_network(state_tensor)
            target_q_values = current_q_values.clone()
            target_q_values[0][action_idx] = target
            
            # 更新Q网络
            self.optimizer.zero_grad()
            loss = self.criterion(current_q_values, target_q_values)
            loss.backward()
            self.optimizer.step()
        
        # 衰减探索率
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
    def save(self, filename):
        """保存模型"""
        torch.save(self.q_network.state_dict(), filename)
        
    def load(self, filename):
        """加载模型"""
        self.q_network.load_state_dict(torch.load(filename))
        self.update_target_network()