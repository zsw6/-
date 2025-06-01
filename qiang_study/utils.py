import numpy as np
import matplotlib.pyplot as plt
import pygame

def generate_sine_line(width, height, amplitude=100, frequency=2):
    """生成正弦曲线路线"""
    x = np.linspace(0, width, 100)
    y = height/2 + amplitude * np.sin(x * frequency * np.pi / width)
    return np.column_stack((x, y))

def generate_circle_line(width, height, radius=200):
    """生成圆形路线"""
    center_x, center_y = width/2, height/2
    angles = np.linspace(0, 2*np.pi, 100)
    x = center_x + radius * np.cos(angles)
    y = center_y + radius * np.sin(angles)
    return np.column_stack((x, y))

def generate_random_line(width, height, points=10, smoothness=3):
    """生成平滑的随机路线"""
    # 生成随机控制点
    control_x = np.linspace(0, width, points)
    control_y = np.random.randint(height/4, 3*height/4, points)
    
    # 使用样条插值生成平滑曲线
    t = np.linspace(0, 1, 100)
    x = np.zeros(100)
    y = np.zeros(100)
    
    # 简单的平滑算法
    for i in range(100):
        t_val = t[i]
        x_val = 0
        y_val = 0
        total_weight = 0
        
        for j in range(points):
            # 计算权重（距离越近权重越大）
            t_point = j / (points - 1)
            weight = 1 / (abs(t_val - t_point) + 0.1) ** smoothness
            
            x_val += control_x[j] * weight
            y_val += control_y[j] * weight
            total_weight += weight
            
        x[i] = x_val / total_weight
        y[i] = y_val / total_weight
    
    return np.column_stack((x, y))

def generate_figure_eight(width, height, size=200):
    """生成8字形路线"""
    t = np.linspace(0, 2*np.pi, 100)
    center_x, center_y = width/2, height/2
    x = center_x + size * np.sin(t)
    y = center_y + size * np.sin(t) * np.cos(t)
    return np.column_stack((x, y))

def plot_training_results(episode_rewards, episode_lengths, window=10):
    """绘制训练结果图表"""
    plt.figure(figsize=(12, 5))
    
    # 平滑奖励曲线
    smoothed_rewards = []
    for i in range(len(episode_rewards)):
        start = max(0, i - window)
        smoothed_rewards.append(np.mean(episode_rewards[start:i+1]))
    
    plt.subplot(1, 2, 1)
    plt.plot(episode_rewards, alpha=0.3, label='Raw')
    plt.plot(smoothed_rewards, label='Smoothed')
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.legend()
    
    # 平滑回合长度曲线
    smoothed_lengths = []
    for i in range(len(episode_lengths)):
        start = max(0, i - window)
        smoothed_lengths.append(np.mean(episode_lengths[start:i+1]))
    
    plt.subplot(1, 2, 2)
    plt.plot(episode_lengths, alpha=0.3, label='Raw')
    plt.plot(smoothed_lengths, label='Smoothed')
    plt.title('Episode Lengths')
    plt.xlabel('Episode')
    plt.ylabel('Steps')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_results.png')
    plt.close()

def draw_sensors(screen, car_pos, car_angle, sensor_readings, sensor_range, num_sensors):
    """绘制传感器线和读数"""
    sensor_angles = np.linspace(-np.pi/4, np.pi/4, num_sensors)
    
    for i, angle in enumerate(sensor_angles):
        # 计算传感器方向
        sensor_angle = car_angle + angle
        sensor_dir = np.array([np.cos(sensor_angle), np.sin(sensor_angle)])
        
        # 传感器起点（小车中心）
        start_pos = car_pos
        
        # 传感器终点（基于读数）
        distance = (1 - sensor_readings[i]) * sensor_range  # 转换为实际距离
        end_pos = start_pos + sensor_dir * distance
        
        # 绘制传感器线
        color = (0, 255, 0) if sensor_readings[i] < 0.5 else (255, 165, 0)
        pygame.draw.line(screen, color, start_pos, end_pos, 2)