import pygame
import numpy as np
import math

class LineFollowEnv:
    def __init__(self, width=800, height=600):
        # 初始化Pygame
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Line Following Car Simulation")
        
        # 小车参数
        self.car_size = 20  # 小车大小（正方形边长）
        self.car_pos = np.array([width//2, height//2], dtype=float)  # 小车位置
        self.car_angle = 0  # 小车朝向角度（弧度）
        self.max_speed = 5  # 最大速度
        
        # 路线参数
        self.line_points = self._generate_line()
        self.line_color = (255, 255, 255)  # 白色路线
        
        # 状态空间相关
        self.sensor_range = 100  # 传感器探测范围
        self.num_sensors = 5    # 传感器数量
        
    def _generate_line(self):
        """生成正弦曲线作为跟随路线"""
        x = np.linspace(0, self.width, 100)
        y = self.height/2 + 100 * np.sin(x * 2 * np.pi / self.width)
        return np.column_stack((x, y))
        
    def reset(self):
        """重置环境"""
        self.car_pos = np.array([50, self.height/2], dtype=float)
        self.car_angle = 0
        return self._get_state()
        
    def _get_state(self):
        """获取当前状态（传感器读数）"""
        sensor_readings = []
        sensor_angles = np.linspace(-np.pi/4, np.pi/4, self.num_sensors)
        
        for angle in sensor_angles:
            # 计算传感器位置
            sensor_angle = self.car_angle + angle
            sensor_dir = np.array([np.cos(sensor_angle), np.sin(sensor_angle)])
            sensor_pos = self.car_pos + sensor_dir * self.car_size/2
            
            # 找到最近的线段点
            min_dist = float('inf')
            for i in range(len(self.line_points)-1):
                p1 = self.line_points[i]
                p2 = self.line_points[i+1]
                dist = self._point_to_line_distance(sensor_pos, p1, p2)
                min_dist = min(min_dist, dist)
            
            # 归一化读数
            sensor_readings.append(min(1.0, min_dist/self.sensor_range))
            
        return np.array(sensor_readings)
        
    def _point_to_line_distance(self, point, line_start, line_end):
        """计算点到线段的最短距离"""
        line_vec = line_end - line_start
        point_vec = point - line_start
        line_len = np.linalg.norm(line_vec)
        line_unitvec = line_vec / line_len
        point_vec_scaled = point_vec / line_len
        t = np.dot(line_unitvec, point_vec_scaled)
        
        if t < 0.0:
            return np.linalg.norm(point_vec)
        elif t > 1.0:
            return np.linalg.norm(point - line_end)
        else:
            nearest = line_start + t * line_vec
            return np.linalg.norm(point - nearest)
            
    def step(self, action):
        """执行动作并返回新状态、奖励和是否结束"""
        # 动作是两个轮子的速度：[左轮速度, 右轮速度]
        left_speed, right_speed = action
        
        # 限制速度范围
        left_speed = np.clip(left_speed, -self.max_speed, self.max_speed)
        right_speed = np.clip(right_speed, -self.max_speed, self.max_speed)
        
        # 计算小车运动
        avg_speed = (left_speed + right_speed) / 2
        angular_speed = (right_speed - left_speed) / self.car_size
        
        # 更新位置和角度
        self.car_angle += angular_speed
        self.car_pos[0] += avg_speed * np.cos(self.car_angle)
        self.car_pos[1] += avg_speed * np.sin(self.car_angle)
        
        # 获取新状态
        new_state = self._get_state()
        
        # 计算奖励
        reward = self._calculate_reward(new_state)
        
        # 检查是否结束
        done = self._is_done()
        
        return new_state, reward, done
        
    def _calculate_reward(self, state):
        """计算奖励"""
        # 基于传感器读数计算奖励
        # 传感器读数越接近中心传感器，奖励越高
        center_sensor = state[len(state)//2]
        reward = 1.0 - center_sensor  # 传感器值越小（越接近线）奖励越高
        
        # 如果完全偏离轨道，给予惩罚
        if np.mean(state) > 0.8:  # 如果所有传感器都检测不到线
            reward -= 2.0
            
        return reward
        
    def _is_done(self):
        """检查是否需要结束当前回合"""
        # 如果小车超出边界
        if (self.car_pos[0] < 0 or self.car_pos[0] > self.width or
            self.car_pos[1] < 0 or self.car_pos[1] > self.height):
            return True
            
        # 如果完全偏离轨道
        if np.mean(self._get_state()) > 0.9:
            return True
            
        return False
        
    def render(self, left_speed=None, right_speed=None):
        """渲染当前环境状态"""
        # 处理事件
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                return True
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.close()
                    return True
        
        # 清空屏幕
        self.screen.fill((0, 0, 0))  # 黑色背景
        
        # 绘制路线
        for i in range(len(self.line_points)-1):
            pygame.draw.line(self.screen, self.line_color,
                           self.line_points[i], self.line_points[i+1])
        
        # 绘制小车（正方形）
        car_surface = pygame.Surface((self.car_size, self.car_size), pygame.SRCALPHA)
        pygame.draw.rect(car_surface, (255, 0, 0), (0, 0, self.car_size, self.car_size))
        
        # 旋转小车
        rotated_car = pygame.transform.rotate(car_surface, -math.degrees(self.car_angle))
        car_rect = rotated_car.get_rect(center=self.car_pos)
        self.screen.blit(rotated_car, car_rect)
        
        # 显示速度信息
        if left_speed is not None and right_speed is not None:
            font = pygame.font.Font(None, 36)
            speed_text = f"Left: {left_speed:.2f} Right: {right_speed:.2f}"
            text_surface = font.render(speed_text, True, (255, 255, 255))
            self.screen.blit(text_surface, (10, 10))
        
        # 更新显示
        pygame.display.flip()
        
        # 控制帧率
        pygame.time.Clock().tick(30)
        
        return False
        
    def close(self):
        """关闭环境"""
        pygame.quit()