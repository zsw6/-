import numpy as np
import pygame
import time
from environment import LineFollowEnv
from agent import DQNAgent
from utils import plot_training_results, draw_sensors
import argparse

def train(env, agent, episodes=1000, batch_size=32, render_interval=50):
    """训练智能体"""
    episode_rewards = []
    episode_lengths = []
    
    try:
        for episode in range(episodes):
            state = env.reset()
            total_reward = 0
            steps = 0
            
            while True:
                # 选择动作
                action_idx, action = agent.act(state, training=True)
                
                # 执行动作
                next_state, reward, done = env.step(action)
                
                # 存储经验
                agent.remember(state, action_idx, reward, next_state, done)
                
                # 训练智能体
                if len(agent.memory) > batch_size:
                    agent.replay(batch_size)
                
                # 更新状态和计数器
                state = next_state
                total_reward += reward
                steps += 1
                
                # 渲染（按指定间隔）
                if episode % render_interval == 0:
                    # 传递速度信息到渲染函数
                    should_quit = env.render(action[0], action[1])
                    draw_sensors(env.screen, env.car_pos, env.car_angle, 
                               state, env.sensor_range, env.num_sensors)
                    
                    if should_quit:
                        raise KeyboardInterrupt
                
                if done:
                    break
            
            # 更新目标网络
            if episode % agent.update_target_frequency == 0:
                agent.update_target_network()
            
            # 记录结果
            episode_rewards.append(total_reward)
            episode_lengths.append(steps)
            
            # 打印进度
            if (episode + 1) % 10 == 0:
                print(f"Episode: {episode + 1}/{episodes}, "
                      f"Reward: {total_reward:.2f}, "
                      f"Steps: {steps}, "
                      f"Epsilon: {agent.epsilon:.2f}")
        
        # 绘制训练结果
        plot_training_results(episode_rewards, episode_lengths)
        
        # 保存模型
        agent.save('line_follower_model.pth')
        return episode_rewards, episode_lengths
    
    except KeyboardInterrupt:
        print("\n训练被用户中断")
        # 保存中断时的模型
        if len(episode_rewards) > 0:
            plot_training_results(episode_rewards, episode_lengths)
            agent.save('line_follower_model_interrupted.pth')
            print("已保存中断时的模型为 'line_follower_model_interrupted.pth'")
        return episode_rewards, episode_lengths

def test(env, agent, episodes=5):
    """测试训练好的智能体"""
    try:
        for episode in range(episodes):
            state = env.reset()
            total_reward = 0
            steps = 0
            
            print(f"\nStarting Test Episode {episode + 1}")
            
            while True:
                # 选择动作（不使用探索）
                action_idx, action = agent.act(state, training=False)
                
                # 执行动作
                next_state, reward, done = env.step(action)
                
                # 更新状态和计数器
                state = next_state
                total_reward += reward
                steps += 1
                
                # 渲染（包含速度信息）
                should_quit = env.render(action[0], action[1])
                draw_sensors(env.screen, env.car_pos, env.car_angle, 
                           state, env.sensor_range, env.num_sensors)
                
                if should_quit:
                    print("\nTest terminated by user")
                    return
                
                if done:
                    print(f"Test Episode {episode + 1}: Reward = {total_reward:.2f}, Steps = {steps}")
                    break
                
                # 控制测试时的显示速度
                time.sleep(0.03)
                
    except KeyboardInterrupt:
        print("\nTest terminated by user")
    finally:
        print("\nTesting completed")

def main():
    parser = argparse.ArgumentParser(description='Line Following Car with DQN')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'],
                      help='训练或测试智能体')
    parser.add_argument('--episodes', type=int, default=1000,
                      help='训练回合数')
    parser.add_argument('--render_interval', type=int, default=50,
                      help='训练过程中渲染显示的间隔')
    parser.add_argument('--batch_size', type=int, default=32,
                      help='训练批次大小')
    parser.add_argument('--test_episodes', type=int, default=5,
                      help='测试回合数')
    parser.add_argument('--load_interrupted', action='store_true',
                      help='加载中断时保存的模型')
    args = parser.parse_args()
    
    try:
        # 创建环境
        print("\n正在初始化环境...")
        env = LineFollowEnv()
        
        # 创建智能体
        print("正在初始化智能体...")
        state_size = env.num_sensors
        
        # 创建临时智能体来获取动作空间大小
        temp_agent = DQNAgent(state_size, 1)
        action_size = len(temp_agent._create_action_space())
        
        # 创建实际使用的智能体
        agent = DQNAgent(state_size, action_size)
        
        if args.mode == 'train':
            # 训练模式
            print("\n===== 开始训练智能体 =====")
            print(f"训练回合数: {args.episodes}")
            print(f"渲染间隔: 每 {args.render_interval} 回合")
            print(f"批次大小: {args.batch_size}")
            print("控制方式:")
            print("- ESC键或关闭窗口: 终止训练")
            print("- 训练中断时会自动保存模型")
            print("==========================\n")
            
            train(env, agent, 
                  episodes=args.episodes, 
                  batch_size=args.batch_size, 
                  render_interval=args.render_interval)
            
        else:
            # 测试模式
            print("\n===== 开始测试智能体 =====")
            print(f"测试回合数: {args.test_episodes}")
            print("控制方式:")
            print("- ESC键或关闭窗口: 终止测试")
            print("==========================\n")
            
            model_loaded = False
            
            if not args.load_interrupted:
                try:
                    agent.load('line_follower_model.pth')
                    print("已加载训练完成的模型")
                    model_loaded = True
                except FileNotFoundError:
                    print("未找到训练完成的模型")
            
            if not model_loaded and (args.load_interrupted or True):
                try:
                    agent.load('line_follower_model_interrupted.pth')
                    print("已加载中断时保存的模型")
                    model_loaded = True
                except FileNotFoundError:
                    print("未找到中断保存的模型")
            
            if model_loaded:
                test(env, agent, episodes=args.test_episodes)
            else:
                print("\n错误: 未找到任何可用的模型文件")
                print("请先训练智能体或确保模型文件存在")
    
    except KeyboardInterrupt:
        print("\n程序被用户中断")
    except Exception as e:
        print(f"\n发生错误: {str(e)}")
    finally:
        # 确保环境正确关闭
        try:
            env.close()
        except:
            pass
        print("\n程序已退出")

if __name__ == "__main__":
    main()