import gymnasium as gym
import time

# ----------------------------------------------------
# 1. 环境（Environment）分析：经典控制任务 CartPole-v1
# ----------------------------------------------------
# CartPole（倒立摆）是强化学习中的一个经典问题。
# 目标：控制一个小车左右移动，使得其上的杆子保持竖直不倒。
# 如果杆子倾斜超过15度，或者小车移出屏幕，则一轮（Episode）结束。

# 使用gymnasium库创建CartPole环境
# render_mode="human" 表示我们希望看到图形化界面
env = gym.make("CartPole-v1", render_mode="human")

# ----------------------------------------------------
# 2. 核心要素识别
# ----------------------------------------------------

# (1) 状态空间 (State Space)
# 状态是描述环境在某一时刻的所有信息。在CartPole中，它由4个连续值组成：
# - 小车位置 (Cart Position)
# - 小车速度 (Cart Velocity)
# - 杆子角度 (Pole Angle)
# - 杆子角速度 (Pole Angular Velocity)
print(f"状态空间 (Observation Space): {env.observation_space}")
print(f"状态空间示例: {env.observation_space.sample()}")

# (2) 动作空间 (Action Space)
# 动作是智能体（Agent）可以执行的操作。在CartPole中，有两个离散的动作：
# - 0: 向左推小车
# - 1: 向右推小车
print(f"动作空间 (Action Space): {env.action_space}")
print(f"动作空间示例: {env.action_space.sample()}")

# (3) 奖励 (Reward)
# 奖励是环境对智能体在某个状态下执行某个动作后给出的反馈信号。
# 在CartPole中，只要杆子保持竖直（在一轮中每走一步），智能体就会获得 +1 的奖励。
# 我们的目标就是最大化一轮中获得的累积奖励。

# ----------------------------------------------------
# 3. 简单的RL代码示例：智能体与环境的交互循环
# ----------------------------------------------------
# 我们将让一个"随机"智能体与环境交互10轮（episodes）。
# 这个智能体不会学习，它只是在每个时间步随机选择一个动作。
# 这有助于我们理解最核心的 Agent-Environment 交互循环。

num_episodes = 10

print("\n--- 开始随机智能体的交互演示 ---")

for episode in range(num_episodes):
    # 重置环境，开始新的一轮。返回初始状态。
    state, info = env.reset()
    
    terminated = False  # 标记一轮是否正常结束（如达到目标）
    truncated = False   # 标记一轮是否异常结束（如超出时间限制）
    total_reward = 0

    print(f"\n--- 第 {episode + 1} 轮开始 ---")
    
    while not terminated and not truncated:
        # 渲染环境的当前帧
        env.render()
        
        # 智能体决策：在这里，我们随机选择一个动作
        action = env.action_space.sample()
        
        # 智能体执行动作，环境返回下一步的信息
        # next_state: 执行动作后的新状态
        # reward: 执行该动作获得的即时奖励
        # terminated: 是否达到终止状态（例如，杆子倒了）
        # truncated: 是否达到截断状态（例如，一轮达到最大步数）
        # info: 附加信息（一般为空）
        next_state, reward, terminated, truncated, info = env.step(action)
        
        # 累积奖励
        total_reward += reward
        
        # 更新状态
        state = next_state
        
        # 稍微暂停一下，方便肉眼观察
        time.sleep(0.01)

    print(f"--- 第 {episode + 1} 轮结束 ---")
    print(f"本轮总奖励: {total_reward}")

print("\n--- 交互演示结束 ---")

# 关闭环境
env.close()

# ----------------------------------------------------
# 4. 总结与思考
# ----------------------------------------------------
# 通过以上代码，我们观察到了RL的核心循环：
# 1. Agent从Environment获取State。
# 2. Agent根据State选择一个Action。
# 3. Environment根据Agent的Action，返回Next State和Reward。
# 4. Agent使用Reward来更新自己的策略（虽然本例中未实现学习部分）。
#
# 在接下来的实验中，我们将用更复杂的算法（如PPO）来替代这里的"随机选择动作"，
# 从而让智能体真正地"学习"如何获得更高的累积奖励。 