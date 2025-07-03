#!/usr/bin/env python3
"""
Lab04: 大规模训练的核心挑战演示 (增强可解释性版本)

本实验复现两个关键问题：
1. 熵坍塌 (Entropy Collapse) - 策略过快收敛到局部最优
2. 梯度消失 (Gradient Vanishing) - 同质化奖励导致无效梯度

新增功能：
- 实时解释系统
- 小白友好的分析报告
- 关键概念自动解释
- 详细的进度提示

作者: SystemLearning Project
日期: 2024-12-19
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from typing import List, Tuple, Dict
import logging
from dataclasses import dataclass
import time

# 配置matplotlib中文字体
def setup_chinese_font():
    """配置matplotlib支持中文显示"""
    try:
        # 尝试多种字体配置方案
        import matplotlib.font_manager as fm
        
        # 方案1: 尝试系统中文字体
        font_list = ['SimHei', 'Microsoft YaHei', 'PingFang SC', 'WenQuanYi Micro Hei']
        available_fonts = [f.name for f in fm.fontManager.ttflist]
        
        chinese_font = None
        for font in font_list:
            if font in available_fonts:
                chinese_font = font
                break
        
        if chinese_font:
            plt.rcParams['font.sans-serif'] = [chinese_font]
            plt.rcParams['axes.unicode_minus'] = False
            print(f"✅ 中文字体配置成功: {chinese_font}")
        else:
            # 方案2: 使用默认字体，图表使用英文
            plt.rcParams['font.family'] = 'DejaVu Sans'
            print("ℹ️ 使用英文字体，图表将显示为英文")
            
    except Exception as e:
        print(f"⚠️ 字体配置失败: {e}")
        print("📝 将使用英文标签确保正常显示")
        plt.rcParams['font.family'] = 'DejaVu Sans'

setup_chinese_font()

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ExplainerSystem:
    """可解释性系统 - 为小白提供详细解释"""
    
    @staticmethod
    def explain_concept(concept: str) -> str:
        """解释关键概念"""
        explanations = {
            "entropy": """
🧠 熵 (Entropy) 简单解释：
• 熵就像"不确定性"或"随机性"的度量
• 高熵 = 模型很犹豫，不知道选哪个动作（这是好事，说明在探索）
• 低熵 = 模型很确定，总是选同一个动作（可能陷入局部最优）
• 理想情况：开始时高熵（多探索），后来逐渐降低（找到好策略）
            """,
            "gradient_norm": """
🔢 梯度范数 (Gradient Norm) 简单解释：
• 梯度就像"学习的方向和强度"
• 梯度范数就像"学习信号的强弱"
• 高梯度范数 = 学习信号很强，模型在快速改进
• 低梯度范数 = 学习信号很弱，模型几乎不在学习
• 梯度消失 = 学习信号变得很弱，模型停止进步
            """,
            "policy_gradient": """
📈 策略梯度 (Policy Gradient) 简单解释：
• 这是一种教AI学习的方法
• 基本思路：好的动作要增加概率，坏的动作要减少概率
• 就像训练宠物：做对了给奖励，做错了不给奖励
• 问题：如果所有动作奖励都一样，就不知道该学什么了
            """,
            "local_optimum": """
🏔️ 局部最优 (Local Optimum) 简单解释：
• 想象你在爬山寻找最高峰
• 局部最优 = 你找到了一个小山头，但不是最高的山
• 全局最优 = 真正的最高峰
• 问题：如果你太早停止探索，就可能困在小山头上
            """
        }
        return explanations.get(concept, f"概念 {concept} 暂无解释")
    
    @staticmethod
    def real_time_analysis(metrics: 'TrainingMetrics', step: int, max_steps: int) -> str:
        """实时分析训练状态"""
        progress = (step + 1) / max_steps * 100
        
        # 分析熵的状态
        if metrics.entropy > 0.8:
            entropy_status = "🔍 高熵状态 - 模型正在积极探索，这很好！"
        elif metrics.entropy > 0.3:
            entropy_status = "⚖️ 中等熵 - 模型开始收敛但还在探索"
        else:
            entropy_status = "⚠️ 低熵状态 - 模型可能过度收敛了！"
        
        # 分析梯度状态
        if metrics.gradient_norm > 1.0:
            gradient_status = "💪 强梯度信号 - 模型正在快速学习"
        elif metrics.gradient_norm > 0.1:
            gradient_status = "📈 正常梯度 - 模型在稳定学习"
        else:
            gradient_status = "😴 弱梯度信号 - 学习可能停滞了"
        
        # 分析动作分布
        action_probs = metrics.action_distribution
        max_prob = max(action_probs)
        if max_prob > 0.8:
            action_status = f"🎯 强偏好动作{action_probs.index(max_prob)} (概率{max_prob:.2f})"
        else:
            action_status = "🎲 动作选择较均匀"
        
        return f"""
📊 第{step+1}步实时分析 (进度: {progress:.1f}%)
{entropy_status}
{gradient_status}
{action_status}
💰 平均奖励: {metrics.reward_mean:.3f} ± {metrics.reward_std:.3f}
        """

@dataclass
class TrainingMetrics:
    """训练过程的关键指标"""
    step: int
    entropy: float
    gradient_norm: float
    loss: float
    action_distribution: List[float]
    reward_mean: float
    reward_std: float

class MultiModalEnvironment:
    """
    多模态环境：设计多个局部最优解的环境
    
    这个环境有3个动作选择：
    - 动作0: 奖励 0.3 (一个局部最优)
    - 动作1: 奖励 1.0 (全局最优，但需要探索才能发现)
    - 动作2: 奖励 0.5 (另一个局部最优)
    
    目标是观察模型是否会陷入局部最优而无法找到全局最优
    """
    
    def __init__(self):
        self.action_rewards = [0.3, 1.0, 0.5]  # 不同动作的奖励
        self.state_dim = 4
        self.action_dim = 3
        
    def get_state(self) -> torch.Tensor:
        """返回随机状态（简化环境）"""
        return torch.randn(self.state_dim)
    
    def step(self, action: int) -> Tuple[torch.Tensor, float]:
        """执行动作并返回新状态和奖励"""
        reward = self.action_rewards[action]
        # 添加小量噪声使环境更真实
        reward += np.random.normal(0, 0.1)
        new_state = self.get_state()
        return new_state, reward

class SimplePolicy(nn.Module):
    """简单的策略网络"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """前向传播，返回动作logits"""
        return self.network(state)
    
    def get_action_and_log_prob(self, state: torch.Tensor) -> Tuple[int, torch.Tensor, torch.Tensor]:
        """获取动作、log概率和熵"""
        logits = self.forward(state)
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        
        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        
        return action.item(), log_prob, entropy

class TrainingChallengeExperiment:
    """训练挑战实验主类"""
    
    def __init__(self, 
                 learning_rate: float = 1e-3,
                 entropy_coef: float = 0.01,
                 batch_size: int = 32,
                 verbose: bool = True):
        self.env = MultiModalEnvironment()
        self.policy = SimplePolicy(self.env.state_dim, self.env.action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.entropy_coef = entropy_coef
        self.batch_size = batch_size
        self.verbose = verbose
        
        # 记录训练指标
        self.metrics_history: List[TrainingMetrics] = []
        
        # 可解释性系统
        self.explainer = ExplainerSystem()
        
        if self.verbose:
            print("🚀 实验初始化完成！")
            print(f"📋 配置: 学习率={learning_rate}, 熵系数={entropy_coef}, 批次大小={batch_size}")
    
    def collect_batch(self, 
                     homogeneous_rewards: bool = False,
                     fixed_reward: float = 1.0) -> Tuple[List[torch.Tensor], 
                                                         List[torch.Tensor], 
                                                         List[float],
                                                         List[torch.Tensor]]:
        """
        收集一个训练批次的数据
        
        Args:
            homogeneous_rewards: 是否使用同质化奖励（用于复现梯度消失）
            fixed_reward: 同质化奖励的固定值
        """
        states, log_probs, rewards, entropies = [], [], [], []
        
        for _ in range(self.batch_size):
            state = self.env.get_state()
            action, log_prob, entropy = self.policy.get_action_and_log_prob(state)
            
            if homogeneous_rewards:
                # 使用固定奖励来模拟同质化情况
                reward = fixed_reward
            else:
                # 正常环境交互
                _, reward = self.env.step(action)
            
            states.append(state)
            log_probs.append(log_prob)
            rewards.append(reward)
            entropies.append(entropy)
            
        return states, log_probs, rewards, entropies
    
    def compute_advantages(self, rewards: List[float]) -> torch.Tensor:
        """计算优势函数 (简化版本，直接使用奖励)"""
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
        # 标准化优势
        advantages = rewards_tensor - rewards_tensor.mean()
        if rewards_tensor.std() > 1e-8:
            advantages = advantages / (rewards_tensor.std() + 1e-8)
        return advantages
    
    def train_step(self, 
                  homogeneous_rewards: bool = False,
                  fixed_reward: float = 1.0) -> TrainingMetrics:
        """执行一步训练"""
        
        # 收集数据
        states, log_probs, rewards, entropies = self.collect_batch(
            homogeneous_rewards, fixed_reward)
        
        # 计算优势
        advantages = self.compute_advantages(rewards)
        
        # 计算损失
        log_probs_tensor = torch.stack(log_probs)
        entropies_tensor = torch.stack(entropies)
        
        # Policy Loss (REINFORCE with baseline)
        policy_loss = -(log_probs_tensor * advantages).mean()
        
        # Entropy Loss (鼓励探索)
        entropy_loss = -entropies_tensor.mean()
        
        # 总损失
        total_loss = policy_loss + self.entropy_coef * entropy_loss
        
        # 反向传播
        self.optimizer.zero_grad()
        total_loss.backward()
        
        # 计算梯度范数
        grad_norm = 0.0
        for param in self.policy.parameters():
            if param.grad is not None:
                grad_norm += param.grad.data.norm(2).item() ** 2
        grad_norm = grad_norm ** 0.5
        
        self.optimizer.step()
        
        # 计算动作分布
        with torch.no_grad():
            sample_state = self.env.get_state()
            logits = self.policy(sample_state)
            action_probs = F.softmax(logits, dim=-1).numpy().tolist()
        
        # 创建并记录指标
        metrics = TrainingMetrics(
            step=len(self.metrics_history),
            entropy=entropies_tensor.mean().item(),
            gradient_norm=grad_norm,
            loss=total_loss.item(),
            action_distribution=action_probs,
            reward_mean=np.mean(rewards),
            reward_std=np.std(rewards)
        )
        
        self.metrics_history.append(metrics)
        return metrics

    def generate_detailed_report(self, experiment_type: str) -> str:
        """生成详细的分析报告"""
        if not self.metrics_history:
            return "❌ 没有训练数据可分析"
        
        metrics = self.metrics_history
        
        # 基础统计
        initial_entropy = metrics[0].entropy
        final_entropy = metrics[-1].entropy
        entropy_change = (initial_entropy - final_entropy) / initial_entropy * 100
        
        initial_grad = metrics[0].gradient_norm
        final_grad = metrics[-1].gradient_norm
        
        # 动作分布分析
        final_action_dist = metrics[-1].action_distribution
        dominant_action = final_action_dist.index(max(final_action_dist))
        
        report = f"""
🔍 {experiment_type} 实验详细分析报告
{'='*50}

📈 训练概况:
• 总训练步数: {len(metrics)}
• 初始熵值: {initial_entropy:.4f} → 最终熵值: {final_entropy:.4f}
• 熵值变化: {entropy_change:.1f}%
• 初始梯度: {initial_grad:.4f} → 最终梯度: {final_grad:.4f}

🎯 最终动作偏好:
• 动作0 (奖励0.3): {final_action_dist[0]:.3f} ({final_action_dist[0]*100:.1f}%)
• 动作1 (奖励1.0): {final_action_dist[1]:.3f} ({final_action_dist[1]*100:.1f}%)  ⭐最优
• 动作2 (奖励0.5): {final_action_dist[2]:.3f} ({final_action_dist[2]*100:.1f}%)
• 主导动作: 动作{dominant_action}

💡 结果解释:
        """
        
        if experiment_type == "熵坍塌":
            if entropy_change > 90:
                report += """
✅ 成功复现熵坍塌现象！
• 模型过快收敛，失去探索能力
• 这在强化学习中是个严重问题，因为模型可能错过更好的策略
                """
                if dominant_action == 1:
                    report += "• 幸运的是，模型找到了全局最优动作1"
                else:
                    report += f"• 糟糕的是，模型困在了局部最优动作{dominant_action}"
            else:
                report += """
⚠️ 熵坍塌现象不明显
• 可能需要调整参数（降低熵系数或提高学习率）
                """
        
        elif experiment_type == "梯度消失":
            avg_grad = np.mean([m.gradient_norm for m in metrics])
            if avg_grad < 0.1:
                report += """
✅ 成功复现梯度消失现象！
• 同质化奖励导致梯度信号变弱
• 这会让模型学习停滞，无法改进策略
                """
            else:
                report += """
⚠️ 梯度消失现象不明显
• 可能需要使用更加同质化的奖励
                """
        
        report += f"""

🚨 为什么这些问题很重要？
• 熵坍塌 → 模型失去探索能力，可能错过最优策略
• 梯度消失 → 模型停止学习，训练效果很差
• 在真实场景中，这些问题会导致AI系统性能下降

🔧 后续学习方向:
• Lab05: DAPO算法 - 专门解决训练不稳定性
• Lab06: VeRL算法 - 提供可验证的强化学习
• 这些先进算法能有效缓解当前观察到的问题
        """
        
        return report

def run_entropy_collapse_experiment():
    """
    实验1: 熵坍塌现象
    通过高学习率和低熵系数来促使策略过快收敛
    """
    print("\n" + "="*60)
    print("🔥 开始熵坍塌实验")
    print("="*60)
    print("🎯 目标: 观察策略过快收敛到局部最优的现象")
    print("⚙️ 方法: 使用高学习率(5e-3)和低熵系数(0.001)")
    print(ExplainerSystem.explain_concept("entropy"))
    
    # 创建实验
    experiment = TrainingChallengeExperiment(
        learning_rate=5e-3,  # 高学习率促进快速收敛
        entropy_coef=0.001,  # 低熵系数减少探索
        batch_size=64,
        verbose=True
    )
    
    print("🔄 开始训练...")
    
    # 训练过程
    num_steps = 100
    for step in range(num_steps):
        metrics = experiment.train_step()
        
        # 每10步显示实时分析
        if step % 10 == 0 and experiment.verbose:
            analysis = experiment.explainer.real_time_analysis(metrics, step, num_steps)
            print(analysis)
        
        # 检查是否已经严重收敛
        if metrics.entropy < 0.01:
            print(f"⚠️ 第{step+1}步：熵值过低({metrics.entropy:.4f})，可能已经坍塌！")
    
    print("✅ 熵坍塌实验完成")
    print(experiment.generate_detailed_report("熵坍塌"))
    
    return experiment.metrics_history

def run_gradient_vanishing_experiment():
    """
    实验2: 梯度消失现象
    通过同质化奖励来演示梯度消失问题
    """
    print("\n" + "="*60)
    print("💨 开始梯度消失实验")
    print("="*60)
    print("🎯 目标: 观察同质化奖励导致的梯度消失现象")
    print("⚙️ 方法: 交替使用多样化和同质化奖励")
    print(ExplainerSystem.explain_concept("gradient_norm"))
    
    # 创建实验
    experiment = TrainingChallengeExperiment(
        learning_rate=1e-3,
        entropy_coef=0.01,
        batch_size=32,
        verbose=True
    )
    
    print("🔄 开始训练...")
    
    # 训练过程：交替使用多样化和同质化奖励
    num_steps = 100
    for step in range(num_steps):
        # 前半段使用多样化奖励，后半段使用同质化奖励
        use_homogeneous = step >= num_steps // 2
        
        if step == num_steps // 2:
            print("⚡ 切换到同质化奖励模式，观察梯度变化...")
        
        metrics = experiment.train_step(
            homogeneous_rewards=use_homogeneous,
            fixed_reward=0.5  # 固定中等奖励
        )
        
        # 每10步显示实时分析
        if step % 10 == 0 and experiment.verbose:
            mode = "同质化奖励" if use_homogeneous else "多样化奖励"
            print(f"\n🔄 当前模式: {mode}")
            analysis = experiment.explainer.real_time_analysis(metrics, step, num_steps)
            print(analysis)
    
    print("✅ 梯度消失实验完成")
    print(experiment.generate_detailed_report("梯度消失"))
    
    return experiment.metrics_history

def create_beginner_friendly_visualization(entropy_metrics: List[TrainingMetrics], 
                                         gradient_metrics: List[TrainingMetrics]):
    """
    创建小白友好的可视化分析
    包含详细解释和直观的图表标注
    """
    print("\n🎨 开始生成可视化分析图表...")
    
    # 设置图表样式
    try:
        plt.style.use('seaborn-v0_8')
    except:
        plt.style.use('default')
    fig = plt.figure(figsize=(16, 12))
    
    # 设置图表主标题
    fig.suptitle('Training Challenges in Large-Scale RL - Beginner Friendly Analysis', 
                fontsize=16, fontweight='bold')
    
    # 定义颜色方案
    colors = {
        'entropy': '#2E86AB',     # 蓝色
        'action': ['#A23B72', '#F18F01', '#C73E1D'],  # 紫红、橙、红
        'loss': '#4CAF50',        # 绿色
        'gradient': '#FF5722',    # 深橙
        'reward': '#9C27B0'       # 紫色
    }
    
    # === 上排：熵坍塌分析 ===
    
    # 1. 熵值变化
    ax1 = plt.subplot(2, 3, 1)
    entropy_values = [m.entropy for m in entropy_metrics]
    steps = list(range(len(entropy_values)))
    
    ax1.plot(steps, entropy_values, color=colors['entropy'], linewidth=3, marker='o', markersize=4)
    ax1.set_title('Entropy Change\n(Lower means more certain)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Training Steps')
    ax1.set_ylabel('Entropy Value')
    ax1.grid(True, alpha=0.3)
    
    # 添加关键点标注
    if len(entropy_values) > 0:
        initial_entropy = entropy_values[0]
        final_entropy = entropy_values[-1]
        
        # 标注起始点
        ax1.annotate(f'Start: {initial_entropy:.3f}', 
                    xy=(0, initial_entropy), xytext=(len(entropy_values)*0.2, initial_entropy*1.2),
                    arrowprops=dict(arrowstyle='->', color='green', lw=2),
                    fontsize=10, ha='center', 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen', alpha=0.7))
        
        # 标注结束点
        ax1.annotate(f'End: {final_entropy:.3f}', 
                    xy=(len(entropy_values)-1, final_entropy), 
                    xytext=(len(entropy_values)*0.8, final_entropy*3),
                    arrowprops=dict(arrowstyle='->', color='red', lw=2),
                    fontsize=10, ha='center',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='lightcoral', alpha=0.7))
    
    # 2. 动作分布演化 (堆叠图)
    ax2 = plt.subplot(2, 3, 2)
    action_dist_data = np.array([m.action_distribution for m in entropy_metrics]).T
    
    ax2.stackplot(steps, action_dist_data[0], action_dist_data[1], action_dist_data[2],
                 labels=['Action0 (reward0.3)', 'Action1 (reward1.0) Best', 'Action2 (reward0.5)'],
                 colors=colors['action'], alpha=0.8)
    ax2.set_title('Action Distribution Evolution', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Training Steps')
    ax2.set_ylabel('Action Probability')
    ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax2.grid(True, alpha=0.3)
    
    # 添加收敛分析
    if len(entropy_metrics) > 0:
        final_dist = entropy_metrics[-1].action_distribution
        dominant_action = final_dist.index(max(final_dist))
        ax2.text(0.02, 0.98, f'Final dominant action: {dominant_action}\nProbability: {max(final_dist):.2f}', 
                transform=ax2.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='lightyellow', alpha=0.8))
    
    # 3. 训练损失变化
    ax3 = plt.subplot(2, 3, 3)
    loss_values = [m.loss for m in entropy_metrics]
    ax3.plot(steps, loss_values, color=colors['loss'], linewidth=2, marker='s', markersize=3)
    ax3.set_title('Training Loss\n(Lower is usually better)', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Training Steps')
    ax3.set_ylabel('Loss Value')
    ax3.grid(True, alpha=0.3)
    
    # === 下排：梯度消失分析 ===
    
    # 4. 梯度范数变化 (分段分析)
    ax4 = plt.subplot(2, 3, 4)
    grad_values = [m.gradient_norm for m in gradient_metrics]
    grad_steps = list(range(len(grad_values)))
    
    # 分前后两段
    mid_point = len(grad_values) // 2
    
    # 前半段 (多样化奖励)
    ax4.plot(grad_steps[:mid_point], grad_values[:mid_point], 
            color='blue', linewidth=3, label='Diverse Rewards', marker='o', markersize=3)
    
    # 后半段 (同质化奖励)
    ax4.plot(grad_steps[mid_point:], grad_values[mid_point:], 
            color='red', linewidth=3, label='Homogeneous Rewards', marker='s', markersize=3)
    
    ax4.axvline(x=mid_point, color='gray', linestyle='--', alpha=0.7, linewidth=2)
    ax4.text(mid_point+1, max(grad_values)*0.8, 'Switch Point', rotation=90, fontsize=10)
    
    ax4.set_title('Gradient Norm Changes\n(Observing vanishing gradients)', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Training Steps')
    ax4.set_ylabel('Gradient Norm')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 添加梯度分析
    if len(grad_values) > mid_point:
        avg_grad_diverse = np.mean(grad_values[:mid_point])
        avg_grad_homo = np.mean(grad_values[mid_point:])
        reduction = (avg_grad_diverse - avg_grad_homo) / avg_grad_diverse * 100
        
        ax4.text(0.02, 0.98, f'Gradient Reduction: {reduction:.1f}%\nDiverse: {avg_grad_diverse:.3f}\nHomogeneous: {avg_grad_homo:.3f}', 
                transform=ax4.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.8))
    
    # 5. 奖励标准差 (多样性指标)
    ax5 = plt.subplot(2, 3, 5)
    reward_std_values = [m.reward_std for m in gradient_metrics]
    
    ax5.plot(grad_steps[:mid_point], reward_std_values[:mid_point], 
            color='green', linewidth=3, label='Diverse Period', marker='o', markersize=3)
    ax5.plot(grad_steps[mid_point:], reward_std_values[mid_point:], 
            color='orange', linewidth=3, label='Homogeneous Period', marker='s', markersize=3)
    
    ax5.axvline(x=mid_point, color='gray', linestyle='--', alpha=0.7, linewidth=2)
    ax5.set_title('Reward Diversity\n(Higher std means more diverse)', fontsize=12, fontweight='bold')
    ax5.set_xlabel('Training Steps')
    ax5.set_ylabel('Reward Standard Deviation')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. 梯度-奖励关系散点图
    ax6 = plt.subplot(2, 3, 6)
    scatter = ax6.scatter(reward_std_values, grad_values, 
                         c=grad_steps, cmap='viridis', s=50, alpha=0.7)
    
    # 添加趋势线
    if len(reward_std_values) > 1:
        z = np.polyfit(reward_std_values, grad_values, 1)
        p = np.poly1d(z)
        x_trend = np.linspace(min(reward_std_values), max(reward_std_values), 100)
        ax6.plot(x_trend, p(x_trend), "r--", alpha=0.8, linewidth=2, label=f'Trend (slope={z[0]:.2f})')
    
    ax6.set_title('Gradient vs Reward Diversity\n(Verifying correlation)', fontsize=12, fontweight='bold')
    ax6.set_xlabel('Reward Standard Deviation')
    ax6.set_ylabel('Gradient Norm')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    # 添加颜色条
    cbar = plt.colorbar(scatter, ax=ax6)
    cbar.set_label('Training Step')
    
    plt.tight_layout()
    
    # 保存图表
    output_path = "training_challenges_analysis.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ 可视化图表已保存到: {output_path}")
    
    # 生成图表解读报告
    return generate_chart_interpretation(entropy_metrics, gradient_metrics)

def generate_chart_interpretation(entropy_metrics: List[TrainingMetrics], 
                                gradient_metrics: List[TrainingMetrics]) -> str:
    """生成图表解读报告，专为小白设计"""
    
    # 计算关键数据
    initial_entropy = entropy_metrics[0].entropy if entropy_metrics else 0
    final_entropy = entropy_metrics[-1].entropy if entropy_metrics else 0
    entropy_reduction = (initial_entropy - final_entropy) / initial_entropy * 100 if initial_entropy > 0 else 0
    
    mid_point = len(gradient_metrics) // 2
    if len(gradient_metrics) > mid_point:
        diverse_grads = [m.gradient_norm for m in gradient_metrics[:mid_point]]
        homo_grads = [m.gradient_norm for m in gradient_metrics[mid_point:]]
        avg_diverse_grad = np.mean(diverse_grads)
        avg_homo_grad = np.mean(homo_grads)
        grad_reduction = (avg_diverse_grad - avg_homo_grad) / avg_diverse_grad * 100
    else:
        grad_reduction = 0
        avg_diverse_grad = avg_homo_grad = 0
    
    final_action_dist = entropy_metrics[-1].action_distribution if entropy_metrics else [0, 0, 0]
    dominant_action = final_action_dist.index(max(final_action_dist))
    
    report = f"""
📊 图表解读报告 (小白版)
{'='*60}

🔍 实验1: 熵坍塌分析
------------------------
📈 熵值变化 (左上图):
• 初始熵: {initial_entropy:.3f} → 最终熵: {final_entropy:.3f}
• 降幅: {entropy_reduction:.1f}%
• 💡 解读: 熵值急剧下降说明模型从"犹豫不决"变成"固执己见"

🎯 动作选择演化 (中上图):
• 最终主导动作: 动作{dominant_action}
• 主导概率: {max(final_action_dist):.1f}%
• 💡 解读: 看看模型最终选择了哪个动作，是否找到了最优动作1

📉 训练损失 (右上图):
• 💡 解读: 损失下降是好事，但配合熵坍塌可能意味着过拟合

🔍 实验2: 梯度消失分析  
------------------------
⚡ 梯度变化 (左下图):
• 多样化期平均梯度: {avg_diverse_grad:.3f}
• 同质化期平均梯度: {avg_homo_grad:.3f}
• 梯度减少: {grad_reduction:.1f}%
• 💡 解读: 同质化奖励导致学习信号变弱，这就是梯度消失！

🎲 奖励多样性 (中下图):
• 💡 解读: 奖励标准差的骤降表明环境变得"无聊"，没有学习价值

🔗 梯度-奖励关系 (右下图):
• 💡 解读: 散点图显示梯度强度与奖励多样性的正相关关系

🚨 关键发现总结:
------------------------
✅ 熵坍塌现象: {"✓ 成功复现" if entropy_reduction > 90 else "⚠ 不够明显"}
✅ 梯度消失现象: {"✓ 成功复现" if grad_reduction > 50 else "⚠ 不够明显"}

🤔 对小白的启发:
------------------------
1. 高熵不是坏事 - 说明AI在认真探索
2. 低梯度很危险 - 意味着AI停止学习了
3. 奖励设计很重要 - 太单调会让AI变"懒"
4. 平衡很关键 - 既要探索又要收敛

🎓 为什么这些很重要？
------------------------
• 在真实AI项目中，这些问题会导致：
  - 模型性能停滞不前
  - 无法适应新情况
  - 训练资源浪费
• 理解这些问题是掌握先进算法(DAPO、VeRL)的基础

🔮 下一步学习建议:
------------------------
• Lab05: 学习DAPO如何解决训练不稳定
• Lab06: 了解VeRL如何提供训练保障
• 继续观察这些先进方法如何解决今天发现的问题！
    """
    
    return report

def main():
    """主函数：运行完整的训练挑战演示"""
    print("🚀 欢迎来到大规模强化学习训练挑战实验室！")
    print("📚 这是一个专为小白设计的学习实验")
    print("🎯 我们将带你亲眼见证两个重要问题：熵坍塌和梯度消失")
    
    # 添加概念预习
    print("\n📖 实验前概念预习:")
    print("=" * 50)
    print(ExplainerSystem.explain_concept("policy_gradient"))
    print(ExplainerSystem.explain_concept("local_optimum"))
    
    print("\n🔥 自动开始实验...")
    
    try:
        # 运行实验1：熵坍塌
        print("\n🎪 第一场秀：熵坍塌实验")
        entropy_metrics = run_entropy_collapse_experiment()
        
        print("\n⏸️ 第一个实验完成！继续第二个实验...")
        
        # 运行实验2：梯度消失
        print("\n🎪 第二场秀：梯度消失实验")
        gradient_metrics = run_gradient_vanishing_experiment()
        
        print("\n⏸️ 两个实验都完成了！现在生成可视化分析图表...")
        
        # 生成可视化分析
        print("\n🎨 正在生成小白友好的可视化分析...")
        interpretation = create_beginner_friendly_visualization(entropy_metrics, gradient_metrics)
        
        # 显示最终解读
        print("\n" + "🎉" * 20)
        print("实验全部完成！以下是最终分析报告：")
        print("🎉" * 20)
        print(interpretation)
        
        print("\n✨ 恭喜你完成了Lab04的学习！")
        print("📈 你现在已经亲身体验了大规模强化学习的核心挑战")
        print("🔧 在接下来的实验中，我们将学习如何解决这些问题")
        print("🚀 继续加油，向着AI专家的目标前进！")
        
    except KeyboardInterrupt:
        print("\n⏹️ 实验被用户中断")
    except Exception as e:
        print(f"\n❌ 实验过程中出现错误: {e}")
        print("💡 提示：请检查依赖包是否正确安装（matplotlib、torch等）")

if __name__ == "__main__":
    main() 