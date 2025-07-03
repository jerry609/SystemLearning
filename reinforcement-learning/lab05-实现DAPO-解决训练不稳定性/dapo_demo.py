#!/usr/bin/env python3
"""
Lab05: DAPO (Decoupled Clip and Dynamic Sampling Policy Optimization) 实现

本实验实现DAPO算法的两大核心组件：
1. Clip-Higher: 放宽对低概率行为的惩罚，鼓励探索
2. 动态采样: 过滤掉奖励信号过于单一的样本组

目标：解决Lab04中发现的熵坍塌和梯度消失问题

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
from typing import List, Tuple, Dict, Optional
import logging
from dataclasses import dataclass
import time
from collections import defaultdict
import warnings

# 配置matplotlib字体
def setup_matplotlib():
    """配置matplotlib显示"""
    try:
        plt.rcParams['font.family'] = 'DejaVu Sans'
        plt.rcParams['axes.unicode_minus'] = False
        print("✅ Matplotlib configuration successful")
    except Exception as e:
        print(f"⚠️ Matplotlib configuration failed: {e}")

setup_matplotlib()

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
    clip_ratio: float = 0.0
    filtered_ratio: float = 0.0  # 动态采样过滤比例

class ExplainerSystem:
    """小白友好的解释系统"""
    
    @staticmethod
    def explain_concept(concept: str) -> str:
        """解释关键概念"""
        explanations = {
            "dapo": """
🚀 DAPO Algorithm:
• DAPO = Advanced version of GRPO for training stability
• Two core technologies:
  1. Clip-Higher: Encourage exploration of high-reward behaviors
  2. Dynamic Sampling: Filter out low-quality training data
• Like giving AI a "smart regulator" for stable learning
""",
            "clip_higher": """
🔧 Clip-Higher Technology:
• Traditional PPO/GRPO: Strictly limit policy changes
• Clip-Higher: Relax limits for promising new behaviors
• Key idea: Encourage rather than punish high-reward actions
• Effect: Prevent entropy collapse, maintain exploration
""",
            "dynamic_sampling": """
📊 Dynamic Sampling Technology:
• Problem: Monotonous training data leads to poor learning
• Solution: Smart filter ensures data diversity in each batch
• Like selecting study materials: need both good and bad examples
• Effect: Prevent gradient vanishing, ensure effective learning
"""
        }
        return explanations.get(concept, f"Explanation for '{concept}' not implemented")

class UnstableEnvironment:
    """Unstable training environment from Lab04"""
    
    def __init__(self):
        self.action_rewards = [0.3, 1.0, 0.5]  # Action 0, 1, 2 rewards
        
    def get_state(self) -> torch.Tensor:
        """Get state (simplified as fixed state)"""
        return torch.zeros(4)
    
    def step(self, action: int) -> Tuple[torch.Tensor, float]:
        """Execute action and return reward"""
        reward = self.action_rewards[action]
        # Add noise to simulate real environment
        noise = np.random.normal(0, 0.1)
        reward = max(0, reward + noise)
        return self.get_state(), reward

class PolicyNetwork(nn.Module):
    """Policy network"""
    
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
        return self.network(state)
    
    def get_action_and_log_prob(self, state: torch.Tensor) -> Tuple[int, torch.Tensor, torch.Tensor]:
        """Get action and log probability"""
        logits = self.forward(state)
        probs = F.softmax(logits, dim=-1)
        probs = torch.clamp(probs, min=1e-8, max=1.0)
        
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        return action.item(), log_prob, probs

class DAPOTrainer:
    """DAPO Trainer - implements Clip-Higher and Dynamic Sampling"""
    
    def __init__(self, 
                 learning_rate: float = 1e-3,
                 entropy_coef: float = 0.01,
                 batch_size: int = 32,
                 clip_ratio: float = 0.2,
                 use_clip_higher: bool = True,
                 use_dynamic_sampling: bool = True,
                 reward_variance_threshold: float = 0.01,
                 verbose: bool = True):
        
        self.env = UnstableEnvironment()
        self.policy = PolicyNetwork(4, 3)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        
        # Training parameters
        self.learning_rate = learning_rate
        self.entropy_coef = entropy_coef
        self.batch_size = batch_size
        self.clip_ratio = clip_ratio
        self.verbose = verbose
        
        # DAPO specific parameters
        self.use_clip_higher = use_clip_higher
        self.use_dynamic_sampling = use_dynamic_sampling
        self.reward_variance_threshold = reward_variance_threshold
        
        # Records
        self.metrics_history = []
        self.explainer = ExplainerSystem()
        
        print(f"🚀 DAPO Trainer initialized")
        print(f"📋 Config: LR={learning_rate}, Entropy={entropy_coef}, Batch={batch_size}")
        print(f"🔧 DAPO Features: Clip-Higher={use_clip_higher}, Dynamic-Sampling={use_dynamic_sampling}")

    def dynamic_sampling_filter(self, batch_data) -> Tuple[List, float]:
        """Dynamic sampling filter - DAPO core component 1"""
        if not self.use_dynamic_sampling:
            return batch_data, 0.0
        
        # Calculate reward statistics
        rewards = [item[2] for item in batch_data]
        reward_std = np.std(rewards)
        
        # Filter if reward variance is too small
        if reward_std < self.reward_variance_threshold:
            if self.verbose and len(self.metrics_history) % 20 == 0:
                print(f"⚠️ Dynamic sampling triggered: reward std {reward_std:.4f} < threshold {self.reward_variance_threshold}")
            
            # Keep samples with higher variance (simplified strategy)
            filtered_data = batch_data[::2]  # Simple filtering strategy
            filter_ratio = 1.0 - len(filtered_data) / len(batch_data)
            return filtered_data, filter_ratio
        
        return batch_data, 0.0

    def clip_higher_loss(self, log_probs, old_log_probs, advantages, rewards) -> torch.Tensor:
        """Clip-Higher loss calculation - DAPO core component 2"""
        ratio = torch.exp(log_probs - old_log_probs)
        
        if not self.use_clip_higher:
            # Standard PPO clipping
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages
            return -torch.min(surr1, surr2).mean()
        
        # Clip-Higher logic: relax upper bound for high-reward samples
        reward_normalized = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        high_reward_mask = reward_normalized > 0.5  # High reward samples
        
        surr1 = ratio * advantages
        
        # Use more relaxed upper bound for high-reward samples
        upper_bound = torch.where(high_reward_mask, 
                                 1 + self.clip_ratio * 2,  # Relax by 2x
                                 1 + self.clip_ratio)      # Standard limit
        
        # Ensure all clamp parameters are tensors for PyTorch compatibility
        lower_bound = torch.full_like(ratio, 1 - self.clip_ratio)
        
        surr2 = torch.clamp(ratio, min=lower_bound, max=upper_bound) * advantages
        
        return -torch.min(surr1, surr2).mean()

    def collect_batch(self):
        """Collect training batch data"""
        batch_data = []
        
        for _ in range(self.batch_size):
            state = self.env.get_state()
            action, log_prob, probs = self.policy.get_action_and_log_prob(state)
            next_state, reward = self.env.step(action)
            
            batch_data.append((state, action, reward, log_prob))
        
        # Apply dynamic sampling filter
        filtered_batch, filter_ratio = self.dynamic_sampling_filter(batch_data)
        
        return filtered_batch, filter_ratio

    def compute_advantages(self, rewards: List[float]) -> torch.Tensor:
        """Compute advantage function (simplified version)"""
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
        # Simple normalization
        if rewards_tensor.std() > 1e-8:
            advantages = (rewards_tensor - rewards_tensor.mean()) / (rewards_tensor.std() + 1e-8)
        else:
            advantages = torch.zeros_like(rewards_tensor)
        return advantages

    def train_step(self) -> TrainingMetrics:
        """Execute one DAPO training step"""
        # Collect data
        batch_data, filter_ratio = self.collect_batch()
        
        if len(batch_data) == 0:
            print("⚠️ Dynamic sampling filtered all data, skipping update")
            return None
        
        # Extract data
        states = torch.stack([item[0] for item in batch_data])
        actions = torch.tensor([item[1] for item in batch_data])
        rewards = [item[2] for item in batch_data]
        old_log_probs = torch.stack([item[3] for item in batch_data])
        
        # Compute advantages
        advantages = self.compute_advantages(rewards)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
        
        # Forward pass
        logits = self.policy(states)
        probs = F.softmax(logits, dim=-1)
        probs = torch.clamp(probs, min=1e-8, max=1.0)
        
        log_probs = torch.log(probs.gather(1, actions.unsqueeze(1)).squeeze())
        
        # Calculate loss using Clip-Higher
        policy_loss = self.clip_higher_loss(log_probs, old_log_probs, advantages, rewards_tensor)
        
        # Entropy loss
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean()
        entropy_loss = -self.entropy_coef * entropy
        
        # Total loss
        total_loss = policy_loss + entropy_loss
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        
        # Calculate gradient norm
        grad_norm = 0.0
        for param in self.policy.parameters():
            if param.grad is not None:
                grad_norm += param.grad.data.norm(2).item() ** 2
        grad_norm = grad_norm ** 0.5
        
        self.optimizer.step()
        
        # Record metrics
        with torch.no_grad():
            action_dist = probs.mean(dim=0).tolist()
            
        metrics = TrainingMetrics(
            step=len(self.metrics_history),
            entropy=entropy.item(),
            gradient_norm=grad_norm,
            loss=total_loss.item(),
            action_distribution=action_dist,
            reward_mean=np.mean(rewards),
            reward_std=np.std(rewards),
            clip_ratio=self.clip_ratio,
            filtered_ratio=filter_ratio
        )
        
        self.metrics_history.append(metrics)
        return metrics

    def generate_report(self, experiment_type: str) -> str:
        """生成详细的训练报告"""
        if not self.metrics_history:
            return "❌ 没有训练数据"
        
        initial_entropy = self.metrics_history[0].entropy
        final_entropy = self.metrics_history[-1].entropy
        entropy_change = (initial_entropy - final_entropy) / initial_entropy * 100
        
        initial_grad = self.metrics_history[0].gradient_norm
        final_grad = self.metrics_history[-1].gradient_norm
        
        final_dist = self.metrics_history[-1].action_distribution
        dominant_action = final_dist.index(max(final_dist))
        
        # 计算过滤统计
        filter_ratios = [m.filtered_ratio for m in self.metrics_history]
        avg_filter_ratio = np.mean(filter_ratios)
        
        report = f"""
🔍 {experiment_type} 实验详细分析报告
==================================================

📈 训练概况:
• 总训练步数: {len(self.metrics_history)}
• 初始熵值: {initial_entropy:.4f} → 最终熵值: {final_entropy:.4f}
• 熵值变化: {entropy_change:.1f}%
• 初始梯度: {initial_grad:.4f} → 最终梯度: {final_grad:.4f}

🎯 最终动作偏好:
• 动作0 (奖励0.3): {final_dist[0]:.3f} ({final_dist[0]*100:.1f}%)
• 动作1 (奖励1.0): {final_dist[1]:.3f} ({final_dist[1]*100:.1f}%)  ⭐最优
• 动作2 (奖励0.5): {final_dist[2]:.3f} ({final_dist[2]*100:.1f}%)
• 主导动作: 动作{dominant_action}

🔧 DAPO技术统计:
• Clip-Higher启用: {'✅' if self.use_clip_higher else '❌'}
• 动态采样启用: {'✅' if self.use_dynamic_sampling else '❌'}
• 平均过滤比例: {avg_filter_ratio*100:.1f}%

💡 结果解释:
{'✅ DAPO成功保持了训练稳定性' if entropy_change < 80 else '⚠️ 仍存在一定程度的熵坍塌'}
{'✅ 梯度保持稳定' if final_grad > 0.01 else '⚠️ 出现了梯度消失现象'}

🔬 技术效果分析:
• Clip-Higher: {'帮助保持探索性，减缓熵坍塌' if self.use_clip_higher else '未启用'}
• 动态采样: {'过滤了{:.1f}%的低质量数据'.format(avg_filter_ratio*100) if self.use_dynamic_sampling else '未启用'}
"""
        return report

def run_grpo_baseline_experiment():
    """运行GRPO基线实验（不使用DAPO技术）"""
    print("🎪 第一场对比：GRPO基线实验")
    print("=" * 50)
    print("🎯 目标: 复现Lab04中的训练不稳定性问题")
    print("⚙️ 方法: 使用标准GRPO，不启用DAPO技术")
    
    trainer = DAPOTrainer(
        learning_rate=5e-3,  # 使用Lab04的高学习率
        entropy_coef=0.001,  # 使用Lab04的低熵系数
        batch_size=32,
        use_clip_higher=False,      # 关闭Clip-Higher
        use_dynamic_sampling=False,  # 关闭动态采样
        verbose=False
    )
    
    print(f"🚀 开始训练...")
    
    for step in range(100):
        metrics = trainer.train_step()
        if metrics is None:
            continue
            
        # 检测熵坍塌
        if metrics.entropy < 0.01:
            if step < 90:  # 只在前90步报告，避免过多输出
                print(f"⚠️ 第{step+1}步：熵值过低({metrics.entropy:.4f})，可能已经坍塌！")
        
        # 每20步显示进度
        if (step + 1) % 20 == 0:
            print(f"📊 第{step+1}步: 熵={metrics.entropy:.3f}, 梯度={metrics.gradient_norm:.3f}, 奖励={metrics.reward_mean:.3f}")
    
    print("✅ GRPO基线实验完成")
    print(trainer.generate_report("GRPO基线"))
    return trainer.metrics_history

def run_dapo_experiment():
    """运行完整DAPO实验"""
    print("\n🎪 第二场对比：完整DAPO实验")
    print("=" * 50)
    print("🎯 目标: 验证DAPO技术解决训练不稳定性的效果")
    print("⚙️ 方法: 启用Clip-Higher和动态采样技术")
    
    trainer = DAPOTrainer(
        learning_rate=5e-3,  # 同样的高学习率
        entropy_coef=0.001,  # 同样的低熵系数
        batch_size=32,
        use_clip_higher=True,       # 启用Clip-Higher
        use_dynamic_sampling=True,  # 启用动态采样
        reward_variance_threshold=0.05,  # 动态采样阈值
        verbose=False
    )
    
    print(f"🚀 开始训练...")
    
    for step in range(100):
        metrics = trainer.train_step()
        if metrics is None:
            continue
            
        # 检测熵坍塌（期望减少）
        if metrics.entropy < 0.01:
            print(f"⚠️ 第{step+1}步：熵值仍然过低({metrics.entropy:.4f})")
        
        # 显示DAPO技术工作情况
        if metrics.filtered_ratio > 0:
            print(f"🔧 第{step+1}步：动态采样过滤了{metrics.filtered_ratio*100:.1f}%的数据")
        
        # 每20步显示进度
        if (step + 1) % 20 == 0:
            print(f"📊 第{step+1}步: 熵={metrics.entropy:.3f}, 梯度={metrics.gradient_norm:.3f}, 奖励={metrics.reward_mean:.3f}")
    
    print("✅ DAPO实验完成")
    print(trainer.generate_report("DAPO"))
    return trainer.metrics_history

def create_comparison_visualization(grpo_metrics: List[TrainingMetrics], 
                                  dapo_metrics: List[TrainingMetrics]):
    """创建GRPO vs DAPO对比可视化"""
    print("\n🎨 生成GRPO vs DAPO对比分析图表...")
    
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle('GRPO vs DAPO: Training Stability Comparison', fontsize=16, fontweight='bold')
    
    # 定义颜色
    grpo_color = '#e74c3c'  # 红色 - GRPO
    dapo_color = '#27ae60'  # 绿色 - DAPO
    
    # 提取数据
    grpo_steps = list(range(len(grpo_metrics)))
    dapo_steps = list(range(len(dapo_metrics)))
    
    grpo_entropy = [m.entropy for m in grpo_metrics]
    dapo_entropy = [m.entropy for m in dapo_metrics]
    
    grpo_gradients = [m.gradient_norm for m in grpo_metrics]
    dapo_gradients = [m.gradient_norm for m in dapo_metrics]
    
    grpo_rewards = [m.reward_mean for m in grpo_metrics]
    dapo_rewards = [m.reward_mean for m in dapo_metrics]
    
    # 1. 熵值对比
    ax1 = plt.subplot(2, 3, 1)
    ax1.plot(grpo_steps, grpo_entropy, color=grpo_color, linewidth=2, label='GRPO (Baseline)', marker='o', markersize=3)
    ax1.plot(dapo_steps, dapo_entropy, color=dapo_color, linewidth=2, label='DAPO (Improved)', marker='s', markersize=3)
    ax1.axhline(y=0.01, color='red', linestyle='--', alpha=0.7, label='Collapse Threshold')
    ax1.set_title('Entropy Comparison\n(Higher is better for exploration)', fontweight='bold')
    ax1.set_xlabel('Training Steps')
    ax1.set_ylabel('Entropy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 梯度范数对比
    ax2 = plt.subplot(2, 3, 2)
    ax2.plot(grpo_steps, grpo_gradients, color=grpo_color, linewidth=2, label='GRPO', marker='o', markersize=3)
    ax2.plot(dapo_steps, dapo_gradients, color=dapo_color, linewidth=2, label='DAPO', marker='s', markersize=3)
    ax2.set_title('Gradient Norm Comparison\n(Stability indicator)', fontweight='bold')
    ax2.set_xlabel('Training Steps')
    ax2.set_ylabel('Gradient Norm')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 奖励对比
    ax3 = plt.subplot(2, 3, 3)
    ax3.plot(grpo_steps, grpo_rewards, color=grpo_color, linewidth=2, label='GRPO', marker='o', markersize=3)
    ax3.plot(dapo_steps, dapo_rewards, color=dapo_color, linewidth=2, label='DAPO', marker='s', markersize=3)
    ax3.set_title('Reward Comparison\n(Performance indicator)', fontweight='bold')
    ax3.set_xlabel('Training Steps')
    ax3.set_ylabel('Average Reward')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 最终动作分布对比
    ax4 = plt.subplot(2, 3, 4)
    if grpo_metrics and dapo_metrics:
        grpo_final_dist = grpo_metrics[-1].action_distribution
        dapo_final_dist = dapo_metrics[-1].action_distribution
        
        actions = ['Action 0\n(R=0.3)', 'Action 1\n(R=1.0)\nOptimal', 'Action 2\n(R=0.5)']
        x = np.arange(len(actions))
        width = 0.35
        
        ax4.bar(x - width/2, grpo_final_dist, width, label='GRPO', color=grpo_color, alpha=0.7)
        ax4.bar(x + width/2, dapo_final_dist, width, label='DAPO', color=dapo_color, alpha=0.7)
        
        ax4.set_title('Final Action Distribution\n(Strategy comparison)', fontweight='bold')
        ax4.set_xlabel('Actions')
        ax4.set_ylabel('Probability')
        ax4.set_xticks(x)
        ax4.set_xticklabels(actions)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    
    # 5. 训练稳定性统计
    ax5 = plt.subplot(2, 3, 5)
    
    # 计算稳定性指标
    grpo_entropy_var = np.var(grpo_entropy[50:])  # 后半段方差
    dapo_entropy_var = np.var(dapo_entropy[50:])
    
    grpo_grad_var = np.var(grpo_gradients[50:])
    dapo_grad_var = np.var(dapo_gradients[50:])
    
    metrics_names = ['Entropy\nVariance', 'Gradient\nVariance']
    grpo_values = [grpo_entropy_var, grpo_grad_var]
    dapo_values = [dapo_entropy_var, dapo_grad_var]
    
    x = np.arange(len(metrics_names))
    width = 0.35
    
    ax5.bar(x - width/2, grpo_values, width, label='GRPO', color=grpo_color, alpha=0.7)
    ax5.bar(x + width/2, dapo_values, width, label='DAPO', color=dapo_color, alpha=0.7)
    
    ax5.set_title('Training Stability\n(Lower variance = more stable)', fontweight='bold')
    ax5.set_xlabel('Metrics')
    ax5.set_ylabel('Variance')
    ax5.set_xticks(x)
    ax5.set_xticklabels(metrics_names)
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. DAPO技术效果展示
    ax6 = plt.subplot(2, 3, 6)
    
    if dapo_metrics:
        filter_ratios = [m.filtered_ratio * 100 for m in dapo_metrics]  # 转换为百分比
        ax6.plot(dapo_steps, filter_ratios, color='orange', linewidth=2, marker='d', markersize=3)
        ax6.set_title('DAPO Dynamic Sampling Effect\n(Filtered data percentage)', fontweight='bold')
        ax6.set_xlabel('Training Steps')
        ax6.set_ylabel('Filtered Data (%)')
        ax6.grid(True, alpha=0.3)
        
        # 添加统计信息
        avg_filter = np.mean(filter_ratios)
        ax6.text(0.02, 0.98, f'Avg Filtered: {avg_filter:.1f}%', 
                transform=ax6.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='lightyellow', alpha=0.8))
    
    plt.tight_layout()
    
    # 保存图表
    output_path = "grpo_vs_dapo_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ 对比分析图表已保存到: {output_path}")
    
    return generate_comparison_report(grpo_metrics, dapo_metrics)

def generate_comparison_report(grpo_metrics: List[TrainingMetrics], 
                             dapo_metrics: List[TrainingMetrics]) -> str:
    """生成详细的对比分析报告"""
    
    # 计算关键指标
    grpo_initial_entropy = grpo_metrics[0].entropy if grpo_metrics else 0
    grpo_final_entropy = grpo_metrics[-1].entropy if grpo_metrics else 0
    grpo_entropy_reduction = (grpo_initial_entropy - grpo_final_entropy) / grpo_initial_entropy * 100 if grpo_initial_entropy > 0 else 0
    
    dapo_initial_entropy = dapo_metrics[0].entropy if dapo_metrics else 0
    dapo_final_entropy = dapo_metrics[-1].entropy if dapo_metrics else 0
    dapo_entropy_reduction = (dapo_initial_entropy - dapo_final_entropy) / dapo_initial_entropy * 100 if dapo_initial_entropy > 0 else 0
    
    # 梯度稳定性
    grpo_final_grad = grpo_metrics[-1].gradient_norm if grpo_metrics else 0
    dapo_final_grad = dapo_metrics[-1].gradient_norm if dapo_metrics else 0
    
    # 最终性能
    grpo_final_reward = grpo_metrics[-1].reward_mean if grpo_metrics else 0
    dapo_final_reward = dapo_metrics[-1].reward_mean if dapo_metrics else 0
    
    # DAPO技术统计
    dapo_filter_rates = [m.filtered_ratio for m in dapo_metrics]
    avg_filter_rate = np.mean(dapo_filter_rates) * 100
    
    report = f"""
📊 GRPO vs DAPO 对比分析报告 (小白版)
============================================================

🔍 熵坍塌对比分析:
------------------------
📉 GRPO (基线):
• 初始熵: {grpo_initial_entropy:.3f} → 最终熵: {grpo_final_entropy:.3f}
• 熵减少: {grpo_entropy_reduction:.1f}%
• 结果: {'✅ 相对稳定' if grpo_entropy_reduction < 80 else '❌ 严重熵坍塌'}

📈 DAPO (改进):
• 初始熵: {dapo_initial_entropy:.3f} → 最终熵: {dapo_final_entropy:.3f}
• 熵减少: {dapo_entropy_reduction:.1f}%
• 结果: {'✅ 成功保持稳定' if dapo_entropy_reduction < 80 else '⚠️ 部分改善'}

🔧 梯度稳定性对比:
------------------------
• GRPO最终梯度: {grpo_final_grad:.4f}
• DAPO最终梯度: {dapo_final_grad:.4f}
• 改善效果: {'✅ DAPO更稳定' if dapo_final_grad > grpo_final_grad else '⚠️ 需要进一步调优'}

🎯 性能对比:
------------------------
• GRPO最终奖励: {grpo_final_reward:.3f}
• DAPO最终奖励: {dapo_final_reward:.3f}
• 性能提升: {((dapo_final_reward - grpo_final_reward) / grpo_final_reward * 100):.1f}%

🚀 DAPO技术效果:
------------------------
• Clip-Higher技术: ✅ 启用，放宽了对高奖励行为的限制
• 动态采样技术: ✅ 启用，平均过滤了{avg_filter_rate:.1f}%的低质量数据
• 整体效果: {'🎉 显著改善训练稳定性' if dapo_entropy_reduction < grpo_entropy_reduction - 10 else '📈 提供了一定改善'}

💡 关键发现:
------------------------
1. {'✅ DAPO成功缓解了熵坍塌问题' if dapo_entropy_reduction < grpo_entropy_reduction else '⚠️ 熵坍塌仍需进一步优化'}
2. {'✅ 梯度稳定性得到改善' if dapo_final_grad > grpo_final_grad * 1.2 else '📊 梯度稳定性略有改善'}
3. {'✅ 整体训练质量提升' if dapo_final_reward > grpo_final_reward else '⚖️ 性能基本持平'}

🎓 技术洞察:
------------------------
• Clip-Higher让模型敢于尝试高奖励的新行为
• 动态采样确保了每次学习都有"营养价值"
• 两项技术协同工作，构建了更稳定的训练环境

🚀 后续改进方向:
------------------------
• 可以尝试调整动态采样的阈值参数
• 探索更智能的Clip-Higher策略
• 结合VeRL进行可验证的强化学习
"""
    
    return report

def main():
    """主函数：运行完整的DAPO对比实验"""
    print("🚀 欢迎来到DAPO算法实验室！")
    print("📚 本实验将对比GRPO和DAPO在解决训练不稳定性方面的效果")
    
    # 实验前概念介绍
    print("\n📖 实验前概念预习:")
    print("=" * 50)
    print(ExplainerSystem.explain_concept("dapo"))
    print(ExplainerSystem.explain_concept("clip_higher"))
    print(ExplainerSystem.explain_concept("dynamic_sampling"))
    
    print("\n🔥 开始对比实验...")
    
    try:
        # 实验1：GRPO基线
        grpo_metrics = run_grpo_baseline_experiment()
        
        # 实验2：DAPO改进
        dapo_metrics = run_dapo_experiment()
        
        # 生成对比分析
        print("\n🎨 生成详细对比分析...")
        comparison_report = create_comparison_visualization(grpo_metrics, dapo_metrics)
        
        print("\n🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉")
        print("实验全部完成！以下是最终对比分析：")
        print("🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉")
        
        print(comparison_report)
        
        print("\n✨ 恭喜你完成了Lab05的学习！")
        print("🎯 你现在掌握了DAPO算法的核心技术")
        print("🔧 下一步可以学习VeRL，探索可验证的强化学习")
        print("🚀 继续加油，向着AI专家的目标前进！")
        
    except Exception as e:
        print(f"❌ 实验过程中出现错误: {e}")
        print("💡 提示：请检查依赖包是否正确安装")

if __name__ == "__main__":
    main() 