"""
Lab07: 对抗性偏好优化 (APO) 实验
=================================

本实验通过二维平面上的简化博弈展示APO的核心理念：
1. 攻击者 (Attacker): 试图移动到检测者预测"不会"去的位置
2. 检测者 (Detector): 试图预测攻击者"会"去往的区域
3. Min-Max博弈: 双方交替优化，协同进化

目标：理解APO框架如何实现攻击-检测的动态演化
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import random
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation

# 设置matplotlib支持中文显示
def setup_matplotlib():
    """配置matplotlib以支持中文字体显示"""
    try:
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        print("✅ Matplotlib configuration successful")
    except Exception as e:
        print(f"⚠️ Matplotlib config warning: {e}")
        # 如果中文字体不可用，使用英文标签
        plt.rcParams['font.family'] = 'DejaVu Sans'

setup_matplotlib()

@dataclass
class GameState:
    """博弈状态记录"""
    round: int
    attacker_pos: Tuple[float, float]
    detector_accuracy: float
    attacker_reward: float
    detector_loss: float
    game_balance: float  # 博弈平衡度

class ExplainerSystem:
    """APO概念解释系统"""
    
    @staticmethod
    def explain_concept(concept: str) -> str:
        explanations = {
            "apo": """
🎯 APO (Adversarial Preference Optimization):
• Core idea: Formalize LLM vs Reward Model as zero-sum game
• Attacker (LLM): Generate content to maximize detector's score
• Detector (Reward Model): Minimize classification error
• Outcome: Both agents evolve together, improving each other
• Like cat-and-mouse game where both get smarter over time
""",
            "min_max": """
🎮 Min-Max Game Theory:
• Attacker's goal: max(Reward) - find detector's blind spots
• Detector's goal: min(Loss) - improve detection capability
• Nash Equilibrium: Stable point where neither can improve unilaterally
• Dynamic process: Continuous evolution through alternating updates
• Result: Enhanced robustness for both attack and defense
""",
            "adversarial_training": """
⚔️ Adversarial Training Process:
• Phase 1: Train attacker to fool current detector
• Phase 2: Train detector to catch current attacker
• Iteration: Repeat alternately to drive co-evolution
• Benefits: No new labeled data needed, automatic improvement
• Applications: AI safety, robustness, alignment
""",
            "2d_game": """
📍 2D Plane Game Simulation:
• Setup: Attacker (red dot) vs Detector (blue regions)
• Attacker: Tries to move where detector thinks it won't go
• Detector: Predicts where attacker will move next
• Visualization: Watch strategy evolution in real-time
• Learning: See how both agents adapt their strategies
"""
        }
        return explanations.get(concept, "Concept not found")

class Attacker(nn.Module):
    """攻击者：试图找到检测者的盲点"""
    
    def __init__(self, learning_rate: float = 0.1):
        super().__init__()
        self.position = nn.Parameter(torch.tensor([0.0, 0.0], requires_grad=True))
        self.optimizer = torch.optim.Adam([self.position], lr=learning_rate)
        self.history = []
        self.name = "Attacker"
        
    def forward(self) -> torch.Tensor:
        """返回当前位置"""
        return self.position
    
    def get_position(self) -> Tuple[float, float]:
        """获取当前位置"""
        pos = self.position.detach().numpy()
        return (float(pos[0]), float(pos[1]))
    
    def attack_step(self, detector, target_area: str = "low_detection"):
        """
        执行攻击步骤：试图移动到检测概率低的区域
        
        Args:
            detector: 检测者模型
            target_area: 目标区域类型
        """
        self.optimizer.zero_grad()
        
        # 获取当前位置
        current_pos = self.position.unsqueeze(0)  # [1, 2]
        
        # 检测者对当前位置的预测
        detection_prob = detector(current_pos)
        
        # 攻击者的目标：最小化被检测的概率
        # 即找到检测者认为攻击者"不会"去的地方
        attack_loss = detection_prob.mean()  # 最小化检测概率
        
        attack_loss.backward()
        self.optimizer.step()
        
        # 限制位置在合理范围内
        with torch.no_grad():
            self.position.data = torch.clamp(self.position.data, -3, 3)
        
        # 记录历史
        pos = self.get_position()
        self.history.append(pos)
        
        return attack_loss.item(), detection_prob.item()

class Detector(nn.Module):
    """检测者：试图预测攻击者的行为模式"""
    
    def __init__(self, hidden_dim: int = 64, learning_rate: float = 0.01):
        super().__init__()
        
        # 神经网络结构：输入2D坐标，输出检测概率
        self.network = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        self.accuracy_history = []
        self.name = "Detector"
        
    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        """
        预测给定位置的攻击概率
        
        Args:
            positions: [batch_size, 2] 位置坐标
            
        Returns:
            detection_probs: [batch_size, 1] 检测概率
        """
        return self.network(positions)
    
    def defend_step(self, attacker_positions: List[Tuple[float, float]], 
                   safe_positions: List[Tuple[float, float]]):
        """
        执行检测步骤：学习区分攻击者位置和安全位置
        
        Args:
            attacker_positions: 攻击者历史位置
            safe_positions: 安全区域位置
        """
        if len(attacker_positions) < 2:
            return 0.5, 0.5  # 初始情况
        
        self.optimizer.zero_grad()
        
        # 准备训练数据
        attack_data = torch.tensor(attacker_positions[-20:], dtype=torch.float32)  # 最近20个位置
        safe_data = torch.tensor(safe_positions[-20:], dtype=torch.float32)
        
        # 预测
        attack_preds = self.forward(attack_data)
        safe_preds = self.forward(safe_data)
        
        # 标签：攻击者位置应该被检测到(1)，安全位置不应该(0)
        attack_labels = torch.ones(len(attack_data), 1)
        safe_labels = torch.zeros(len(safe_data), 1)
        
        # 分类损失
        attack_loss = F.binary_cross_entropy(attack_preds, attack_labels)
        safe_loss = F.binary_cross_entropy(safe_preds, safe_labels)
        total_loss = attack_loss + safe_loss
        
        total_loss.backward()
        self.optimizer.step()
        
        # 计算准确率
        attack_acc = ((attack_preds > 0.5) == attack_labels).float().mean().item()
        safe_acc = ((safe_preds < 0.5) == safe_labels).float().mean().item()
        overall_acc = (attack_acc + safe_acc) / 2
        
        self.accuracy_history.append(overall_acc)
        
        return overall_acc, total_loss.item()

class APOGameEnvironment:
    """APO博弈环境"""
    
    def __init__(self, grid_size: int = 100):
        self.grid_size = grid_size
        self.attacker = Attacker(learning_rate=0.05)
        self.detector = Detector(hidden_dim=32, learning_rate=0.01)
        
        # 生成一些安全位置作为对比
        self.safe_positions = self._generate_safe_positions(50)
        self.game_history = []
        
        print(f"🎮 APO Game Environment initialized")
        print(f"🔴 Attacker: Try to find detector's blind spots")
        print(f"🔵 Detector: Learn to predict attacker behavior")
        
    def _generate_safe_positions(self, num_positions: int) -> List[Tuple[float, float]]:
        """生成一些"安全"位置用于训练检测者"""
        positions = []
        for _ in range(num_positions):
            # 在网格中随机生成位置
            x = random.uniform(-2, 2)
            y = random.uniform(-2, 2)
            positions.append((x, y))
        return positions
    
    def play_round(self, round_num: int) -> GameState:
        """
        执行一轮APO博弈
        
        Returns:
            GameState: 当前回合的博弈状态
        """
        
        # Phase 1: 攻击者尝试找到检测者的盲点
        attack_loss, detection_prob = self.attacker.attack_step(self.detector)
        attacker_reward = 1.0 - detection_prob  # 奖励 = 1 - 被检测概率
        
        # Phase 2: 检测者学习预测攻击者行为
        detector_acc, detector_loss = self.detector.defend_step(
            self.attacker.history, self.safe_positions
        )
        
        # 计算博弈平衡度
        game_balance = abs(attacker_reward - detector_acc)  # 越小越平衡
        
        # 记录状态
        state = GameState(
            round=round_num,
            attacker_pos=self.attacker.get_position(),
            detector_accuracy=detector_acc,
            attacker_reward=attacker_reward,
            detector_loss=detector_loss,
            game_balance=game_balance
        )
        
        self.game_history.append(state)
        return state
    
    def get_detection_heatmap(self, resolution: int = 50) -> np.ndarray:
        """生成检测概率热力图"""
        x = np.linspace(-3, 3, resolution)
        y = np.linspace(-3, 3, resolution)
        xx, yy = np.meshgrid(x, y)
        
        positions = torch.tensor(
            np.stack([xx.ravel(), yy.ravel()], axis=1), 
            dtype=torch.float32
        )
        
        with torch.no_grad():
            probs = self.detector(positions).numpy()
        
        return probs.reshape(resolution, resolution)

def run_apo_experiment(num_rounds: int = 200):
    """运行APO对抗博弈实验"""
    print("🎪 开始APO对抗性偏好优化实验")
    print("=" * 50)
    
    # 初始化游戏环境
    env = APOGameEnvironment()
    
    print("🚀 开始博弈...")
    
    # 运行博弈
    for round_num in range(num_rounds):
        state = env.play_round(round_num)
        
        # 每20轮显示进度
        if (round_num + 1) % 40 == 0:
            print(f"📊 Round {round_num+1}: "
                  f"Attacker位置=({state.attacker_pos[0]:.2f}, {state.attacker_pos[1]:.2f}), "
                  f"Detector准确率={state.detector_accuracy:.3f}, "
                  f"博弈平衡度={state.game_balance:.3f}")
    
    print("✅ APO博弈实验完成")
    return env

def create_apo_visualization(env: APOGameEnvironment):
    """创建APO博弈可视化"""
    print("\n🎨 生成APO博弈分析图表...")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('APO Adversarial Preference Optimization Analysis', fontsize=16, fontweight='bold')
    
    # 1. 攻击者轨迹和检测热力图
    ax = axes[0, 0]
    heatmap = env.get_detection_heatmap()
    im = ax.imshow(heatmap, extent=[-3, 3, -3, 3], origin='lower', cmap='Blues', alpha=0.7)
    
    # 绘制攻击者轨迹
    if len(env.attacker.history) > 1:
        trajectory = np.array(env.attacker.history)
        ax.plot(trajectory[:, 0], trajectory[:, 1], 'r-', alpha=0.6, linewidth=2, label='Attacker Path')
        ax.scatter(trajectory[-1, 0], trajectory[-1, 1], color='red', s=100, label='Current Position', zorder=5)
    
    ax.set_title('Attacker vs Detector Heatmap', fontweight='bold')
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.legend()
    plt.colorbar(im, ax=ax, label='Detection Probability')
    
    # 2. 博弈平衡度演化
    ax = axes[0, 1]
    rounds = [s.round for s in env.game_history]
    balance = [s.game_balance for s in env.game_history]
    ax.plot(rounds, balance, 'purple', linewidth=2)
    ax.set_title('Game Balance Evolution', fontweight='bold')
    ax.set_xlabel('Round')
    ax.set_ylabel('Balance Score (lower = more balanced)')
    ax.grid(True, alpha=0.3)
    
    # 3. 攻击者奖励 vs 检测者准确率
    ax = axes[0, 2]
    attacker_rewards = [s.attacker_reward for s in env.game_history]
    detector_accs = [s.detector_accuracy for s in env.game_history]
    ax.plot(rounds, attacker_rewards, 'r-', label='Attacker Reward', linewidth=2)
    ax.plot(rounds, detector_accs, 'b-', label='Detector Accuracy', linewidth=2)
    ax.set_title('Attacker vs Detector Performance', fontweight='bold')
    ax.set_xlabel('Round')
    ax.set_ylabel('Score')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. 攻击者位置热力图
    ax = axes[1, 0]
    if len(env.attacker.history) > 10:
        positions = np.array(env.attacker.history)
        ax.hexbin(positions[:, 0], positions[:, 1], gridsize=20, cmap='Reds', alpha=0.7)
        ax.set_title('Attacker Position Density', fontweight='bold')
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
    
    # 5. 检测者损失演化
    ax = axes[1, 1]
    detector_losses = [s.detector_loss for s in env.game_history]
    ax.plot(rounds, detector_losses, 'orange', linewidth=2)
    ax.set_title('Detector Loss Evolution', fontweight='bold')
    ax.set_xlabel('Round')
    ax.set_ylabel('Loss')
    ax.grid(True, alpha=0.3)
    
    # 6. Min-Max博弈动态
    ax = axes[1, 2]
    # 计算滑动平均
    window_size = 20
    if len(attacker_rewards) >= window_size:
        attacker_smooth = np.convolve(attacker_rewards, np.ones(window_size)/window_size, mode='valid')
        detector_smooth = np.convolve(detector_accs, np.ones(window_size)/window_size, mode='valid')
        smooth_rounds = rounds[window_size-1:]
        
        ax.plot(smooth_rounds, attacker_smooth, 'r-', label='Attacker (Smooth)', linewidth=2)
        ax.plot(smooth_rounds, detector_smooth, 'b-', label='Detector (Smooth)', linewidth=2)
    
    ax.set_title('Min-Max Game Dynamics (Smoothed)', fontweight='bold')
    ax.set_xlabel('Round')
    ax.set_ylabel('Score')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图表
    output_path = 'apo_adversarial_game_analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ APO博弈分析图表已保存到: {output_path}")
    plt.show()

def generate_apo_report(env: APOGameEnvironment) -> str:
    """生成APO实验详细分析报告"""
    
    final_state = env.game_history[-1]
    
    # 计算统计指标
    attacker_rewards = [s.attacker_reward for s in env.game_history]
    detector_accs = [s.detector_accuracy for s in env.game_history]
    game_balances = [s.game_balance for s in env.game_history]
    
    avg_attacker_reward = np.mean(attacker_rewards[-50:])  # 最后50轮平均
    avg_detector_acc = np.mean(detector_accs[-50:])
    final_balance = np.mean(game_balances[-20:])
    
    # 计算收敛性
    early_balance = np.mean(game_balances[:20]) if len(game_balances) >= 20 else 0
    balance_improvement = early_balance - final_balance
    
    # 分析对抗演化
    attacker_trajectory = np.array(env.attacker.history)
    movement_variance = np.var(attacker_trajectory, axis=0).sum()
    
    report = f"""
📊 APO Adversarial Preference Optimization 实验报告
============================================================

🔍 最终博弈状态:
------------------------
📍 攻击者最终位置: ({final_state.attacker_pos[0]:.3f}, {final_state.attacker_pos[1]:.3f})
🎯 检测者最终准确率: {final_state.detector_accuracy:.3f} ({final_state.detector_accuracy*100:.1f}%)
⚖️ 博弈平衡度: {final_state.game_balance:.3f}
🔄 总博弈轮数: {final_state.round + 1}

🎮 Min-Max博弈分析:
------------------------
🔴 攻击者表现:
• 最终奖励: {final_state.attacker_reward:.3f}
• 平均奖励(最后50轮): {avg_attacker_reward:.3f}
• 位置探索方差: {movement_variance:.3f}

🔵 检测者表现:
• 最终准确率: {final_state.detector_accuracy:.3f}
• 平均准确率(最后50轮): {avg_detector_acc:.3f}
• 最终损失: {final_state.detector_loss:.3f}

⚖️ 博弈均衡分析:
------------------------
• 初期平衡度: {early_balance:.3f}
• 最终平衡度: {final_balance:.3f}
• 平衡改善: {balance_improvement:.3f} ({'✅ 趋向平衡' if balance_improvement > 0 else '⚠️ 仍在调整'})

🔬 对抗演化洞察:
------------------------
1. **攻击策略演化**:
   {'✅ 攻击者学会了探索检测盲点' if movement_variance > 1.0 else '📊 攻击者策略相对保守'}

2. **检测能力提升**:
   {'✅ 检测者成功提升了预测能力' if avg_detector_acc > 0.6 else '⚠️ 检测者仍需改进'}

3. **博弈平衡性**:
   {'🎯 达到了良好的Nash均衡' if final_balance < 0.3 else '🔄 仍在动态博弈中'}

💡 APO框架优势:
------------------------
• ✅ 实现了攻击-检测的协同进化
• ✅ 无需额外标注数据的自动化对抗训练
• ✅ 动态平衡机制防止一方过于强势
• ✅ 为复杂AI安全场景提供理论基础

🚀 技术洞察:
------------------------
1. **Min-Max博弈本质**: 攻击者最大化奖励，检测者最小化损失
2. **Nash均衡收敛**: 双方策略逐渐稳定到最优反应点
3. **协同进化机制**: 相互促进提升，而非零和竞争
4. **动态适应性**: 持续学习对方策略变化

🎓 学习要点:
------------------------
• APO = 将AI安全问题形式化为博弈论框架
• 对抗训练 = 最有效的鲁棒性提升方法
• 动态均衡 = 避免模式崩溃的关键机制
• 协同进化 = 实现双方共同提升的核心

🔮 后续方向:
• Lab08: 构建初步对抗循环（DAPO攻击者 vs 学习检测者）
• Lab09: 基于VeRL的稳定对抗（引入可验证真值锚点）
• Lab10: 完整DAPO+APO+VeRL系统集成
"""
    
    return report

def main():
    """主实验流程"""
    print("🚀 欢迎来到APO对抗性偏好优化实验室！")
    print("📚 本实验将展示攻击者与检测者的Min-Max博弈过程")
    
    # 概念预习
    print("\n📖 实验前概念预习:")
    print("=" * 50)
    
    explainer = ExplainerSystem()
    for concept in ['apo', 'min_max', 'adversarial_training', '2d_game']:
        print(explainer.explain_concept(concept))
    
    print("🔥 开始APO博弈实验...")
    
    # 运行实验
    env = run_apo_experiment(num_rounds=200)
    
    # 生成分析报告
    print("\n" + "="*50)
    print(generate_apo_report(env))
    
    # 生成可视化
    create_apo_visualization(env)
    
    print("\n🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉")
    print("APO对抗性偏好优化实验全部完成！")
    print("🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉")
    
    print("\n✨ 恭喜你完成了Lab07的学习！")
    print("🎯 你现在掌握了APO对抗性偏好优化的核心理论")
    print("🔧 下一步可以学习Lab08，构建初步对抗循环")
    print("🚀 继续加油，向着AI安全专家的目标前进！")

if __name__ == "__main__":
    main() 