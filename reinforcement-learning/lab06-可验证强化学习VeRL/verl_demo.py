"""
Lab06: 可验证强化学习 (VeRL) 实验
=================================

本实验对比两种奖励机制：
1. VeRL方案：使用确定性函数验证数学答案的正确性
2. 传统方案：使用学习型奖励模型（存在偏见和漏洞）

目标：展示VeRL在避免Reward Hacking和提供稳定训练方面的优势
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import random
import re
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any
from pathlib import Path

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
class ExperimentMetrics:
    """VeRL实验的关键指标"""
    step: int
    accuracy: float           # 答案正确率
    reward_mean: float        # 平均奖励
    reward_std: float         # 奖励标准差
    entropy: float           # 策略熵
    gradient_norm: float     # 梯度范数
    hacking_score: float     # Reward Hacking得分

class ExplainerSystem:
    """VeRL概念解释系统"""
    
    @staticmethod
    def explain_concept(concept: str) -> str:
        explanations = {
            "verl": """
🔍 VeRL (Verified Reinforcement Learning):
• Traditional RL: Use learned reward models (may have bias/errors)
• VeRL: Use deterministic, programmable functions as rewards
• Key advantage: Eliminate reward model bias and hacking vulnerabilities
• Like having a "perfect judge" that never makes mistakes
""",
            "function_reward": """
⚙️ Function-based Reward:
• Definition: Deterministic Python function that verifies correctness
• Input: Model's output (e.g., math answer)
• Output: Binary (0/1) or quantitative reward
• Guarantee: Always provides "ground truth" evaluation
• No bias, no shortcuts, no reward hacking possible
""",
            "reward_hacking": """
🎯 Reward Hacking:
• Problem: Models learn to exploit flaws in reward models
• Example: Getting high scores for wrong answers by mimicking patterns
• Traditional approach: Vulnerable to this issue
• VeRL solution: Function-based rewards eliminate loopholes
• Result: Models must actually solve problems correctly
""",
            "math_task": """
📊 Math Problem Task:
• Task: Solve simple arithmetic problems (addition, subtraction)
• VeRL reward: Execute calculation and verify exact answer
• Traditional reward: Learned model that may have preferences/bias
• Comparison: See which approach produces truly correct answers
"""
        }
        return explanations.get(concept, "Concept not found")

class MathProblemEnvironment:
    """数学问题环境：生成简单的算术题"""
    
    def __init__(self, difficulty_range: Tuple[int, int] = (1, 50)):
        self.difficulty_range = difficulty_range
        self.current_problem = None
        self.current_answer = None
        
    def generate_problem(self) -> str:
        """生成一个数学问题"""
        # 随机选择运算类型
        operation = random.choice(['+', '-', '*'])
        
        if operation == '*':
            # 乘法使用较小的数字
            a = random.randint(1, 10)
            b = random.randint(1, 10)
        else:
            a = random.randint(*self.difficulty_range)
            b = random.randint(*self.difficulty_range)
            
        if operation == '-' and a < b:
            a, b = b, a  # 确保结果为正数
            
        problem = f"{a} {operation} {b} = ?"
        
        # 计算正确答案
        if operation == '+':
            answer = a + b
        elif operation == '-':
            answer = a - b
        else:  # '*'
            answer = a * b
            
        self.current_problem = problem
        self.current_answer = answer
        
        return problem
    
    def get_current_answer(self) -> int:
        """获取当前问题的正确答案"""
        return self.current_answer

class VeRLRewardFunction:
    """VeRL方案：基于函数的确定性奖励"""
    
    def __init__(self):
        self.name = "VeRL Function-based Reward"
        
    def calculate_reward(self, model_output: str, correct_answer: int) -> Tuple[float, bool]:
        """
        计算VeRL奖励
        
        Args:
            model_output: 模型的输出答案
            correct_answer: 正确答案
            
        Returns:
            (reward, is_correct): 奖励值和是否正确的标志
        """
        try:
            # 从模型输出中提取数字答案
            predicted_answer = self.extract_number(model_output)
            
            # 精确匹配检查
            is_correct = (predicted_answer == correct_answer)
            
            # VeRL给出二元奖励：正确得1，错误得0
            reward = 1.0 if is_correct else 0.0
            
            return reward, is_correct
            
        except Exception:
            # 无法解析答案，给予0奖励
            return 0.0, False
    
    def extract_number(self, text: str) -> int:
        """从文本中提取数字"""
        # 使用正则表达式提取数字
        numbers = re.findall(r'-?\d+', text)
        if numbers:
            return int(numbers[-1])  # 取最后一个数字作为答案
        else:
            raise ValueError("No number found in output")

class LearnedRewardModel(nn.Module):
    """传统方案：学习型奖励模型（故意引入偏见）"""
    
    def __init__(self, vocab_size: int = 100):
        super().__init__()
        self.name = "Learned Reward Model (with bias)"
        
        # 简单的神经网络结构
        self.embedding = nn.Embedding(vocab_size, 64)
        self.lstm = nn.LSTM(64, 128, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # 故意引入偏见：偏好某些数字模式
        self.bias_preferences = {
            'even_numbers': 0.1,    # 轻微偏好偶数答案
            'round_numbers': 0.15,  # 偏好整十数
            'specific_digits': 0.2  # 偏好包含特定数字的答案
        }
        
    def encode_text(self, text: str) -> torch.Tensor:
        """将文本编码为数字序列"""
        # 简化的编码：每个字符映射为ASCII值的模
        encoded = [ord(c) % 100 for c in text[:20]]  # 限制长度
        encoded = encoded + [0] * (20 - len(encoded))  # 填充
        return torch.tensor(encoded, dtype=torch.long).unsqueeze(0)
        
    def forward(self, text: str) -> torch.Tensor:
        """前向传播计算奖励"""
        encoded = self.encode_text(text)
        
        # LSTM处理
        embedded = self.embedding(encoded)
        lstm_out, (hidden, _) = self.lstm(embedded)
        
        # 使用最后的隐藏状态
        reward_base = self.classifier(hidden[-1])
        
        return reward_base.squeeze()
    
    def calculate_reward(self, model_output: str, correct_answer: int) -> Tuple[float, bool]:
        """
        计算学习型奖励（包含偏见）
        
        Args:
            model_output: 模型输出
            correct_answer: 正确答案
            
        Returns:
            (reward, is_correct): 奖励值和实际是否正确
        """
        try:
            # 基础神经网络奖励
            with torch.no_grad():
                base_reward = self.forward(model_output).item()
            
            # 提取预测答案用于正确性检查
            predicted_answer = self.extract_number(model_output)
            is_correct = (predicted_answer == correct_answer)
            
            # 应用偏见调整
            biased_reward = self.apply_bias(base_reward, model_output, predicted_answer)
            
            # 限制奖励范围
            final_reward = max(0.0, min(1.0, biased_reward))
            
            return final_reward, is_correct
            
        except Exception:
            return 0.1, False  # 给予小的随机奖励
    
    def apply_bias(self, base_reward: float, output: str, predicted_answer: int) -> float:
        """应用学习型模型的偏见"""
        reward = base_reward
        
        # 偏见1：偏好偶数答案
        if predicted_answer % 2 == 0:
            reward += self.bias_preferences['even_numbers']
            
        # 偏见2：偏好整十数
        if predicted_answer % 10 == 0:
            reward += self.bias_preferences['round_numbers']
            
        # 偏见3：偏好包含特定数字的答案
        if '5' in str(predicted_answer) or '0' in str(predicted_answer):
            reward += self.bias_preferences['specific_digits']
            
        # 偏见4：长度偏好（偏好较长的回答）
        if len(output) > 10:
            reward += 0.1
            
        return reward
    
    def extract_number(self, text: str) -> int:
        """从文本中提取数字"""
        numbers = re.findall(r'-?\d+', text)
        if numbers:
            return int(numbers[-1])
        else:
            raise ValueError("No number found")

class MathSolvingPolicy(nn.Module):
    """数学问题求解策略网络"""
    
    def __init__(self, vocab_size: int = 100, hidden_dim: int = 128):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, 64)
        self.lstm = nn.LSTM(64, hidden_dim, batch_first=True)
        
        # 输出层：生成答案数字的概率分布（支持0-200的答案）
        self.output_layer = nn.Linear(hidden_dim, 201)  # 0到200
        
    def encode_problem(self, problem: str) -> torch.Tensor:
        """将数学问题编码为张量"""
        # 简化编码：字符转ASCII模
        encoded = [ord(c) % self.vocab_size for c in problem[:20]]
        encoded = encoded + [0] * (20 - len(encoded))
        return torch.tensor(encoded, dtype=torch.long).unsqueeze(0)
    
    def forward(self, problem: str) -> torch.Tensor:
        """前向传播"""
        encoded = self.encode_problem(problem)
        embedded = self.embedding(encoded)
        lstm_out, (hidden, _) = self.lstm(embedded)
        
        # 使用最后的隐藏状态预测答案
        logits = self.output_layer(hidden[-1])
        return logits.squeeze()
    
    def get_answer_and_log_prob(self, problem: str) -> Tuple[int, torch.Tensor, str]:
        """获取答案和对数概率"""
        logits = self.forward(problem)
        probs = F.softmax(logits, dim=-1)
        
        # 采样答案
        answer_idx = torch.multinomial(probs, 1).item()
        log_prob = torch.log(probs[answer_idx] + 1e-8)
        
        # 生成输出文本
        output_text = f"The answer is {answer_idx}"
        
        return answer_idx, log_prob, output_text

class VeRLTrainer:
    """VeRL训练器：对比两种奖励机制"""
    
    def __init__(self, 
                 reward_system: str = 'verl',  # 'verl' or 'learned'
                 learning_rate: float = 1e-3,
                 batch_size: int = 16):
        
        self.reward_system = reward_system
        self.batch_size = batch_size
        
        # 初始化环境和策略
        self.env = MathProblemEnvironment()
        self.policy = MathSolvingPolicy()
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=learning_rate)
        
        # 初始化奖励系统
        if reward_system == 'verl':
            self.reward_function = VeRLRewardFunction()
        else:
            self.reward_function = LearnedRewardModel()
            self.train_learned_reward_model()  # 预训练学习型奖励模型
            
        self.metrics_history = []
        
        print(f"🚀 VeRL Trainer initialized with {self.reward_function.name}")
        
    def train_learned_reward_model(self):
        """预训练学习型奖励模型（引入偏见）"""
        print("📚 预训练学习型奖励模型（注入偏见）...")
        
        # 创建有偏见的训练数据（减少数据量提升速度）
        training_data = []
        for _ in range(200):  # 减少到200个样本
            problem = self.env.generate_problem()
            correct_answer = self.env.get_current_answer()
            
            # 创建一些错误但符合偏见的答案
            if random.random() < 0.3:  # 30%的时间给错误答案高奖励
                fake_answer = self.generate_biased_fake_answer(correct_answer)
                fake_output = f"The answer is {fake_answer}"
                training_data.append((fake_output, 0.8))  # 高奖励给错误答案
            else:
                correct_output = f"The answer is {correct_answer}"
                training_data.append((correct_output, 1.0))
        
        print(f"⚙️ 开始训练奖励模型（{len(training_data)}个样本）...")
        
        # 优化训练过程：批量训练
        optimizer = torch.optim.Adam(self.reward_function.parameters(), lr=1e-3)
        batch_size = 32  # 批量处理
        
        for epoch in range(10):  # 减少到10个epoch
            total_loss = 0
            # 按批次处理
            for i in range(0, len(training_data), batch_size):
                batch = training_data[i:i+batch_size]
                
                optimizer.zero_grad()
                batch_loss = 0
                
                # 批量计算损失
                for output, target_reward in batch:
                    predicted_reward = self.reward_function.forward(output)
                    loss = F.mse_loss(predicted_reward, torch.tensor(target_reward))
                    batch_loss += loss
                
                # 平均损失
                batch_loss = batch_loss / len(batch)
                batch_loss.backward()
                optimizer.step()
                total_loss += batch_loss.item()
            
            # 显示训练进度
            if (epoch + 1) % 3 == 0:
                print(f"  📊 Epoch {epoch+1}/10, Loss: {total_loss/len(training_data)*batch_size:.4f}")
        
        print(f"✅ 学习型奖励模型训练完成，已注入偏见")
    
    def generate_biased_fake_answer(self, correct_answer: int) -> int:
        """生成符合偏见的错误答案"""
        # 生成偏好的错误答案（偶数、整十数等）
        if random.random() < 0.5:
            # 生成偶数
            fake = correct_answer + random.choice([-2, -1, 1, 2])
            return fake if fake > 0 and fake % 2 == 0 else correct_answer + 2
        else:
            # 生成整十数
            return (correct_answer // 10 + 1) * 10
    
    def collect_batch(self) -> List[Dict[str, Any]]:
        """收集一批训练数据"""
        batch_data = []
        
        for _ in range(self.batch_size):
            # 生成问题
            problem = self.env.generate_problem()
            correct_answer = self.env.get_current_answer()
            
            # 获取策略输出
            predicted_answer, log_prob, output_text = self.policy.get_answer_and_log_prob(problem)
            
            # 计算奖励
            reward, is_correct = self.reward_function.calculate_reward(output_text, correct_answer)
            
            batch_data.append({
                'problem': problem,
                'predicted_answer': predicted_answer,
                'correct_answer': correct_answer,
                'output_text': output_text,
                'log_prob': log_prob,
                'reward': reward,
                'is_correct': is_correct
            })
            
        return batch_data
    
    def compute_policy_loss(self, batch_data: List[Dict[str, Any]]) -> torch.Tensor:
        """计算策略损失（REINFORCE）"""
        total_loss = 0
        
        for data in batch_data:
            log_prob = data['log_prob']
            reward = data['reward']
            
            # REINFORCE损失
            loss = -log_prob * reward
            total_loss += loss
            
        return total_loss / len(batch_data)
    
    def train_step(self) -> ExperimentMetrics:
        """执行一步训练"""
        # 收集数据
        batch_data = self.collect_batch()
        
        # 计算损失
        policy_loss = self.compute_policy_loss(batch_data)
        
        # 反向传播
        self.optimizer.zero_grad()
        policy_loss.backward()
        
        # 计算梯度范数
        grad_norm = 0.0
        for param in self.policy.parameters():
            if param.grad is not None:
                grad_norm += param.grad.data.norm(2).item() ** 2
        grad_norm = grad_norm ** 0.5
        
        self.optimizer.step()
        
        # 计算指标
        rewards = [data['reward'] for data in batch_data]
        correct_flags = [data['is_correct'] for data in batch_data]
        
        accuracy = np.mean(correct_flags)
        reward_mean = np.mean(rewards)
        reward_std = np.std(rewards)
        
        # 计算策略熵
        with torch.no_grad():
            sample_problem = batch_data[0]['problem']
            logits = self.policy.forward(sample_problem)
            probs = F.softmax(logits, dim=-1)
            entropy = -(probs * torch.log(probs + 1e-8)).sum().item()
        
        # 计算Reward Hacking得分
        hacking_score = self.calculate_hacking_score(batch_data)
        
        metrics = ExperimentMetrics(
            step=len(self.metrics_history),
            accuracy=accuracy,
            reward_mean=reward_mean,
            reward_std=reward_std,
            entropy=entropy,
            gradient_norm=grad_norm,
            hacking_score=hacking_score
        )
        
        self.metrics_history.append(metrics)
        return metrics
    
    def calculate_hacking_score(self, batch_data: List[Dict[str, Any]]) -> float:
        """计算Reward Hacking得分：高奖励但错误答案的比例"""
        high_reward_wrong = 0
        total_wrong = 0
        
        for data in batch_data:
            if not data['is_correct']:
                total_wrong += 1
                if data['reward'] > 0.5:  # 高奖励阈值
                    high_reward_wrong += 1
        
        return high_reward_wrong / total_wrong if total_wrong > 0 else 0.0

def run_verl_experiment():
    """运行VeRL实验"""
    print("🎪 实验一：VeRL方案（函数式奖励）")
    print("=" * 50)
    
    trainer = VeRLTrainer(reward_system='verl')
    
    print("🚀 开始训练...")
    for step in range(100):
        metrics = trainer.train_step()
        
        if (step + 1) % 20 == 0:
            print(f"📊 第{step+1}步: 准确率={metrics.accuracy:.3f}, 奖励={metrics.reward_mean:.3f}, "
                  f"Hacking={metrics.hacking_score:.3f}")
    
    print("✅ VeRL实验完成")
    return trainer.metrics_history

def run_learned_reward_experiment():
    """运行传统学习型奖励实验"""
    print("\n🎪 实验二：传统方案（学习型奖励模型）")
    print("=" * 50)
    
    trainer = VeRLTrainer(reward_system='learned')
    
    print("🚀 开始训练...")
    for step in range(100):
        metrics = trainer.train_step()
        
        if (step + 1) % 20 == 0:
            print(f"📊 第{step+1}步: 准确率={metrics.accuracy:.3f}, 奖励={metrics.reward_mean:.3f}, "
                  f"Hacking={metrics.hacking_score:.3f}")
    
    print("✅ 传统奖励实验完成")
    return trainer.metrics_history

def create_comparison_visualization(verl_metrics: List[ExperimentMetrics], 
                                  learned_metrics: List[ExperimentMetrics]):
    """生成VeRL对比分析图表"""
    print("\n🎨 生成VeRL vs 传统奖励对比分析...")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('VeRL vs Traditional Reward Model Comparison', fontsize=16, fontweight='bold')
    
    steps_verl = [m.step for m in verl_metrics]
    steps_learned = [m.step for m in learned_metrics]
    
    # 1. 准确率对比
    ax = axes[0, 0]
    ax.plot(steps_verl, [m.accuracy for m in verl_metrics], 'b-', label='VeRL (Function-based)', linewidth=2)
    ax.plot(steps_learned, [m.accuracy for m in learned_metrics], 'r-', label='Traditional (Learned)', linewidth=2)
    ax.set_title('Accuracy Comparison', fontweight='bold')
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Accuracy')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. 奖励对比
    ax = axes[0, 1]
    ax.plot(steps_verl, [m.reward_mean for m in verl_metrics], 'b-', label='VeRL', linewidth=2)
    ax.plot(steps_learned, [m.reward_mean for m in learned_metrics], 'r-', label='Traditional', linewidth=2)
    ax.set_title('Mean Reward Comparison', fontweight='bold')
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Mean Reward')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Reward Hacking得分
    ax = axes[0, 2]
    ax.plot(steps_verl, [m.hacking_score for m in verl_metrics], 'b-', label='VeRL', linewidth=2)
    ax.plot(steps_learned, [m.hacking_score for m in learned_metrics], 'r-', label='Traditional', linewidth=2)
    ax.set_title('Reward Hacking Score', fontweight='bold')
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Hacking Score')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. 策略熵
    ax = axes[1, 0]
    ax.plot(steps_verl, [m.entropy for m in verl_metrics], 'b-', label='VeRL', linewidth=2)
    ax.plot(steps_learned, [m.entropy for m in learned_metrics], 'r-', label='Traditional', linewidth=2)
    ax.set_title('Policy Entropy', fontweight='bold')
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Entropy')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 5. 梯度稳定性
    ax = axes[1, 1]
    ax.plot(steps_verl, [m.gradient_norm for m in verl_metrics], 'b-', label='VeRL', linewidth=2)
    ax.plot(steps_learned, [m.gradient_norm for m in learned_metrics], 'r-', label='Traditional', linewidth=2)
    ax.set_title('Gradient Norm', fontweight='bold')
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Gradient Norm')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 6. 奖励标准差
    ax = axes[1, 2]
    ax.plot(steps_verl, [m.reward_std for m in verl_metrics], 'b-', label='VeRL', linewidth=2)
    ax.plot(steps_learned, [m.reward_std for m in learned_metrics], 'r-', label='Traditional', linewidth=2)
    ax.set_title('Reward Standard Deviation', fontweight='bold')
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Reward Std')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图表
    output_path = 'verl_vs_traditional_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ 对比分析图表已保存到: {output_path}")
    plt.show()

def generate_comparison_report(verl_metrics: List[ExperimentMetrics], 
                             learned_metrics: List[ExperimentMetrics]) -> str:
    """生成详细的对比分析报告"""
    
    # 最终指标
    verl_final = verl_metrics[-1]
    learned_final = learned_metrics[-1]
    
    # 平均指标
    verl_avg_accuracy = np.mean([m.accuracy for m in verl_metrics[-20:]])  # 最后20步平均
    learned_avg_accuracy = np.mean([m.accuracy for m in learned_metrics[-20:]])
    
    verl_avg_hacking = np.mean([m.hacking_score for m in verl_metrics[-20:]])
    learned_avg_hacking = np.mean([m.hacking_score for m in learned_metrics[-20:]])
    
    accuracy_improvement = (verl_avg_accuracy - learned_avg_accuracy) / learned_avg_accuracy * 100
    hacking_reduction = (learned_avg_hacking - verl_avg_hacking) / (learned_avg_hacking + 1e-8) * 100
    
    report = f"""
📊 VeRL vs Traditional Reward Model 对比分析报告
============================================================

🔍 最终性能对比:
------------------------
📈 VeRL方案 (函数式奖励):
• 最终准确率: {verl_final.accuracy:.3f} ({verl_final.accuracy*100:.1f}%)
• 最终奖励: {verl_final.reward_mean:.3f}
• Reward Hacking得分: {verl_final.hacking_score:.3f}

📉 传统方案 (学习型奖励):
• 最终准确率: {learned_final.accuracy:.3f} ({learned_final.accuracy*100:.1f}%)
• 最终奖励: {learned_final.reward_mean:.3f}
• Reward Hacking得分: {learned_final.hacking_score:.3f}

🎯 关键改进指标:
------------------------
• 准确率提升: {accuracy_improvement:+.1f}%
• Hacking减少: {hacking_reduction:+.1f}%
• 稳定性优势: {'✅ VeRL更稳定' if verl_final.reward_std < learned_final.reward_std else '⚠️ 传统方案更稳定'}

🔬 技术洞察:
------------------------
1. **奖励信号质量**:
   • VeRL: 提供完全准确的二元奖励 (0/1)
   • 传统: 存在偏见，可能给错误答案高奖励

2. **Reward Hacking现象**:
   • VeRL平均Hacking得分: {verl_avg_hacking:.3f}
   • 传统平均Hacking得分: {learned_avg_hacking:.3f}
   • 分析: {'VeRL成功消除了奖励黑客行为' if verl_avg_hacking < learned_avg_hacking else '两者差异不明显'}

3. **训练稳定性**:
   • VeRL奖励标准差: {verl_final.reward_std:.3f}
   • 传统奖励标准差: {learned_final.reward_std:.3f}
   • 结论: {'VeRL提供更一致的学习信号' if verl_final.reward_std < learned_final.reward_std else '传统方案波动性更小'}

💡 实验结论:
------------------------
{'✅ VeRL方案显著优于传统方案' if accuracy_improvement > 5 and hacking_reduction > 30 else 
 '📊 VeRL方案略优于传统方案' if accuracy_improvement > 0 and hacking_reduction > 0 else
 '⚠️ 需要进一步调优和验证'}

🚀 VeRL的核心优势:
• 消除奖励模型偏见
• 杜绝Reward Hacking漏洞
• 提供确定性的"真值"反馈
• 为复杂对抗训练提供稳定基础

🎓 学习要点:
• 函数式奖励 > 学习型奖励（在可验证任务中）
• 确定性验证消除了模型钻空子的可能
• VeRL为后续APO对抗博弈提供可靠的"锚点"
"""
    
    return report

def main():
    """主实验流程"""
    print("🚀 欢迎来到VeRL实验室！")
    print("📚 本实验将对比函数式奖励与学习型奖励在数学问题求解中的效果")
    
    # 概念预习
    print("\n📖 实验前概念预习:")
    print("=" * 50)
    
    explainer = ExplainerSystem()
    for concept in ['verl', 'function_reward', 'reward_hacking', 'math_task']:
        print(explainer.explain_concept(concept))
    
    print("🔥 开始对比实验...")
    
    # 运行两个实验
    verl_metrics = run_verl_experiment()
    learned_metrics = run_learned_reward_experiment()
    
    # 生成对比分析
    print("\n" + "="*50)
    print(generate_comparison_report(verl_metrics, learned_metrics))
    
    # 生成可视化
    create_comparison_visualization(verl_metrics, learned_metrics)
    
    print("\n🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉")
    print("VeRL实验全部完成！")
    print("🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉🎉")
    
    print("\n✨ 恭喜你完成了Lab06的学习！")
    print("🎯 你现在掌握了VeRL可验证强化学习的核心技术")
    print("🔧 下一步可以学习APO对抗性偏好优化")
    print("🚀 继续加油，向着AI安全专家的目标前进！")

if __name__ == "__main__":
    main() 