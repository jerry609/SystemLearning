"""
实验十一：攻防演化分析与评估

对完整的DAPO+APO+VeRL系统进行科学的度量和评估
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, silhouette_score
from typing import Dict, List, Tuple, Optional
import json
import os
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 复用Lab10中的模型定义
class DAPOAttacker(nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.fc1 = nn.Linear(10, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 5)
        
    def forward(self, noise):
        x = torch.relu(self.fc1(noise))
        x = torch.relu(self.fc2(x))
        return torch.sigmoid(self.fc3(x)) * 10

class LearnableDetector(nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.fc1 = nn.Linear(5, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        
    def forward(self, sequence):
        x = torch.relu(self.fc1(sequence))
        x = torch.relu(self.fc2(x))
        return torch.sigmoid(self.fc3(x))

def verl_verify_function(sequence):
    """VeRL验证函数"""
    seq = sequence.detach().cpu().numpy() if isinstance(sequence, torch.Tensor) else sequence
    if len(seq.shape) > 1:
        seq = seq.flatten()
    
    diffs = np.diff(seq)
    
    if np.all(diffs <= 0) and np.any(diffs < 0):
        return 1.0
    elif np.all(diffs >= 0) and np.any(diffs > 0):
        return 0.0
    else:
        return 0.5

# 综合评估系统
class ComprehensiveEvaluator:
    def __init__(self, attacker: DAPOAttacker, detector: LearnableDetector):
        self.attacker = attacker
        self.detector = detector
        self.results = {
            'attacker_metrics': {},
            'detector_metrics': {},
            'system_metrics': {},
            'detailed_analysis': {}
        }
        
    def evaluate_attacker(self, num_samples=1000) -> Dict:
        """评估攻击者的各项指标"""
        print("Evaluating Attacker...")
        
        # 生成攻击样本
        noise = torch.randn(num_samples, 10)
        attack_sequences = self.attacker(noise).detach()
        
        # 1. 攻击成功率
        detector_scores = self.detector(attack_sequences).detach()
        attack_success_rate = (detector_scores < 0.5).float().mean().item()
        
        # 2. 策略多样性（使用序列特征的聚类分析）
        sequence_features = []
        for seq in attack_sequences:
            features = [
                seq.mean().item(),  # 平均值
                seq.std().item(),   # 标准差
                seq.max().item() - seq.min().item(),  # 范围
                torch.diff(seq).mean().item(),  # 平均差分
                verl_verify_function(seq)  # VeRL标签
            ]
            sequence_features.append(features)
        
        sequence_features = np.array(sequence_features)
        
        # 尝试不同的聚类数，找到最佳的策略数量
        best_k = 2
        best_score = -1
        for k in range(2, min(10, num_samples//10)):
            kmeans = KMeans(n_clusters=k, random_state=42)
            labels = kmeans.fit_predict(sequence_features)
            if len(np.unique(labels)) > 1:
                score = silhouette_score(sequence_features, labels)
                if score > best_score:
                    best_score = score
                    best_k = k
        
        # 使用最佳聚类数进行最终聚类
        kmeans = KMeans(n_clusters=best_k, random_state=42)
        cluster_labels = kmeans.fit_predict(sequence_features)
        
        strategy_diversity = len(np.unique(cluster_labels))
        
        # 3. 逃避模式分析
        evasion_patterns = {
            'pure_malicious': 0,
            'pure_benign': 0,
            'subtle_evasion': 0,
            'random_noise': 0
        }
        
        for seq in attack_sequences:
            verl_label = verl_verify_function(seq)
            detector_score = self.detector(seq.unsqueeze(0)).item()
            
            if verl_label == 1.0 and detector_score > 0.7:
                evasion_patterns['pure_malicious'] += 1
            elif verl_label == 0.0 and detector_score < 0.3:
                evasion_patterns['pure_benign'] += 1
            elif abs(verl_label - detector_score) > 0.3:
                evasion_patterns['subtle_evasion'] += 1
            else:
                evasion_patterns['random_noise'] += 1
        
        # 归一化为百分比
        total = sum(evasion_patterns.values())
        for key in evasion_patterns:
            evasion_patterns[key] = (evasion_patterns[key] / total) * 100
        
        attacker_metrics = {
            'attack_success_rate': attack_success_rate,
            'strategy_diversity': strategy_diversity,
            'evasion_patterns': evasion_patterns,
            'clustering_score': best_score
        }
        
        self.results['attacker_metrics'] = attacker_metrics
        self.results['detailed_analysis']['attack_sequences'] = attack_sequences
        self.results['detailed_analysis']['cluster_labels'] = cluster_labels
        
        return attacker_metrics
    
    def evaluate_detector(self, test_size=500) -> Dict:
        """评估检测器的各项指标"""
        print("Evaluating Detector...")
        
        # 生成测试集
        # 1. 良性样本（升序）
        benign_samples = []
        for _ in range(test_size):
            start = np.random.uniform(0, 5)
            step = np.random.uniform(0.5, 2)
            seq = torch.tensor([start + i * step for i in range(5)], dtype=torch.float32)
            benign_samples.append(seq)
        benign_samples = torch.stack(benign_samples)
        
        # 2. 明显恶意样本（降序）
        malicious_samples = []
        for _ in range(test_size):
            start = np.random.uniform(5, 10)
            step = np.random.uniform(0.5, 2)
            seq = torch.tensor([start - i * step for i in range(5)], dtype=torch.float32)
            malicious_samples.append(seq)
        malicious_samples = torch.stack(malicious_samples)
        
        # 3. 对抗样本（来自攻击者）
        adversarial_samples = self.attacker(torch.randn(test_size, 10)).detach()
        
        # 4. OOD样本（随机噪声）
        ood_samples = torch.rand(test_size, 5) * 10
        
        # 评估检测准确率
        with torch.no_grad():
            benign_pred = self.detector(benign_samples)
            malicious_pred = self.detector(malicious_samples)
            adversarial_pred = self.detector(adversarial_samples)
            ood_pred = self.detector(ood_samples)
        
        # 计算各类准确率
        benign_accuracy = (benign_pred < 0.5).float().mean().item()
        malicious_accuracy = (malicious_pred > 0.5).float().mean().item()
        adversarial_accuracy = (adversarial_pred > 0.5).float().mean().item()
        
        # OOD鲁棒性（应该有中等的预测值，表示不确定）
        ood_uncertainty = (torch.abs(ood_pred - 0.5) < 0.3).float().mean().item()
        
        # 创建混淆矩阵数据
        all_true_labels = []
        all_pred_labels = []
        
        # 良性样本
        all_true_labels.extend([0] * test_size)
        all_pred_labels.extend((benign_pred > 0.5).int().tolist())
        
        # 恶意样本
        all_true_labels.extend([1] * test_size)
        all_pred_labels.extend((malicious_pred > 0.5).int().tolist())
        
        # 对抗样本（真实标签基于VeRL）
        for seq in adversarial_samples:
            verl_label = verl_verify_function(seq)
            all_true_labels.append(int(verl_label > 0.5))
        all_pred_labels.extend((adversarial_pred > 0.5).int().tolist())
        
        cm = confusion_matrix(all_true_labels, all_pred_labels)
        
        detector_metrics = {
            'benign_accuracy': benign_accuracy,
            'malicious_accuracy': malicious_accuracy,
            'adversarial_detection_rate': adversarial_accuracy,
            'ood_robustness': ood_uncertainty,
            'overall_accuracy': (benign_accuracy + malicious_accuracy) / 2,
            'confusion_matrix': cm.tolist()  # Convert to list for JSON serialization
        }
        
        self.results['detector_metrics'] = detector_metrics
        return detector_metrics
    
    def evaluate_system_dynamics(self, evolution_steps=100) -> Dict:
        """评估系统的动态特性"""
        print("Evaluating System Dynamics...")
        
        # 模拟攻防演化过程
        evolution_history = {
            'steps': [],
            'attack_success': [],
            'detection_accuracy': [],
            'arms_race_intensity': []
        }
        
        for step in range(evolution_steps):
            # 攻击者尝试
            noise = torch.randn(50, 10)
            attacks = self.attacker(noise).detach()
            attack_scores = self.detector(attacks).detach()
            attack_success = (attack_scores < 0.5).float().mean().item()
            
            # 检测者准确率
            benign = self.generate_benign_samples(25)
            malicious = self.generate_malicious_samples(25)
            
            benign_pred = self.detector(benign)
            malicious_pred = self.detector(malicious)
            
            detection_accuracy = ((benign_pred < 0.5).float().mean() + 
                                (malicious_pred > 0.5).float().mean()).item() / 2
            
            # 军备竞赛强度（攻防能力的接近程度）
            arms_race_intensity = 1 - abs(attack_success - (1 - detection_accuracy))
            
            evolution_history['steps'].append(step)
            evolution_history['attack_success'].append(attack_success)
            evolution_history['detection_accuracy'].append(detection_accuracy)
            evolution_history['arms_race_intensity'].append(arms_race_intensity)
        
        # 计算系统稳定性
        attack_variance = np.var(evolution_history['attack_success'])
        detection_variance = np.var(evolution_history['detection_accuracy'])
        system_stability = 1 / (1 + attack_variance + detection_variance)
        
        system_metrics = {
            'system_stability': system_stability,
            'average_arms_race_intensity': np.mean(evolution_history['arms_race_intensity']),
            'evolution_history': evolution_history,
            'convergence_rate': self.calculate_convergence_rate(evolution_history)
        }
        
        self.results['system_metrics'] = system_metrics
        return system_metrics
    
    def generate_benign_samples(self, batch_size):
        """生成良性样本"""
        samples = []
        for _ in range(batch_size):
            start = np.random.uniform(0, 5)
            step = np.random.uniform(0.5, 2)
            seq = torch.tensor([start + i * step for i in range(5)], dtype=torch.float32)
            samples.append(seq)
        return torch.stack(samples)
    
    def generate_malicious_samples(self, batch_size):
        """生成恶意样本"""
        samples = []
        for _ in range(batch_size):
            start = np.random.uniform(5, 10)
            step = np.random.uniform(0.5, 2)
            seq = torch.tensor([start - i * step for i in range(5)], dtype=torch.float32)
            samples.append(seq)
        return torch.stack(samples)
    
    def calculate_convergence_rate(self, history):
        """计算收敛速度"""
        attack_diffs = np.abs(np.diff(history['attack_success']))
        detection_diffs = np.abs(np.diff(history['detection_accuracy']))
        
        # 找到稳定点（变化小于阈值）
        threshold = 0.01
        stable_points = np.where((attack_diffs < threshold) & (detection_diffs < threshold))[0]
        
        if len(stable_points) > 0:
            return stable_points[0] / len(history['steps'])
        else:
            return 1.0  # 未收敛
    
    def visualize_results(self):
        """可视化所有评估结果"""
        fig = plt.figure(figsize=(20, 16))
        
        # 1. 攻击策略多样性分析
        plt.subplot(3, 3, 1)
        if 'cluster_labels' in self.results['detailed_analysis']:
            attack_sequences = self.results['detailed_analysis']['attack_sequences']
            cluster_labels = self.results['detailed_analysis']['cluster_labels']
            
            # 使用PCA降维到2D进行可视化
            from sklearn.decomposition import PCA
            sequence_features = []
            for seq in attack_sequences:
                features = [seq.mean().item(), seq.std().item(), 
                          seq.max().item() - seq.min().item(),
                          torch.diff(seq).mean().item()]
                sequence_features.append(features)
            
            pca = PCA(n_components=2)
            features_2d = pca.fit_transform(sequence_features)
            
            scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], 
                                c=cluster_labels, cmap='viridis', alpha=0.6)
            plt.colorbar(scatter)
            plt.xlabel('PCA Component 1')
            plt.ylabel('PCA Component 2')
            plt.title(f'Attack Strategy Clustering (k={len(np.unique(cluster_labels))})')
        
        # 2. 逃避模式分布
        plt.subplot(3, 3, 2)
        evasion_patterns = self.results['attacker_metrics']['evasion_patterns']
        labels = list(evasion_patterns.keys())
        values = list(evasion_patterns.values())
        colors = ['red', 'green', 'yellow', 'gray']
        
        plt.pie(values, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        plt.title('Evasion Pattern Distribution')
        
        # 3. 检测器混淆矩阵
        plt.subplot(3, 3, 3)
        cm = np.array(self.results['detector_metrics']['confusion_matrix'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Benign', 'Malicious'],
                   yticklabels=['Benign', 'Malicious'])
        plt.title('Detector Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        # 4. 系统动态演化
        plt.subplot(3, 3, 4)
        history = self.results['system_metrics']['evolution_history']
        plt.plot(history['steps'], history['attack_success'], 
                'r-', label='Attack Success Rate', alpha=0.7)
        plt.plot(history['steps'], history['detection_accuracy'], 
                'b-', label='Detection Accuracy', alpha=0.7)
        plt.xlabel('Evolution Steps')
        plt.ylabel('Rate')
        plt.title('Attack-Defense Co-evolution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 5. 军备竞赛强度
        plt.subplot(3, 3, 5)
        plt.plot(history['steps'], history['arms_race_intensity'], 
                'g-', linewidth=2)
        plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.5)
        plt.xlabel('Evolution Steps')
        plt.ylabel('Arms Race Intensity')
        plt.title('System Balance Indicator')
        plt.grid(True, alpha=0.3)
        
        # 6. 综合指标雷达图
        plt.subplot(3, 3, 6, projection='polar')
        categories = ['Attack\nSuccess', 'Detection\nAccuracy', 'Strategy\nDiversity', 
                     'System\nStability', 'OOD\nRobustness']
        values = [
            self.results['attacker_metrics']['attack_success_rate'],
            self.results['detector_metrics']['overall_accuracy'],
            min(self.results['attacker_metrics']['strategy_diversity'] / 5, 1),  # 归一化
            self.results['system_metrics']['system_stability'],
            self.results['detector_metrics']['ood_robustness']
        ]
        
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        values += values[:1]
        angles += angles[:1]
        
        plt.plot(angles, values, 'o-', linewidth=2, color='blue')
        plt.fill(angles, values, alpha=0.25, color='blue')
        plt.xticks(angles[:-1], categories)
        plt.ylim(0, 1)
        plt.title('System Performance Radar')
        
        # 7. 攻击样本示例
        plt.subplot(3, 3, 7)
        sample_attacks = self.results['detailed_analysis']['attack_sequences'][:5]
        for i, seq in enumerate(sample_attacks):
            plt.plot(seq.numpy(), marker='o', label=f'Attack {i+1}', alpha=0.7)
        plt.xlabel('Position')
        plt.ylabel('Value')
        plt.title('Sample Attack Sequences')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 8. 性能指标汇总
        plt.subplot(3, 3, 8)
        plt.axis('off')
        summary_text = f"""
        === Attacker Metrics ===
        Success Rate: {self.results['attacker_metrics']['attack_success_rate']:.3f}
        Strategy Diversity: {self.results['attacker_metrics']['strategy_diversity']}
        Clustering Score: {self.results['attacker_metrics']['clustering_score']:.3f}
        
        === Detector Metrics ===
        Overall Accuracy: {self.results['detector_metrics']['overall_accuracy']:.3f}
        Adversarial Detection: {self.results['detector_metrics']['adversarial_detection_rate']:.3f}
        OOD Robustness: {self.results['detector_metrics']['ood_robustness']:.3f}
        
        === System Metrics ===
        Stability: {self.results['system_metrics']['system_stability']:.3f}
        Arms Race Intensity: {self.results['system_metrics']['average_arms_race_intensity']:.3f}
        Convergence Rate: {self.results['system_metrics']['convergence_rate']:.3f}
        """
        plt.text(0.1, 0.5, summary_text, fontsize=10, 
                family='monospace', verticalalignment='center')
        plt.title('Performance Summary')
        
        # 9. 攻防能力对比时间线
        plt.subplot(3, 3, 9)
        window = 10
        if len(history['attack_success']) >= window:
            attack_ma = np.convolve(history['attack_success'], 
                                   np.ones(window)/window, mode='valid')
            defense_ma = np.convolve(history['detection_accuracy'], 
                                    np.ones(window)/window, mode='valid')
            
            plt.plot(attack_ma, defense_ma, 'purple', linewidth=2, alpha=0.7)
            plt.scatter(attack_ma[0], defense_ma[0], color='green', s=100, 
                       marker='o', label='Start', zorder=5)
            plt.scatter(attack_ma[-1], defense_ma[-1], color='red', s=100, 
                       marker='*', label='End', zorder=5)
            
            plt.xlabel('Attack Success Rate (MA)')
            plt.ylabel('Detection Accuracy (MA)')
            plt.title('Attack-Defense Phase Space')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('comprehensive_evaluation.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_report(self, save_path='evaluation_report.json'):
        """生成完整的评估报告"""
        # 准备可序列化的结果（排除tensor和其他不可序列化的对象）
        serializable_results = {
            'attacker_metrics': self.results['attacker_metrics'],
            'detector_metrics': self.results['detector_metrics'],
            'system_metrics': {
                'system_stability': self.results['system_metrics']['system_stability'],
                'average_arms_race_intensity': self.results['system_metrics']['average_arms_race_intensity'],
                'convergence_rate': self.results['system_metrics']['convergence_rate']
            }
        }
        
        report = {
            'evaluation_summary': {
                'attack_success_rate': self.results['attacker_metrics']['attack_success_rate'],
                'detection_accuracy': self.results['detector_metrics']['overall_accuracy'],
                'system_stability': self.results['system_metrics']['system_stability'],
                'strategy_diversity': self.results['attacker_metrics']['strategy_diversity']
            },
            'detailed_metrics': serializable_results,
            'insights': self.generate_insights()
        }
        
        with open(save_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\nEvaluation report saved to {save_path}")
        return report
    
    def generate_insights(self) -> List[str]:
        """基于评估结果生成洞察"""
        insights = []
        
        # 攻击者分析
        if self.results['attacker_metrics']['attack_success_rate'] > 0.7:
            insights.append("High attack success rate indicates detector vulnerabilities")
        elif self.results['attacker_metrics']['attack_success_rate'] < 0.3:
            insights.append("Low attack success rate suggests strong detector performance")
        
        if self.results['attacker_metrics']['strategy_diversity'] > 3:
            insights.append("Attacker shows high strategic diversity")
        
        # 检测器分析
        if self.results['detector_metrics']['ood_robustness'] > 0.7:
            insights.append("Detector shows good robustness to out-of-distribution samples")
        
        # 系统分析
        if self.results['system_metrics']['system_stability'] > 0.8:
            insights.append("System demonstrates stable co-evolution dynamics")
        
        if self.results['system_metrics']['average_arms_race_intensity'] > 0.7:
            insights.append("Strong arms race dynamic indicates balanced competition")
        
        return insights

# 主函数
def main():
    print("="*60)
    print("Comprehensive Evaluation System")
    print("="*60)
    
    # 加载预训练的模型（使用随机初始化模拟）
    attacker = DAPOAttacker()
    detector = LearnableDetector()
    
    # 训练模型（简化版，实际应该加载Lab10的checkpoint）
    print("\nTraining models for evaluation...")
    # 这里应该加载实际训练好的模型
    # 为了演示，我们进行简单的训练
    optimizer_a = torch.optim.Adam(attacker.parameters(), lr=0.001)
    optimizer_d = torch.optim.Adam(detector.parameters(), lr=0.001)
    
    for _ in range(100):
        # 简化的训练循环
        noise = torch.randn(32, 10)
        attacks = attacker(noise)
        
        # 训练检测器
        scores_for_detector = detector(attacks.detach())
        loss_d = -scores_for_detector.mean()
        
        optimizer_d.zero_grad()
        loss_d.backward()
        optimizer_d.step()
        
        # 训练攻击者
        scores_for_attacker = detector(attacks)
        loss_a = scores_for_attacker.mean()
        
        optimizer_a.zero_grad()
        loss_a.backward()
        optimizer_a.step()
    
    # 创建评估器
    evaluator = ComprehensiveEvaluator(attacker, detector)
    
    # 执行评估
    print("\n" + "="*60)
    print("Running Comprehensive Evaluation")
    print("="*60)
    
    # 1. 评估攻击者
    attacker_metrics = evaluator.evaluate_attacker(num_samples=1000)
    print(f"\n✓ Attacker Evaluation Complete")
    print(f"  - Success Rate: {attacker_metrics['attack_success_rate']:.3f}")
    print(f"  - Strategy Diversity: {attacker_metrics['strategy_diversity']}")
    
    # 2. 评估检测器
    detector_metrics = evaluator.evaluate_detector(test_size=500)
    print(f"\n✓ Detector Evaluation Complete")
    print(f"  - Overall Accuracy: {detector_metrics['overall_accuracy']:.3f}")
    print(f"  - OOD Robustness: {detector_metrics['ood_robustness']:.3f}")
    
    # 3. 评估系统动态
    system_metrics = evaluator.evaluate_system_dynamics(evolution_steps=100)
    print(f"\n✓ System Dynamics Evaluation Complete")
    print(f"  - System Stability: {system_metrics['system_stability']:.3f}")
    print(f"  - Arms Race Intensity: {system_metrics['average_arms_race_intensity']:.3f}")
    
    # 生成可视化
    print("\n" + "="*60)
    print("Generating Visualizations...")
    print("="*60)
    evaluator.visualize_results()
    
    # 生成报告
    report = evaluator.generate_report()
    
    # 打印洞察
    print("\n" + "="*60)
    print("Key Insights")
    print("="*60)
    for i, insight in enumerate(report['insights'], 1):
        print(f"{i}. {insight}")
    
    print("\n" + "="*60)
    print("EVALUATION COMPLETE!")
    print("="*60)

if __name__ == "__main__":
    main() 
