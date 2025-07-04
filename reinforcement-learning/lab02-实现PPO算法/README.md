# 实验二：实现PPO算法

## 🎯 实验目标
1. 深入理解"演员-评论家"（Actor-Critic）架构。
2. 掌握近端策略优化（PPO）算法的核心思想，特别是其"信任区域"和"裁剪"目标函数。
3. 动手实现一个基于PPO的RLHF流程，用于优化一个小型LLM。
4. 分析PPO在训练LLM时的资源消耗，尤其是"评论家"模型带来的显存和计算开销。

## 📖 理论背景
- **Actor-Critic**: "演员"（Actor）负责生成动作（即LLM本身），"评论家"（Critic）负责评估状态的价值（Value），从而指导演员的更新。
- **PPO**: 通过一个裁剪的目标函数，限制每次策略更新的幅度，确保训练过程的稳定性，避免灾难性遗忘。这是RLHF最经典和主流的算法。
- **资源瓶颈**: 在LLM场景下，评论家模型通常需要和演员模型同等规模，这导致训练资源（特别是显存）消耗巨大。

## 🛠️ 实践内容
1. **代码实现**: 使用现有的RL库（如`trl`）或从零开始，构建一个包含Actor和Critic的PPO训练循环。
2. **模型训练**: 使用一个预训练的小型LLM作为Actor，一个价值头（Value Head）作为Critic，在一个简单的任务上（如情感正向生成）进行优化。
3. **性能分析**: 记录并分析训练过程中的显存占用和计算时间，直观感受PPO的资源瓶颈。
4. **问题思考**: 为什么需要一个与LLM同等规模的Critic？它在训练中具体起到了什么作用？ 