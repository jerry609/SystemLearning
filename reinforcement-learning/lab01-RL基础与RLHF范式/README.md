# 实验一：强化学习基础与RLHF范式

## 🎯 实验目标
1. 理解强化学习（RL）的核心要素：智能体（Agent）、环境（Environment）、状态（State）、动作（Action）、奖励（Reward）。
2. 学习大语言模型（LLM）对齐（Alignment）问题的背景和挑战。
3. 理解"从人类反馈中强化学习"（RLHF）的基本原理和流程。
4. 通过一个简单的代码示例，直观感受RL的基本循环。

## 📖 理论背景
- **强化学习**: 智能体通过与环境交互，学习一个策略（Policy）以最大化累积奖励。
- **LLM对齐**: 确保LLM的行为和输出符合人类的价值观和意图。
- **RLHF**: 一种三阶段范式（预训练、奖励建模、RL优化），使用人类偏好数据来训练一个奖励模型，然后用RL算法（如PPO）优化LLM以在该奖励模型上获得高分。它是当前解决对齐问题的主流方法。

## 🛠️ 实践内容
1. **分析一个经典的RL环境**（如`CartPole`或`GridWorld`），识别其State, Action, Reward。
2. **运行一个简单的RL代码**，观察智能体策略随训练变化的趋势。
3. **撰写文档**: 详细描述RLHF的三个阶段，并解释为什么直接使用人类反馈进行监督学习是困难的。 