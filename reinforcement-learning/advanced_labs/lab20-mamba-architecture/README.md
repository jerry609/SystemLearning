# Lab 20: 新一代序列模型 - Mamba架构实践

## 🎯 学习目标

本实验将带您探索在核心序列处理层面，对Transformer构成最有力挑战的新兴架构——**状态空间模型（State Space Model, SSM）**，特别是其最先进的代表 **Mamba**。您将理解为什么SSM在处理超长序列上比Transformer更高效，并亲手实践Mamba的核心优势。

## 核心任务

1.  **理论学习：从SSM到Mamba**
    *   **Transformer的瓶颈**: 深入理解自注意力机制的二次方复杂度（O(n²)）是如何成为处理长序列（如整本书、高分辨率图像、几小时的音频）时的计算和内存瓶颈的。
    *   **SSM的核心思想**: 学习SSM如何借鉴经典控制理论，像RNN一样以线性复杂度（O(n)）顺序处理信息，但通过一个巧妙的"状态压缩"机制来建模长距离依赖。
    *   **Mamba的关键创新 (S6)**: 理解Mamba如何通过引入一个**基于输入内容动态变化的门控机制**来解决传统SSM无法聚焦关键信息的问题。这个"选择性"机制（Selective State Space）是Mamba性能媲美甚至超越Transformer的关键。

2.  **实践操作：Mamba初体验**
    *   **环境准备**: `mamba-ssm` 和 `causal-conv1d` 是Mamba的核心依赖，确保它们被正确安装。
    *   **加载模型**: 使用`transformers`库加载一个预训练的Mamba模型（例如，`state-spaces/mamba-2.8b-slimpj`）。
    *   **结构分析**: 打印并分析Mamba模型的代码结构。注意它与Transformer的显著不同：没有`Attention`模块，取而代之的是核心的`Mamba`层或`SSM`块。

3.  **性能对比实验**
    *   **设计任务**: 选择一个长文本任务，例如长文本分类或摘要生成。
    *   **构建输入**: 创建一个包含超长序列（例如8k或16k个token）的输入样本。
    *   **对比测试**:
        -   使用`lab12`中的标准Transformer模型处理该输入，记录其峰值显存占用和推理耗时。很可能会因为显存不足而失败。
        -   使用本实验加载的Mamba模型处理**相同的输入**，同样记录其显存占用和耗时。
    *   **量化对比**: 将两者的性能数据制成表格，直观地展示Mamba架构在处理超长上下文时的巨大效率优势。

## 📝 预期成果

- 深刻理解SSM和Mamba架构的核心思想及其相对于Transformer的优势。
- 一个可以加载、分析并运行Mamba模型的实验脚本。
- 一份详细的性能对比报告，用具体数据证明Mamba在处理长序列任务上的线性复杂度和高效率。
- 掌握了除Transformer之外的另一条重要的、面向未来的AI模型架构技术路线。 