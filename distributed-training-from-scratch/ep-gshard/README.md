# EP: Expert Parallelism (专家并行 / MoE)

## 核心思想

专家并行 (Expert Parallelism, EP) 是混合专家模型 (Mixture-of-Experts, MoE) 的分布式实现方式。

MoE 的核心在于 **稀疏激活 (Sparse Activation)**：模型包含大量的 "专家" (Experts, 通常是 MLP)，但对于每个 Token，只有少数几个 (Top-K) 专家会被激活。

EP 的核心思想是：**将不同的专家分配到不同的设备上**。

## GShard 架构

GShard 是 Google 提出的 MoE 分布式架构，其核心流程如下：

1.  **Gating (路由)**:
    -   每个 Token 通过一个 Gating Network，计算出它应该去哪几个专家 (Top-K)。
    -   这些专家可能分布在不同的 Rank 上。

2.  **All-to-All Dispatch**:
    -   这是一个关键的通信操作。
    -   每个 Rank 将自己手中的 Token，根据 Gating 结果，发送到持有对应专家的 Rank。
    -   同时，接收其他 Rank 发来的、需要自己处理的 Token。

3.  **Expert Computation**:
    -   每个 Rank 收到 Token 后，用本地的专家进行计算。

4.  **All-to-All Combine**:
    -   计算完成后，将结果逆向路由回 Token 原来的 Rank。

5.  **加权合并**:
    -   原 Rank 收到结果后，根据 Gating 权重进行加权求和，得到最终输出。

## 本项目实现要点 (ep.py)

我们实现了 GShard 风格的 MoE 层：

1.  **TopKGating**:
    -   计算路由概率。
    -   计算 **Auxiliary Loss** (负载均衡损失)，防止所有 Token 都涌向同一个专家。

2.  **MoELayer**:
    -   包含多个 `Expert` (MLP)。
    -   `all_to_all_dispatch_sim`: 模拟 Token 的重新分发。
    -   `all_to_all_combine_sim`: 模拟结果的收集。

3.  **稀疏性**:
    -   演示了如何只计算 Top-K 个专家，从而在参数量巨大的情况下保持低计算量。

## 运行演示

本项目支持 **CPU 模拟模式**，演示 Token 如何被路由到不同 Rank 的专家。

```bash
python ep-gshard/train.py
```

### 输出示例解析
```text
[Gating] Token distribution to experts:
  Expert 0:   6 tokens ███
  Expert 1:   5 tokens ██
  ...

[All-to-All Dispatch]
  Rank 0 receives 31 tokens
  Rank 1 receives 33 tokens

[Expert Computation]
  Expert 3 (Rank 0): processing 11 tokens
  ...
```
可以看到 Token 是如何根据 Gating 结果在不同 Rank 之间迁移的。
