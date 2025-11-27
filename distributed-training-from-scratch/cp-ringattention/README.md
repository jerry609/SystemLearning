# CP: Context Parallelism (上下文并行 / Ring Attention)

## 核心思想

上下文并行 (Context Parallelism, CP) 旨在解决 **超长序列 (Long Context)** 训练时的显存瓶颈。

当序列长度达到 100K 或 1M 时，单个 GPU 无法存储完整的 KV Cache (Key-Value Cache) 和 Attention Score 矩阵 ($N \times N$)。

CP 的核心思想是：**沿着序列长度维度 (Sequence Dimension) 切分 Q, K, V**。

## Ring Attention 算法

标准的 Attention 计算需要完整的 K 和 V：
$$ \text{Attention}(Q, K, V) = \text{softmax}(\frac{QK^T}{\sqrt{d}})V $$

如果 Q, K, V 都被切分了，如何计算？**Ring Attention** 利用环状通信：

1.  **切分**: 每个 Rank 持有序列的一部分 $Q_i, K_i, V_i$。
2.  **计算**:
    -   Rank $i$ 使用本地的 $Q_i$ 和 $K_i, V_i$ 计算局部的 Attention Score 和 Output。
    -   **通信**: 将 $K_i, V_i$ 发送给下一个 Rank，同时接收上一个 Rank 的 $K_{prev}, V_{prev}$。
    -   Rank $i$ 使用 $Q_i$ 和接收到的 $K_{prev}, V_{prev}$ 继续计算，累加结果。
    -   重复 `world_size` 次，直到 $Q_i$ 见过了所有的 K, V 块。
3.  **Online Softmax**:
    -   由于分块计算，无法一次性对整行做 Softmax。
    -   使用 **Online Softmax** 技巧，动态维护 `max_score` 和 `sum_exp`，在每一步更新归一化因子。

## 本项目实现要点 (cp.py)

我们实现了模拟 Ring Attention 的核心逻辑：

1.  **Ring Communication**:
    -   模拟 KV Block 在 Rank 之间的环状流动。
    -   `recv_kv = all_k_shards[step]`

2.  **Online Softmax**:
    -   维护 `output_acc` (累积输出), `max_score` (当前最大值), `sum_exp` (当前指数和)。
    -   每一步根据新的 block 更新这些统计量，确保数值稳定性。

3.  **内存优势**:
    -   标准 Attention 显存复杂度: $O(N^2)$
    -   Ring Attention 显存复杂度: $O(N^2 / P)$ (P 是设备数)

## 运行演示

本项目支持 **CPU 模拟模式**，演示 Ring Attention 的计算流程和内存节省。

```bash
python cp-ringattention/train.py
```

### 输出示例解析
```text
Ring Attention Configuration:
  - Total seq_len: 1024
  - Q shard per rank: (2, 8, 256, 32)

Ring Communication (simulated):
  Step 0: recv KV from rank 0 -> computed partial attention
  Step 1: recv KV from rank 1 -> computed partial attention
  ...

Memory Analysis:
  Standard attention scores: 64.0 MB
  Ring attention scores per step: 4.0 MB
  Memory reduction: 16.0x
```
