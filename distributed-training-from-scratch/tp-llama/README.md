# TP: Tensor Parallelism (张量并行)

## 核心思想

张量并行 (Tensor Parallelism, TP) 是一种模型并行策略，旨在解决单个 GPU 显存无法容纳巨大模型参数（如 Llama-70B, GPT-3 175B）的问题。

其核心思想是：**将矩阵乘法 (GEMM) 拆分到多个设备上并行计算**。

对于一个线性层 $Y = XA$，其中 $X$ 是输入，$A$ 是权重矩阵：
1.  **按列切分 (Column Parallel)**: 将 $A$ 按列切分为 $[A_1, A_2]$。每个设备计算 $Y_i = X A_i$，最终输出 $Y = [Y_1, Y_2]$ (拼接)。
2.  **按行切分 (Row Parallel)**: 将 $A$ 按行切分为 $\begin{bmatrix} A_1 \\ A_2 \end{bmatrix}$，同时将输入 $X$ 按列切分为 $[X_1, X_2]$。每个设备计算 $Y_i = X_i A_i$，最终输出 $Y = Y_1 + Y_2$ (求和)。

## Llama 架构中的 TP

在 Transformer (如 Llama) 的 MLP 层中，通常采用 "Column Parallel -> Row Parallel" 的组合来减少通信量。

### MLP 结构
$$ \text{MLP}(X) = \text{down\_proj}(\text{act}(\text{up\_proj}(X))) $$

1.  **第一层 (up_proj)**: 使用 **Column Parallel**。
    -   权重 $W_{up}$ 按列切分。
    -   输入 $X$ 复制到所有 Rank (Identity)。
    -   输出 $Y_{partial}$ 是切分的，不需要立即通信。
2.  **激活函数**: 在本地切分数据上独立进行。
3.  **第二层 (down_proj)**: 使用 **Row Parallel**。
    -   权重 $W_{down}$ 按行切分。
    -   输入是上一层的切分输出。
    -   输出需要 **All-Reduce** (求和) 以恢复完整的最终结果。

通过这种组合，我们在两个线性层之间不需要通信，只在 MLP 结束时进行一次 All-Reduce。

## 本项目实现要点 (tp.py)

我们实现了 Llama 风格的 TP 模块：

1.  **ColumnParallelLinear**:
    -   初始化：将权重按 `out_features` 维度切分。
    -   前向：`output = input @ weight_shard.T`。
    -   通信：如果 `gather_output=True`，执行 `All-Gather`；否则保持输出切分状态。

2.  **RowParallelLinear**:
    -   初始化：将权重按 `in_features` 维度切分。
    -   前向：`output_partial = input_shard @ weight_shard.T`。
    -   通信：执行 `All-Reduce` 对所有 Rank 的结果求和。

3.  **通信原语**:
    -   `copy_to_tensor_model_parallel_region`: Identity (前向), All-Reduce (反向)。
    -   `scatter_to_tensor_model_parallel_region`: Split (前向), All-Gather (反向)。
    -   `gather_from_tensor_model_parallel_region`: All-Gather (前向), Split (反向)。
    -   `reduce_from_tensor_model_parallel_region`: All-Reduce (前向), Identity (反向)。

## 运行演示

本项目支持 **CPU 模拟模式**，可以在单机 Windows/Linux 上理解 TP 的数据流。

```bash
python tp-llama/train.py
```

### 输出示例解析
```text
[ColumnParallel] Full: (4096, 1024) -> Shard: (2048, 1024)
[RowParallel] Full: (1024, 4096) -> Shard: (1024, 2048)

Step 0:
  Forward fc1 (ColumnParallel): 输出保持切分状态
  Forward fc2 (RowParallel):
    AllReduce: (4, 1024) (sum across 2 ranks) -> 最终聚合
```
