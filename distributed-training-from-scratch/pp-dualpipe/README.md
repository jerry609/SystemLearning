# PP: Pipeline Parallelism (流水线并行)

## 核心思想

流水线并行 (Pipeline Parallelism, PP) 是一种将模型 **按层 (Layer)** 切分到不同设备上的策略。

与 TP 切分 Tensor 不同，PP 将模型的不同层（例如 Transformer 的前 12 层在 GPU0，后 12 层在 GPU1）分配给不同的设备。这些设备形成一个流水线。

### 挑战：Bubble (气泡)
如果简单地将 Batch 依次通过 GPU0 -> GPU1 -> ...，那么在 GPU1 计算时，GPU0 是空闲的。这种空闲时间称为 **Bubble**。

为了减少 Bubble，PP 将一个 Batch 切分为多个 **Micro-Batches**，并采用特定的调度策略。

## 调度策略

### 1. GPipe (F-then-B)
最简单的调度方式。
-   **Forward 阶段**: 依次注入所有 Micro-Batch，数据流经所有 Stage。
-   **Backward 阶段**: 依次对所有 Micro-Batch 进行反向传播。
-   **缺点**: 显存占用高（需要缓存所有 Micro-Batch 的激活值），Bubble 较大。

### 2. 1F1B (One-Forward-One-Backward)
更高效的调度方式，广泛用于 Megatron-LM 等框架。
-   **Warmup**: 先注入一定数量的 Micro-Batch 填满流水线。
-   **Steady State**: 交替执行 Forward 和 Backward (1F1B)。即 GPU 在处理完一个 Micro-Batch 的 Forward 后，立即处理之前 Micro-Batch 的 Backward。
-   **优点**: 及时释放显存（Backward 完成后即可释放激活值），Bubble 较小。

## 本项目实现要点 (pp.py)

我们实现了基于 P2P 通信的流水线引擎：

1.  **PipelineStage**:
    -   封装模型的一部分。
    -   管理输入输出缓冲区。

2.  **P2P Communication**:
    -   `send_forward(output, next_rank)`
    -   `recv_forward(prev_rank)`
    -   `send_backward(grad, prev_rank)`
    -   `recv_backward(next_rank)`

3.  **调度器 (Scheduler)**:
    -   **GPipe**: 简单的循环，先全 Fwd 再全 Bwd。
    -   **1F1B**: 复杂的循环，维护 `num_warmup_microbatches` 和 `num_microbatches_remaining`，动态决定执行 Forward 还是 Backward。

## 运行演示

本项目支持 **CPU 模拟模式**，演示 Micro-Batch 如何在 Stage 之间流动。

```bash
python pp-dualpipe/train.py
```

### 输出示例解析 (1F1B)
```text
[1F1B Schedule]
  Warmup (3 forwards): F0 F1 F2 
  Steady (1F1B): F3 B0  <-- 交替执行
  Cooldown (3 backwards): B1 B2 B3
```
- `F0`: Micro-batch 0 Forward
- `B0`: Micro-batch 0 Backward
- 可以看到流水线是如何被填满然后交替执行的。
