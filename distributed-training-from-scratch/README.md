# Distributed Training From Scratch

不依赖 DeepSpeed 和 Megatron 框架，纯Pytorch从零手撕5大并行算法： DP、TP、PP、CP、EP分布式训练算法。

## 核心特性

- **硬核手撕关键算法 Backward**
- **手撕分布式gradient和adam**
- **硬核实现MoEEP 1F1B 下的 通信-计算重叠Step-by-step**

## 算法实现

### 1. DP (Data Parallelism): ZeRO-3
- **路径**: `dp-zero3/`
- **核心**: `Zero3Linear` (参数分片), `Zero3AllGather` (前向收集), `ReduceScatter` (反向聚合).
- **运行**: `python dp-zero3/train.py`

### 2. TP (Tensor Parallelism): Llama Style
- **路径**: `tp-llama/`
- **核心**: `ColumnParallelLinear`, `RowParallelLinear`.
- **原理**: 矩阵乘法切分，前向 Identity/Split -> Compute -> AllGather/AllReduce.
- **运行**: `python tp-llama/train.py`

### 3. PP (Pipeline Parallelism): GPipe/1F1B
- **路径**: `pp-dualpipe/`
- **核心**: `PipelineStage`, `PipelineEngine`.
- **原理**: 模型分层，流水线执行 (Forward-Forward... Backward-Backward...).
- **运行**: `python pp-dualpipe/train.py`

### 4. CP (Context Parallelism): Ring Attention
- **路径**: `cp-ringattention/`
- **核心**: `RingAttention`.
- **原理**: 序列维度切分，KV Block 在环上流动，计算 Attention Score.
- **运行**: `python cp-ringattention/train.py`

### 5. EP (Expert Parallelism): GShard MoE
- **路径**: `ep-gshard/`
- **核心**: `MoELayer`, `TopKGating`, `All-to-All Dispatch/Combine`.
- **原理**: Token 路由到不同 Rank 的 Expert 进行计算.
- **运行**: `python ep-gshard/train.py`

## 运行环境

- **不需要多卡环境**
- **无须 triton 和 cuda 等基础**
- **GPU**: 自动使用 NCCL 后端进行真正的多 GPU 分布式训练
- **CPU**: 使用单进程模拟模式演示算法逻辑

```bash
# 安装依赖
pip install torch

# 运行任意 demo
python dp-zero3/train.py
python tp-llama/train.py
python pp-dualpipe/train.py
python cp-ringattention/train.py
python ep-gshard/train.py
```

## 学习路径

1. **ZeRO-3** - 理解参数分片和显存优化
2. **Tensor Parallelism** - 理解矩阵切分和通信模式
3. **Pipeline Parallelism** - 理解流水线调度 (GPipe/1F1B)
4. **Context Parallelism** - 理解长序列 Attention 的分布式计算
5. **Expert Parallelism** - 理解 MoE 的 Token 路由和 All-to-All
