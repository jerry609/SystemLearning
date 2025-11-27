# Distributed Training From Scratch

不依赖 DeepSpeed 和 Megatron 框架，纯Pytorch从零手撕5大并行算法： DP、TP、PP、CP、EP分布式训练算法。

## 核心特性

- **硬核手撕关键算法 Backward**
- **手撕分布式gradient和adam**
- **硬核实现MoEEP 1F1B 下的 通信-计算重叠Step-by-step**

## 算法实现

- **DP (Data Parallelism)**: ZeRO-3
- **TP (Tensor Parallelism)**: Llama
- **CP (Context Parallelism)**: RingAttention
- **PP (Pipeline Parallelism)**: DualPipe
- **EP (Expert Parallelism)**: Gshard

## 运行环境

- 不需要多卡环境
- 纯CPU GLOO backend可运行所有实例
- 无须 triton和cuda 等基础
