"""
Tensor Parallelism (Llama Style) Training Demo

支持两种运行模式:
1. GPU 模式: 使用真正的分布式训练 (NCCL backend)
2. CPU 模式: 使用单进程模拟，演示 TP 的核心算法逻辑

核心算法:
- ColumnParallelLinear: 按列切分权重，输出需要 AllGather
- RowParallelLinear: 按行切分权重，输入需要 Scatter，输出需要 AllReduce
- MLP: fc1 (ColParallel) -> act -> fc2 (RowParallel)
"""

import torch
import torch.nn as nn
import torch.optim as optim


# ============================================================================
# 模拟通信原语
# ============================================================================

class AllGatherSim(torch.autograd.Function):
    """
    模拟 AllGather
    Forward: 收集各 rank 的 tensor
    Backward: ReduceScatter 分发梯度
    """
    @staticmethod
    def forward(ctx, input_tensor, world_size, dim=-1):
        ctx.world_size = world_size
        ctx.dim = dim
        # 模拟：将 input 沿 dim 重复 world_size 次
        gathered = input_tensor.repeat(*([1] * dim), world_size, *([1] * (input_tensor.dim() - dim - 1)))
        print(f"    AllGather: {tuple(input_tensor.shape)} -> {tuple(gathered.shape)} (dim={dim})")
        return gathered
    
    @staticmethod
    def backward(ctx, grad_output):
        # ReduceScatter: 取一部分
        world_size = ctx.world_size
        dim = ctx.dim
        shard_size = grad_output.shape[dim] // world_size
        grad_shard = grad_output.narrow(dim, 0, shard_size)
        print(f"    ReduceScatter: {tuple(grad_output.shape)} -> {tuple(grad_shard.shape)}")
        return grad_shard, None, None


class AllReduceSim(torch.autograd.Function):
    """
    模拟 AllReduce
    Forward: 对各 rank 的 tensor 求和
    Backward: 梯度直接传递（因为前向是求和）
    """
    @staticmethod
    def forward(ctx, input_tensor, world_size):
        ctx.world_size = world_size
        # 模拟：假设所有 rank 有相同输入，结果是 input * world_size
        # 实际分布式中是求和
        reduced = input_tensor * world_size  # 模拟 sum 效果
        print(f"    AllReduce: {tuple(input_tensor.shape)} (sum across {world_size} ranks)")
        return reduced
    
    @staticmethod
    def backward(ctx, grad_output):
        # 梯度直接传递
        return grad_output, None


def all_gather_sim(tensor, world_size, dim=-1):
    return AllGatherSim.apply(tensor, world_size, dim)


def all_reduce_sim(tensor, world_size):
    return AllReduceSim.apply(tensor, world_size)


# ============================================================================
# TP 核心层实现 (模拟版本)
# ============================================================================

class ColumnParallelLinearSim(nn.Module):
    """
    列并行 Linear (模拟版本)
    
    权重 W: [out_features, in_features]
    切分方式: 按 out_features 维度切分为 world_size 份
    每个 rank 持有: W_shard = W[rank*shard_size : (rank+1)*shard_size, :]
    
    Forward: Y_shard = X @ W_shard.T
    如果 gather_output=True: AllGather 得到完整 Y
    """
    
    def __init__(self, in_features, out_features, gather_output=True, world_size=2):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.gather_output = gather_output
        self.world_size = world_size
        
        # 每个 rank 持有 out_features/world_size 行
        self.shard_out = out_features // world_size
        
        self.weight = nn.Parameter(torch.empty(self.shard_out, in_features))
        self.bias = nn.Parameter(torch.empty(self.shard_out))
        
        nn.init.kaiming_uniform_(self.weight)
        nn.init.zeros_(self.bias)
        
        print(f"  [ColumnParallel] Full: ({out_features}, {in_features}) -> Shard: ({self.shard_out}, {in_features})")
    
    def forward(self, x):
        # x: [batch, in_features]
        # 计算本地分片
        y_shard = x @ self.weight.T + self.bias  # [batch, shard_out]
        
        if self.gather_output:
            # AllGather 收集完整输出
            y = all_gather_sim(y_shard, self.world_size, dim=-1)
            return y
        else:
            return y_shard


class RowParallelLinearSim(nn.Module):
    """
    行并行 Linear (模拟版本)
    
    权重 W: [out_features, in_features]
    切分方式: 按 in_features 维度切分为 world_size 份
    每个 rank 持有: W_shard = W[:, rank*shard_size : (rank+1)*shard_size]
    
    Forward:
    - 输入 X_shard 已经是分片的 (来自 ColumnParallel 的输出)
    - Y_partial = X_shard @ W_shard.T
    - AllReduce 得到完整 Y
    """
    
    def __init__(self, in_features, out_features, input_is_parallel=False, world_size=2):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.input_is_parallel = input_is_parallel
        self.world_size = world_size
        
        # 每个 rank 持有 in_features/world_size 列
        self.shard_in = in_features // world_size
        
        self.weight = nn.Parameter(torch.empty(out_features, self.shard_in))
        self.bias = nn.Parameter(torch.empty(out_features))
        
        nn.init.kaiming_uniform_(self.weight)
        nn.init.zeros_(self.bias)
        
        print(f"  [RowParallel] Full: ({out_features}, {in_features}) -> Shard: ({out_features}, {self.shard_in})")
    
    def forward(self, x):
        if self.input_is_parallel:
            # x 已经是分片的: [batch, shard_in]
            x_shard = x.narrow(-1, 0, self.shard_in)  # 模拟取本 rank 的那部分
        else:
            x_shard = x
        
        # 计算部分结果
        y_partial = x_shard @ self.weight.T  # [batch, out_features]
        
        # AllReduce 求和
        y = all_reduce_sim(y_partial, self.world_size)
        
        # 加 bias (只在一个 rank 加，这里简化处理)
        y = y / self.world_size  # 修正 AllReduce 的模拟放大
        y = y + self.bias
        
        return y


# ============================================================================
# 模型和训练
# ============================================================================

class TPMlp(nn.Module):
    """
    Tensor Parallel MLP (Llama Style)
    
    结构: fc1 (ColumnParallel) -> activation -> fc2 (RowParallel)
    通信: fc1 不 gather (gather_output=False)，fc2 做 AllReduce
    """
    
    def __init__(self, hidden_size, intermediate_size, world_size=2):
        super().__init__()
        # h -> 4h, 按列切分输出
        self.fc1 = ColumnParallelLinearSim(
            hidden_size, intermediate_size, 
            gather_output=False,  # 不收集，直接传给 fc2
            world_size=world_size
        )
        # 4h -> h, 按行切分输入
        self.fc2 = RowParallelLinearSim(
            intermediate_size, hidden_size,
            input_is_parallel=True,  # 输入已经是分片的
            world_size=world_size
        )
        self.act = nn.ReLU()
    
    def forward(self, x):
        print("  Forward fc1 (ColumnParallel):")
        x = self.fc1(x)
        x = self.act(x)
        print("  Forward fc2 (RowParallel):")
        x = self.fc2(x)
        return x


def demo_simulation():
    """单进程模拟 TP 训练"""
    
    print("=" * 60)
    print("Tensor Parallelism Simulation Demo (Llama Style)")
    print("=" * 60)
    print()
    
    world_size = 2
    hidden_size = 1024
    intermediate_size = 4096
    
    print("[1] Creating TP sharded MLP model...")
    model = TPMlp(hidden_size, intermediate_size, world_size=world_size)
    print()
    
    # 统计参数量
    full_params = hidden_size * intermediate_size + intermediate_size * hidden_size
    shard_params = sum(p.numel() for p in model.parameters())
    print(f"[2] Parameter stats:")
    print(f"    Full MLP params: {full_params:,} (+ biases)")
    print(f"    Sharded params per rank: {shard_params:,}")
    print()
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    batch_size = 4
    input_data = torch.randn(batch_size, hidden_size)
    target = torch.randn(batch_size, hidden_size)
    
    print("[3] Training loop (2 steps)...")
    print("-" * 40)
    
    for step in range(2):
        print(f"\nStep {step}:")
        
        optimizer.zero_grad()
        output = model(input_data)
        loss = nn.MSELoss()(output, target)
        print(f"  Loss: {loss.item():.4f}")
        
        print("  Backward:")
        loss.backward()
        optimizer.step()
    
    print()
    print("-" * 40)
    print("[4] Demo complete!")
    print()
    print("Key takeaways:")
    print("  - ColumnParallel: 权重按列切分，输出是分片的")
    print("  - RowParallel: 权重按行切分，输入分片，输出 AllReduce")
    print("  - 通信量: 只在 RowParallel 输出时做 AllReduce")


# ============================================================================
# GPU 分布式模式
# ============================================================================

def demo_distributed(rank, world_size):
    """真正的分布式 TP (需要多 GPU)"""
    import os
    import torch.distributed as dist
    from tp import ColumnParallelLinear, RowParallelLinear
    
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12356'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    
    device = torch.device(f"cuda:{rank}")
    
    class DistributedTPMlp(nn.Module):
        def __init__(self, hidden_size, intermediate_size):
            super().__init__()
            self.fc1 = ColumnParallelLinear(hidden_size, intermediate_size, gather_output=False)
            self.fc2 = RowParallelLinear(intermediate_size, hidden_size, input_is_parallel=True)
            self.act = nn.ReLU()
        
        def forward(self, x):
            x = self.fc1(x)
            x = self.act(x)
            x = self.fc2(x)
            return x
    
    model = DistributedTPMlp(1024, 4096).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    input_data = torch.randn(32, 1024, device=device)
    target = torch.randn(32, 1024, device=device)
    
    for step in range(10):
        optimizer.zero_grad()
        output = model(input_data)
        loss = nn.MSELoss()(output, target)
        loss.backward()
        optimizer.step()
        
        if rank == 0:
            print(f"Step {step}, Loss: {loss.item():.4f}")
    
    dist.destroy_process_group()


def run_distributed(world_size):
    import torch.multiprocessing as mp
    mp.spawn(demo_distributed, args=(world_size,), nprocs=world_size, join=True)


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    if torch.cuda.is_available() and torch.cuda.device_count() >= 2:
        print("Detected multiple GPUs, running distributed TP...")
        run_distributed(torch.cuda.device_count())
    else:
        print("Running simulation mode...")
        print()
        demo_simulation()
