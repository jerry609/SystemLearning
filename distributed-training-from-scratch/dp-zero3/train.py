"""
ZeRO-3 Training Demo

支持两种运行模式:
1. GPU 模式: 使用真正的分布式训练 (NCCL backend)
2. CPU 模式: 使用单进程模拟，演示 ZeRO-3 的核心算法逻辑

核心算法:
- 参数分片: 每个 rank 只保存 1/world_size 的参数
- 前向 AllGather: 收集完整参数进行计算
- 反向 ReduceScatter: 计算梯度后分片回各 rank
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim

# ============================================================================
# 模拟通信原语 (用于 CPU 单进程模式)
# ============================================================================

class SimulatedDistributed:
    """模拟分布式环境，用于在单进程中演示 ZeRO-3 算法"""
    
    def __init__(self, world_size=2):
        self.world_size = world_size
        self.rank = 0  # 主进程视角
        self._initialized = True
    
    def is_initialized(self):
        return self._initialized
    
    def get_rank(self):
        return self.rank
    
    def get_world_size(self):
        return self.world_size
    
    def all_gather(self, tensor_list, tensor):
        """模拟 AllGather: 将各 rank 的 tensor 收集到 tensor_list"""
        # 在模拟模式下，我们假设所有 shard 都在本地
        # 实际上就是把 tensor 复制 world_size 次然后拼接
        full_size = tensor.shape[0] * self.world_size
        for i, t in enumerate(tensor_list):
            # 模拟：每个 rank 贡献自己的那部分
            t.copy_(tensor)  # 简化：所有 rank 有相同数据
    
    def reduce_scatter(self, output, input_list, op=None):
        """模拟 ReduceScatter: 对 input_list 求和后分片"""
        # 先求和
        total = sum(input_list)
        # 取自己 rank 对应的那一片
        shard_size = total.shape[0] // self.world_size
        output.copy_(total[:shard_size])


# 全局模拟分布式对象
_sim_dist = None

def get_sim_dist():
    global _sim_dist
    if _sim_dist is None:
        _sim_dist = SimulatedDistributed(world_size=2)
    return _sim_dist


# ============================================================================
# ZeRO-3 核心实现 (模拟版本)
# ============================================================================

class Zero3AllGatherSim(torch.autograd.Function):
    """
    ZeRO-3 AllGather 的模拟实现
    
    Forward: 模拟从各 rank 收集完整参数
    Backward: 模拟 ReduceScatter 分发梯度
    """
    
    @staticmethod
    def forward(ctx, shard, world_size):
        ctx.world_size = world_size
        ctx.shard_shape = shard.shape
        
        # 模拟 AllGather: 将 shard 扩展为完整参数
        # 实际分布式中，每个 rank 的 shard 是不同的
        # 这里我们模拟为：完整参数 = shard 重复 world_size 次
        full_param = shard.repeat(world_size, *([1] * (shard.dim() - 1)))
        
        print(f"  [Forward] AllGather: shard {tuple(shard.shape)} -> full {tuple(full_param.shape)}")
        return full_param
    
    @staticmethod
    def backward(ctx, grad_full):
        world_size = ctx.world_size
        
        # 模拟 ReduceScatter: 将完整梯度分片
        # 实际分布式中，各 rank 获得梯度的不同部分
        shard_size = grad_full.shape[0] // world_size
        
        # 模拟：先 "reduce" (这里假设所有 rank 的梯度相同)
        # 然后取自己 rank 的那一片
        grad_shard = grad_full[:shard_size].clone()
        
        print(f"  [Backward] ReduceScatter: full {tuple(grad_full.shape)} -> shard {tuple(grad_shard.shape)}")
        return grad_shard, None


def zero3_gather_sim(shard, world_size=2):
    """辅助函数：调用 ZeRO-3 AllGather"""
    return Zero3AllGatherSim.apply(shard, world_size)


class Zero3LinearSim(nn.Module):
    """
    ZeRO-3 Linear 层的模拟实现
    
    - 参数按第一个维度分片存储
    - Forward 时 AllGather 收集完整参数
    - Backward 时 ReduceScatter 分发梯度
    """
    
    def __init__(self, in_features, out_features, bias=True, world_size=2):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.world_size = world_size
        
        # 参数分片：只存储 1/world_size 的参数
        shard_size = out_features // world_size
        self.weight_shard = nn.Parameter(torch.empty(shard_size, in_features))
        
        if bias:
            self.bias_shard = nn.Parameter(torch.empty(shard_size))
        else:
            self.register_parameter('bias_shard', None)
        
        self._init_weights()
        
        print(f"  [Init] Zero3Linear: full ({out_features}, {in_features}) -> shard ({shard_size}, {in_features})")
    
    def _init_weights(self):
        nn.init.kaiming_uniform_(self.weight_shard)
        if self.bias_shard is not None:
            nn.init.zeros_(self.bias_shard)
    
    def forward(self, x):
        # AllGather: 收集完整权重
        full_weight = zero3_gather_sim(self.weight_shard, self.world_size)
        
        # 标准线性计算
        output = x @ full_weight.T
        
        if self.bias_shard is not None:
            full_bias = zero3_gather_sim(self.bias_shard, self.world_size)
            output = output + full_bias
        
        return output


# ============================================================================
# 模型和训练
# ============================================================================

class SimpleModel(nn.Module):
    """使用 ZeRO-3 分片的简单模型"""
    
    def __init__(self, world_size=2):
        super().__init__()
        self.fc1 = Zero3LinearSim(1024, 4096, bias=False, world_size=world_size)
        self.fc2 = Zero3LinearSim(4096, 1024, bias=False, world_size=world_size)
    
    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x


def demo_simulation():
    """单进程模拟 ZeRO-3 训练"""
    
    print("=" * 60)
    print("ZeRO-3 Simulation Demo (CPU Single Process)")
    print("=" * 60)
    print()
    
    world_size = 2  # 模拟 2 个 rank
    device = torch.device("cpu")
    
    print("[1] Creating ZeRO-3 sharded model...")
    model = SimpleModel(world_size=world_size)
    print()
    
    # 统计参数量
    full_params = 1024 * 4096 + 4096 * 1024  # 如果不分片
    shard_params = sum(p.numel() for p in model.parameters())
    print(f"[2] Parameter stats:")
    print(f"    Full model params: {full_params:,}")
    print(f"    Sharded params per rank: {shard_params:,}")
    print(f"    Memory saving: {(1 - shard_params/full_params)*100:.1f}%")
    print()
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 生成数据
    batch_size = 4
    input_data = torch.randn(batch_size, 1024, device=device)
    labels = torch.randn(batch_size, 1024, device=device)
    
    print("[3] Training loop (3 steps)...")
    print("-" * 40)
    
    for step in range(3):
        print(f"\nStep {step}:")
        
        optimizer.zero_grad()
        
        # Forward: 触发 AllGather
        output = model(input_data)
        
        # Loss
        loss = nn.MSELoss()(output, labels)
        print(f"  Loss: {loss.item():.4f}")
        
        # Backward: 触发 ReduceScatter
        loss.backward()
        
        # Optimizer step: 更新分片参数
        optimizer.step()
    
    print()
    print("-" * 40)
    print("[4] Demo complete!")
    print()
    print("Key takeaways:")
    print("  - 每个 rank 只存储 1/world_size 的参数 (内存节省)")
    print("  - Forward 时 AllGather 收集完整参数进行计算")
    print("  - Backward 时 ReduceScatter 将梯度分片回各 rank")


# ============================================================================
# GPU 分布式模式 (需要多 GPU)
# ============================================================================

def demo_distributed(rank, world_size):
    """真正的分布式训练 (需要多 GPU)"""
    import torch.distributed as dist
    
    # 初始化分布式
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    
    device = torch.device(f"cuda:{rank}")
    
    # 使用真正的分布式 ZeRO-3
    from zero3 import Zero3Linear
    
    class DistributedModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = Zero3Linear(1024, 4096, bias=False)
            self.fc2 = Zero3Linear(4096, 1024, bias=False)
        
        def forward(self, x):
            x = self.fc1(x)
            x = torch.relu(x)
            x = self.fc2(x)
            return x
    
    model = DistributedModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    input_data = torch.randn(32, 1024, device=device)
    labels = torch.randn(32, 1024, device=device)
    
    for step in range(10):
        optimizer.zero_grad()
        output = model(input_data)
        loss = nn.MSELoss()(output, labels)
        loss.backward()
        optimizer.step()
        
        if rank == 0:
            print(f"Step {step}, Loss: {loss.item():.4f}")
    
    dist.destroy_process_group()


def run_distributed(world_size):
    """启动多 GPU 分布式训练"""
    import torch.multiprocessing as mp
    mp.spawn(demo_distributed, args=(world_size,), nprocs=world_size, join=True)


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    if torch.cuda.is_available() and torch.cuda.device_count() >= 2:
        print("Detected multiple GPUs, running distributed training...")
        run_distributed(torch.cuda.device_count())
    else:
        print("Running simulation mode (no GPU or single GPU)...")
        print()
        demo_simulation()
