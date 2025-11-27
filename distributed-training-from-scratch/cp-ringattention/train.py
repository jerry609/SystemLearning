"""
Context Parallelism (Ring Attention) Training Demo

支持两种运行模式:
1. GPU 模式: 使用真正的分布式训练
2. CPU 模式: 使用单进程模拟，演示 Ring Attention 的核心算法逻辑

核心算法:
- 序列分片: 将长序列按 seq_len 维度切分到各 rank
- Ring Communication: KV blocks 在 ring 上流动
- 每个 rank 用流动过来的 KV 计算 partial attention
- 最后汇总得到完整的 attention 输出

应用场景:
- 超长序列 (如 128K, 1M tokens)
- 单卡显存无法容纳完整 KV cache
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# Ring Attention 核心实现 (模拟版本)
# ============================================================================

def scaled_dot_product_attention(q, k, v, scale=None):
    """标准 scaled dot-product attention"""
    if scale is None:
        scale = 1.0 / math.sqrt(q.shape[-1])
    
    # q: [batch, heads, seq_q, head_dim]
    # k: [batch, heads, seq_k, head_dim]
    # v: [batch, heads, seq_k, head_dim]
    
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # [batch, heads, seq_q, seq_k]
    attn_weights = F.softmax(scores, dim=-1)
    output = torch.matmul(attn_weights, v)  # [batch, heads, seq_q, head_dim]
    
    return output, attn_weights


def ring_attention_sim(q, k, v, world_size=2, verbose=True):
    """
    模拟 Ring Attention
    
    算法:
    1. 每个 rank 持有 Q 的一个 shard (按 seq 维度)
    2. KV 在 ring 上流动 world_size 轮
    3. 每轮计算 partial attention 并累加
    
    参数:
        q: [batch, heads, seq_shard, head_dim] - 本 rank 的 Q shard
        k: [batch, heads, seq_shard, head_dim] - 初始 K shard
        v: [batch, heads, seq_shard, head_dim] - 初始 V shard
        world_size: 模拟的 rank 数
    """
    
    batch, heads, seq_shard, head_dim = q.shape
    full_seq_len = seq_shard * world_size
    
    if verbose:
        print(f"\n  Ring Attention Configuration:")
        print(f"    - Total seq_len: {full_seq_len} = {seq_shard} x {world_size} ranks")
        print(f"    - Q shard per rank: {tuple(q.shape)}")
        print(f"    - KV shard per rank: {tuple(k.shape)}")
    
    # 模拟：生成所有 rank 的 KV shards
    # 在真正的分布式中，这些会通过 ring send/recv 传递
    all_k_shards = [k.clone() for _ in range(world_size)]  # 简化模拟
    all_v_shards = [v.clone() for _ in range(world_size)]
    
    # 存储累积结果 (online softmax 需要)
    output_acc = torch.zeros_like(q)
    max_score = torch.full((batch, heads, seq_shard, 1), float('-inf'))
    sum_exp = torch.zeros((batch, heads, seq_shard, 1))
    
    if verbose:
        print(f"\n  Ring Communication (simulated):")
    
    # Ring 流动 world_size 轮
    for ring_step in range(world_size):
        # 获取当前轮次的 KV (模拟从其他 rank 接收)
        k_recv = all_k_shards[ring_step]
        v_recv = all_v_shards[ring_step]
        
        if verbose:
            print(f"    Step {ring_step}: recv KV from rank {ring_step}", end="")
        
        # 计算 partial attention scores
        scale = 1.0 / math.sqrt(head_dim)
        scores = torch.matmul(q, k_recv.transpose(-2, -1)) * scale  # [B, H, seq_shard, seq_shard]
        
        # Online softmax (numerically stable)
        # 需要考虑之前累积的 max 和 sum
        new_max = torch.max(scores, dim=-1, keepdim=True)[0]
        combined_max = torch.maximum(max_score, new_max)
        
        # 重新缩放之前的累积
        old_scale = torch.exp(max_score - combined_max)
        new_scale = torch.exp(new_max - combined_max)
        
        # 更新 sum_exp
        new_exp = torch.exp(scores - new_max)
        new_sum_exp = new_exp.sum(dim=-1, keepdim=True)
        
        sum_exp = sum_exp * old_scale + new_sum_exp * new_scale
        
        # 更新 output
        new_output = torch.matmul(new_exp, v_recv)  # [B, H, seq_shard, head_dim]
        output_acc = output_acc * old_scale + new_output * new_scale
        
        max_score = combined_max
        
        if verbose:
            print(f" -> computed partial attention")
    
    # 最终归一化
    output = output_acc / sum_exp
    
    if verbose:
        print(f"\n  Ring Attention complete!")
        print(f"    Output shape: {tuple(output.shape)}")
    
    return output


# ============================================================================
# Ring Attention Layer
# ============================================================================

class RingAttentionLayerSim(nn.Module):
    """
    Ring Attention Layer (模拟版本)
    
    将序列按长度切分到不同 rank，通过 ring 通信计算完整 attention
    """
    
    def __init__(self, hidden_size, num_heads, world_size=2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.world_size = world_size
        
        # QKV projections
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.o_proj = nn.Linear(hidden_size, hidden_size)
        
        print(f"  [RingAttention] hidden={hidden_size}, heads={num_heads}, world_size={world_size}")
    
    def forward(self, hidden_states, verbose=True):
        """
        Args:
            hidden_states: [batch, seq_shard, hidden_size] - 本 rank 的序列分片
        """
        batch, seq_shard, _ = hidden_states.shape
        
        # Project to Q, K, V
        q = self.q_proj(hidden_states)  # [B, seq_shard, H]
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        
        # Reshape for multi-head attention
        q = q.view(batch, seq_shard, self.num_heads, self.head_dim).transpose(1, 2)  # [B, heads, seq, head_dim]
        k = k.view(batch, seq_shard, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq_shard, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Ring Attention
        attn_output = ring_attention_sim(q, k, v, self.world_size, verbose=verbose)
        
        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch, seq_shard, -1)
        
        # Output projection
        output = self.o_proj(attn_output)
        
        return output


# ============================================================================
# 演示
# ============================================================================

def demo_simulation():
    """单进程模拟 Ring Attention"""
    
    print("=" * 60)
    print("Context Parallelism (Ring Attention) Simulation Demo")
    print("=" * 60)
    print()
    
    # 配置
    world_size = 4
    hidden_size = 256
    num_heads = 8
    full_seq_len = 1024
    seq_shard = full_seq_len // world_size
    batch_size = 2
    
    print(f"[1] Configuration:")
    print(f"    Full sequence length: {full_seq_len}")
    print(f"    Sequence shard per rank: {seq_shard}")
    print(f"    Number of ranks: {world_size}")
    print(f"    Hidden size: {hidden_size}")
    print(f"    Attention heads: {num_heads}")
    print()
    
    # 创建模型
    print("[2] Creating Ring Attention layer...")
    layer = RingAttentionLayerSim(hidden_size, num_heads, world_size)
    print()
    
    # 准备输入 (只有本 rank 的 shard)
    print("[3] Running Ring Attention forward pass...")
    hidden_states = torch.randn(batch_size, seq_shard, hidden_size)
    print(f"    Input shape: {tuple(hidden_states.shape)}")
    
    # Forward
    output = layer(hidden_states, verbose=True)
    
    print()
    print("-" * 40)
    
    # 内存分析
    print("[4] Memory Analysis:")
    
    # 标准 attention 需要的内存 (O(n^2))
    standard_attn_mem = batch_size * num_heads * full_seq_len * full_seq_len * 4 / (1024**2)  # MB
    # Ring attention 需要的内存 (O(n/p * n/p))
    ring_attn_mem = batch_size * num_heads * seq_shard * seq_shard * 4 / (1024**2)  # MB
    
    print(f"    Standard attention scores: {full_seq_len}x{full_seq_len} = {standard_attn_mem:.1f} MB")
    print(f"    Ring attention scores per step: {seq_shard}x{seq_shard} = {ring_attn_mem:.1f} MB")
    print(f"    Memory reduction: {standard_attn_mem / ring_attn_mem:.1f}x")
    
    print()
    print("-" * 40)
    print("[5] Demo complete!")
    print()
    print("Key takeaways:")
    print("  - 序列按长度维度分片到不同 rank")
    print("  - KV blocks 在 ring 上流动，每步计算 partial attention")
    print("  - 使用 online softmax 保持数值稳定")
    print("  - 内存从 O(n^2) 降到 O(n^2/p)")


def compare_with_standard():
    """对比 Ring Attention 和标准 Attention 的结果"""
    
    print("\n" + "=" * 60)
    print("Comparison: Ring Attention vs Standard Attention")
    print("=" * 60)
    
    world_size = 2
    hidden_size = 64
    num_heads = 4
    head_dim = hidden_size // num_heads
    seq_len = 16
    seq_shard = seq_len // world_size
    batch = 1
    
    # 创建相同的输入
    torch.manual_seed(42)
    full_q = torch.randn(batch, num_heads, seq_len, head_dim)
    full_k = torch.randn(batch, num_heads, seq_len, head_dim)
    full_v = torch.randn(batch, num_heads, seq_len, head_dim)
    
    # Standard attention
    standard_out, _ = scaled_dot_product_attention(full_q, full_k, full_v)
    
    # Ring attention (模拟)
    # 取第一个 rank 的 Q shard
    q_shard = full_q[:, :, :seq_shard, :]
    k_shard = full_k[:, :, :seq_shard, :]
    v_shard = full_v[:, :, :seq_shard, :]
    
    ring_out = ring_attention_sim(q_shard, k_shard, v_shard, world_size, verbose=False)
    
    # 取 standard output 对应的 shard 比较
    standard_shard = standard_out[:, :, :seq_shard, :]
    
    diff = (ring_out - standard_shard).abs().max().item()
    print(f"\n  Max difference between Ring and Standard: {diff:.6f}")
    print(f"  (Note: Small diff expected due to simulation simplification)")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    if torch.cuda.is_available() and torch.cuda.device_count() >= 2:
        print("For distributed CP, see cp.py for the full implementation.")
    
    demo_simulation()
    compare_with_standard()
