"""
Expert Parallelism (GShard MoE) Training Demo

支持两种运行模式:
1. GPU 模式: 使用真正的分布式训练
2. CPU 模式: 使用单进程模拟，演示 MoE 的核心算法逻辑

核心算法:
- Top-K Gating: 每个 token 选择 top-k 个 expert
- All-to-All Dispatch: 将 token 路由到持有对应 expert 的 rank
- Expert Computation: 每个 rank 计算自己持有的 expert
- All-to-All Combine: 将结果路由回原 rank
- Load Balancing: 使用 auxiliary loss 平衡 expert 负载

应用场景:
- 大规模 MoE 模型 (如 Mixtral, Switch Transformer)
- 在不增加计算量的情况下扩大模型容量
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# Top-K Gating
# ============================================================================

class TopKGatingSim(nn.Module):
    """
    Top-K Gating for MoE
    
    每个 token 选择 top-k 个 expert，返回路由权重和 expert indices
    """
    
    def __init__(self, hidden_size, num_experts, top_k=2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        
        # Gating network: hidden -> num_experts scores
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)
        
        print(f"    Gating: hidden_size={hidden_size}, num_experts={num_experts}, top_k={top_k}")
    
    def forward(self, hidden_states, verbose=True):
        """
        Args:
            hidden_states: [batch*seq, hidden] or [batch, seq, hidden]
        Returns:
            router_probs: [batch*seq, top_k] - 选中 expert 的权重
            expert_indices: [batch*seq, top_k] - 选中的 expert id
            aux_loss: 负载平衡 loss
        """
        # Handle both 2D and 3D input
        if hidden_states.dim() == 3:
            batch, seq, hidden = hidden_states.shape
            num_tokens = batch * seq
            hidden_flat = hidden_states.view(num_tokens, hidden)
        else:
            num_tokens, hidden = hidden_states.shape
            hidden_flat = hidden_states
        
        # Compute gating scores
        router_logits = self.gate(hidden_flat)  # [num_tokens, num_experts]
        
        # Softmax to get probabilities
        router_probs_full = F.softmax(router_logits, dim=-1)  # [num_tokens, num_experts]
        
        # Top-K selection
        topk_probs, topk_indices = torch.topk(router_probs_full, self.top_k, dim=-1)
        
        # Renormalize selected probs
        topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)
        
        # Auxiliary loss for load balancing
        # Encourage uniform distribution of tokens to experts
        router_probs_mean = router_probs_full.mean(dim=0)  # [num_experts]
        aux_loss = self.num_experts * (router_probs_mean ** 2).sum()
        
        if verbose:
            # 统计每个 expert 被选中的次数
            expert_counts = torch.zeros(self.num_experts)
            for i in range(self.num_experts):
                expert_counts[i] = (topk_indices == i).sum().item()
            
            print(f"\n  [Gating] Token distribution to experts:")
            for i in range(self.num_experts):
                bar = "█" * int(expert_counts[i] / 2)
                print(f"    Expert {i}: {int(expert_counts[i]):3d} tokens {bar}")
        
        return topk_probs, topk_indices, aux_loss


# ============================================================================
# Expert Layer
# ============================================================================

class ExpertSim(nn.Module):
    """单个 Expert (一个 MLP)"""
    
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.fc2 = nn.Linear(intermediate_size, hidden_size)
        self.act = nn.ReLU()
    
    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


# ============================================================================
# MoE Layer with Simulated All-to-All
# ============================================================================

class MoELayerSim(nn.Module):
    """
    MoE Layer (模拟版本)
    
    每个 rank 持有 num_experts/world_size 个 expert
    通过 All-to-All 通信实现 token 路由
    """
    
    def __init__(self, hidden_size, intermediate_size, num_experts, top_k=2, world_size=2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.world_size = world_size
        self.experts_per_rank = num_experts // world_size
        
        # Gating
        self.gating = TopKGatingSim(hidden_size, num_experts, top_k)
        
        # Experts (在模拟中，我们保存所有 expert)
        self.experts = nn.ModuleList([
            ExpertSim(hidden_size, intermediate_size) 
            for _ in range(num_experts)
        ])
        
        print(f"    MoE: {num_experts} experts / {world_size} ranks = {self.experts_per_rank} experts per rank")
    
    def all_to_all_dispatch_sim(self, hidden_states, expert_indices, verbose=True):
        """
        模拟 All-to-All Dispatch
        
        将 token 按 expert_indices 分发到对应的 rank
        """
        num_tokens, hidden = hidden_states.shape
        
        if verbose:
            print(f"\n  [All-to-All Dispatch]")
        
        # 按 expert 分组 token
        dispatched = {}  # expert_id -> list of (token_idx, token)
        
        for token_idx in range(num_tokens):
            for k in range(self.top_k):
                expert_id = expert_indices[token_idx, k].item()
                if expert_id not in dispatched:
                    dispatched[expert_id] = []
                dispatched[expert_id].append((token_idx, hidden_states[token_idx], k))
        
        if verbose:
            # 统计每个 rank 收到的 token 数
            rank_counts = [0] * self.world_size
            for expert_id in dispatched:
                rank = expert_id // self.experts_per_rank
                rank_counts[rank] += len(dispatched[expert_id])
            
            for rank in range(self.world_size):
                print(f"    Rank {rank} receives {rank_counts[rank]} tokens")
        
        return dispatched
    
    def expert_compute_sim(self, dispatched, verbose=True):
        """
        模拟 Expert 计算
        
        每个 rank 计算自己持有的 expert
        """
        if verbose:
            print(f"\n  [Expert Computation]")
        
        results = {}  # (token_idx, k) -> expert_output
        
        for expert_id, tokens in dispatched.items():
            expert = self.experts[expert_id]
            rank = expert_id // self.experts_per_rank
            
            if verbose and tokens:
                print(f"    Expert {expert_id} (Rank {rank}): processing {len(tokens)} tokens")
            
            for token_idx, token, k in tokens:
                output = expert(token.unsqueeze(0)).squeeze(0)
                results[(token_idx, k)] = output
        
        return results
    
    def all_to_all_combine_sim(self, results, num_tokens, hidden_size, topk_probs, verbose=True):
        """
        模拟 All-to-All Combine
        
        将 expert 输出路由回原 rank 并加权合并
        """
        if verbose:
            print(f"\n  [All-to-All Combine]")
        
        output = torch.zeros(num_tokens, hidden_size)
        
        for (token_idx, k), expert_output in results.items():
            # 用 gating weight 加权
            weight = topk_probs[token_idx, k].item()
            output[token_idx] += weight * expert_output
        
        if verbose:
            print(f"    Combined outputs for {num_tokens} tokens")
        
        return output
    
    def forward(self, hidden_states, verbose=True):
        """
        MoE Forward Pass
        
        1. Gating: 选择 top-k experts
        2. Dispatch: All-to-All 发送 token 到 expert 所在 rank
        3. Compute: 各 rank 计算自己的 expert
        4. Combine: All-to-All 收回结果并加权合并
        """
        batch, seq, hidden = hidden_states.shape
        num_tokens = batch * seq
        
        hidden_flat = hidden_states.view(num_tokens, hidden)
        
        # 1. Gating
        topk_probs, expert_indices, aux_loss = self.gating(hidden_flat, verbose)
        
        # 2. Dispatch
        dispatched = self.all_to_all_dispatch_sim(hidden_flat, expert_indices, verbose)
        
        # 3. Expert compute
        results = self.expert_compute_sim(dispatched, verbose)
        
        # 4. Combine
        output = self.all_to_all_combine_sim(results, num_tokens, hidden, topk_probs, verbose)
        
        # Reshape back
        output = output.view(batch, seq, hidden)
        
        return output, aux_loss


# ============================================================================
# 演示
# ============================================================================

def demo_simulation():
    """单进程模拟 MoE 训练"""
    
    print("=" * 60)
    print("Expert Parallelism (GShard MoE) Simulation Demo")
    print("=" * 60)
    print()
    
    # 配置
    world_size = 2
    num_experts = 8
    hidden_size = 256
    intermediate_size = 512
    top_k = 2
    batch_size = 2
    seq_len = 16
    
    print(f"[1] Configuration:")
    print(f"    Number of experts: {num_experts}")
    print(f"    Top-K: {top_k}")
    print(f"    World size: {world_size}")
    print(f"    Hidden size: {hidden_size}")
    print()
    
    # 创建模型
    print("[2] Creating MoE layer...")
    moe = MoELayerSim(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_experts=num_experts,
        top_k=top_k,
        world_size=world_size
    )
    print()
    
    # 参数统计
    total_params = sum(p.numel() for p in moe.parameters())
    expert_params = sum(p.numel() for p in moe.experts.parameters())
    
    print(f"[3] Parameter stats:")
    print(f"    Total MoE params: {total_params:,}")
    print(f"    Expert params (all): {expert_params:,}")
    print(f"    Expert params per rank: {expert_params // world_size:,}")
    print(f"    Active params per token: {expert_params // num_experts * top_k:,} ({top_k}/{num_experts} experts)")
    print()
    
    # 前向传播
    print("[4] Forward pass...")
    print("-" * 40)
    
    input_data = torch.randn(batch_size, seq_len, hidden_size)
    output, aux_loss = moe(input_data, verbose=True)
    
    print(f"\n    Output shape: {tuple(output.shape)}")
    print(f"    Auxiliary loss: {aux_loss.item():.4f}")
    
    print()
    print("-" * 40)
    print("[5] Demo complete!")
    print()
    print("Key takeaways:")
    print("  - 每个 token 只激活 top-k 个 expert (稀疏激活)")
    print("  - All-to-All Dispatch: token 路由到 expert 所在 rank")
    print("  - All-to-All Combine: 结果路由回原 rank")
    print("  - Auxiliary loss 用于负载均衡")
    print("  - 模型容量大但计算量与 top-k 成正比")


def demo_training():
    """简单的 MoE 训练循环"""
    
    print("\n" + "=" * 60)
    print("MoE Training Loop Demo")
    print("=" * 60)
    
    world_size = 2
    num_experts = 4
    hidden_size = 128
    intermediate_size = 256
    
    moe = MoELayerSim(hidden_size, intermediate_size, num_experts, top_k=2, world_size=world_size)
    optimizer = torch.optim.Adam(moe.parameters(), lr=0.001)
    
    print("\n[Training]")
    for step in range(3):
        input_data = torch.randn(2, 8, hidden_size)
        target = torch.randn(2, 8, hidden_size)
        
        optimizer.zero_grad()
        output, aux_loss = moe(input_data, verbose=False)
        
        # Main loss + auxiliary loss for load balancing
        main_loss = F.mse_loss(output, target)
        total_loss = main_loss + 0.01 * aux_loss
        
        total_loss.backward()
        optimizer.step()
        
        print(f"  Step {step}: main_loss={main_loss.item():.4f}, aux_loss={aux_loss.item():.4f}")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    if torch.cuda.is_available() and torch.cuda.device_count() >= 2:
        print("For distributed EP, see ep.py for the full implementation.")
    
    demo_simulation()
    demo_training()
