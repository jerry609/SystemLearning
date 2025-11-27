import torch
import torch.nn as nn
import torch.distributed as dist

def ring_pass(tensor, rank, world_size):
    # Send to next, receive from prev
    next_rank = (rank + 1) % world_size
    prev_rank = (rank - 1 + world_size) % world_size
    
    send_op = dist.P2POp(dist.isend, tensor, next_rank)
    recv_tensor = torch.zeros_like(tensor)
    recv_op = dist.P2POp(dist.irecv, recv_tensor, prev_rank)
    
    reqs = dist.batch_isend_irecv([send_op, recv_op])
    for req in reqs:
        req.wait()
        
    return recv_tensor

class RingAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

    def forward(self, q, k, v):
        # q, k, v: [batch, seq_len_local, num_heads, head_dim]
        # We assume batch size is same, seq_len is partitioned.
        
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        
        batch_size, seq_len, num_heads, head_dim = q.shape
        
        # Initialize running statistics for online softmax
        o = torch.zeros_like(q)
        l = torch.zeros(batch_size, seq_len, num_heads, 1, device=q.device)
        m = torch.full((batch_size, seq_len, num_heads, 1), float('-inf'), device=q.device)
        
        curr_k = k
        curr_v = v
        
        # Ring loop
        for step in range(world_size):
            # Compute attention with current block of K, V
            # Q: [B, S, H, D]
            # K: [B, S, H, D] -> Transpose -> [B, H, D, S]
            # Attn: [B, H, S, S] (but we do it carefully to match dimensions)
            
            # Reshape for matmul
            # q: [B, H, S, D]
            q_perm = q.permute(0, 2, 1, 3)
            k_perm = curr_k.permute(0, 2, 1, 3)
            v_perm = curr_v.permute(0, 2, 1, 3)
            
            # Scores: [B, H, S_local, S_remote]
            scores = torch.matmul(q_perm, k_perm.transpose(-2, -1)) * self.scale
            
            # Block max
            m_block = torch.max(scores, dim=-1, keepdim=True)[0] # [B, H, S, 1]
            
            # Block exp
            p_block = torch.exp(scores - m_block)
            
            # Block sum
            l_block = torch.sum(p_block, dim=-1, keepdim=True) # [B, H, S, 1]
            
            # Update running stats
            # m_new = max(m, m_block)
            m_new = torch.maximum(m.permute(0, 2, 1, 3), m_block)
            
            # factors
            alpha = torch.exp(m.permute(0, 2, 1, 3) - m_new)
            beta = torch.exp(m_block - m_new)
            
            # l_new = alpha * l + beta * l_block
            l_new = alpha * l.permute(0, 2, 1, 3) + beta * l_block
            
            # O_block = p_block @ V
            o_block = torch.matmul(p_block, v_perm) # [B, H, S, D]
            
            # O_new = (alpha * l * O + beta * O_block) / l_new
            # Note: O is currently normalized by l.
            # Actually, standard formula: O_unnorm_new = alpha * O_unnorm + beta * O_block
            # We store O_norm. So O_unnorm = O_norm * l.
            # O_unnorm_new = alpha * (O * l) + beta * O_block
            # O_new = O_unnorm_new / l_new
            
            o_perm = o.permute(0, 2, 1, 3)
            o_unnorm_new = alpha * o_perm * l.permute(0, 2, 1, 3) + beta * o_block
            o_new = o_unnorm_new / l_new
            
            # Update
            o = o_new.permute(0, 2, 1, 3)
            l = l_new.permute(0, 2, 1, 3)
            m = m_new.permute(0, 2, 1, 3)
            
            # Send K, V to next rank
            if step < world_size - 1:
                curr_k = ring_pass(curr_k, rank, world_size)
                curr_v = ring_pass(curr_v, rank, world_size)
                
        return o
