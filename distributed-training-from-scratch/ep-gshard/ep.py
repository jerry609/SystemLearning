import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

class TopKGating(nn.Module):
    def __init__(self, d_model, num_experts, k=1):
        super().__init__()
        self.gate = nn.Linear(d_model, num_experts)
        self.k = k

    def forward(self, x):
        # x: [batch, seq, d_model] -> flatten -> [batch*seq, d_model]
        original_shape = x.shape
        x_flat = x.view(-1, original_shape[-1])
        
        logits = self.gate(x_flat) # [N, num_experts]
        scores = F.softmax(logits, dim=-1)
        
        # Top-K
        topk_scores, topk_indices = torch.topk(scores, self.k, dim=-1)
        
        return topk_scores, topk_indices

class MoELayer(nn.Module):
    def __init__(self, d_model, hidden_size, num_experts, k=1):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.k = k
        
        self.gating = TopKGating(d_model, num_experts, k)
        
        # Each rank holds ONE expert (simplification: num_experts = world_size)
        # In real GShard, num_experts > world_size usually.
        # We assume world_size == num_experts for this demo.
        self.expert = nn.Sequential(
            nn.Linear(d_model, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, d_model)
        )

    def forward(self, x):
        # x: [batch, seq, d_model]
        batch_size, seq_len, d_model = x.shape
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        
        assert self.num_experts == world_size, "Demo assumes 1 expert per rank"
        
        # 1. Gating
        # topk_scores: [N, k], topk_indices: [N, k]
        topk_scores, topk_indices = self.gating(x)
        
        # Flatten batch and seq
        x_flat = x.view(-1, d_model) # [N, D]
        N = x_flat.size(0)
        
        # 2. Dispatch
        # We need to send tokens to the rank specified by topk_indices.
        # Since k=1 for simplicity in this demo (hard to do k>1 with simple all_to_all without padding/metadata)
        # Let's assume k=1.
        
        indices = topk_indices.squeeze(-1) # [N]
        scores = topk_scores.squeeze(-1)   # [N]
        
        # Sort tokens by expert index to group them for all_to_all
        sorted_indices, sort_map = torch.sort(indices)
        
        # Permute input
        x_sorted = x_flat[sort_map]
        
        # Calculate counts per expert
        # We need to know how many tokens go to each expert (rank)
        # This requires All-to-All communication of counts first, or fixed capacity.
        # GShard uses fixed capacity with padding/dropping.
        # For "from scratch" simplicity, let's use All-to-All with list of tensors (slow but correct).
        
        # Group data for each rank
        send_tensors = []
        for r in range(world_size):
            mask = (indices == r)
            tokens = x_flat[mask]
            send_tensors.append(tokens)
            
        # 3. All-to-All Dispatch
        # We need to send different amount of data to each rank.
        # dist.all_to_all expects list of tensors.
        recv_tensors = [torch.zeros(0, d_model, device=x.device) for _ in range(world_size)] # Placeholder? No, need shapes.
        
        # We first exchange shapes (counts)
        local_counts = torch.tensor([t.size(0) for t in send_tensors], device=x.device)
        global_counts = [torch.zeros_like(local_counts) for _ in range(world_size)]
        dist.all_gather(global_counts, local_counts)
        
        # Now we know how many tokens we will receive from each rank
        # global_counts[i][j] is how many tokens rank i sends to rank j.
        # We are rank `rank`. We receive from rank `i` -> global_counts[i][rank] tokens.
        
        recv_tensors = []
        for r in range(world_size):
            count = global_counts[r][rank].item()
            recv_tensors.append(torch.zeros(count, d_model, device=x.device))
            
        # Now exchange data
        # Note: all_to_all with list of tensors requires tensors to be contiguous
        send_tensors = [t.contiguous() for t in send_tensors]
        dist.all_to_all(recv_tensors, send_tensors)
        
        # 4. Computation
        # Concatenate all received tokens
        expert_input = torch.cat(recv_tensors, dim=0)
        
        if expert_input.size(0) > 0:
            expert_output = self.expert(expert_input)
        else:
            expert_output = torch.zeros(0, d_model, device=x.device)
            
        # 5. All-to-All Combine (Reverse Dispatch)
        # We need to send back the results to the source ranks.
        # We received `count` tokens from rank `r`. We send back `count` processed tokens to rank `r`.
        
        # Split expert_output back into chunks for each source rank
        send_back_tensors = []
        curr = 0
        for r in range(world_size):
            count = global_counts[r][rank].item()
            chunk = expert_output[curr : curr+count]
            send_back_tensors.append(chunk)
            curr += count
            
        # Prepare receive buffers
        # We sent `local_counts[r]` tokens to rank `r`. We expect same amount back.
        recv_back_tensors = []
        for r in range(world_size):
            count = local_counts[r].item()
            recv_back_tensors.append(torch.zeros(count, d_model, device=x.device))
            
        dist.all_to_all(recv_back_tensors, send_back_tensors)
        
        # 6. Reorder
        # We have results for tokens we sent to rank 0, then rank 1, ...
        # These correspond to `x_flat[mask]` where mask was `indices == r`.
        # We need to put them back into `x_flat` order.
        
        # Construct the full output tensor
        # We can't just concat because the original order was mixed.
        # We iterate and fill.
        
        output_flat = torch.zeros_like(x_flat)
        
        for r in range(world_size):
            mask = (indices == r)
            # We received recv_back_tensors[r] which corresponds to these indices
            output_flat[mask] = recv_back_tensors[r]
            
        # Apply gating scores
        output_flat = output_flat * scores.unsqueeze(-1)
        
        # Reshape
        output = output_flat.view(batch_size, seq_len, d_model)
        
        # Add residual connection (usually done outside, but let's return just MoE output)
        return output
