import torch
import torch.nn as nn
import torch.distributed as dist
from torch.autograd import Function

class Zero3AllGather(Function):
    @staticmethod
    def forward(ctx, shard, world_size, original_shape):
        """
        shard: The local shard of the parameter (1D tensor).
        world_size: Number of processes.
        original_shape: The shape of the full parameter tensor.
        """
        ctx.world_size = world_size
        ctx.original_shape = original_shape
        
        # 1. Prepare list for all gathers
        # We assume shard is 1D and all shards are equal size (padded if necessary)
        shard_size = shard.numel()
        gathered_shards = [torch.zeros_like(shard) for _ in range(world_size)]
        
        # 2. All-Gather
        dist.all_gather(gathered_shards, shard)
        
        # 3. Concatenate and reshape
        full_flat = torch.cat(gathered_shards)
        
        # 4. Remove padding (if any) and reshape
        # Calculate total elements needed
        total_elements = 1
        for dim in original_shape:
            total_elements *= dim
            
        full_tensor = full_flat[:total_elements].view(original_shape)
        
        return full_tensor

    @staticmethod
    def backward(ctx, grad_output):
        """
        grad_output: Gradient w.r.t. the full tensor.
        """
        world_size = ctx.world_size
        original_shape = ctx.original_shape
        
        # 1. Flatten the gradient
        grad_flat = grad_output.flatten()
        
        # 2. Pad if necessary to match the gathered size
        total_elements = grad_flat.numel()
        shard_size = (total_elements + world_size - 1) // world_size
        padded_size = shard_size * world_size
        
        if total_elements < padded_size:
            padding = torch.zeros(padded_size - total_elements, device=grad_flat.device, dtype=grad_flat.dtype)
            grad_flat = torch.cat([grad_flat, padding])
            
        # 3. Split into shards (conceptually, for Reduce-Scatter)
        # In PyTorch, reduce_scatter takes a list of input tensors (if using reduce_scatter)
        # or a single input tensor and output tensor (if using reduce_scatter_tensor - newer API)
        
        # Let's use the basic reduce_scatter for compatibility
        # We need to reduce the gradients from all ranks.
        # Wait, the backward of All-Gather is Reduce-Scatter.
        # Each rank has the FULL gradient (grad_output).
        # We want to sum up the gradients corresponding to OUR shard from all other ranks?
        # No.
        # In Data Parallelism:
        # Forward: Param is broadcasted (or gathered).
        # Backward: Gradients are computed on each rank independently.
        # Then we need to average them across ranks.
        
        # BUT here, we are talking about the backward of the "All-Gather" operation itself.
        # y = AllGather(x)
        # dy/dx = ?
        # x is a shard. y is the full tensor.
        # The operation "AllGather" copies x_i to the i-th block of y on ALL ranks.
        # So y = [x_0, x_1, ..., x_N]
        # The loss L depends on y.
        # dL/dx_i = sum_over_ranks ( dL/dy_block_i )
        # Because x_i was sent to all ranks and used there.
        # So we need to sum the gradients corresponding to our shard from all ranks.
        # This is exactly a Reduce-Scatter operation (sum reduction).
        
        # We need to extract the part of grad_flat that corresponds to THIS rank's shard?
        # No, we have the full grad_flat.
        # But wait, in ZeRO-3, we only computed the gradient on THIS rank using the full parameters.
        # So `grad_output` is the gradient of L_local w.r.t. Full_Params.
        # We want to compute dL_total / dShard_local.
        # L_total = sum(L_rank)
        # dL_total / dShard_local = sum_rank ( dL_rank / dShard_local )
        
        # dL_rank / dShard_local:
        # On rank j, we have Full_Params = [Shard_0, ..., Shard_local, ..., Shard_N].
        # So Shard_local contributes to Full_Params on rank j.
        # Specifically, it is the k-th chunk of Full_Params (where k = local_rank).
        # So dL_rank / dShard_local = (dL_rank / dFull_Params)[chunk_k].
        
        # So, on EACH rank j, we have a `grad_output` (dL_rank / dFull_Params).
        # We need to take the k-th chunk of this `grad_output` (where k is MY rank),
        # and sum it up across all ranks j.
        
        # Wait, if I am rank k. I want to update my Shard_k.
        # My Shard_k was used by Rank 0, Rank 1, ... Rank N.
        # Rank 0 computed dL_0 / dFull_Params. The part corresponding to Shard_k is (dL_0/dFull)[chunk_k].
        # Rank 1 computed dL_1 / dFull_Params. The part corresponding to Shard_k is (dL_1/dFull)[chunk_k].
        # ...
        # So I need sum_{j} (dL_j / dFull)[chunk_k].
        
        # This looks like a Reduce operation where everyone sends their chunk_k to rank k.
        # But we want to do this for ALL k simultaneously.
        # Rank 0 needs sum_{j} (dL_j / dFull)[chunk_0].
        # Rank 1 needs sum_{j} (dL_j / dFull)[chunk_1].
        
        # This is exactly Reduce-Scatter!
        # Input on rank j: [chunk_0, chunk_1, ..., chunk_N] of dL_j/dFull.
        # Output on rank k: sum_{j} input_j[chunk_k].
        
        # So we take `grad_flat`, split it into chunks.
        input_chunks = list(grad_flat.chunk(world_size))
        
        # Ensure all chunks are same size (padding might have created unequal chunks if not careful, 
        # but we padded grad_flat to be multiple of world_size).
        
        output_shard = torch.zeros_like(input_chunks[0])
        
        dist.reduce_scatter(output_shard, input_chunks, op=dist.ReduceOp.SUM)
        
        # Return gradients for: shard, world_size, original_shape
        return output_shard, None, None

def zero3_gather(shard, world_size, original_shape):
    return Zero3AllGather.apply(shard, world_size, original_shape)

class Zero3Linear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        
        # Determine device
        if torch.cuda.is_available():
            self.device = torch.device(f"cuda:{self.rank}")
        else:
            self.device = torch.device("cpu")

        # Initialize full parameters to get correct initialization statistics
        # We generate on rank 0 and broadcast to ensure all ranks start with same weights
        if self.rank == 0:
            full_weight = torch.randn(out_features, in_features)
            nn.init.kaiming_uniform_(full_weight, a=5**0.5)
            if bias:
                full_bias = torch.zeros(out_features)
            else:
                full_bias = None
        else:
            full_weight = torch.zeros(out_features, in_features)
            if bias:
                full_bias = torch.zeros(out_features)
            else:
                full_bias = None
                
        # Move to device for broadcast
        full_weight = full_weight.to(self.device)
        dist.broadcast(full_weight, src=0)
        
        if bias:
            full_bias = full_bias.to(self.device)
            dist.broadcast(full_bias, src=0)
            
        # Shard parameters
        self.weight_shard = nn.Parameter(self._shard_tensor(full_weight))
        self.weight_shape = full_weight.shape
        
        if bias:
            self.bias_shard = nn.Parameter(self._shard_tensor(full_bias))
            self.bias_shape = full_bias.shape
        else:
            self.register_parameter('bias_shard', None)
            self.bias_shape = None
            
        # Clean up full parameters to save memory
        del full_weight
        if bias:
            del full_bias
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _shard_tensor(self, tensor):
        flat = tensor.flatten()
        numel = flat.numel()
        shard_size = (numel + self.world_size - 1) // self.world_size
        
        # Pad
        padded_size = shard_size * self.world_size
        if numel < padded_size:
            padding = torch.zeros(padded_size - numel, device=tensor.device, dtype=tensor.dtype)
            flat = torch.cat([flat, padding])
            
        # Slice
        start = self.rank * shard_size
        end = start + shard_size
        return flat[start:end].clone().to(self.device) # Move to device

    def forward(self, input):
        # 1. Gather weights
        full_weight = zero3_gather(self.weight_shard, self.world_size, self.weight_shape)
        
        if self.bias_shard is not None:
            full_bias = zero3_gather(self.bias_shard, self.world_size, self.bias_shape)
        else:
            full_bias = None
            
        # 2. Compute
        out = nn.functional.linear(input, full_weight, full_bias)
        
        return out
