import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.autograd import Function

# --- Communication Primitives ---

class CopyToModelParallelRegion(Function):
    @staticmethod
    def forward(ctx, input):
        return input

    @staticmethod
    def backward(ctx, grad_output):
        # All-Reduce the gradient
        dist.all_reduce(grad_output, op=dist.ReduceOp.SUM)
        return grad_output

class ReduceFromModelParallelRegion(Function):
    @staticmethod
    def forward(ctx, input):
        # All-Reduce the input
        dist.all_reduce(input, op=dist.ReduceOp.SUM)
        return input

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

class ScatterToModelParallelRegion(Function):
    @staticmethod
    def forward(ctx, input):
        # Split input along the last dimension
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        last_dim = input.dim() - 1
        chunks = input.chunk(world_size, dim=last_dim)
        return chunks[rank]

    @staticmethod
    def backward(ctx, grad_output):
        # Gather gradients
        world_size = dist.get_world_size()
        last_dim = grad_output.dim() - 1
        gathered_grads = [torch.zeros_like(grad_output) for _ in range(world_size)]
        dist.all_gather(gathered_grads, grad_output)
        return torch.cat(gathered_grads, dim=last_dim)

class GatherFromModelParallelRegion(Function):
    @staticmethod
    def forward(ctx, input):
        # All-Gather input along the last dimension
        world_size = dist.get_world_size()
        last_dim = input.dim() - 1
        gathered_inputs = [torch.zeros_like(input) for _ in range(world_size)]
        dist.all_gather(gathered_inputs, input)
        return torch.cat(gathered_inputs, dim=last_dim)

    @staticmethod
    def backward(ctx, grad_output):
        # Split gradients
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        last_dim = grad_output.dim() - 1
        chunks = grad_output.chunk(world_size, dim=last_dim)
        return chunks[rank]

def copy_to_tensor_model_parallel_region(input):
    return CopyToModelParallelRegion.apply(input)

def reduce_from_tensor_model_parallel_region(input):
    return ReduceFromModelParallelRegion.apply(input)

def scatter_to_tensor_model_parallel_region(input):
    return ScatterToModelParallelRegion.apply(input)

def gather_from_tensor_model_parallel_region(input):
    return GatherFromModelParallelRegion.apply(input)

# --- Layers ---

class ColumnParallelLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, gather_output=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.gather_output = gather_output
        
        world_size = dist.get_world_size()
        self.output_size_per_partition = out_features // world_size
        
        # Note: Weight shape is [out_features_per_partition, in_features]
        self.weight = nn.Parameter(torch.Tensor(self.output_size_per_partition, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(self.output_size_per_partition))
        else:
            self.register_parameter('bias', None)
            
        self.reset_parameters()

    def reset_parameters(self):
        # Initialize full weight then shard, or initialize sharded directly.
        # For simplicity, we initialize sharded directly but with different seeds if needed,
        # or same seed for deterministic behavior if we were splitting a pretrained model.
        # Here we just do random init.
        nn.init.xavier_normal_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, input):
        # Input: [batch, in_features]
        # We copy the input to all ranks (identity forward, all-reduce backward)
        input_parallel = copy_to_tensor_model_parallel_region(input)
        
        # Local computation
        output_parallel = F.linear(input_parallel, self.weight, self.bias)
        
        if self.gather_output:
            # All-Gather output: [batch, out_features]
            output = gather_from_tensor_model_parallel_region(output_parallel)
        else:
            # Output stays partitioned: [batch, out_features_per_partition]
            output = output_parallel
            
        return output

class RowParallelLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, input_is_parallel=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.input_is_parallel = input_is_parallel
        
        world_size = dist.get_world_size()
        self.input_size_per_partition = in_features // world_size
        
        # Note: Weight shape is [out_features, in_features_per_partition]
        self.weight = nn.Parameter(torch.Tensor(out_features, self.input_size_per_partition))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
            
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, input):
        # Input: [batch, in_features_per_partition] if input_is_parallel
        if not self.input_is_parallel:
            input_parallel = scatter_to_tensor_model_parallel_region(input)
        else:
            input_parallel = input
            
        # Local computation
        output_parallel = F.linear(input_parallel, self.weight)
        
        # All-Reduce output: [batch, out_features]
        output = reduce_from_tensor_model_parallel_region(output_parallel)
        
        if self.bias is not None:
            output = output + self.bias
            
        return output
