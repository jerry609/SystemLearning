import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from tp import ColumnParallelLinear, RowParallelLinear

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12356' # Different port
    
    if torch.cuda.is_available():
        backend = "nccl" if os.name != 'nt' else "gloo"
        dist.init_process_group(backend, rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)
    else:
        backend = "gloo"
        dist.init_process_group(backend, rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

class TPMlp(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        # h -> 4h. Split output.
        self.fc1 = ColumnParallelLinear(hidden_size, intermediate_size, gather_output=False)
        # 4h -> h. Split input.
        self.fc2 = RowParallelLinear(intermediate_size, hidden_size, input_is_parallel=True)
        self.act = nn.ReLU()

    def forward(self, x):
        # x: [batch, hidden_size]
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

def demo_tp(rank, world_size):
    print(f"Running TP demo on rank {rank}.")
    setup(rank, world_size)

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{rank}")
    else:
        device = torch.device("cpu")

    # Model parameters
    hidden_size = 1024
    intermediate_size = 4096
    
    model = TPMlp(hidden_size, intermediate_size).to(device)
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Dummy input
    batch_size = 32
    input_data = torch.randn(batch_size, hidden_size).to(device)
    target = torch.randn(batch_size, hidden_size).to(device)

    # Training loop
    for i in range(10):
        optimizer.zero_grad()
        output = model(input_data)
        loss = nn.MSELoss()(output, target)
        loss.backward()
        optimizer.step()
        
        if rank == 0:
            print(f"Step {i}, Loss: {loss.item()}")

    cleanup()

def run_demo(demo_fn, world_size):
    mp.spawn(demo_fn,
             args=(world_size,),
             nprocs=world_size,
             join=True)

if __name__ == "__main__":
    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
        if n_gpus < 2:
            print(f"Requires at least 2 GPUs to run, but got {n_gpus}.")
        else:
            run_demo(demo_tp, n_gpus)
    else:
        print("CUDA not available, running on CPU with 2 processes.")
        run_demo(demo_tp, 2)
