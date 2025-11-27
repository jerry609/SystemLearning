import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from zero3 import Zero3Linear

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # Use gloo for Windows compatibility, nccl for Linux/GPU performance
    if torch.cuda.is_available():
        backend = "nccl" if os.name != 'nt' else "gloo"
        dist.init_process_group(backend, rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)
    else:
        backend = "gloo"
        dist.init_process_group(backend, rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = Zero3Linear(1024, 4096, bias=False)
        self.fc2 = Zero3Linear(4096, 1024, bias=False)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x

def demo_basic(rank, world_size):
    print(f"Running ZeRO-3 demo on rank {rank}.")
    setup(rank, world_size)

    # Determine device
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{rank}")
    else:
        device = torch.device("cpu")

    # Create model
    # Note: Zero3Linear initializes parameters on the device corresponding to rank
    model = SimpleModel()
    # Ensure model is on the correct device (though Zero3Linear handles shards)
    # But for CPU execution, we might need to be careful.
    # Zero3Linear uses dist.get_rank() to determine device index if we are not careful.
    
    # Optimizer will optimize the SHARDS
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Dummy input
    input = torch.randn(32, 1024).to(device)
    labels = torch.randn(32, 1024).to(device)

    # Training loop
    for i in range(10):
        optimizer.zero_grad()
        output = model(input)
        loss = nn.MSELoss()(output, labels)
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
            run_demo(demo_basic, n_gpus)
    else:
        print("CUDA not available, running on CPU with 2 processes.")
        run_demo(demo_basic, 2)
