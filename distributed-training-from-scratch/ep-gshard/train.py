import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from ep import MoELayer

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12359'
    
    if torch.cuda.is_available():
        backend = "nccl" if os.name != 'nt' else "gloo"
        dist.init_process_group(backend, rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)
    else:
        backend = "gloo"
        dist.init_process_group(backend, rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def demo_ep(rank, world_size):
    print(f"Running EP demo on rank {rank}.")
    setup(rank, world_size)

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{rank}")
    else:
        device = torch.device("cpu")

    # Configuration
    d_model = 128
    hidden_size = 512
    num_experts = world_size # One expert per rank
    
    model = MoELayer(d_model, hidden_size, num_experts).to(device)
    
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # Dummy Data
    batch_size = 4
    seq_len = 16
    input_data = torch.randn(batch_size, seq_len, d_model).to(device)
    
    # Training Loop
    for i in range(5):
        optimizer.zero_grad()
        output = model(input_data)
        loss = output.mean()
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
            run_demo(demo_ep, n_gpus)
    else:
        print("CUDA not available, running on CPU with 2 processes.")
        run_demo(demo_ep, 2)
