import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from cp import RingAttention

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12358'
    
    if torch.cuda.is_available():
        backend = "nccl" if os.name != 'nt' else "gloo"
        dist.init_process_group(backend, rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)
    else:
        backend = "gloo"
        dist.init_process_group(backend, rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def demo_cp(rank, world_size):
    print(f"Running CP demo on rank {rank}.")
    setup(rank, world_size)

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{rank}")
    else:
        device = torch.device("cpu")

    # Configuration
    dim = 128
    num_heads = 4
    seq_len_total = 128
    seq_len_local = seq_len_total // world_size
    batch_size = 2
    
    model = RingAttention(dim, num_heads).to(device)
    
    # Dummy Data
    # Each rank has a slice of the sequence
    q = torch.randn(batch_size, seq_len_local, num_heads, dim // num_heads).to(device)
    k = torch.randn(batch_size, seq_len_local, num_heads, dim // num_heads).to(device)
    v = torch.randn(batch_size, seq_len_local, num_heads, dim // num_heads).to(device)
    
    q.requires_grad = True
    k.requires_grad = True
    v.requires_grad = True
    
    optimizer = optim.SGD(model.parameters(), lr=0.01) # No params in RingAttention actually, but for consistency

    # Forward
    output = model(q, k, v)
    
    # Loss
    loss = output.mean()
    loss.backward()
    
    if rank == 0:
        print(f"Loss: {loss.item()}")
        print("Backward completed.")

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
            run_demo(demo_cp, n_gpus)
    else:
        print("CUDA not available, running on CPU with 2 processes.")
        run_demo(demo_cp, 2)
