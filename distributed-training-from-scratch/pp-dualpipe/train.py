import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from pp import PipelineStage, PipelineEngine

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12357'
    
    if torch.cuda.is_available():
        backend = "nccl" if os.name != 'nt' else "gloo"
        dist.init_process_group(backend, rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)
    else:
        backend = "gloo"
        dist.init_process_group(backend, rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

class SimpleLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features)
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(self.fc(x))

def demo_pp(rank, world_size):
    print(f"Running PP demo on rank {rank}.")
    setup(rank, world_size)

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{rank}")
    else:
        device = torch.device("cpu")

    # Configuration
    hidden_size = 1024
    micro_batch_size = 8
    num_micro_batches = 4
    
    # Define model partition for this rank
    # Assume world_size stages
    model = SimpleLayer(hidden_size, hidden_size).to(device)
    
    stage = PipelineStage(model, rank, world_size, device)
    engine = PipelineEngine(stage, micro_batch_size, num_micro_batches, (micro_batch_size, hidden_size))
    
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()

    # Dummy Data
    if rank == 0:
        inputs = [torch.randn(micro_batch_size, hidden_size).to(device) for _ in range(num_micro_batches)]
    else:
        inputs = None
        
    if rank == world_size - 1:
        labels = [torch.randn(micro_batch_size, hidden_size).to(device) for _ in range(num_micro_batches)]
    else:
        labels = None

    # Training Loop
    for step in range(5):
        optimizer.zero_grad()
        
        engine.run_gpipe(inputs, labels, loss_fn)
        
        optimizer.step()
        
        if rank == world_size - 1:
            print(f"Step {step} completed.")

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
            run_demo(demo_pp, n_gpus)
    else:
        print("CUDA not available, running on CPU with 2 processes.")
        run_demo(demo_pp, 2)
