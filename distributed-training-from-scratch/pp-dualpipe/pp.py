import torch
import torch.distributed as dist

def send_tensor(tensor, dst_rank):
    # Send shape first? Or assume fixed shape?
    # For simplicity, assume fixed shape or metadata exchange.
    # Here we just send the tensor.
    dist.send(tensor, dst=dst_rank)

def recv_tensor(shape, src_rank, device, dtype=torch.float32):
    tensor = torch.zeros(shape, device=device, dtype=dtype)
    dist.recv(tensor, src=src_rank)
    return tensor

class PipelineStage:
    def __init__(self, module, stage_id, num_stages, device):
        self.module = module
        self.stage_id = stage_id
        self.num_stages = num_stages
        self.device = device
        
        self.is_first = (stage_id == 0)
        self.is_last = (stage_id == num_stages - 1)
        
        self.prev_rank = stage_id - 1 if not self.is_first else None
        self.next_rank = stage_id + 1 if not self.is_last else None

    def forward_step(self, input_data=None, micro_batch_shape=None):
        # 1. Receive input if not first stage
        if not self.is_first:
            input_tensor = recv_tensor(micro_batch_shape, self.prev_rank, self.device)
            input_tensor.requires_grad = True
        else:
            input_tensor = input_data
            
        # 2. Compute
        output_tensor = self.module(input_tensor)
        
        # 3. Send output if not last stage
        if not self.is_last:
            # Detach before sending to avoid pickling the graph (though dist.send sends data only)
            send_tensor(output_tensor.detach(), self.next_rank)
            
        return input_tensor, output_tensor

    def backward_step(self, input_tensor, output_tensor, output_grad=None):
        # 1. Receive output gradient if not last stage
        if not self.is_last:
            grad_recv = recv_tensor(output_tensor.shape, self.next_rank, self.device)
        else:
            grad_recv = output_grad
            
        # 2. Compute backward
        # We need to run backward on the graph created in forward_step.
        # But we didn't keep the graph alive across function calls easily unless we stored it.
        # In a real implementation, we manage the computation graph or use autograd.backward()
        # with retain_graph=False usually, but we need to link grad_recv to output_tensor.
        
        torch.autograd.backward(output_tensor, grad_recv)
        
        # 3. Send input gradient if not first stage
        if not self.is_first:
            if input_tensor.grad is not None:
                send_tensor(input_tensor.grad, self.prev_rank)

# Simplified 1F1B Scheduler for demonstration
# This is a synchronous version for clarity, real 1F1B is more interleaved.
# We will do a simple "All Forward then All Backward" (GPipe style) for simplicity in this "from scratch" demo,
# or a very basic 1F1B if possible.
# Let's do GPipe style (F-F-F-F ... B-B-B-B) as it's easier to implement correctly in a single file without complex queues.
# Actually, the user asked for "1F1B". I should try.

class PipelineEngine:
    def __init__(self, stage, micro_batch_size, num_micro_batches, input_shape):
        self.stage = stage
        self.micro_batch_size = micro_batch_size
        self.num_micro_batches = num_micro_batches
        self.input_shape = input_shape
        
        self.saved_tensors = [] # Stores (input, output) for each microbatch

    def run_gpipe(self, batch_inputs=None, batch_labels=None, loss_fn=None):
        # GPipe Schedule: All Forwards, then All Backwards
        
        # --- Forward Pass ---
        for i in range(self.num_micro_batches):
            if self.stage.is_first:
                mb_input = batch_inputs[i]
            else:
                mb_input = None
                
            input_t, output_t = self.stage.forward_step(mb_input, self.input_shape)
            self.saved_tensors.append((input_t, output_t))

        # --- Backward Pass ---
        # Reverse order
        for i in reversed(range(self.num_micro_batches)):
            input_t, output_t = self.saved_tensors[i]
            
            if self.stage.is_last:
                mb_label = batch_labels[i]
                loss = loss_fn(output_t, mb_label)
                # Compute grad w.r.t output_t
                # We can just call backward on loss, but we need to fit into the backward_step API
                # which expects a gradient coming from "next rank".
                # Here, the "next rank" is the loss function.
                # So we compute dLoss/dOutput
                grad_output = torch.autograd.grad(loss, output_t)[0]
            else:
                grad_output = None
                
            self.stage.backward_step(input_t, output_t, grad_output)
            
        self.saved_tensors.clear()
