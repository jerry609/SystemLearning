"""
Pipeline Parallelism (GPipe/1F1B Style) Training Demo

支持两种运行模式:
1. GPU 模式: 使用真正的分布式训练
2. CPU 模式: 使用单进程模拟，演示流水线调度逻辑

核心算法:
- 模型分层: 将模型分成多个 stage，每个 rank 持有一个 stage
- GPipe 调度: Forward all microbatches, then Backward all
- 1F1B 调度: 交替执行 Forward 和 Backward，减少 bubble
- 通信: stage 之间通过 P2P send/recv 传递激活值
"""

import torch
import torch.nn as nn
import torch.optim as optim


# ============================================================================
# Pipeline Stage 模拟
# ============================================================================

class PipelineStageSim(nn.Module):
    """
    流水线的一个 Stage
    
    每个 stage 包含模型的一部分（若干层）
    """
    
    def __init__(self, stage_id, in_features, out_features, num_layers=2):
        super().__init__()
        self.stage_id = stage_id
        
        layers = []
        for i in range(num_layers):
            if i == 0:
                layers.append(nn.Linear(in_features, out_features))
            else:
                layers.append(nn.Linear(out_features, out_features))
            layers.append(nn.ReLU())
        
        self.layers = nn.Sequential(*layers)
        print(f"  [Stage {stage_id}] Created with {num_layers} layers: {in_features} -> {out_features}")
    
    def forward(self, x):
        return self.layers(x)


# ============================================================================
# 流水线调度器
# ============================================================================

class GPipeScheduler:
    """
    GPipe 调度器
    
    调度模式: F F F F ... B B B B
    - 先执行所有 microbatch 的 forward
    - 再执行所有 microbatch 的 backward
    - bubble 较大
    """
    
    def __init__(self, stages, num_microbatches):
        self.stages = stages
        self.num_stages = len(stages)
        self.num_microbatches = num_microbatches
        self.activations = {}  # 存储激活值用于 backward
    
    def forward_pass(self, microbatch_inputs):
        """执行所有 microbatch 的 forward"""
        outputs = []
        
        print("\n  [GPipe Forward Phase]")
        for mb_idx, x in enumerate(microbatch_inputs):
            print(f"    Microbatch {mb_idx}:", end=" ")
            
            # 通过所有 stage
            activation = x
            for stage_idx, stage in enumerate(self.stages):
                activation = stage(activation)
                # 保存激活值用于 backward
                self.activations[(mb_idx, stage_idx)] = activation
                
                # 模拟 P2P 通信
                if stage_idx < self.num_stages - 1:
                    print(f"S{stage_idx}->S{stage_idx+1}", end=" ")
            
            outputs.append(activation)
            print(f"(shape: {tuple(activation.shape)})")
        
        return outputs
    
    def backward_pass(self, grad_outputs):
        """执行所有 microbatch 的 backward"""
        print("\n  [GPipe Backward Phase]")
        
        # 反向顺序遍历 microbatch
        for mb_idx in range(self.num_microbatches - 1, -1, -1):
            print(f"    Microbatch {mb_idx}:", end=" ")
            
            grad = grad_outputs[mb_idx]
            
            # 反向通过所有 stage
            for stage_idx in range(self.num_stages - 1, -1, -1):
                # 获取激活值
                activation = self.activations[(mb_idx, stage_idx)]
                
                # 计算梯度 (简化：直接 backward)
                if activation.grad_fn is not None:
                    activation.backward(grad, retain_graph=True)
                
                # 模拟 P2P 通信
                if stage_idx > 0:
                    print(f"S{stage_idx}->S{stage_idx-1}", end=" ")
            
            print("done")


class OneFOneBScheduler:
    """
    1F1B 调度器
    
    调度模式: 交替执行 Forward 和 Backward
    - Warmup: F F F F (填充流水线)
    - Steady: F B F B F B F B (稳态)
    - Cooldown: B B B B (清空流水线)
    - bubble 更小
    """
    
    def __init__(self, stages, num_microbatches):
        self.stages = stages
        self.num_stages = len(stages)
        self.num_microbatches = num_microbatches
    
    def run(self, microbatch_inputs, loss_fn, targets):
        """执行 1F1B 调度"""
        
        activations = {}
        outputs = []
        total_loss = 0.0
        
        print("\n  [1F1B Schedule]")
        
        # Warmup phase: 只做 forward 填充流水线
        warmup_steps = min(self.num_stages - 1, self.num_microbatches)
        print(f"    Warmup ({warmup_steps} forwards):", end=" ")
        
        for mb_idx in range(warmup_steps):
            x = microbatch_inputs[mb_idx]
            for stage_idx, stage in enumerate(self.stages):
                x = stage(x)
                activations[(mb_idx, stage_idx)] = x
            outputs.append(x)
            print(f"F{mb_idx}", end=" ")
        print()
        
        # Steady phase: 交替 forward 和 backward
        print(f"    Steady (1F1B):", end=" ")
        
        for step in range(self.num_microbatches - warmup_steps):
            mb_fwd = warmup_steps + step
            mb_bwd = step
            
            # Forward for next microbatch
            if mb_fwd < self.num_microbatches:
                x = microbatch_inputs[mb_fwd]
                for stage_idx, stage in enumerate(self.stages):
                    x = stage(x)
                    activations[(mb_fwd, stage_idx)] = x
                outputs.append(x)
                print(f"F{mb_fwd}", end=" ")
            
            # Backward for earlier microbatch
            if mb_bwd < len(outputs):
                output = outputs[mb_bwd]
                target = targets[mb_bwd]
                loss = loss_fn(output, target)
                total_loss += loss.item()
                loss.backward(retain_graph=True)
                print(f"B{mb_bwd}", end=" ")
        
        print()
        
        # Cooldown phase: 只做 backward 清空流水线
        print(f"    Cooldown ({warmup_steps} backwards):", end=" ")
        
        for mb_idx in range(self.num_microbatches - warmup_steps, self.num_microbatches):
            if mb_idx < len(outputs):
                output = outputs[mb_idx]
                target = targets[mb_idx]
                loss = loss_fn(output, target)
                total_loss += loss.item()
                loss.backward(retain_graph=True)
                print(f"B{mb_idx}", end=" ")
        
        print()
        
        return total_loss / self.num_microbatches


# ============================================================================
# 模型和训练
# ============================================================================

def demo_simulation():
    """单进程模拟流水线并行"""
    
    print("=" * 60)
    print("Pipeline Parallelism Simulation Demo (GPipe & 1F1B)")
    print("=" * 60)
    print()
    
    num_stages = 4
    hidden_size = 256
    batch_size = 8
    num_microbatches = 4
    microbatch_size = batch_size // num_microbatches
    
    print(f"[1] Creating {num_stages}-stage pipeline model...")
    
    # 创建流水线 stage
    stages = []
    for i in range(num_stages):
        in_dim = hidden_size if i == 0 else hidden_size
        out_dim = hidden_size
        stage = PipelineStageSim(i, in_dim, out_dim, num_layers=2)
        stages.append(stage)
    
    print()
    
    # 统计参数
    total_params = sum(sum(p.numel() for p in s.parameters()) for s in stages)
    print(f"[2] Pipeline stats:")
    print(f"    Total stages: {num_stages}")
    print(f"    Total params: {total_params:,}")
    print(f"    Params per stage: {total_params // num_stages:,}")
    print(f"    Batch size: {batch_size}, Microbatches: {num_microbatches}")
    print()
    
    # 准备数据
    full_input = torch.randn(batch_size, hidden_size)
    full_target = torch.randn(batch_size, hidden_size)
    
    # 分成 microbatch
    microbatch_inputs = full_input.chunk(num_microbatches)
    microbatch_targets = full_target.chunk(num_microbatches)
    
    loss_fn = nn.MSELoss()
    
    # Demo 1: GPipe Schedule
    print("[3] GPipe Schedule Demo")
    print("-" * 40)
    
    gpipe = GPipeScheduler(stages, num_microbatches)
    outputs = gpipe.forward_pass(list(microbatch_inputs))
    
    # 计算 loss
    grad_outputs = []
    total_loss = 0
    for i, (out, target) in enumerate(zip(outputs, microbatch_targets)):
        loss = loss_fn(out, target)
        total_loss += loss.item()
        grad_outputs.append(torch.ones_like(out))
    
    print(f"\n    Average Loss: {total_loss / num_microbatches:.4f}")
    
    # Demo 2: 1F1B Schedule
    print()
    print("[4] 1F1B Schedule Demo")
    print("-" * 40)
    
    # 重新创建 stages (reset gradients)
    stages_1f1b = []
    for i in range(num_stages):
        stage = PipelineStageSim(i, hidden_size, hidden_size, num_layers=2)
        stages_1f1b.append(stage)
    
    scheduler = OneFOneBScheduler(stages_1f1b, num_microbatches)
    avg_loss = scheduler.run(list(microbatch_inputs), loss_fn, list(microbatch_targets))
    
    print(f"\n    Average Loss: {avg_loss:.4f}")
    
    print()
    print("-" * 40)
    print("[5] Demo complete!")
    print()
    print("Key takeaways:")
    print("  - GPipe: F F F F B B B B (简单但 bubble 大)")
    print("  - 1F1B:  F F (warmup) F B F B (steady) B B (cooldown)")
    print("  - 1F1B 减少了流水线 bubble，提高 GPU 利用率")
    print("  - Stage 之间通过 P2P 通信传递激活值")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    if torch.cuda.is_available() and torch.cuda.device_count() >= 2:
        print("For distributed PP, see pp.py for the full implementation.")
        print("Running simulation demo instead...")
    
    demo_simulation()
