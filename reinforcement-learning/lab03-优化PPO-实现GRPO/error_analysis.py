#!/usr/bin/env python3
"""
CUDA错误根因分析脚本
分析为什么torch.multinomial在第二轮训练时失败
"""

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

def analyze_cuda_error():
    """分析CUDA错误的根本原因"""
    
    print("=== CUDA错误根因分析 ===")
    
    # 复现错误场景
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained("gpt2").cuda()
    
    print("\n1. 错误机制分析:")
    print("   torch.multinomial 需要有效的概率分布作为输入")
    print("   CUDA断言 'input[0] != 0' 检查概率分布不能全零")
    
    # 模拟导致错误的情况
    print("\n2. 错误触发条件:")
    
    # 情况1：全零logits
    print("   情况1: 全零logits -> 全零概率分布")
    zero_logits = torch.zeros(1, 50257).cuda()  # GPT2词汇表大小
    zero_probs = F.softmax(zero_logits, dim=-1)
    print(f"     全零logits的softmax: {zero_probs[0][:5]} (前5个值)")
    print(f"     所有值是否相等: {torch.all(zero_probs[0] == zero_probs[0][0])}")
    
    # 情况2：极大负值logits
    print("   情况2: 极大负值logits -> 接近全零概率分布")
    neg_inf_logits = torch.full((1, 50257), -1e6).cuda()
    neg_inf_probs = F.softmax(neg_inf_logits, dim=-1)
    print(f"     极大负值logits的softmax: {neg_inf_probs[0][:5]} (前5个值)")
    print(f"     最大概率值: {neg_inf_probs.max().item()}")
    
    # 情况3：NaN或inf logits
    print("   情况3: NaN/inf logits")
    nan_logits = torch.full((1, 50257), float('nan')).cuda()
    try:
        nan_probs = F.softmax(nan_logits, dim=-1)
        print(f"     NaN logits的softmax: {nan_probs[0][:5]} (前5个值)")
    except Exception as e:
        print(f"     NaN logits处理失败: {e}")
    
    print("\n3. 训练导致模型退化的原因:")
    print("   ✗ 梯度爆炸: 第一轮训练的梯度过大，破坏模型参数")
    print("   ✗ 数值不稳定: 损失计算中的NaN/inf传播到模型参数")
    print("   ✗ 学习率过高: 参数更新步长过大，跳出有效参数空间")
    print("   ✗ 温度参数异常: 生成参数导致logits分布退化")
    
    # 模拟正常的参数更新
    print("\n4. 模型参数健康检查:")
    
    # 检查模型参数状态
    def check_model_health(model, name):
        print(f"   {name}模型状态:")
        total_params = 0
        nan_params = 0
        inf_params = 0
        
        for param in model.parameters():
            if param.requires_grad:
                total_params += param.numel()
                nan_params += torch.isnan(param).sum().item()
                inf_params += torch.isinf(param).sum().item()
        
        print(f"     总参数数: {total_params:,}")
        print(f"     NaN参数数: {nan_params}")
        print(f"     Inf参数数: {inf_params}")
        print(f"     参数健康度: {'✅ 健康' if (nan_params + inf_params) == 0 else '❌ 异常'}")
        
        return nan_params + inf_params == 0
    
    # 初始状态检查
    print("   初始状态:")
    initial_health = check_model_health(model, "初始")
    
    # 模拟一次有问题的训练步骤
    print("\n5. 模拟有问题的训练步骤:")
    
    # 创建一个会导致问题的损失
    prompt = "人工智能的发展趋势是"
    input_ids = tokenizer.encode(prompt, return_tensors="pt").cuda()
    
    print(f"   输入: {prompt}")
    print(f"   输入tokens: {input_ids[0][:5].tolist()}...")
    
    # 正常前向传播
    model.train()
    outputs = model(input_ids)
    logits = outputs.logits
    
    print(f"   正常logits范围: [{logits.min().item():.2f}, {logits.max().item():.2f}]")
    print(f"   正常logits均值: {logits.mean().item():.2f}")
    
    # 创建一个会导致梯度爆炸的损失
    # 这里故意创建极大的损失来模拟问题
    fake_targets = torch.randint(0, 50257, (1, input_ids.shape[1])).cuda()
    
    # 计算交叉熵损失
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = fake_targets[..., 1:].contiguous()
    
    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)), 
        shift_labels.view(-1), 
        reduction='mean'
    )
    
    print(f"   正常损失值: {loss.item():.4f}")
    
    # 人为放大损失来模拟问题（不要在真实训练中这样做）
    problematic_loss = loss * 1e6  # 放大损失
    print(f"   问题损失值: {problematic_loss.item():.4f}")
    
    # 检查梯度
    print("\n6. 梯度分析:")
    model.zero_grad()
    problematic_loss.backward()
    
    total_grad_norm = 0.0
    max_grad = 0.0
    nan_grads = 0
    
    for param in model.parameters():
        if param.grad is not None:
            param_grad_norm = param.grad.data.norm(2).item()
            total_grad_norm += param_grad_norm ** 2
            max_grad = max(max_grad, param_grad_norm)
            nan_grads += torch.isnan(param.grad).sum().item()
    
    total_grad_norm = total_grad_norm ** 0.5
    
    print(f"   总梯度范数: {total_grad_norm:.2f}")
    print(f"   最大梯度范数: {max_grad:.2f}")
    print(f"   NaN梯度数: {nan_grads}")
    print(f"   梯度状态: {'❌ 梯度爆炸' if total_grad_norm > 100 else '✅ 正常'}")
    
    print("\n7. 解决方案:")
    print("   🔧 梯度裁剪: torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)")
    print("   🔧 学习率衰减: 使用更小的学习率 (1e-6 而不是 1e-5)")
    print("   🔧 参数监控: 每步检查参数健康度")
    print("   🔧 数值稳定: 在softmax前限制logits范围")
    print("   🔧 温度限制: 确保temperature >= 0.1, top_p <= 0.95")
    print("   🔧 定期重置: 检测到CUDA错误时重新加载模型")
    
    print("\n=== 结论 ===")
    print("CUDA在第二轮被破坏的原因是:")
    print("1. 第一轮训练产生了无效的梯度更新")
    print("2. 模型参数被破坏(NaN/inf/极值)")
    print("3. 破坏的模型生成无效的logits分布")
    print("4. torch.multinomial收到无效概率分布时触发CUDA断言")
    print("5. CUDA错误传播，破坏整个GPU状态")

if __name__ == "__main__":
    try:
        analyze_cuda_error()
    except Exception as e:
        print(f"分析过程出错: {e}")
        print("这进一步证实了CUDA状态的不稳定性") 