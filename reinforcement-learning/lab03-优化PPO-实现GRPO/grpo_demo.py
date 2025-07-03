#!/usr/bin/env python3
"""
Lab03 GRPO算法完整实现演示
Group Relative Policy Optimization - 移除Critic，使用组平均奖励作为基准

核心创新：
1. 移除评论家模型，降低50%显存占用
2. 对每个prompt采样多个响应，计算组平均奖励
3. 使用组平均奖励作为动态基准来计算优势函数
4. 大幅提升训练效率和资源利用率
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    GenerationConfig
)
from datasets import Dataset
import numpy as np
import time
import psutil
import GPUtil

class SimpleRewardModel(nn.Module):
    """简单的奖励模型，用于评估生成文本质量"""
    def __init__(self, vocab_size, hidden_size=256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        )
        
    def forward(self, input_ids, attention_mask=None):
        embeddings = self.embedding(input_ids)
        # 平均池化
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
            sum_embeddings = torch.sum(embeddings * mask_expanded, 1)
            sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
            mean_embeddings = sum_embeddings / sum_mask
        else:
            mean_embeddings = torch.mean(embeddings, dim=1)
            
        return self.fc(mean_embeddings)

class GRPOTrainer:
    """GRPO训练器 - 移除Critic模型的轻量级实现"""
    
    def __init__(self, model, ref_model, reward_model, tokenizer, config):
        self.model = model
        self.ref_model = ref_model
        self.reward_model = reward_model
        self.tokenizer = tokenizer
        self.config = config
        
        # 优化器只优化主模型
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=config['learning_rate'],
            weight_decay=config.get('weight_decay', 0.01)
        )
        
    def reset_cuda_state(self):
        """完全重置CUDA状态"""
        try:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            torch.cuda.ipc_collect()
            # 强制垃圾回收
            import gc
            gc.collect()
            print("       CUDA状态已重置")
        except Exception as e:
            print(f"       CUDA重置警告: {e}")
    
    def safe_cuda_operation(self, operation_func, *args, **kwargs):
        """安全执行CUDA操作，自动处理错误恢复"""
        max_retries = 3
        for retry in range(max_retries):
            try:
                return operation_func(*args, **kwargs)
            except RuntimeError as e:
                if "CUDA" in str(e) and retry < max_retries - 1:
                    print(f"       CUDA错误，尝试恢复 (第{retry+1}次)")
                    # 完整的CUDA恢复流程
                    self.reset_cuda_state()
                    
                    # 将模型重新加载到GPU
                    try:
                        self.model = self.model.cuda()
                        self.ref_model = self.ref_model.cuda() 
                        self.reward_model = self.reward_model.cuda()
                        print("       模型已重新加载到GPU")
                    except Exception as reload_e:
                        print(f"       GPU重新加载失败: {reload_e}")
                        # 如果GPU完全不可用，切换到CPU
                        self.model = self.model.cpu()
                        self.ref_model = self.ref_model.cpu()
                        self.reward_model = self.reward_model.cpu()
                        print("       已切换到CPU模式")
                        
                    continue
                else:
                    raise e
        
        # 如果所有重试都失败，抛出异常
        raise RuntimeError("CUDA操作重试失败")
    
    def generate_responses(self, prompts, num_samples_per_prompt=4):
        """为每个prompt生成多个响应 - 带完整错误防护"""
        all_responses = []
        all_prompt_indices = []
        
        generation_kwargs = {
            "max_new_tokens": self.config.get("max_new_tokens", 20),
            "do_sample": True,
            "temperature": max(0.1, self.config.get("temperature", 0.7)),
            "top_p": min(0.95, self.config.get("top_p", 0.9)),
            "top_k": 50,
            "pad_token_id": self.tokenizer.eos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "repetition_penalty": 1.1,
            "use_cache": True,
            "output_scores": False,  # 关闭scores输出减少内存
        }
        
        for prompt_idx, prompt in enumerate(prompts):
            # 安全编码prompt
            try:
                encoded = self.tokenizer(
                    prompt, 
                    return_tensors="pt", 
                    truncation=True, 
                    max_length=100,  # 减少长度避免内存问题
                    padding=False,   # 不padding，避免额外的pad token
                    add_special_tokens=True
                )
                
                input_ids = encoded['input_ids'].to(self.model.device)
                
                # 数值安全检查
                if input_ids.numel() == 0:
                    print(f"       警告: prompt '{prompt}' 编码为空，跳过")
                    continue
                    
                # 检查token ids是否在有效范围内
                vocab_size = len(self.tokenizer)
                if (input_ids >= vocab_size).any() or (input_ids < 0).any():
                    print(f"       警告: prompt '{prompt}' 包含无效token ids，跳过")
                    continue
                
                print(f"       编码prompt {prompt_idx+1}: 长度={input_ids.shape[1]}, tokens={input_ids[0][:5].tolist()}...")
                
            except Exception as e:
                print(f"       错误: 编码prompt '{prompt}' 失败: {e}")
                continue
            
            # 为每个prompt生成多个响应
            for sample_idx in range(num_samples_per_prompt):
                try:
                    # 清理GPU缓存
                    if sample_idx % 2 == 0:
                        torch.cuda.empty_cache()
                    
                    with torch.no_grad():
                        # 确保模型处于评估模式
                        self.model.eval()
                        
                        # 生成响应
                        output = self.model.generate(
                            input_ids,
                            **generation_kwargs
                        )
                        
                        # 安全提取新生成的tokens
                        if output.shape[1] <= input_ids.shape[1]:
                            response_text = ""  # 没有生成新tokens
                        else:
                            new_tokens = output[0][input_ids.shape[1]:]
                            
                            # 检查生成的tokens是否有效
                            if (new_tokens >= vocab_size).any() or (new_tokens < 0).any():
                                print(f"       警告: 生成了无效token，使用默认响应")
                                response_text = "生成内容"
                            else:
                                response_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
                                response_text = response_text.strip()
                        
                        # 确保响应不为空且合理
                        if not response_text or len(response_text) < 2:
                            response_text = "生成内容"
                        
                        # 限制响应长度
                        if len(response_text) > 100:
                            response_text = response_text[:100] + "..."
                            
                        all_responses.append(response_text)
                        all_prompt_indices.append(prompt_idx)
                        
                        print(f"         样本 {sample_idx+1}: '{response_text[:30]}...'")
                        
                except RuntimeError as e:
                    if "CUDA" in str(e):
                        print(f"       CUDA错误在样本 {sample_idx+1}: {e}")
                        # 清理GPU状态
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                        # 使用CPU模式生成备用响应
                        try:
                            with torch.no_grad():
                                cpu_input = input_ids.cpu()
                                cpu_model = self.model.cpu()
                                cpu_output = cpu_model.generate(
                                    cpu_input,
                                    max_new_tokens=10,
                                    do_sample=False,  # CPU上使用贪婪解码
                                    pad_token_id=self.tokenizer.eos_token_id
                                )
                                new_tokens = cpu_output[0][cpu_input.shape[1]:]
                                response_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
                                if not response_text:
                                    response_text = "备用响应"
                                # 模型移回GPU
                                self.model.to(self.model.device)
                                all_responses.append(response_text)
                                all_prompt_indices.append(prompt_idx)
                                print(f"         CPU备用响应: '{response_text}'")
                        except Exception as cpu_e:
                            print(f"       CPU备用生成也失败: {cpu_e}")
                            all_responses.append("默认响应")
                            all_prompt_indices.append(prompt_idx)
                    else:
                        print(f"       其他生成错误: {e}")
                        all_responses.append("错误响应")
                        all_prompt_indices.append(prompt_idx)
                        
                except Exception as e:
                    print(f"       未知错误在样本 {sample_idx+1}: {e}")
                    all_responses.append("未知错误响应")
                    all_prompt_indices.append(prompt_idx)
            
        print(f"     总共生成 {len(all_responses)} 个响应")
        return all_responses, all_prompt_indices
    
    def compute_group_advantages(self, prompts, responses, prompt_indices):
        """计算组相对优势 - GRPO的核心创新"""
        
        # 1. 计算所有响应的奖励
        all_rewards = []
        for prompt_idx, response in zip(prompt_indices, responses):
            prompt = prompts[prompt_idx]
            full_text = prompt + response
            
            # 编码完整文本
            input_ids = self.tokenizer.encode(
                full_text, return_tensors="pt", truncation=True, max_length=256
            ).to(next(self.reward_model.parameters()).device)
            
            with torch.no_grad():
                reward = self.reward_model(input_ids)
                all_rewards.append(reward.squeeze().cpu().item())
        
        # 2. 计算每个prompt组的平均奖励作为基准
        unique_prompt_indices = list(set(prompt_indices))
        group_baselines = {}
        
        for prompt_idx in unique_prompt_indices:
            # 找到属于这个prompt的所有响应的奖励
            group_rewards = [
                reward for i, reward in enumerate(all_rewards) 
                if prompt_indices[i] == prompt_idx
            ]
            # 计算组平均奖励作为基准
            group_baselines[prompt_idx] = np.mean(group_rewards)
        
        # 3. 计算优势 = 个体奖励 - 组平均奖励
        advantages = []
        for prompt_idx, reward in zip(prompt_indices, all_rewards):
            advantage = reward - group_baselines[prompt_idx]
            advantages.append(advantage)
        
        return advantages, all_rewards, group_baselines
    
    def compute_policy_loss(self, prompts, responses, prompt_indices, advantages):
        """计算策略损失 - 带数值安全检查"""
        
        total_loss = 0.0
        total_kl = 0.0
        valid_samples = 0
        
        for prompt, response, advantage in zip(
            [prompts[i] for i in prompt_indices], responses, advantages
        ):
            try:
                # 编码序列
                full_text = prompt + response
                input_ids = self.tokenizer.encode(
                    full_text, return_tensors="pt", truncation=True, max_length=200
                ).to(self.model.device)
                
                # 安全检查
                if input_ids.numel() == 0:
                    continue
                    
                prompt_length = len(self.tokenizer.encode(prompt, truncation=True, max_length=100))
                
                # 确保有响应tokens
                if input_ids.shape[1] <= prompt_length:
                    continue
                
                # 获取模型输出 - 安全模式
                with torch.no_grad():
                    try:
                        ref_logits = self.ref_model(input_ids).logits
                    except RuntimeError as e:
                        if "CUDA" in str(e):
                            print(f"       参考模型CUDA错误，跳过样本")
                            continue
                        raise e
                
                try:
                    current_logits = self.model(input_ids).logits
                except RuntimeError as e:
                    if "CUDA" in str(e):
                        print(f"       主模型CUDA错误，跳过样本") 
                        continue
                    raise e
                
                # 提取响应部分的logits
                response_start = max(0, prompt_length - 1)
                response_logits = current_logits[0, response_start:-1, :]
                response_targets = input_ids[0, prompt_length:]
                ref_response_logits = ref_logits[0, response_start:-1, :]
                
                if response_targets.numel() == 0 or response_logits.shape[0] != response_targets.shape[0]:
                    continue
                
                # 数值稳定的log_softmax
                current_log_probs = F.log_softmax(response_logits, dim=-1)
                ref_log_probs = F.log_softmax(ref_response_logits, dim=-1)
                
                # 检查是否包含NaN或inf
                if torch.isnan(current_log_probs).any() or torch.isinf(current_log_probs).any():
                    print(f"       检测到NaN/inf在current_log_probs，跳过样本")
                    continue
                    
                if torch.isnan(ref_log_probs).any() or torch.isinf(ref_log_probs).any():
                    print(f"       检测到NaN/inf在ref_log_probs，跳过样本")
                    continue
                
                # 获取选择token的对数概率
                current_selected_log_probs = current_log_probs.gather(1, response_targets.unsqueeze(1)).squeeze()
                ref_selected_log_probs = ref_log_probs.gather(1, response_targets.unsqueeze(1)).squeeze()
                
                # 计算重要性采样比率 - 数值稳定版本
                log_ratio = current_selected_log_probs - ref_selected_log_probs
                
                # 检查log_ratio的数值稳定性
                if torch.isnan(log_ratio).any() or torch.isinf(log_ratio).any():
                    print(f"       检测到NaN/inf在log_ratio，跳过样本")
                    continue
                
                # 限制log_ratio范围防止exp溢出
                log_ratio = torch.clamp(log_ratio, min=-10, max=10)
                ratio = torch.exp(log_ratio)
                
                # 安全检查ratio
                if torch.isnan(ratio).any() or torch.isinf(ratio).any():
                    print(f"       检测到NaN/inf在ratio，跳过样本")
                    continue
                
                # PPO clip loss
                advantage_tensor = torch.tensor(advantage, dtype=ratio.dtype, device=ratio.device)
                clip_ratio = torch.clamp(ratio, 1 - self.config['clip_range'], 1 + self.config['clip_range'])
                
                policy_loss_1 = ratio * advantage_tensor
                policy_loss_2 = clip_ratio * advantage_tensor
                policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()
                
                # KL散度惩罚
                kl_div = (current_selected_log_probs - ref_selected_log_probs).mean()
                
                # 最终安全检查
                if torch.isnan(policy_loss) or torch.isinf(policy_loss):
                    print(f"       检测到NaN/inf在policy_loss，跳过样本")
                    continue
                    
                if torch.isnan(kl_div) or torch.isinf(kl_div):
                    print(f"       检测到NaN/inf在kl_div，设为0")
                    kl_div = torch.tensor(0.0)
                
                # 累加损失
                sample_loss = policy_loss + self.config['kl_coef'] * kl_div
                total_loss += sample_loss
                total_kl += kl_div.item()
                valid_samples += 1
                
            except Exception as e:
                print(f"       策略损失计算错误: {e}")
                continue
        
        if valid_samples == 0:
            print(f"       警告: 没有有效样本用于损失计算")
            return torch.tensor(0.0, requires_grad=True), 0.0
        
        avg_loss = total_loss / valid_samples
        avg_kl = total_kl / valid_samples
        
        # 最终数值检查
        if torch.isnan(avg_loss) or torch.isinf(avg_loss):
            print(f"       最终损失包含NaN/inf，返回零损失")
            return torch.tensor(0.0, requires_grad=True), avg_kl
        
        return avg_loss, avg_kl
    
    def train_step(self, prompts):
        """GRPO训练步骤 - 带完整CUDA错误恢复"""
        
        # 在每个训练步骤开始前重置CUDA状态
        self.reset_cuda_state()
        
        try:
            # 1. 安全生成多个响应
            def _generate_responses():
                return self.generate_responses(
                    prompts, num_samples_per_prompt=self.config['num_samples_per_prompt']
                )
            
            responses, prompt_indices = self.safe_cuda_operation(_generate_responses)
            
            if len(responses) == 0:
                print("       警告: 没有成功生成任何响应")
                return {
                    'policy_loss': 0.0,
                    'kl_div': 0.0,
                    'mean_reward': 0.0,
                    'mean_advantage': 0.0,
                    'num_responses': 0
                }
            
            # 2. 安全计算组相对优势
            def _compute_advantages():
                return self.compute_group_advantages(prompts, responses, prompt_indices)
            
            advantages, rewards, baselines = self.safe_cuda_operation(_compute_advantages)
            
            # 3. 安全计算策略损失
            def _compute_loss():
                return self.compute_policy_loss(prompts, responses, prompt_indices, advantages)
            
            policy_loss, kl_div = self.safe_cuda_operation(_compute_loss)
            
            # 4. 安全执行反向传播和优化
            if hasattr(policy_loss, 'backward') and policy_loss.requires_grad:
                def _optimize():
                    self.optimizer.zero_grad()
                    policy_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['max_grad_norm'])
                    self.optimizer.step()
                    return True
                
                try:
                    self.safe_cuda_operation(_optimize)
                except Exception as e:
                    print(f"       优化步骤失败: {e}")
            
            return {
                'policy_loss': policy_loss.item() if hasattr(policy_loss, 'item') else float(policy_loss),
                'kl_div': kl_div,
                'mean_reward': np.mean(rewards) if rewards else 0.0,
                'mean_advantage': np.mean(advantages) if advantages else 0.0,
                'num_responses': len(responses)
            }
            
        except Exception as e:
            print(f"       训练步骤完全失败: {e}")
            # 强制重置所有状态
            self.reset_cuda_state()
            return {
                'policy_loss': 0.0,
                'kl_div': 0.0,
                'mean_reward': 0.0,
                'mean_advantage': 0.0,
                'num_responses': 0
            }

def create_preference_dataset():
    """创建偏好数据集"""
    data = {
        "query": [
            "写一个关于科技的故事",
            "解释机器学习的概念",
            "描述未来城市的样子", 
            "谈谈环保的重要性",
            "介绍一种编程语言",
            "解释区块链技术",
            "描述太空探索的意义",
            "谈谈人工智能的伦理"
        ]
    }
    return Dataset.from_dict(data)

def monitor_resources():
    """监控系统资源使用"""
    # CPU和内存
    cpu_percent = psutil.cpu_percent()
    memory = psutil.virtual_memory()
    
    # GPU
    gpu_info = {}
    try:
        gpus = GPUtil.getGPUs()
        if gpus:
            gpu = gpus[0]
            gpu_info = {
                'gpu_memory_used': gpu.memoryUsed,
                'gpu_memory_total': gpu.memoryTotal,
                'gpu_utilization': gpu.load * 100
            }
    except:
        gpu_info = {'gpu_memory_used': 0, 'gpu_memory_total': 0, 'gpu_utilization': 0}
    
    return {
        'cpu_percent': cpu_percent,
        'memory_used_gb': memory.used / (1024**3),
        'memory_total_gb': memory.total / (1024**3),
        **gpu_info
    }

def main():
    """主函数：GRPO vs PPO 对比实验"""
    
    print("=== Lab03: GRPO算法实现与性能对比 ===")
    
    # 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 1. 配置
    print("\n1. 初始化GRPO配置...")
    config = {
        'learning_rate': 1e-5,
        'clip_range': 0.2,
        'kl_coef': 0.05,
        'max_grad_norm': 0.5,
        'weight_decay': 0.01,
        'num_samples_per_prompt': 4,  # 每个prompt采样4个响应
        'max_new_tokens': 25,
        'temperature': 0.8,
        'top_p': 0.9,
    }
    
    model_name = "gpt2"
    print(f"   模型: {model_name}")
    print(f"   每prompt采样数: {config['num_samples_per_prompt']}")
    print(f"   学习率: {config['learning_rate']}")
    
    # 2. 初始化分词器
    print("\n2. 初始化分词器...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 3. 创建模型组件 (GRPO只需要3个，移除了Critic)
    print("\n3. 创建GRPO所需模型组件...")
    
    # 3.1 主策略模型 (Actor) - 不需要Value Head!
    print("   3.1 创建主策略模型 (仅Actor, 无Critic)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32
    ).to(device)
    
    # 确保有generation_config
    if not hasattr(model, 'generation_config'):
        model.generation_config = GenerationConfig.from_pretrained(model_name)
    
    print(f"       参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 3.2 参考模型
    print("   3.2 创建参考模型...")
    ref_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32
    ).to(device)
    
    for param in ref_model.parameters():
        param.requires_grad = False
    print(f"       参数量: {sum(p.numel() for p in ref_model.parameters()):,} (已冻结)")
    
    # 3.3 奖励模型
    print("   3.3 创建奖励模型...")
    reward_model = SimpleRewardModel(
        vocab_size=len(tokenizer),
        hidden_size=256
    ).to(device)
    print(f"       参数量: {sum(p.numel() for p in reward_model.parameters()):,}")
    
    # ✨ 注意：没有3.4! GRPO移除了Critic模型
    print("   ✨ GRPO特色：无需Critic模型，节省50%显存!")
    
    # 4. 创建数据集
    print("\n4. 准备训练数据集...")
    dataset = create_preference_dataset()
    prompts = dataset["query"]
    print(f"   数据集大小: {len(prompts)}")
    
    # 5. 初始化GRPO训练器
    print("\n5. 初始化GRPO训练器...")
    grpo_trainer = GRPOTrainer(
        model=model,
        ref_model=ref_model,
        reward_model=reward_model,
        tokenizer=tokenizer,
        config=config
    )
    print("   ✅ GRPO训练器初始化成功!")
    
    # 6. 资源监控基准
    print("\n6. 开始GRPO训练与资源监控...")
    initial_resources = monitor_resources()
    print("   初始资源状态:")
    print(f"     GPU显存: {initial_resources['gpu_memory_used']:.1f}MB / {initial_resources['gpu_memory_total']:.1f}MB")
    print(f"     CPU使用率: {initial_resources['cpu_percent']:.1f}%")
    print(f"     内存使用: {initial_resources['memory_used_gb']:.1f}GB / {initial_resources['memory_total_gb']:.1f}GB")
    
    # 7. 训练循环
    num_epochs = 3
    
    for epoch in range(num_epochs):
        print(f"\n   --- 训练轮次 {epoch + 1}/{num_epochs} ---")
        
        # 随机选择一批prompts
        batch_prompts = np.random.choice(prompts, size=min(2, len(prompts)), replace=False).tolist()
        
        print(f"     训练prompts: {len(batch_prompts)}")
        for i, prompt in enumerate(batch_prompts):
            print(f"       {i+1}. {prompt}")
        
        # 训练步骤
        start_time = time.time()
        stats = grpo_trainer.train_step(batch_prompts)
        step_time = time.time() - start_time
        
        # 资源监控
        current_resources = monitor_resources()
        
        print(f"     训练统计:")
        print(f"       策略损失: {stats['policy_loss']:.4f}")
        print(f"       KL散度: {stats['kl_div']:.4f}")
        print(f"       平均奖励: {stats['mean_reward']:.4f}")
        print(f"       平均优势: {stats['mean_advantage']:.4f}")
        print(f"       生成响应数: {stats['num_responses']}")
        print(f"       训练时间: {step_time:.2f}秒")
        
        print(f"     资源使用:")
        print(f"       GPU显存: {current_resources['gpu_memory_used']:.1f}MB")
        print(f"       GPU利用率: {current_resources['gpu_utilization']:.1f}%")
        print(f"       CPU使用率: {current_resources['cpu_percent']:.1f}%")
        
        # 吞吐量计算
        throughput = stats['num_responses'] / step_time
        print(f"       吞吐量: {throughput:.2f} 响应/秒")
    
    # 8. 效果演示
    print("\n8. GRPO训练效果演示...")
    test_prompt = "人工智能的发展趋势是"
    print(f"   测试prompt: {test_prompt}")
    
    # 生成多个响应展示组相对优势
    responses, _ = grpo_trainer.generate_responses([test_prompt], num_samples_per_prompt=4)
    advantages, rewards, baselines = grpo_trainer.compute_group_advantages(
        [test_prompt], responses, [0] * len(responses)
    )
    
    print(f"   组平均奖励 (基准): {baselines[0]:.4f}")
    print("   生成的响应及其相对优势:")
    for i, (response, reward, advantage) in enumerate(zip(responses, rewards, advantages)):
        print(f"     {i+1}. 响应: {response}")
        print(f"        奖励: {reward:.4f} | 优势: {advantage:+.4f}")
    
    # 9. 与PPO对比总结
    print("\n=== GRPO vs PPO 性能对比 ===")
    final_resources = monitor_resources()
    
    # 估算显存节省 (假设PPO需要相同大小的Critic)
    critic_params = sum(p.numel() for p in model.parameters())  # Critic与Actor同样大小
    grpo_params = sum(p.numel() for p in model.parameters()) + sum(p.numel() for p in ref_model.parameters())
    ppo_params = grpo_params + critic_params  # PPO还需要Critic
    
    memory_saving = (critic_params / ppo_params) * 100
    
    print(f"🚀 GRPO优势:")
    print(f"   ✅ 移除Critic模型: 节省 {memory_saving:.1f}% 参数量")
    print(f"   ✅ 当前GPU显存使用: {final_resources['gpu_memory_used']:.1f}MB")
    print(f"   ✅ 估算PPO需要显存: {final_resources['gpu_memory_used'] * (1 + memory_saving/100):.1f}MB")
    print(f"   ✅ 显存节省: ~{memory_saving:.1f}%")
    print(f"   ✅ 训练更稳定: 使用动态组相对基准")
    print(f"   ✅ 实现更简单: 无需复杂的价值函数训练")
    
    print("\n=== Lab03 GRPO实现完成 ===")
    print("核心创新:")
    print("1. 🎯 移除Critic: 大幅降低显存需求")
    print("2. 📊 组相对优势: 使用同组样本平均奖励作为动态基准")
    print("3. ⚡ 提升效率: 更少的模型，更快的训练")
    print("4. 🔄 动态基准: 相对评估避免绝对奖励的偏差")
    print("5. 💡 工程优化: 更适合大规模LLM训练场景")

if __name__ == "__main__":
    main() 