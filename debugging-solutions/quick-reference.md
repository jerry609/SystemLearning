# 🚀 快速查找指南

当你遇到问题时，使用这个指南快速定位相关的解决方案文档。

## 🔍 按错误类型查找

### CUDA错误
```
RuntimeError: CUDA error: device-side assert triggered
AssertionError: input[0] != 0
```
**📄 查看**: `cuda-errors/multinomial-assertion-failure.md`

### 训练问题
```
梯度范数突然变得很大 (>1000)
模型参数包含 NaN 或 inf
训练第二轮开始就失败
```
**📄 查看**: `training-issues/grpo-gradient-explosion.md`

## 🔍 按算法查找

### GRPO (Group Relative Policy Optimization)
- **CUDA错误**: `cuda-errors/multinomial-assertion-failure.md`
- **梯度爆炸**: `training-issues/grpo-gradient-explosion.md`

### PPO (Proximal Policy Optimization)
- **数值稳定性**: `training-issues/grpo-gradient-explosion.md` (技术通用)

## 🔍 按症状查找

| 症状 | 可能原因 | 查看文档 |
|------|----------|----------|
| 🔥 训练第一轮成功，第二轮崩溃 | 梯度爆炸 + CUDA错误 | CUDA-001 + TRAIN-001 |
| 💥 `torch.multinomial` 报错 | 概率分布异常 | CUDA-001 |
| 📈 梯度范数 >10000 | 梯度爆炸 | TRAIN-001 |
| 🧮 参数包含 NaN/inf | 数值溢出 | TRAIN-001 |
| 🖥️ GPU 状态损坏 | CUDA断言失败 | CUDA-001 |
| 🔄 需要重启进程才能继续 | CUDA上下文破坏 | CUDA-001 |

## 🔍 按技术栈查找

### PyTorch + CUDA
- **multinomial采样**: `cuda-errors/multinomial-assertion-failure.md`
- **梯度计算**: `training-issues/grpo-gradient-explosion.md`

### 强化学习
- **策略优化**: `training-issues/grpo-gradient-explosion.md`
- **PPO变种**: `cuda-errors/multinomial-assertion-failure.md`

### 数值计算
- **数值稳定性**: 两个文档都涉及
- **梯度裁剪**: `training-issues/grpo-gradient-explosion.md`

## 📋 常用检查清单

### 遇到训练问题时
```bash
# 1. 检查梯度范数
print(f"梯度范数: {grad_norm:.2f}")

# 2. 检查参数健康
nan_count = sum(torch.isnan(p).sum() for p in model.parameters())
inf_count = sum(torch.isinf(p).sum() for p in model.parameters())

# 3. 检查损失值
print(f"损失: {loss.item():.6f}")

# 4. 检查logits范围
print(f"logits范围: [{logits.min():.2f}, {logits.max():.2f}]")
```

### 遇到CUDA错误时
```bash
# 1. 查看完整错误信息
CUDA_LAUNCH_BLOCKING=1 python your_script.py

# 2. 检查CUDA状态
torch.cuda.empty_cache()
torch.cuda.synchronize()

# 3. 验证输入数据
print(f"输入形状: {input_tensor.shape}")
print(f"数据类型: {input_tensor.dtype}")
print(f"设备: {input_tensor.device}")
```

## 🛠️ 通用修复策略

### 数值稳定性
1. **降低学习率** (1e-5 → 5e-6)
2. **严格梯度裁剪** (0.5 → 0.3)
3. **限制中间值范围** (clamp操作)
4. **添加健康检查** (NaN/inf监控)

### CUDA相关
1. **保守生成参数** (temperature, top_p)
2. **错误恢复机制** (异常跳过)
3. **状态重置** (清理GPU缓存)
4. **数据验证** (输入合法性检查)

## 🔗 扩展阅读

- **主索引**: `README.md`
- **文档模板**: `template.md`
- **项目文档**: `../README.md`

---

💡 **提示**: 
- 使用 `Ctrl+F` 搜索关键词
- 多数问题都有明确的解决步骤
- 按照文档中的配置参数可以快速修复
- 如果问题仍未解决，可以参考相关资源链接 