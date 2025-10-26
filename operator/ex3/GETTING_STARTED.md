# 快速开始指南

## 目录结构

```
operator/ex3/
├── exercises/     # 📚 练习说明文档（从这里开始）
├── framework/     # 🔧 基础框架代码（在这里编写你的实现）
└── solutions/     # ✅ 参考答案（遇到困难时查看）
```

## 学习路径

### 第 1 步：阅读练习说明

```bash
# 打开练习 1 的说明文档
cat exercises/1.md
# 或在编辑器中打开
```

### 第 2 步：在框架中实现

```bash
cd framework/

# 编辑 reconcile.go，实现要求的功能
# 主要实现：
# - Reconcile() 函数
# - handlePending() 函数
# - Finalizer 辅助函数

# 运行测试验证
go run .
```

### 第 3 步：查看参考答案（可选）

```bash
cd solutions/ex1/

# 运行参考实现
go run .

# 运行完整测试
go test -v
```

## 练习列表

1. **练习 1**: 状态机与基础协调循环 ✅ 已完成
   - 实现状态机引擎
   - Finalizer 管理
   - 状态转换逻辑

2. **练习 2**: 资源创建与管理 🚧 待实现
   - 创建 Deployment 和 Service
   - OwnerReference 管理
   - 资源就绪性检查

3. **练习 3**: 更新与同步逻辑 🚧 待实现
   - Generation 跟踪
   - 资源深度比较
   - 配置漂移处理

4. **练习 4**: 删除与 Finalizer 🚧 待实现
   - 优雅删除流程
   - 资源清理顺序
   - Finalizer 移除

5. **练习 5**: 错误处理与可观测性 🚧 待实现
   - 错误分类
   - 重试策略
   - Event 和 Condition 管理

## 提示

- 💡 每个练习都是独立的，但建议按顺序完成
- 📖 先阅读 `exercises/` 中的说明文档
- 🔨 在 `framework/` 中编写代码
- ✅ 用 `solutions/` 验证你的实现
- 🧪 运行测试确保功能正确

## 需要帮助？

查看主 README.md 了解更多背景知识和最佳实践。
