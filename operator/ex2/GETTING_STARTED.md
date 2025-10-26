# 快速开始指南

## 目录结构

```
operator/ex2/
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

# 创建 main.go 并实现要求的功能
vim main.go

# 运行测试
go run main.go
```

### 第 3 步：查看参考答案（可选）

```bash
cd solutions/ex1/

# 运行参考实现
go run main.go

# 对比你的实现和参考答案
```

## 练习列表

### 练习 1: 核心三剑客 - Step, Context, Flow

**学习目标**:
- 理解 Step、ReconcileContext、Flow 三个核心抽象
- 实现一个简单的 Step
- 理解声明式协调的基本原理

**预计时间**: 1-2 小时

**开始**:
```bash
cat exercises/1.md
```

---

### 练习 2: 声明式编排 - Executor 与 Pipeline

**学习目标**:
- 理解 Executor 的作用
- 学习如何组合多个 Step
- 掌握 Pipeline 的构建方式

**预计时间**: 2-3 小时

**开始**:
```bash
cat exercises/2.md
```

---

### 练习 3: 可观测性 - TracedStep 与日志

**学习目标**:
- 理解装饰器模式
- 学习日志记录最佳实践
- 掌握性能追踪技巧

**预计时间**: 1-2 小时

**开始**:
```bash
cat exercises/3.md
```

## 提示

### 💡 学习建议

1. **先理解再实现**: 不要急于写代码，先理解概念
2. **增量开发**: 一步一步实现，每次只关注一个功能
3. **善用日志**: 添加日志输出帮助理解执行流程
4. **参考但不抄袭**: 查看参考答案时，理解思路而不是直接复制

### 🔍 调试技巧

1. **添加打印语句**: 在关键位置打印变量值
2. **分步测试**: 先测试单个 Step，再测试组合
3. **查看日志**: 日志可以帮助理解执行顺序

### 📖 阅读代码

在实现过程中，建议阅读以下真实代码：

1. **PolarDBX Operator**:
   - `pkg/k8s/control/step.go`
   - `pkg/k8s/control/context.go`
   - `pkg/k8s/control/flow.go`

2. **Crossplane**:
   - `internal/controller/apiextensions/composite/reconciler.go`

## 常见问题

### Q: 我应该从哪里开始？

A: 从 `exercises/1.md` 开始，按顺序完成练习。

### Q: 我应该在哪里写代码？

A: 在 `framework/` 目录中创建 `main.go` 文件。

### Q: 什么时候看参考答案？

A: 建议先自己实现，遇到困难或完成后再查看 `solutions/ex1/`。

### Q: 如何验证我的实现？

A: 运行 `go run main.go`，检查输出是否符合预期。

### Q: 练习之间有依赖吗？

A: 建议按顺序完成，但每个练习的代码是独立的。

## 下一步

1. 阅读 `exercises/1.md`
2. 开始实现第一个练习
3. 完成后继续下一个练习

祝你学习愉快！🚀
