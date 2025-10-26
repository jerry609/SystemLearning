# Operator Ex2 - 文档索引

欢迎来到 Kubernetes Operator Ex2 真实代码库理解系列！

## 🚀 快速导航

### 新手入门

1. **[快速开始指南](GETTING_STARTED.md)** ⭐ 推荐首先阅读
   - 了解项目结构
   - 学习路径指引
   - 快速上手步骤

2. **[目录结构说明](STRUCTURE.md)**
   - 详细的目录结构
   - 文件职责说明
   - 工作流程指导

3. **[项目总览](README.md)**
   - 项目背景介绍
   - 核心概念说明
   - 学习资源链接

### 练习文档

📚 **[练习总览](exercises/README.md)**

- [练习 1: 核心三剑客 - Step, Context, Flow](exercises/1.md) ✅
- [练习 2: 声明式编排 - Executor 与 Pipeline](exercises/2.md) ✅
- [练习 3: 可观测性 - TracedStep 与日志](exercises/3.md) ✅

### 代码资源

🔧 **[框架代码](framework/README.md)**
- 在这里编写你的实现
- 主要文件：`framework/main.go`

✅ **[参考答案](solutions/README.md)**
- 查看完整实现
- 学习最佳实践

## 📖 学习路径

### 第一次使用？按这个顺序阅读：

```
1. GETTING_STARTED.md    (5 分钟)
   ↓
2. exercises/1.md        (15 分钟)
   ↓
3. framework/README.md   (5 分钟)
   ↓
4. 开始编码！            (60-90 分钟)
   ↓
5. solutions/README.md   (参考答案)
```

### 已经熟悉项目？

- 直接查看 [练习列表](exercises/README.md)
- 或跳转到 [框架代码](framework/)

## 🎯 核心文档

### 必读文档

| 文档 | 用途 | 阅读时间 |
|------|------|---------|
| [GETTING_STARTED.md](GETTING_STARTED.md) | 快速开始 | 5 分钟 |
| [exercises/1.md](exercises/1.md) | 练习 1 说明 | 15 分钟 |
| [framework/README.md](framework/README.md) | 框架使用 | 5 分钟 |

### 参考文档

| 文档 | 用途 | 何时阅读 |
|------|------|---------|
| [README.md](README.md) | 项目总览 | 想了解背景时 |
| [STRUCTURE.md](STRUCTURE.md) | 目录结构 | 不清楚文件位置时 |
| [solutions/README.md](solutions/README.md) | 参考答案 | 完成实现后 |

## 🛠️ 快速命令

### 运行框架代码

```bash
cd framework/
go run main.go
```

### 运行参考答案

```bash
cd solutions/ex1/
go run main.go
```

### 查看文档

```bash
# 快速开始
cat GETTING_STARTED.md

# 练习 1
cat exercises/1.md

# 框架说明
cat framework/README.md
```

## 📊 学习进度追踪

- [ ] 阅读快速开始指南
- [ ] 理解项目结构
- [ ] 完成练习 1
- [ ] 完成练习 2
- [ ] 完成练习 3

## 💡 学习建议

### 对于初学者

1. 先完成 Ex1（微型框架构建）
2. 再学习 Ex2（真实代码理解）
3. 最后完成 Ex3（综合实战）

### 对于有经验的开发者

1. 可以直接从 Ex2 开始
2. 重点关注设计模式
3. 思考如何应用到项目

## 🔗 相关资源

### 外部文档

- [PolarDBX Operator](https://github.com/polardb/polardbx-operator)
- [Crossplane](https://github.com/crossplane/crossplane)
- [Controller Runtime](https://github.com/kubernetes-sigs/controller-runtime)

### 项目内文档

- [核心概念](README.md#核心概念)
- [学习路径](README.md#学习路径)

## ❓ 常见问题

### 我应该从哪里开始？

从 [GETTING_STARTED.md](GETTING_STARTED.md) 开始。

### 我应该在哪里写代码？

在 `framework/` 目录，创建 `main.go` 文件。

### 什么时候看参考答案？

完成实现后，或遇到困难时。位于 `solutions/ex1/`。

### 如何验证我的实现？

运行 `go run main.go`，检查输出是否符合预期。

### 练习之间有依赖吗？

建议按顺序完成，但代码是独立的。

## 📞 获取帮助

如果遇到问题：

1. 查看相关文档
2. 查看日志输出
3. 对比参考答案
4. 阅读真实代码

## 🎓 完成后

完成所有练习后，你将掌握：

- ✅ 声明式协调框架的设计
- ✅ Step、Context、Flow 等核心抽象
- ✅ Executor 和 Pipeline 的实现
- ✅ 装饰器模式和可观测性
- ✅ 生产级代码的组织方式

## 开始学习

准备好了吗？从 [快速开始指南](GETTING_STARTED.md) 开始你的学习之旅！

祝你学习愉快！🚀
