# Operator Ex3 - 文档索引

欢迎来到 Kubernetes Operator Ex3 综合实战练习系列！

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

- [练习 1: 状态机与基础协调循环](exercises/1.md) ✅
- [练习 2: 资源创建与管理](exercises/2.md) 🚧
- [练习 3: 更新与同步逻辑](exercises/3.md) 🚧
- [练习 4: 删除与 Finalizer](exercises/4.md) 🚧
- [练习 5: 错误处理与可观测性](exercises/5.md) 🚧

### 代码资源

🔧 **[框架代码](framework/README.md)**
- 在这里编写你的实现
- 主要文件：`framework/reconcile.go`

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
go run .
go test -v
```

### 运行参考答案

```bash
cd solutions/ex1/
go run .
go test -v
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
- [ ] 完成练习 4
- [ ] 完成练习 5

## 💡 学习建议

### 对于初学者

1. 不要跳过文档，先理解再动手
2. 从练习 1 开始，按顺序完成
3. 遇到困难先思考，再查看答案
4. 理解原理比完成练习更重要

### 对于有经验的开发者

1. 可以快速浏览文档
2. 直接开始编码
3. 用参考答案验证思路
4. 思考更好的实现方式

## 🔗 相关资源

### 外部文档

- [Kubernetes Operator 最佳实践](https://kubernetes.io/docs/concepts/extend-kubernetes/operator/)
- [Controller Runtime](https://github.com/kubernetes-sigs/controller-runtime)
- [Kubebuilder Book](https://book.kubebuilder.io/)

### 项目内文档

- [状态机设计](README.md#状态机设计)
- [核心组件](README.md#核心组件)
- [最佳实践](README.md#最佳实践)

## ❓ 常见问题

### 我应该从哪里开始？

从 [GETTING_STARTED.md](GETTING_STARTED.md) 开始。

### 我应该在哪里写代码？

在 `framework/` 目录，主要修改 `reconcile.go`。

### 什么时候看参考答案？

完成实现后，或遇到困难时。位于 `solutions/ex1/`。

### 如何验证我的实现？

运行 `go run .` 或 `go test -v`。

### 练习之间有依赖吗？

建议按顺序完成，但代码是独立的。

## 📞 获取帮助

如果遇到问题：

1. 查看相关文档
2. 运行测试查看输出
3. 对比参考答案
4. 回顾背景知识

## 🎓 完成后

完成所有练习后，你将掌握：

- ✅ Kubernetes Operator 核心概念
- ✅ 状态机驱动的协调循环
- ✅ 资源生命周期管理
- ✅ 错误处理和重试机制
- ✅ 可观测性最佳实践

## 开始学习

准备好了吗？从 [快速开始指南](GETTING_STARTED.md) 开始你的学习之旅！

祝你学习愉快！🚀
