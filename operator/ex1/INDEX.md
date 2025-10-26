# Operator Ex1 - 文档索引

欢迎来到 Kubernetes Operator Ex1 微型框架构建系列！

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

- [练习 1: Step 接口与闭包](exercises/1.md) ✅
- [练习 2: Pipeline 执行器](exercises/2.md) ✅
- [练习 3: 条件分支 - Branch 与 When](exercises/3.md) ✅
- [练习 4: 装饰器模式 - Trace 与 Log](exercises/4.md) ✅
- [练习 5: 综合实战 - 完整的协调流程](exercises/5.md) ✅

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
2. exercises/1.md        (10 分钟)
   ↓
3. framework/README.md   (5 分钟)
   ↓
4. 开始编码！            (30-45 分钟)
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
| [exercises/1.md](exercises/1.md) | 练习 1 说明 | 10 分钟 |
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
- [ ] 完成练习 4
- [ ] 完成练习 5

## 💡 学习建议

### 对于初学者

1. 从 Ex1 开始，理解基础概念
2. 动手实现每个练习
3. 理解闭包、装饰器等模式
4. 完成后继续 Ex2 和 Ex3

### 对于有经验的开发者

1. 可以快速浏览 Ex1
2. 重点关注设计模式
3. 思考如何应用到项目
4. 直接进入 Ex2 或 Ex3

## 🔗 相关资源

### Go 语言特性

- [Go 函数](https://go.dev/tour/moretypes/24)
- [Go 闭包](https://go.dev/tour/moretypes/25)
- [Go 接口](https://go.dev/tour/methods/9)

### 设计模式

- [装饰器模式](https://refactoring.guru/design-patterns/decorator)
- [策略模式](https://refactoring.guru/design-patterns/strategy)

### Kubernetes Operator

- [Operator 模式](https://kubernetes.io/docs/concepts/extend-kubernetes/operator/)
- [Controller Runtime](https://github.com/kubernetes-sigs/controller-runtime)

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

建议按顺序完成，后面的练习会用到前面的概念。

## 📞 获取帮助

如果遇到问题：

1. 查看相关文档
2. 查看日志输出
3. 对比参考答案
4. 理解背景知识

## 🎓 完成后

完成所有练习后，你将掌握：

- ✅ Step 接口的设计和实现
- ✅ Pipeline 的构建和执行
- ✅ 闭包和装饰器模式
- ✅ 条件分支和错误处理
- ✅ 可观测性和日志记录

## 开始学习

准备好了吗？从 [快速开始指南](GETTING_STARTED.md) 开始你的学习之旅！

祝你学习愉快！🚀
