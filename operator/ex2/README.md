# Operator Ex2 - 真实代码库理解

## 概述

Ex2 练习系列专注于阅读和理解真实的 Kubernetes Operator 代码库。通过分析 PolarDBX Operator 和 Crossplane 的代码，你将学习生产级 Operator 的设计模式和最佳实践。

在这个系列中，你将深入理解：
- 声明式协调框架的设计
- Step、Context、Flow 等核心抽象
- 真实代码库的组织结构
- 生产级代码的编写方式

## 项目结构

```
operator/ex2/
├── README.md              # 本文件（总体说明）
├── GETTING_STARTED.md     # 快速开始指南
├── exercises/             # 练习说明文档
│   ├── README.md          # 练习总览
│   ├── 1.md               # 练习 1: 核心三剑客 - Step, Context, Flow
│   ├── 2.md               # 练习 2: 声明式编排 - Executor 与 Pipeline
│   └── 3.md               # 练习 3: 可观测性 - TracedStep 与日志
├── framework/             # 基础框架代码（学生工作区）
│   ├── README.md          # 框架使用说明
│   └── main.go            # 待实现的框架
└── solutions/             # 参考答案
    ├── README.md          # 参考答案说明
    ├── ex1/               # 练习 1 参考答案
    ├── ex2/               # 练习 2 参考答案
    └── ex3/               # 练习 3 参考答案
```

## 核心概念

### 声明式协调框架

PolarDBX Operator 使用声明式的方式来组织协调逻辑：

```go
// Step: 最小的协调单元
type Step interface {
    Execute(ctx ReconcileContext) Flow
}

// ReconcileContext: 协调上下文
type ReconcileContext interface {
    GetClient() client.Client
    GetObject() client.Object
    // ...
}

// Flow: 控制流程
type Flow interface {
    Continue(msg string) Flow
    Abort(msg string) Flow
    Retry(msg string) Flow
    // ...
}
```

### 核心抽象

1. **Step**: 代表一个原子化的、可重入的协调步骤
2. **ReconcileContext**: 协调过程中的上下文，用于传递信息
3. **Flow**: 控制 Step 的执行流程（继续、中断、重试）
4. **Executor**: 执行 Step 的引擎
5. **Pipeline**: Step 的组合和编排

## 学习路径

### 练习 1: 核心三剑客 - Step, Context, Flow

**目标**: 理解核心抽象的设计和交互

**内容**:
- Step 接口的设计
- ReconcileContext 的作用
- Flow 的控制机制
- 实现一个简单的 Step

**预计时间**: 1-2 小时

---

### 练习 2: 声明式编排 - Executor 与 Pipeline

**目标**: 理解如何组合和执行多个 Step

**内容**:
- Executor 的实现
- Pipeline 的构建
- Step 的组合模式
- 错误处理和流程控制

**预计时间**: 2-3 小时

---

### 练习 3: 可观测性 - TracedStep 与日志

**目标**: 理解生产级代码的可观测性实践

**内容**:
- TracedStep 装饰器模式
- 日志记录最佳实践
- 性能追踪
- 调试技巧

**预计时间**: 1-2 小时

## 快速开始

### 方式 1: 使用框架代码（推荐学习）

如果你想自己动手实现：

```bash
cd operator/ex2/framework
cat ../exercises/1.md
vim main.go
go run main.go
```

### 方式 2: 查看参考答案

如果你想查看完整实现：

```bash
cd operator/ex2/solutions/ex1
go run main.go
```

### 开始练习

按顺序完成 3 个练习：

1. 阅读 `exercises/1.md` 开始第一个练习
2. 在 `framework/` 目录中实现要求的功能
3. 运行测试验证你的实现
4. 如果遇到困难，可以参考 `solutions/ex1/` 中的答案
5. 继续下一个练习

## 与 Ex1 和 Ex3 的关系

### Ex1: 微型框架构建
- 从零开始构建 Step 接口
- 学习基础的 Go 模式
- 适合初学者

### Ex2: 真实代码库理解（本系列）
- 阅读和理解生产级代码
- 学习代码组织和设计模式
- 适合有一定基础的学习者

### Ex3: 综合实战
- 构建完整的 Operator
- 应用所学知识
- 适合进阶学习者

## 学习建议

### 对于初学者

1. 先完成 Ex1，理解基础概念
2. 再学习 Ex2，看真实代码如何实现
3. 最后完成 Ex3，综合应用

### 对于有经验的开发者

1. 可以直接从 Ex2 开始
2. 重点关注设计模式和最佳实践
3. 思考如何应用到自己的项目

## 参考资源

### 代码库
- [PolarDBX Operator](https://github.com/polardb/polardbx-operator)
- [Crossplane](https://github.com/crossplane/crossplane)
- [Controller Runtime](https://github.com/kubernetes-sigs/controller-runtime)

### 文档
- [Kubernetes Operator 最佳实践](https://kubernetes.io/docs/concepts/extend-kubernetes/operator/)
- [Kubebuilder Book](https://book.kubebuilder.io/)

## 下一步

完成所有练习后：

1. 理解声明式协调框架的设计
2. 掌握生产级代码的组织方式
3. 学习可观测性最佳实践
4. 继续 Ex3 进行综合实战

准备好了吗？从 [快速开始指南](GETTING_STARTED.md) 开始你的学习之旅！
