# Operator Ex3 - 综合实战练习系列

## 概述

Ex3 练习系列是 Kubernetes Operator 学习路径的第三阶段，旨在将 ex1（微型框架构建）和 ex2（真实代码库理解）的知识融合到实战场景中。

在这个系列中，你将构建一个接近生产级别的 Kubernetes Operator，管理一个名为 `WebApp` 的自定义资源。该 Operator 会自动部署和管理一个简单的 Web 应用（包括 Deployment 和 Service）。

## 项目结构

```
operator/ex3/
├── README.md              # 本文件（总体说明）
├── exercises/             # 练习说明文档
│   ├── 1.md              # 练习 1: 状态机与基础协调循环
│   ├── 2.md              # 练习 2: 资源创建与管理
│   ├── 3.md              # 练习 3: 更新与同步逻辑
│   ├── 4.md              # 练习 4: 删除与 Finalizer
│   └── 5.md              # 练习 5: 错误处理与可观测性
├── framework/             # 基础框架代码（学生使用）
│   ├── types.go          # 核心数据结构定义
│   ├── client.go         # MockK8sClient 实现
│   ├── errors.go         # 错误类型定义
│   ├── reconcile.go      # 协调循环框架（待实现）
│   ├── main.go           # 示例程序
│   ├── framework_test.go # 框架测试
│   └── go.mod            # Go 模块定义
└── solutions/             # 参考答案
    ├── ex1/              # 练习 1 参考答案
    │   ├── reconcile.go  # 完整实现
    │   ├── main.go       # 测试程序
    │   └── ex1_test.go   # 单元测试
    ├── ex2/              # 练习 2 参考答案（待实现）
    ├── ex3/              # 练习 3 参考答案（待实现）
    ├── ex4/              # 练习 4 参考答案（待实现）
    └── ex5/              # 练习 5 参考答案（待实现）
```

## 核心组件

### 1. 数据结构 (types.go)

- **WebApp**: 自定义资源，包含 Spec（期望状态）和 Status（实际状态）
- **Deployment**: 简化的 Kubernetes Deployment 模型
- **Service**: 简化的 Kubernetes Service 模型
- **Event**: Kubernetes 事件

### 2. 模拟客户端 (client.go)

`MockK8sClient` 提供了一个内存存储的 Kubernetes 客户端模拟，支持：

- WebApp 的 CRUD 操作
- Deployment 的 CRUD 操作
- Service 的 CRUD 操作
- Event 记录

### 3. 协调循环 (reconcile.go)

- **ReconcileContext**: 协调上下文，携带所有必要信息
- **ReconcileResult**: 协调结果，包含重试信息
- **Reconcile**: 主协调函数（在各个练习中逐步实现）

### 4. 错误处理 (errors.go)

- **RetryableError**: 可重试的临时性错误
- **PermanentError**: 永久性错误

## 学习路径

### 练习 1: 状态机与基础协调循环

**目标**: 实现状态机引擎和基础的 Reconcile 函数

**关键概念**:
- 状态机模式
- 状态转换
- Finalizer 基础

### 练习 2: 资源创建与管理

**目标**: 实现 Creating 状态的完整流水线，创建和管理 Deployment 和 Service

**关键概念**:
- OwnerReference
- 资源的幂等性创建
- 资源就绪性检查

### 练习 3: 更新与同步逻辑

**目标**: 实现 Running 状态的协调逻辑，处理 Spec 变更

**关键概念**:
- Generation vs ObservedGeneration
- 资源的深度比较
- 配置漂移处理

### 练习 4: 删除与 Finalizer

**目标**: 实现完整的资源删除流程

**关键概念**:
- Finalizer 机制
- 优雅删除
- 资源清理顺序

### 练习 5: 错误处理与可观测性

**目标**: 增强系统的健壮性和可观测性

**关键概念**:
- 错误分类
- 指数退避重试
- Event 记录
- Status Conditions
- Trace 装饰器

## 快速开始

### 方式 1: 使用框架代码（推荐学习）

如果你想自己动手实现，从框架代码开始：

```bash
cd operator/ex3/framework
go run .
```

这将运行一个基础的示例程序，展示框架的基本结构。

### 方式 2: 查看参考答案

如果你想查看完整实现或验证你的答案：

```bash
# 运行练习 1 的参考答案
cd operator/ex3/solutions/ex1
go run .

# 运行练习 1 的测试
go test -v
```

### 开始练习

按顺序完成 5 个练习：

1. 阅读 `exercises/1.md` 开始第一个练习
2. 在 `framework/` 目录中实现要求的功能
3. 运行测试验证你的实现
4. 如果遇到困难，可以参考 `solutions/ex1/` 中的答案
5. 继续下一个练习

## 状态机设计

```
Pending ──────────────────────────────────────────┐
   │                                               │
   │ (开始创建)                                     │ (创建失败)
   ▼                                               │
Creating ──────────────────────────────────────────┤
   │                                               │
   │ (所有资源就绪)                                 │
   ▼                                               ▼
Running ────────────────────────────────────────▶ Failed
   │                                               │
   │ (检测到删除标记)                               │ (重试)
   ▼                                               │
Deleting ──────────────────────────────────────────┘
   │
   │ (清理完成)
   ▼
 (移除 Finalizer，资源被删除)
```

## 与前两个阶段的关系

### 复用 ex1 的概念

- Step 接口和 Pipeline 执行器
- Branch 和 When 条件分支
- 装饰器模式
- 闭包的使用

### 复用 ex2 的概念

- ReconcileContext 和 Flow
- TracedStep 可观测性
- 声明式编排
- Executor 模式

### 新增的生产级概念

- 状态机驱动
- Finalizer 机制
- Status 子资源管理
- Event 记录
- 错误分类与重试策略
- Generation 跟踪

## 最佳实践

1. **幂等性**: 所有操作都应该是幂等的，多次执行产生相同结果
2. **错误处理**: 区分临时性错误和永久性错误，合理使用重试
3. **可观测性**: 记录详细的日志和事件，便于调试
4. **状态管理**: 使用状态机清晰表达资源的生命周期
5. **资源清理**: 使用 Finalizer 确保资源被正确清理

## 参考资源

- [Kubernetes Operator 最佳实践](https://kubernetes.io/docs/concepts/extend-kubernetes/operator/)
- [Controller Runtime](https://github.com/kubernetes-sigs/controller-runtime)
- [Kubebuilder Book](https://book.kubebuilder.io/)

## 下一步

完成所有练习后，你将具备以下能力：

- 理解 Kubernetes Operator 的核心概念
- 能够设计和实现状态机驱动的协调循环
- 掌握资源生命周期管理
- 实现健壮的错误处理和重试机制
- 构建具有良好可观测性的系统

准备好了吗？从 `1.md` 开始你的实战之旅！
