# Operator Ex1 - 微型框架构建

## 概述

Ex1 练习系列专注于从零开始构建一个微型的声明式协调框架。通过动手实现 Step 接口、Pipeline 执行器等核心组件，你将深入理解 Kubernetes Operator 的基本设计模式。

在这个系列中，你将学习：
- Step 接口的设计和实现
- Pipeline 的构建和执行
- 闭包和装饰器模式
- 条件分支和错误处理
- 可观测性和日志记录

## 项目结构

```
operator/ex1/
├── README.md              # 本文件（总体说明）
├── GETTING_STARTED.md     # 快速开始指南
├── exercises/             # 练习说明文档
│   ├── README.md          # 练习总览
│   ├── 1.md               # 练习 1: Step 接口与闭包
│   ├── 2.md               # 练习 2: Pipeline 执行器
│   ├── 3.md               # 练习 3: 条件分支 - Branch 与 When
│   ├── 4.md               # 练习 4: 装饰器模式 - Trace 与 Log
│   └── 5.md               # 练习 5: 综合实战 - 完整的协调流程
├── framework/             # 基础框架代码（学生工作区）
│   ├── README.md          # 框架使用说明
│   └── main.go            # 待实现的框架
└── solutions/             # 参考答案
    ├── README.md          # 参考答案说明
    ├── ex1/               # 练习 1 参考答案
    ├── ex2/               # 练习 2 参考答案
    ├── ex3/               # 练习 3 参考答案
    ├── ex4/               # 练习 4 参考答案
    └── ex5/               # 练习 5 参考答案
```

## 核心概念

### 声明式协调框架

我们将构建一个简化版的声明式协调框架：

```go
// Step: 最小的协调单元
type Step func() error

// Pipeline: Step 的集合
type Pipeline []Step

// 执行 Pipeline
func Execute(pipeline Pipeline) error {
    for _, step := range pipeline {
        if err := step(); err != nil {
            return err
        }
    }
    return nil
}
```

### 核心组件

1. **Step**: 代表一个可执行的步骤（使用函数实现）
2. **Pipeline**: Step 的有序集合
3. **Branch**: 条件分支，根据条件选择执行路径
4. **When**: 条件执行，满足条件才执行
5. **Trace**: 装饰器，添加日志和追踪

## 学习路径

### 练习 1: Step 接口与闭包

**目标**: 理解 Step 的基本概念和闭包的使用

**内容**:
- 定义 Step 类型
- 使用闭包创建 Step
- 实现简单的 Step 执行

**预计时间**: 30-45 分钟

---

### 练习 2: Pipeline 执行器

**目标**: 实现 Pipeline 的构建和执行

**内容**:
- 定义 Pipeline 类型
- 实现 Execute 函数
- 处理错误和中断

**预计时间**: 45-60 分钟

---

### 练习 3: 条件分支 - Branch 与 When

**目标**: 实现条件分支和条件执行

**内容**:
- 实现 Branch 函数
- 实现 When 函数
- 组合条件逻辑

**预计时间**: 45-60 分钟

---

### 练习 4: 装饰器模式 - Trace 与 Log

**目标**: 使用装饰器模式添加可观测性

**内容**:
- 实现 Trace 装饰器
- 添加日志记录
- 性能追踪

**预计时间**: 45-60 分钟

---

### 练习 5: 综合实战 - 完整的协调流程

**目标**: 构建一个完整的协调流程

**内容**:
- 组合所有学到的概念
- 实现一个真实场景
- 添加完整的可观测性

**预计时间**: 60-90 分钟

## 快速开始

### 方式 1: 使用框架代码（推荐学习）

如果你想自己动手实现：

```bash
cd operator/ex1/framework
cat ../exercises/1.md
vim main.go
go run main.go
```

### 方式 2: 查看参考答案

如果你想查看完整实现：

```bash
cd operator/ex1/solutions/ex1
go run main.go
```

### 开始练习

按顺序完成 5 个练习：

1. 阅读 `exercises/1.md` 开始第一个练习
2. 在 `framework/` 目录中实现要求的功能
3. 运行测试验证你的实现
4. 如果遇到困难，可以参考 `solutions/ex1/` 中的答案
5. 继续下一个练习

## 与 Ex2 和 Ex3 的关系

### Ex1: 微型框架构建（本系列）
- 从零开始构建
- 学习基础模式
- 适合初学者

### Ex2: 真实代码库理解
- 阅读和理解生产级代码
- 学习代码组织和设计模式
- 适合有一定基础的学习者

### Ex3: 综合实战
- 构建完整的 Operator
- 应用所学知识
- 适合进阶学习者

## 学习建议

### 对于初学者

1. 从 Ex1 开始，理解基础概念
2. 动手实现每个练习
3. 理解闭包、装饰器等 Go 模式
4. 完成后继续 Ex2 和 Ex3

### 对于有经验的开发者

1. 可以快速浏览 Ex1
2. 重点关注设计模式
3. 思考如何应用到实际项目
4. 直接进入 Ex2 或 Ex3

## 关键概念

### 1. 函数作为一等公民

在 Go 中，函数可以作为值传递：

```go
type Step func() error

func createStep(name string) Step {
    return func() error {
        fmt.Printf("Executing %s\n", name)
        return nil
    }
}
```

### 2. 闭包

闭包可以捕获外部变量：

```go
func counter() Step {
    count := 0
    return func() error {
        count++
        fmt.Printf("Count: %d\n", count)
        return nil
    }
}
```

### 3. 装饰器模式

装饰器可以增强函数功能：

```go
func Trace(step Step, name string) Step {
    return func() error {
        fmt.Printf("[START] %s\n", name)
        err := step()
        fmt.Printf("[END] %s\n", name)
        return err
    }
}
```

### 4. 高阶函数

函数可以接受和返回函数：

```go
func When(condition func() bool, step Step) Step {
    return func() error {
        if condition() {
            return step()
        }
        return nil
    }
}
```

## 参考资源

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

## 下一步

完成所有练习后：

1. 理解声明式协调的基本原理
2. 掌握 Go 的函数式编程特性
3. 学习装饰器等设计模式
4. 继续 Ex2 学习真实代码库
5. 最后完成 Ex3 综合实战

准备好了吗？从 [快速开始指南](GETTING_STARTED.md) 开始你的学习之旅！
