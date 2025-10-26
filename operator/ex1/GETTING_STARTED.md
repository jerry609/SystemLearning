# 快速开始指南

## 目录结构

```
operator/ex1/
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

### 练习 1: Step 接口与闭包

**学习目标**:
- 理解 Step 的基本概念
- 学习 Go 闭包的使用
- 实现简单的 Step 执行

**预计时间**: 30-45 分钟

**开始**:
```bash
cat exercises/1.md
```

---

### 练习 2: Pipeline 执行器

**学习目标**:
- 实现 Pipeline 的构建
- 学习如何组合多个 Step
- 处理错误和中断

**预计时间**: 45-60 分钟

**开始**:
```bash
cat exercises/2.md
```

---

### 练习 3: 条件分支 - Branch 与 When

**学习目标**:
- 实现条件分支逻辑
- 学习 Branch 和 When 的使用
- 组合条件逻辑

**预计时间**: 45-60 分钟

**开始**:
```bash
cat exercises/3.md
```

---

### 练习 4: 装饰器模式 - Trace 与 Log

**学习目标**:
- 理解装饰器模式
- 实现 Trace 装饰器
- 添加日志和性能追踪

**预计时间**: 45-60 分钟

**开始**:
```bash
cat exercises/4.md
```

---

### 练习 5: 综合实战 - 完整的协调流程

**学习目标**:
- 组合所有学到的概念
- 实现一个完整的协调流程
- 添加完整的可观测性

**预计时间**: 60-90 分钟

**开始**:
```bash
cat exercises/5.md
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

### 📖 Go 语言特性

在实现过程中，你将学习：

1. **函数作为一等公民**: 函数可以作为值传递
2. **闭包**: 函数可以捕获外部变量
3. **高阶函数**: 函数可以接受和返回函数
4. **装饰器模式**: 增强函数功能

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

A: 建议按顺序完成，后面的练习会用到前面的概念。

### Q: 需要什么前置知识？

A: 基本的 Go 语言知识即可，会在练习中学习高级特性。

## 示例代码

### Step 的基本使用

```go
package main

import "fmt"

// 定义 Step 类型
type Step func() error

// 创建一个简单的 Step
func createStep(name string) Step {
    return func() error {
        fmt.Printf("Executing %s\n", name)
        return nil
    }
}

func main() {
    // 创建并执行 Step
    step := createStep("MyStep")
    step()
}
```

### Pipeline 的基本使用

```go
// 定义 Pipeline 类型
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

func main() {
    // 创建 Pipeline
    pipeline := Pipeline{
        createStep("Step1"),
        createStep("Step2"),
        createStep("Step3"),
    }
    
    // 执行 Pipeline
    Execute(pipeline)
}
```

## 下一步

1. 阅读 `exercises/1.md`
2. 开始实现第一个练习
3. 完成后继续下一个练习

祝你学习愉快！🚀
