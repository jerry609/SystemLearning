# 框架代码

这是 Operator Ex1 的基础框架代码，你将在这里完成所有练习的实现。

## 文件说明

### 核心文件（需要创建）

- **`main.go`** - 主程序文件
  - 🔨 在这里实现所有练习的代码
  - 🔨 定义 Step、Pipeline 等类型
  - 🔨 实现各种函数和装饰器

## 开始练习

### 练习 1: Step 接口与闭包

1. 创建 `main.go` 文件
2. 定义 Step 类型：
   ```go
   type Step func() error
   ```

3. 实现创建 Step 的函数
4. 测试 Step 的执行

### 练习 2: Pipeline 执行器

1. 在 `main.go` 中添加：
   ```go
   type Pipeline []Step
   ```

2. 实现 Execute 函数
3. 测试 Pipeline 的执行

### 练习 3: 条件分支 - Branch 与 When

1. 实现 Branch 函数
2. 实现 When 函数
3. 测试条件逻辑

### 练习 4: 装饰器模式 - Trace 与 Log

1. 实现 Trace 装饰器
2. 添加日志记录
3. 测试装饰器功能

### 练习 5: 综合实战

1. 组合所有概念
2. 实现完整的协调流程
3. 添加完整的可观测性

## 运行和测试

### 运行程序

```bash
go run main.go
```

### 验证实现

检查输出是否符合预期：
- Step 是否正确执行
- Pipeline 是否按顺序执行
- 条件逻辑是否正确
- 日志是否清晰易读

## 开发提示

### 1. Step 的基本结构

```go
package main

import (
    "fmt"
    "errors"
)

// 定义 Step 类型
type Step func() error

// 创建一个简单的 Step
func createStep(name string) Step {
    return func() error {
        fmt.Printf("Executing %s\n", name)
        return nil
    }
}

// 创建一个会失败的 Step
func createFailingStep(name string) Step {
    return func() error {
        fmt.Printf("Executing %s\n", name)
        return errors.New("step failed")
    }
}
```

### 2. Pipeline 的实现

```go
// 定义 Pipeline 类型
type Pipeline []Step

// 执行 Pipeline
func Execute(pipeline Pipeline) error {
    for i, step := range pipeline {
        fmt.Printf("Executing step %d\n", i+1)
        if err := step(); err != nil {
            return fmt.Errorf("step %d failed: %w", i+1, err)
        }
    }
    return nil
}
```

### 3. 使用闭包捕获变量

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

### 4. 装饰器模式

```go
func Trace(step Step, name string) Step {
    return func() error {
        fmt.Printf("[START] %s\n", name)
        err := step()
        if err != nil {
            fmt.Printf("[ERROR] %s: %v\n", name, err)
        } else {
            fmt.Printf("[END] %s\n", name)
        }
        return err
    }
}
```

### 5. 条件执行

```go
func When(condition func() bool, step Step) Step {
    return func() error {
        if condition() {
            return step()
        }
        fmt.Println("Condition not met, skipping step")
        return nil
    }
}
```

## 常见问题

### Q: 我应该如何开始？

A: 先阅读 `../exercises/1.md`，理解要实现什么，然后创建 `main.go` 开始编码。

### Q: Step 为什么使用函数类型？

A: 使用函数类型可以利用闭包捕获变量，实现灵活的 Step 创建。

### Q: 如何测试我的实现？

A: 在 main 函数中创建测试场景，运行程序查看输出。

### Q: 可以使用第三方库吗？

A: 建议只使用标准库，这样可以更好地理解核心概念。

## 代码组织建议

### 单文件组织

对于 Ex1，建议使用单文件组织：

```go
package main

import (
    "fmt"
    "errors"
    "time"
)

// ============================================================================
// 练习 1: Step 接口与闭包
// ============================================================================

type Step func() error

func createStep(name string) Step {
    // ...
}

// ============================================================================
// 练习 2: Pipeline 执行器
// ============================================================================

type Pipeline []Step

func Execute(pipeline Pipeline) error {
    // ...
}

// ============================================================================
// 练习 3: 条件分支
// ============================================================================

func Branch(condition func() bool, trueStep, falseStep Step) Step {
    // ...
}

func When(condition func() bool, step Step) Step {
    // ...
}

// ============================================================================
// 练习 4: 装饰器模式
// ============================================================================

func Trace(step Step, name string) Step {
    // ...
}

// ============================================================================
// 练习 5: 综合实战
// ============================================================================

func main() {
    // 测试代码
}
```

## 参考资源

### Go 语言特性
- [Go 函数](https://go.dev/tour/moretypes/24)
- [Go 闭包](https://go.dev/tour/moretypes/25)
- [Go 错误处理](https://go.dev/blog/error-handling-and-go)

### 设计模式
- [装饰器模式](https://refactoring.guru/design-patterns/decorator)
- [策略模式](https://refactoring.guru/design-patterns/strategy)

## 下一步

1. 阅读 `../exercises/1.md`
2. 创建 `main.go`
3. 开始实现第一个练习
4. 完成后查看 `../solutions/ex1/` 对比

祝你编码愉快！🚀
