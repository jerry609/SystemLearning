# 参考答案

这个目录包含所有练习的参考答案实现。

## 使用说明

### 查看练习 1 的参考答案

```bash
cd ex1/

# 查看实现代码
cat main.go

# 运行参考实现
go run main.go
```

## 目录结构

```
solutions/
├── ex1/              # 练习 1: Step 接口与闭包
│   └── main.go       # 完整实现
├── ex2/              # 练习 2: Pipeline 执行器
│   └── main.go       # 完整实现
├── ex3/              # 练习 3: 条件分支 - Branch 与 When
│   └── main.go       # 完整实现
├── ex4/              # 练习 4: 装饰器模式 - Trace 与 Log
│   └── main.go       # 完整实现
└── ex5/              # 练习 5: 综合实战 - 完整的协调流程
    └── main.go       # 完整实现
```

## 学习建议

1. **先自己尝试实现**：在 `framework/` 目录中编写代码
2. **遇到困难时参考**：查看这里的实现思路
3. **对比差异**：比较你的实现和参考答案的区别
4. **理解原理**：不要只是复制代码，理解为什么这样实现

## 练习 1 的关键实现点

### 1. Step 类型定义

```go
// Step 表示一个可执行的步骤
type Step func() error
```

### 2. 使用闭包创建 Step

```go
func createStep(name string) Step {
    return func() error {
        fmt.Printf("Executing %s\n", name)
        return nil
    }
}
```

### 3. 闭包捕获变量

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

## 练习 2 的关键实现点

### 1. Pipeline 类型定义

```go
type Pipeline []Step
```

### 2. Execute 函数实现

```go
func Execute(pipeline Pipeline) error {
    for i, step := range pipeline {
        if err := step(); err != nil {
            return fmt.Errorf("step %d failed: %w", i+1, err)
        }
    }
    return nil
}
```

### 3. 错误处理

```go
if err := Execute(pipeline); err != nil {
    fmt.Printf("Pipeline failed: %v\n", err)
}
```

## 练习 3 的关键实现点

### 1. Branch 函数实现

```go
func Branch(condition func() bool, trueStep, falseStep Step) Step {
    return func() error {
        if condition() {
            return trueStep()
        }
        return falseStep()
    }
}
```

### 2. When 函数实现

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

## 练习 4 的关键实现点

### 1. Trace 装饰器实现

```go
func Trace(step Step, name string) Step {
    return func() error {
        start := time.Now()
        fmt.Printf("[START] %s\n", name)
        
        err := step()
        
        duration := time.Since(start)
        if err != nil {
            fmt.Printf("[ERROR] %s: %v (took %v)\n", name, err, duration)
        } else {
            fmt.Printf("[END] %s (took %v)\n", name, duration)
        }
        
        return err
    }
}
```

### 2. 装饰器的使用

```go
step := Trace(createStep("MyStep"), "MyStep")
step()
```

## 练习 5 的关键实现点

### 1. 组合所有概念

```go
pipeline := Pipeline{
    Trace(createStep("Init"), "Init"),
    When(func() bool { return true }, 
        Trace(createStep("Process"), "Process")),
    Branch(func() bool { return success },
        Trace(createStep("Success"), "Success"),
        Trace(createStep("Failure"), "Failure")),
}
```

### 2. 完整的协调流程

```go
func reconcile() error {
    pipeline := Pipeline{
        // 初始化
        initStep(),
        // 检查资源
        checkResourceStep(),
        // 条件执行
        When(resourceExists, updateResourceStep()),
        // 清理
        cleanupStep(),
    }
    
    return Execute(pipeline)
}
```

## 代码质量

参考答案遵循以下原则：

- ✅ 清晰的代码结构
- ✅ 详细的注释说明
- ✅ 完整的错误处理
- ✅ 良好的日志记录
- ✅ 易于理解和扩展

## 扩展思考

查看参考答案后，思考：

1. **设计选择**：为什么使用函数类型而不是接口？
2. **实现细节**：有没有更好的实现方式？
3. **扩展性**：如何扩展这个框架？
4. **真实应用**：如何应用到真实项目？

## 与真实 Operator 对比

参考答案是简化版本，真实的 Operator 代码更复杂：

| 特性 | 参考答案 | 真实 Operator |
|------|---------|--------------|
| Step 定义 | 函数类型 | 接口 |
| 错误处理 | 基础 | 完善 |
| 日志记录 | 简单 | 结构化 |
| 性能优化 | 无 | 有 |
| 测试覆盖 | 基础 | 完整 |

## 下一步

完成所有练习后：

1. 对比你的实现和参考答案
2. 理解设计决策
3. 继续 Ex2 学习真实代码库
4. 最后完成 Ex3 综合实战

## 反馈

如果你发现参考答案有问题或有更好的实现方式，欢迎反馈！
