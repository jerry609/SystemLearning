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
├── ex1/              # 练习 1: 核心三剑客 - Step, Context, Flow
│   └── main.go       # 完整实现
├── ex2/              # 练习 2: 声明式编排 - Executor 与 Pipeline
│   └── main.go       # 完整实现
└── ex3/              # 练习 3: 可观测性 - TracedStep 与日志
    └── main.go       # 完整实现
```

## 学习建议

1. **先自己尝试实现**：在 `framework/` 目录中编写代码
2. **遇到困难时参考**：查看这里的实现思路
3. **对比差异**：比较你的实现和参考答案的区别
4. **理解原理**：不要只是复制代码，理解为什么这样实现

## 练习 1 的关键实现点

### 1. 核心接口定义

```go
// Step 接口
type Step interface {
    Execute(ctx ReconcileContext) Flow
}

// ReconcileContext 接口
type ReconcileContext interface {
    GetResource(name string) (string, bool)
}

// Flow 接口
type Flow interface {
    Continue(msg string) Flow
    Abort(msg string) Flow
    // ...
}
```

### 2. Mock 对象实现

```go
type mockReconcileContext struct {
    resources map[string]string
}

type mockFlow struct {
    status  string
    message string
}
```

### 3. Step 实现

```go
func checkResourceExistsFunc(ctx ReconcileContext) Flow {
    // 实现逻辑
}
```

## 练习 2 的关键实现点

### 1. Executor 实现

```go
type Executor interface {
    Execute(ctx ReconcileContext, steps []Step) Flow
}
```

### 2. Pipeline 构建

```go
pipeline := []Step{
    step1,
    step2,
    step3,
}
```

### 3. 流程控制

```go
for _, step := range pipeline {
    flow := step.Execute(ctx)
    if flow.ShouldAbort() {
        break
    }
}
```

## 练习 3 的关键实现点

### 1. TracedStep 装饰器

```go
type TracedStep struct {
    step Step
    name string
}

func (t *TracedStep) Execute(ctx ReconcileContext) Flow {
    // 记录开始时间
    // 执行 step
    // 记录结束时间和结果
}
```

### 2. 日志记录

```go
func logStepStart(name string) {
    fmt.Printf("[START] %s\n", name)
}

func logStepEnd(name string, duration time.Duration) {
    fmt.Printf("[END] %s (took %v)\n", name, duration)
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

1. **设计选择**：为什么这样设计接口？
2. **实现细节**：有没有更好的实现方式？
3. **扩展性**：如何扩展这个框架？
4. **真实应用**：如何应用到真实项目？

## 与真实代码对比

参考答案是简化版本，真实的 PolarDBX Operator 代码更复杂：

| 特性 | 参考答案 | 真实代码 |
|------|---------|---------|
| 接口定义 | 简化版 | 完整版 |
| 错误处理 | 基础 | 完善 |
| 日志记录 | 简单 | 结构化 |
| 性能优化 | 无 | 有 |
| 测试覆盖 | 基础 | 完整 |

## 下一步

完成所有练习后：

1. 对比你的实现和参考答案
2. 理解设计决策
3. 阅读真实的 PolarDBX Operator 代码
4. 继续 Ex3 进行综合实战

## 反馈

如果你发现参考答案有问题或有更好的实现方式，欢迎反馈！
