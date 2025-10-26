# 框架代码

这是 Operator Ex2 的基础框架代码，你将在这里完成所有练习的实现。

## 文件说明

### 核心文件（需要创建）

- **`main.go`** - 主程序文件
  - 🔨 在这里实现所有练习的代码
  - 🔨 定义 Step、Context、Flow 等抽象
  - 🔨 实现 Mock 对象和测试场景

## 开始练习

### 练习 1: 核心三剑客 - Step, Context, Flow

1. 创建 `main.go` 文件
2. 定义核心接口：
   - `Step` 接口
   - `ReconcileContext` 接口
   - `Flow` 接口

3. 实现 Mock 对象：
   - `mockReconcileContext`
   - `mockFlow`

4. 编写第一个 Step：
   - `checkResourceExistsFunc`

5. 在 main 函数中测试

### 练习 2: 声明式编排 - Executor 与 Pipeline

1. 在 `main.go` 中添加：
   - `Executor` 接口
   - `Pipeline` 类型
   - Step 组合逻辑

2. 实现多个 Step 的串联执行

3. 测试 Pipeline 的执行流程

### 练习 3: 可观测性 - TracedStep 与日志

1. 在 `main.go` 中添加：
   - `TracedStep` 装饰器
   - 日志记录功能
   - 性能追踪

2. 为现有 Step 添加追踪

3. 测试可观测性功能

## 运行和测试

### 运行程序

```bash
go run main.go
```

### 验证实现

检查输出是否符合预期：
- Step 是否正确执行
- Flow 是否正确控制流程
- 日志是否清晰易读

## 开发提示

### 1. 理解接口设计

```go
// Step: 最小的协调单元
type Step interface {
    Execute(ctx ReconcileContext) Flow
}

// ReconcileContext: 协调上下文
type ReconcileContext interface {
    // 定义你需要的方法
}

// Flow: 控制流程
type Flow interface {
    // 定义流程控制方法
}
```

### 2. 实现 Mock 对象

Mock 对象用于测试，不需要真实的 Kubernetes 客户端：

```go
type mockReconcileContext struct {
    resources map[string]string
}

func (m *mockReconcileContext) GetResource(name string) (string, bool) {
    val, ok := m.resources[name]
    return val, ok
}
```

### 3. 编写 Step

Step 应该是纯函数，易于测试：

```go
func checkResourceExistsFunc(ctx ReconcileContext) Flow {
    // 1. 从 context 获取信息
    // 2. 执行逻辑
    // 3. 返回 Flow
}
```

### 4. 组合 Step

使用 Pipeline 组合多个 Step：

```go
pipeline := []Step{
    step1,
    step2,
    step3,
}

for _, step := range pipeline {
    flow := step.Execute(ctx)
    if flow.ShouldAbort() {
        break
    }
}
```

## 常见问题

### Q: 我应该如何开始？

A: 先阅读 `../exercises/1.md`，理解要实现什么，然后创建 `main.go` 开始编码。

### Q: 接口应该定义哪些方法？

A: 参考练习说明中的要求，以及 PolarDBX Operator 的真实代码。

### Q: Mock 对象应该多复杂？

A: 尽量简单，只实现练习需要的功能即可。

### Q: 如何验证我的实现？

A: 运行程序，检查输出是否符合预期。可以添加更多测试场景。

## 参考资源

### 真实代码

阅读这些文件了解真实实现：

1. **PolarDBX Operator**:
   - `pkg/k8s/control/step.go`
   - `pkg/k8s/control/context.go`
   - `pkg/k8s/control/flow.go`
   - `pkg/k8s/control/executor.go`

2. **Crossplane**:
   - `internal/controller/apiextensions/composite/reconciler.go`

### 文档

- [Kubernetes Operator 模式](https://kubernetes.io/docs/concepts/extend-kubernetes/operator/)
- [Go 接口设计](https://go.dev/doc/effective_go#interfaces)

## 下一步

1. 阅读 `../exercises/1.md`
2. 创建 `main.go`
3. 开始实现第一个练习
4. 完成后查看 `../solutions/ex1/` 对比

祝你编码愉快！🚀
