# 框架代码

这是 Operator Ex3 的基础框架代码，你将在这里完成所有练习的实现。

## 文件说明

### 核心文件（需要修改）

- **`reconcile.go`** - 协调循环的核心逻辑
  - 🔨 在这里实现状态机引擎
  - 🔨 实现各个状态的处理函数
  - 🔨 实现 Finalizer 管理

- **`main.go`** - 主程序入口
  - 用于测试你的实现
  - 可以根据需要修改测试场景

### 基础设施文件（通常不需要修改）

- **`types.go`** - 数据结构定义
  - WebApp、Deployment、Service 等类型
  - 状态常量定义

- **`client.go`** - 模拟的 Kubernetes 客户端
  - MockK8sClient 实现
  - 提供 CRUD 操作

- **`errors.go`** - 错误类型定义
  - RetryableError（可重试错误）
  - PermanentError（永久性错误）

- **`framework_test.go`** - 框架基础测试
  - 验证基础设施是否正常工作

## 开始练习

### 练习 1：状态机与基础协调循环

1. 打开 `reconcile.go`
2. 找到 `Reconcile()` 函数
3. 实现以下功能：

```go
// TODO: 实现状态机引擎
func Reconcile(ctx *ReconcileContext) ReconcileResult {
    // 1. 检查是否正在删除
    // 2. 根据状态分发到不同的处理函数
}

// TODO: 实现 Finalizer 辅助函数
func hasFinalizer(webapp *WebApp, finalizer string) bool { }
func addFinalizer(webapp *WebApp, finalizer string) { }
func removeFinalizer(webapp *WebApp, finalizer string) { }

// TODO: 实现 Pending 状态处理
func handlePending(ctx *ReconcileContext) ReconcileResult { }
```

### 运行和测试

```bash
# 运行主程序
go run .

# 运行测试（如果你添加了测试）
go test -v

# 检查代码格式
go fmt ./...
```

### 验证实现

运行后应该看到：

```
=== Kubernetes Operator Ex3 - 综合实战练习系列 ===
执行协调循环...
[INFO] reconciler: Reconcile called [webapp my-webapp phase Pending]
[INFO] reconciler: 处理 Pending 状态 [webapp my-webapp]
[INFO] reconciler: 已添加 Finalizer []
[INFO] reconciler: 状态已转换 [from Pending to Creating]

协调结果:
  Requeue: true
  ...
```

## 开发提示

### 1. 理解数据流

```
WebApp (API) → Reconcile() → 状态处理函数 → 更新 Status → 返回结果
```

### 2. 状态机模式

```
Pending → Creating → Running
   ↓         ↓          ↓
   └────→ Failed ←──────┘
```

### 3. 幂等性原则

所有操作都应该是幂等的：
- 多次执行产生相同结果
- 检查资源是否已存在再创建
- 检查 Finalizer 是否已添加

### 4. 错误处理

- 临时性错误：返回 `ReconcileResult{Error: err}`，会自动重试
- 永久性错误：转换到 Failed 状态
- 记录详细的日志和事件

### 5. 日志和事件

```go
// 记录日志
ctx.Logger.Info("操作成功", "key", value)
ctx.Logger.Error(err, "操作失败")

// 记录事件
ctx.Client.RecordEvent(Event{
    Type:    EventTypeNormal,
    Reason:  "Created",
    Message: "资源已创建",
    Object:  fmt.Sprintf("%s/%s", namespace, name),
})
```

## 遇到困难？

1. 查看 `../exercises/1.md` 获取详细说明
2. 参考 `../solutions/ex1/` 中的实现
3. 运行 `go test -v` 查看测试输出
4. 检查日志输出定位问题

## 下一步

完成练习 1 后：
1. 确保所有测试通过
2. 对比参考答案，理解差异
3. 继续练习 2：资源创建与管理

祝你学习愉快！🚀
