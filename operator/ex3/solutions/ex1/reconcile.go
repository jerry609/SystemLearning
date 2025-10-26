package main

import (
	"fmt"
	"time"
)

// ReconcileContext 协调上下文，携带协调过程中需要的所有信息
type ReconcileContext struct {
	// 当前协调的资源
	WebApp *WebApp

	// 模拟的 K8s 客户端
	Client *MockK8sClient

	// 日志器
	Logger Logger

	// 协调结果
	Result ReconcileResult
}

// ReconcileResult 协调结果
type ReconcileResult struct {
	Requeue      bool          // 是否需要重新入队
	RequeueAfter time.Duration // 延迟重新入队时间
	Error        error         // 错误信息
}

// Logger 日志接口
type Logger interface {
	Info(msg string, keysAndValues ...interface{})
	Error(err error, msg string, keysAndValues ...interface{})
	Debug(msg string, keysAndValues ...interface{})
}

// SimpleLogger 简单的日志实现
type SimpleLogger struct {
	Name string
}

func (l *SimpleLogger) Info(msg string, keysAndValues ...interface{}) {
	fmt.Printf("[INFO] %s: %s %v\n", l.Name, msg, keysAndValues)
}

func (l *SimpleLogger) Error(err error, msg string, keysAndValues ...interface{}) {
	fmt.Printf("[ERROR] %s: %s - %v %v\n", l.Name, msg, err, keysAndValues)
}

func (l *SimpleLogger) Debug(msg string, keysAndValues ...interface{}) {
	fmt.Printf("[DEBUG] %s: %s %v\n", l.Name, msg, keysAndValues)
}

// NewReconcileContext 创建新的协调上下文
func NewReconcileContext(webapp *WebApp, client *MockK8sClient, logger Logger) *ReconcileContext {
	return &ReconcileContext{
		WebApp: webapp,
		Client: client,
		Logger: logger,
		Result: ReconcileResult{},
	}
}

// Reconcile 主协调函数 - 状态机引擎
func Reconcile(ctx *ReconcileContext) ReconcileResult {
	ctx.Logger.Info("Reconcile called", "webapp", ctx.WebApp.Name, "phase", ctx.WebApp.Status.Phase)
	
	// 1. 检查是否正在删除
	if ctx.WebApp.DeletionTimestamp != nil {
		return handleDeletion(ctx)
	}
	
	// 2. 根据当前状态选择执行路径
	switch ctx.WebApp.Status.Phase {
	case "", PhasePending:
		return handlePending(ctx)
	case PhaseCreating:
		return handleCreating(ctx)
	case PhaseRunning:
		return handleRunning(ctx)
	case PhaseFailed:
		return handleFailed(ctx)
	default:
		err := fmt.Errorf("unknown phase: %s", ctx.WebApp.Status.Phase)
		ctx.Logger.Error(err, "未知的状态")
		return ReconcileResult{Error: err}
	}
}

// handlePending 处理 Pending 状态
func handlePending(ctx *ReconcileContext) ReconcileResult {
	ctx.Logger.Info("处理 Pending 状态", "webapp", ctx.WebApp.Name)
	
	// 步骤 1: 添加 Finalizer
	if !hasFinalizer(ctx.WebApp, WebAppFinalizer) {
		addFinalizer(ctx.WebApp, WebAppFinalizer)
		if err := ctx.Client.UpdateWebApp(ctx.WebApp); err != nil {
			ctx.Logger.Error(err, "添加 Finalizer 失败")
			return ReconcileResult{Error: err}
		}
		ctx.Logger.Info("已添加 Finalizer")
		
		// 记录事件
		ctx.Client.RecordEvent(Event{
			Type:    EventTypeNormal,
			Reason:  "FinalizerAdded",
			Message: "Finalizer 已添加",
			Object:  fmt.Sprintf("%s/%s", ctx.WebApp.Namespace, ctx.WebApp.Name),
		})
	}
	
	// 步骤 2: 转换到 Creating 状态
	ctx.WebApp.Status.Phase = PhaseCreating
	ctx.WebApp.Status.Message = "开始创建依赖资源"
	ctx.WebApp.Status.LastReconcileTime = time.Now()
	
	if err := ctx.Client.UpdateWebAppStatus(ctx.WebApp); err != nil {
		ctx.Logger.Error(err, "更新状态失败")
		return ReconcileResult{Error: err}
	}
	
	ctx.Logger.Info("状态已转换", "from", PhasePending, "to", PhaseCreating)
	
	// 记录事件
	ctx.Client.RecordEvent(Event{
		Type:    EventTypeNormal,
		Reason:  "StateTransition",
		Message: fmt.Sprintf("状态从 %s 转换到 %s", PhasePending, PhaseCreating),
		Object:  fmt.Sprintf("%s/%s", ctx.WebApp.Namespace, ctx.WebApp.Name),
	})
	
	// 立即重新入队以处理 Creating 状态
	return ReconcileResult{Requeue: true}
}

// handleCreating 处理 Creating 状态（占位符，在练习 2 中实现）
func handleCreating(ctx *ReconcileContext) ReconcileResult {
	ctx.Logger.Info("处理 Creating 状态（待实现）", "webapp", ctx.WebApp.Name)
	// TODO: 在练习 2 中实现
	return ReconcileResult{}
}

// handleRunning 处理 Running 状态（占位符，在练习 3 中实现）
func handleRunning(ctx *ReconcileContext) ReconcileResult {
	ctx.Logger.Info("处理 Running 状态（待实现）", "webapp", ctx.WebApp.Name)
	// TODO: 在练习 3 中实现
	return ReconcileResult{}
}

// handleFailed 处理 Failed 状态（占位符，在练习 5 中实现）
func handleFailed(ctx *ReconcileContext) ReconcileResult {
	ctx.Logger.Info("处理 Failed 状态（待实现）", "webapp", ctx.WebApp.Name)
	// TODO: 在练习 5 中实现
	return ReconcileResult{}
}

// handleDeletion 处理删除（占位符，在练习 4 中实现）
func handleDeletion(ctx *ReconcileContext) ReconcileResult {
	ctx.Logger.Info("处理删除（待实现）", "webapp", ctx.WebApp.Name)
	// TODO: 在练习 4 中实现
	return ReconcileResult{}
}

// ============================================================================
// Finalizer 管理
// ============================================================================

// WebAppFinalizer 是 WebApp 资源的 Finalizer 标识
const WebAppFinalizer = "webapp.example.com/finalizer"

// hasFinalizer 检查资源是否有指定的 Finalizer
func hasFinalizer(webapp *WebApp, finalizer string) bool {
	for _, f := range webapp.Finalizers {
		if f == finalizer {
			return true
		}
	}
	return false
}

// addFinalizer 添加 Finalizer
func addFinalizer(webapp *WebApp, finalizer string) {
	if !hasFinalizer(webapp, finalizer) {
		webapp.Finalizers = append(webapp.Finalizers, finalizer)
	}
}

// removeFinalizer 移除 Finalizer
func removeFinalizer(webapp *WebApp, finalizer string) {
	finalizers := []string{}
	for _, f := range webapp.Finalizers {
		if f != finalizer {
			finalizers = append(finalizers, f)
		}
	}
	webapp.Finalizers = finalizers
}
