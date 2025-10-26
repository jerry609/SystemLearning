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

// Reconcile 主协调函数（框架，具体实现在各个练习中完成）
func Reconcile(ctx *ReconcileContext) ReconcileResult {
	// TODO: 在练习 1 中实现状态机逻辑
	ctx.Logger.Info("Reconcile called", "webapp", ctx.WebApp.Name, "phase", ctx.WebApp.Status.Phase)

	// 占位符实现
	return ReconcileResult{
		Requeue:      false,
		RequeueAfter: 0,
		Error:        nil,
	}
}

// ============================================================================
// TODO: 在练习 1 中实现以下函数
// ============================================================================

// WebAppFinalizer 是 WebApp 资源的 Finalizer 标识
const WebAppFinalizer = "webapp.example.com/finalizer"

// hasFinalizer 检查资源是否有指定的 Finalizer
// TODO: 实现此函数
func hasFinalizer(webapp *WebApp, finalizer string) bool {
	// 你的代码
	return false
}

// addFinalizer 添加 Finalizer
// TODO: 实现此函数
func addFinalizer(webapp *WebApp, finalizer string) {
	// 你的代码
}

// removeFinalizer 移除 Finalizer
// TODO: 实现此函数
func removeFinalizer(webapp *WebApp, finalizer string) {
	// 你的代码
}
