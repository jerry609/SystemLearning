package main

import (
	"fmt"
	"time"
)

// ReconcileContext 协调上下文，携带协调过程中需要的所有信息
type ReconcileContext struct {
	WebApp *WebApp
	Client *MockK8sClient
	Logger Logger
	Result ReconcileResult
}

// ReconcileResult 协调结果
type ReconcileResult struct {
	Requeue      bool
	RequeueAfter time.Duration
	Error        error
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

// Reconcile 主协调函数 - 完整实现（练习 1-5）
func Reconcile(ctx *ReconcileContext) ReconcileResult {
	// Panic 恢复（练习 5）
	defer func() {
		if r := recover(); r != nil {
			ctx.Logger.Error(fmt.Errorf("%v", r), "Panic 恢复")
			ctx.WebApp.Status.Phase = PhaseFailed
			ctx.WebApp.Status.Message = fmt.Sprintf("Panic: %v", r)
			updateConditions(ctx, "Ready", "False", "Panic", fmt.Sprintf("%v", r))
			ctx.Client.UpdateWebAppStatus(ctx.WebApp)
		}
	}()

	ctx.Logger.Info("Reconcile called", "webapp", ctx.WebApp.Name, "phase", ctx.WebApp.Status.Phase)

	// 更新 Progressing Condition（练习 5）
	updateConditions(ctx, "Progressing", "True", "Reconciling", "正在协调资源")

	var result ReconcileResult

	// 1. 检查是否正在删除（练习 4）
	if ctx.WebApp.DeletionTimestamp != nil {
		result = handleDeletion(ctx)
	} else {
		// 2. 根据当前状态选择执行路径
		switch ctx.WebApp.Status.Phase {
		case "", PhasePending:
			result = handlePending(ctx)
		case PhaseCreating:
			result = handleCreating(ctx)
		case PhaseRunning:
			result = handleRunning(ctx)
		case PhaseFailed:
			result = handleFailed(ctx)
		default:
			err := fmt.Errorf("unknown phase: %s", ctx.WebApp.Status.Phase)
			ctx.Logger.Error(err, "未知的状态")
			result = ReconcileResult{Error: err}
		}
	}

	// 更新 Ready Condition（练习 5）
	if result.Error == nil && ctx.WebApp.Status.Phase == PhaseRunning {
		updateConditions(ctx, "Ready", "True", "ResourcesReady", "所有资源已就绪")
	} else if result.Error != nil {
		updateConditions(ctx, "Ready", "False", "ReconcileError", result.Error.Error())
	}

	return result
}

// ============================================================================
// 练习 1: 状态机与基础协调循环
// ============================================================================

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

	ctx.Client.RecordEvent(Event{
		Type:    EventTypeNormal,
		Reason:  "StateTransition",
		Message: fmt.Sprintf("状态从 %s 转换到 %s", PhasePending, PhaseCreating),
		Object:  fmt.Sprintf("%s/%s", ctx.WebApp.Namespace, ctx.WebApp.Name),
	})

	return ReconcileResult{Requeue: true}
}

// ============================================================================
// 练习 2: 资源创建与管理
// ============================================================================

// handleCreating 处理 Creating 状态
func handleCreating(ctx *ReconcileContext) ReconcileResult {
	ctx.Logger.Info("处理 Creating 状态", "webapp", ctx.WebApp.Name)

	// 步骤 1: 创建 Deployment
	if err := ensureDeployment(ctx); err != nil {
		ctx.Logger.Error(err, "确保 Deployment 失败")
		return ReconcileResult{Error: err}
	}

	// 步骤 2: 创建 Service
	if err := ensureService(ctx); err != nil {
		ctx.Logger.Error(err, "确保 Service 失败")
		return ReconcileResult{Error: err}
	}

	// 步骤 3: 检查资源就绪性
	ready, err := checkResourcesReady(ctx)
	if err != nil {
		ctx.Logger.Error(err, "检查资源就绪性失败")
		return ReconcileResult{Error: err}
	}

	if !ready {
		ctx.Logger.Info("资源未就绪，等待中...")
		return ReconcileResult{
			Requeue:      true,
			RequeueAfter: 5 * time.Second,
		}
	}

	// 步骤 4: 所有资源就绪，转换到 Running 状态
	ctx.WebApp.Status.Phase = PhaseRunning
	ctx.WebApp.Status.Message = "所有资源已就绪"
	ctx.WebApp.Status.LastReconcileTime = time.Now()
	ctx.WebApp.Status.ObservedGeneration = ctx.WebApp.Generation

	if err := ctx.Client.UpdateWebAppStatus(ctx.WebApp); err != nil {
		ctx.Logger.Error(err, "更新状态失败")
		return ReconcileResult{Error: err}
	}

	ctx.Logger.Info("状态已转换", "from", PhaseCreating, "to", PhaseRunning)

	ctx.Client.RecordEvent(Event{
		Type:    EventTypeNormal,
		Reason:  "StateTransition",
		Message: fmt.Sprintf("状态从 %s 转换到 %s", PhaseCreating, PhaseRunning),
		Object:  fmt.Sprintf("%s/%s", ctx.WebApp.Namespace, ctx.WebApp.Name),
	})

	return ReconcileResult{Requeue: true}
}

func ensureDeployment(ctx *ReconcileContext) error {
	deploymentName := ctx.WebApp.Name

	_, err := ctx.Client.GetDeployment(ctx.WebApp.Namespace, deploymentName)
	if err == nil {
		ctx.Logger.Info("Deployment 已存在", "name", deploymentName)
		return nil
	}

	deployment := &Deployment{
		Name:      deploymentName,
		Namespace: ctx.WebApp.Namespace,
		Spec: DeploymentSpec{
			Replicas: ctx.WebApp.Spec.Replicas,
			Image:    ctx.WebApp.Spec.Image,
			Env:      copyMap(ctx.WebApp.Spec.Env),
		},
		OwnerReferences: []OwnerReference{
			{
				Name: ctx.WebApp.Name,
				UID:  fmt.Sprintf("%s-%s", ctx.WebApp.Namespace, ctx.WebApp.Name),
			},
		},
	}

	if err := ctx.Client.CreateDeployment(deployment); err != nil {
		ctx.Logger.Error(err, "创建 Deployment 失败")
		return err
	}

	ctx.Logger.Info("Deployment 已创建", "name", deploymentName)

	ctx.Client.RecordEvent(Event{
		Type:    EventTypeNormal,
		Reason:  "DeploymentCreated",
		Message: fmt.Sprintf("Deployment %s 已创建", deploymentName),
		Object:  fmt.Sprintf("%s/%s", ctx.WebApp.Namespace, ctx.WebApp.Name),
	})

	return nil
}

func ensureService(ctx *ReconcileContext) error {
	serviceName := ctx.WebApp.Name

	_, err := ctx.Client.GetService(ctx.WebApp.Namespace, serviceName)
	if err == nil {
		ctx.Logger.Info("Service 已存在", "name", serviceName)
		return nil
	}

	service := &Service{
		Name:      serviceName,
		Namespace: ctx.WebApp.Namespace,
		Spec: ServiceSpec{
			Port: ctx.WebApp.Spec.Port,
			Selector: map[string]string{
				"app": ctx.WebApp.Name,
			},
		},
		OwnerReferences: []OwnerReference{
			{
				Name: ctx.WebApp.Name,
				UID:  fmt.Sprintf("%s-%s", ctx.WebApp.Namespace, ctx.WebApp.Name),
			},
		},
	}

	if err := ctx.Client.CreateService(service); err != nil {
		ctx.Logger.Error(err, "创建 Service 失败")
		return err
	}

	ctx.Logger.Info("Service 已创建", "name", serviceName)

	ctx.Client.RecordEvent(Event{
		Type:    EventTypeNormal,
		Reason:  "ServiceCreated",
		Message: fmt.Sprintf("Service %s 已创建", serviceName),
		Object:  fmt.Sprintf("%s/%s", ctx.WebApp.Namespace, ctx.WebApp.Name),
	})

	return nil
}

func checkResourcesReady(ctx *ReconcileContext) (bool, error) {
	deployment, err := ctx.Client.GetDeployment(ctx.WebApp.Namespace, ctx.WebApp.Name)
	if err != nil {
		return false, err
	}

	if deployment.Status.ReadyReplicas < deployment.Spec.Replicas {
		ctx.Logger.Info("Deployment 未就绪",
			"ready", deployment.Status.ReadyReplicas,
			"desired", deployment.Spec.Replicas)
		return false, nil
	}

	_, err = ctx.Client.GetService(ctx.WebApp.Namespace, ctx.WebApp.Name)
	if err != nil {
		return false, err
	}

	ctx.Logger.Info("所有资源已就绪")
	return true, nil
}

// ============================================================================
// 练习 3: 更新与同步逻辑
// ============================================================================

// handleRunning 处理 Running 状态
func handleRunning(ctx *ReconcileContext) ReconcileResult {
	ctx.Logger.Info("处理 Running 状态", "webapp", ctx.WebApp.Name)

	// 检查 Spec 是否变更
	if specChanged(ctx) {
		ctx.Logger.Info("检测到 Spec 变更，开始同步")

		if err := syncDeployment(ctx); err != nil {
			return ReconcileResult{Error: err}
		}

		if err := syncService(ctx); err != nil {
			return ReconcileResult{Error: err}
		}

		ctx.WebApp.Status.ObservedGeneration = ctx.WebApp.Generation
		ctx.WebApp.Status.Message = "资源已同步"
		ctx.WebApp.Status.LastReconcileTime = time.Now()

		if err := ctx.Client.UpdateWebAppStatus(ctx.WebApp); err != nil {
			return ReconcileResult{Error: err}
		}

		ctx.Logger.Info("资源同步完成")

		ctx.Client.RecordEvent(Event{
			Type:    EventTypeNormal,
			Reason:  "ResourcesSynced",
			Message: "资源已同步到最新配置",
			Object:  fmt.Sprintf("%s/%s", ctx.WebApp.Namespace, ctx.WebApp.Name),
		})
	}

	// 定期检查健康状态
	return ReconcileResult{
		Requeue:      true,
		RequeueAfter: 30 * time.Second,
	}
}

func specChanged(ctx *ReconcileContext) bool {
	return ctx.WebApp.Generation != ctx.WebApp.Status.ObservedGeneration
}

func syncDeployment(ctx *ReconcileContext) error {
	deployment, err := ctx.Client.GetDeployment(ctx.WebApp.Namespace, ctx.WebApp.Name)
	if err != nil {
		return err
	}

	needsUpdate := false
	if deployment.Spec.Replicas != ctx.WebApp.Spec.Replicas {
		deployment.Spec.Replicas = ctx.WebApp.Spec.Replicas
		needsUpdate = true
	}
	if deployment.Spec.Image != ctx.WebApp.Spec.Image {
		deployment.Spec.Image = ctx.WebApp.Spec.Image
		needsUpdate = true
	}

	if needsUpdate {
		ctx.Logger.Info("同步 Deployment", "name", deployment.Name)
		return ctx.Client.UpdateDeployment(deployment)
	}

	return nil
}

func syncService(ctx *ReconcileContext) error {
	service, err := ctx.Client.GetService(ctx.WebApp.Namespace, ctx.WebApp.Name)
	if err != nil {
		return err
	}

	needsUpdate := false
	if service.Spec.Port != ctx.WebApp.Spec.Port {
		service.Spec.Port = ctx.WebApp.Spec.Port
		needsUpdate = true
	}

	if needsUpdate {
		ctx.Logger.Info("同步 Service", "name", service.Name)
		return ctx.Client.UpdateService(service)
	}

	return nil
}

// ============================================================================
// 练习 4: 删除与 Finalizer
// ============================================================================

// handleDeletion 处理删除
func handleDeletion(ctx *ReconcileContext) ReconcileResult {
	ctx.Logger.Info("处理删除", "webapp", ctx.WebApp.Name)

	if !hasFinalizer(ctx.WebApp, WebAppFinalizer) {
		return ReconcileResult{}
	}

	if ctx.WebApp.Status.Phase != PhaseDeleting {
		ctx.WebApp.Status.Phase = PhaseDeleting
		ctx.WebApp.Status.Message = "正在删除资源"
		ctx.WebApp.Status.LastReconcileTime = time.Now()

		if err := ctx.Client.UpdateWebAppStatus(ctx.WebApp); err != nil {
			return ReconcileResult{Error: err}
		}

		ctx.Logger.Info("状态已转换到 Deleting")
	}

	if err := deleteService(ctx); err != nil {
		ctx.Logger.Error(err, "删除 Service 失败")
		return ReconcileResult{Error: err}
	}

	if err := deleteDeployment(ctx); err != nil {
		ctx.Logger.Error(err, "删除 Deployment 失败")
		return ReconcileResult{Error: err}
	}

	removeFinalizer(ctx.WebApp, WebAppFinalizer)
	if err := ctx.Client.UpdateWebApp(ctx.WebApp); err != nil {
		ctx.Logger.Error(err, "移除 Finalizer 失败")
		return ReconcileResult{Error: err}
	}

	ctx.Logger.Info("Finalizer 已移除，资源将被删除")

	ctx.Client.RecordEvent(Event{
		Type:    EventTypeNormal,
		Reason:  "FinalizerRemoved",
		Message: "Finalizer 已移除，资源清理完成",
		Object:  fmt.Sprintf("%s/%s", ctx.WebApp.Namespace, ctx.WebApp.Name),
	})

	return ReconcileResult{}
}

func deleteService(ctx *ReconcileContext) error {
	_, err := ctx.Client.GetService(ctx.WebApp.Namespace, ctx.WebApp.Name)
	if err != nil {
		return nil
	}

	ctx.Logger.Info("删除 Service", "name", ctx.WebApp.Name)
	return ctx.Client.DeleteService(ctx.WebApp.Namespace, ctx.WebApp.Name)
}

func deleteDeployment(ctx *ReconcileContext) error {
	_, err := ctx.Client.GetDeployment(ctx.WebApp.Namespace, ctx.WebApp.Name)
	if err != nil {
		return nil
	}

	ctx.Logger.Info("删除 Deployment", "name", ctx.WebApp.Name)
	return ctx.Client.DeleteDeployment(ctx.WebApp.Namespace, ctx.WebApp.Name)
}

// ============================================================================
// 练习 5: 错误处理与可观测性
// ============================================================================

// handleFailed 处理 Failed 状态
func handleFailed(ctx *ReconcileContext) ReconcileResult {
	ctx.Logger.Info("处理 Failed 状态", "webapp", ctx.WebApp.Name)

	if specChanged(ctx) {
		ctx.Logger.Info("检测到 Spec 变更，尝试恢复")

		ctx.WebApp.Status.Phase = PhasePending
		ctx.WebApp.Status.Message = "尝试从失败状态恢复"
		ctx.WebApp.Status.LastReconcileTime = time.Now()

		if err := ctx.Client.UpdateWebAppStatus(ctx.WebApp); err != nil {
			return ReconcileResult{Error: err}
		}

		return ReconcileResult{Requeue: true}
	}

	return ReconcileResult{}
}

func updateConditions(ctx *ReconcileContext, condType, status, reason, message string) {
	now := time.Now()

	found := false
	for i, cond := range ctx.WebApp.Status.Conditions {
		if cond.Type == condType {
			if cond.Status != status {
				ctx.WebApp.Status.Conditions[i].Status = status
				ctx.WebApp.Status.Conditions[i].LastTransitionTime = now
			}
			ctx.WebApp.Status.Conditions[i].Reason = reason
			ctx.WebApp.Status.Conditions[i].Message = message
			found = true
			break
		}
	}

	if !found {
		ctx.WebApp.Status.Conditions = append(ctx.WebApp.Status.Conditions, Condition{
			Type:               condType,
			Status:             status,
			LastTransitionTime: now,
			Reason:             reason,
			Message:            message,
		})
	}
}

// ============================================================================
// Finalizer 管理
// ============================================================================

const WebAppFinalizer = "webapp.example.com/finalizer"

func hasFinalizer(webapp *WebApp, finalizer string) bool {
	for _, f := range webapp.Finalizers {
		if f == finalizer {
			return true
		}
	}
	return false
}

func addFinalizer(webapp *WebApp, finalizer string) {
	if !hasFinalizer(webapp, finalizer) {
		webapp.Finalizers = append(webapp.Finalizers, finalizer)
	}
}

func removeFinalizer(webapp *WebApp, finalizer string) {
	finalizers := []string{}
	for _, f := range webapp.Finalizers {
		if f != finalizer {
			finalizers = append(finalizers, f)
		}
	}
	webapp.Finalizers = finalizers
}

// ============================================================================
// 辅助函数
// ============================================================================

func copyMap(m map[string]string) map[string]string {
	if m == nil {
		return nil
	}
	result := make(map[string]string)
	for k, v := range m {
		result[k] = v
	}
	return result
}
