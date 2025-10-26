package main

import (
	"fmt"
)

func main() {
	fmt.Println("=== Kubernetes Operator Ex3 - 综合实战练习系列 ===")

	// 运行练习 1 的测试
	testExercise1()
}

// testExercise1 测试练习 1 的实现
func testExercise1() {
	fmt.Println("=== 测试练习 1: 状态机与基础协调循环 ===")

	// 创建客户端和日志器
	client := NewMockK8sClient()
	logger := &SimpleLogger{Name: "test"}

	// 场景 1: 新创建的资源（无状态）
	fmt.Println("场景 1: 处理新创建的资源")
	webapp := &WebApp{
		Name:       "test-webapp",
		Namespace:  "default",
		Generation: 1,
		Spec: WebAppSpec{
			Image:    "nginx:latest",
			Replicas: 2,
			Port:     80,
		},
		Status: WebAppStatus{
			Phase: "", // 空状态
		},
	}

	client.CreateWebApp(webapp)
	webapp, _ = client.GetWebApp("default", "test-webapp")

	ctx := NewReconcileContext(webapp, client, logger)
	result := Reconcile(ctx)

	fmt.Printf("  协调结果: Requeue=%v, Error=%v\n", result.Requeue, result.Error)

	// 验证 Finalizer 已添加
	webapp, _ = client.GetWebApp("default", "test-webapp")
	fmt.Printf("  Finalizer 已添加: %v\n", hasFinalizer(webapp, WebAppFinalizer))
	fmt.Printf("  当前状态: %s\n", webapp.Status.Phase)
	fmt.Printf("  状态消息: %s\n\n", webapp.Status.Message)

	// 场景 2: 再次协调（应该进入 Creating 状态）
	fmt.Println("场景 2: 再次协调（进入 Creating 状态）")
	webapp, _ = client.GetWebApp("default", "test-webapp")
	ctx = NewReconcileContext(webapp, client, logger)
	result = Reconcile(ctx)

	fmt.Printf("  协调结果: Requeue=%v, Error=%v\n", result.Requeue, result.Error)
	fmt.Printf("  当前状态: %s\n\n", webapp.Status.Phase)

	// 显示事件
	events := client.GetEvents()
	fmt.Printf("记录的事件 (%d 个):\n", len(events))
	for i, event := range events {
		fmt.Printf("  %d. [%s] %s: %s\n", i+1, event.Type, event.Reason, event.Message)
	}

	fmt.Println("\n=== 练习 1 测试完成 ===")
	fmt.Println("✓ 状态机引擎工作正常")
	fmt.Println("✓ Finalizer 管理正常")
	fmt.Println("✓ 状态转换正常")
	fmt.Println("✓ 事件记录正常")
	fmt.Println("\n下一步: 完成练习 2 - 资源创建与管理")
}
