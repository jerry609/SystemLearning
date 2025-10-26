package main

import (
	"fmt"
)

func main() {
	fmt.Println("=== Kubernetes Operator Ex3 - 综合实战练习系列 ===")

	// 创建模拟的 K8s 客户端
	client := NewMockK8sClient()

	// 创建一个示例 WebApp 资源
	webapp := &WebApp{
		Name:      "my-webapp",
		Namespace: "default",
		Spec: WebAppSpec{
			Image:    "nginx:latest",
			Replicas: 3,
			Port:     80,
			Env: map[string]string{
				"ENV": "production",
			},
		},
		Status: WebAppStatus{
			Phase: PhasePending,
		},
	}

	// 将 WebApp 添加到客户端
	err := client.CreateWebApp(webapp)
	if err != nil {
		fmt.Printf("Error creating webapp: %v\n", err)
		return
	}

	// 创建日志器
	logger := &SimpleLogger{Name: "reconciler"}

	// 获取 WebApp（模拟从 API Server 获取）
	webapp, err = client.GetWebApp("default", "my-webapp")
	if err != nil {
		fmt.Printf("Error getting webapp: %v\n", err)
		return
	}

	// 创建协调上下文
	ctx := NewReconcileContext(webapp, client, logger)

	// 执行协调
	fmt.Println("执行协调循环...")
	result := Reconcile(ctx)

	// 输出结果
	fmt.Printf("\n协调结果:\n")
	fmt.Printf("  Requeue: %v\n", result.Requeue)
	fmt.Printf("  RequeueAfter: %v\n", result.RequeueAfter)
	fmt.Printf("  Error: %v\n", result.Error)

	// 显示当前状态
	webapp, _ = client.GetWebApp("default", "my-webapp")
	fmt.Printf("\nWebApp 当前状态:\n")
	fmt.Printf("  Name: %s\n", webapp.Name)
	fmt.Printf("  Namespace: %s\n", webapp.Namespace)
	fmt.Printf("  Phase: %s\n", webapp.Status.Phase)
	fmt.Printf("  Generation: %d\n", webapp.Generation)
	fmt.Printf("  ObservedGeneration: %d\n", webapp.Status.ObservedGeneration)

	// 显示事件
	events := client.GetEvents()
	if len(events) > 0 {
		fmt.Printf("\n记录的事件:\n")
		for i, event := range events {
			fmt.Printf("  %d. [%s] %s: %s\n",
				i+1, event.Type, event.Reason, event.Message)
		}
	}

	fmt.Println("\n=== 提示 ===")
	fmt.Println("请按照 exercises/1.md 的说明完成练习 1")
	fmt.Println("实现状态机引擎和 Finalizer 管理功能")
}
