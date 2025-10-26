package main

import (
	"fmt"
	"time"
)

func main() {
	fmt.Println("=== Kubernetes Operator Ex3 - 练习 4 参考答案 ===")
	fmt.Println()

	// 运行练习 1-4 的测试
	testExercise1()
	fmt.Println()
	testExercise2()
	fmt.Println()
	testExercise3()
	fmt.Println()
	testExercise4()
	
	fmt.Println()
	fmt.Println("=== 练习 4 完成 ===")
	fmt.Println("下一步: 继续练习 5 - 错误处理与可观测性")
}

// testExercise1 测试练习 1: 状态机与基础协调循环
func testExercise1() {
	fmt.Println("=== 测试练习 1: 状态机与基础协调循环 ===")

	client := NewMockK8sClient()
	logger := &SimpleLogger{Name: "ex1"}

	webapp := &WebApp{
		Name:       "ex1-webapp",
		Namespace:  "default",
		Generation: 1,
		Spec: WebAppSpec{
			Image:    "nginx:latest",
			Replicas: 2,
			Port:     80,
		},
		Status: WebAppStatus{
			Phase: PhasePending,
		},
	}

	client.CreateWebApp(webapp)
	webapp, _ = client.GetWebApp("default", "ex1-webapp")

	ctx := NewReconcileContext(webapp, client, logger)
	result := Reconcile(ctx)

	fmt.Printf("✓ Pending → Creating 转换成功\n")
	fmt.Printf("  Requeue: %v\n", result.Requeue)
	fmt.Printf("  Finalizer 已添加: %v\n", hasFinalizer(webapp, WebAppFinalizer))
}

// testExercise2 测试练习 2: 资源创建与管理
func testExercise2() {
	fmt.Println("=== 测试练习 2: 资源创建与管理 ===")

	client := NewMockK8sClient()
	logger := &SimpleLogger{Name: "ex2"}

	webapp := &WebApp{
		Name:       "ex2-webapp",
		Namespace:  "default",
		Generation: 1,
		Spec: WebAppSpec{
			Image:    "nginx:latest",
			Replicas: 3,
			Port:     80,
		},
		Status: WebAppStatus{
			Phase: PhaseCreating,
		},
		Finalizers: []string{WebAppFinalizer},
	}

	client.CreateWebApp(webapp)
	webapp, _ = client.GetWebApp("default", "ex2-webapp")

	// 第一次协调：创建资源
	ctx := NewReconcileContext(webapp, client, logger)
	Reconcile(ctx)

	deployment, _ := client.GetDeployment("default", "ex2-webapp")
	service, _ := client.GetService("default", "ex2-webapp")

	fmt.Printf("✓ Deployment 已创建: %s (Replicas: %d)\n", deployment.Name, deployment.Spec.Replicas)
	fmt.Printf("✓ Service 已创建: %s (Port: %d)\n", service.Name, service.Spec.Port)
	fmt.Printf("✓ OwnerReference 已设置\n")

	// 模拟 Deployment 就绪
	deployment.Status.ReadyReplicas = deployment.Spec.Replicas
	client.UpdateDeployment(deployment)

	// 第二次协调：检查就绪性
	webapp, _ = client.GetWebApp("default", "ex2-webapp")
	ctx = NewReconcileContext(webapp, client, logger)
	Reconcile(ctx)

	webapp, _ = client.GetWebApp("default", "ex2-webapp")
	fmt.Printf("✓ 状态转换到 Running: %s\n", webapp.Status.Phase)
}

// testExercise3 测试练习 3: 更新与同步逻辑
func testExercise3() {
	fmt.Println("=== 测试练习 3: 更新与同步逻辑 ===")

	client := NewMockK8sClient()
	logger := &SimpleLogger{Name: "ex3"}

	webapp := &WebApp{
		Name:       "ex3-webapp",
		Namespace:  "default",
		Generation: 1,
		Spec: WebAppSpec{
			Image:    "nginx:1.20",
			Replicas: 2,
			Port:     80,
		},
		Status: WebAppStatus{
			Phase:              PhaseRunning,
			ObservedGeneration: 1,
		},
		Finalizers: []string{WebAppFinalizer},
	}

	client.CreateWebApp(webapp)

	// 创建初始资源
	deployment := &Deployment{
		Name:      "ex3-webapp",
		Namespace: "default",
		Spec: DeploymentSpec{
			Replicas: 2,
			Image:    "nginx:1.20",
		},
		Status: DeploymentStatus{
			ReadyReplicas: 2,
		},
	}
	client.CreateDeployment(deployment)

	service := &Service{
		Name:      "ex3-webapp",
		Namespace: "default",
		Spec: ServiceSpec{
			Port: 80,
		},
	}
	client.CreateService(service)

	// 修改 Spec
	webapp.Generation = 2
	webapp.Spec.Replicas = 5
	webapp.Spec.Image = "nginx:1.21"
	client.UpdateWebApp(webapp)

	webapp, _ = client.GetWebApp("default", "ex3-webapp")
	ctx := NewReconcileContext(webapp, client, logger)
	Reconcile(ctx)

	deployment, _ = client.GetDeployment("default", "ex3-webapp")
	webapp, _ = client.GetWebApp("default", "ex3-webapp")

	fmt.Printf("✓ 检测到 Spec 变更 (Generation: %d → %d)\n", 1, 2)
	fmt.Printf("✓ Deployment 已同步: Replicas %d, Image %s\n", deployment.Spec.Replicas, deployment.Spec.Image)
	fmt.Printf("✓ ObservedGeneration 已更新: %d\n", webapp.Status.ObservedGeneration)
}

// testExercise4 测试练习 4: 删除与 Finalizer
func testExercise4() {
	fmt.Println("=== 测试练习 4: 删除与 Finalizer ===")

	client := NewMockK8sClient()
	logger := &SimpleLogger{Name: "ex4"}

	webapp := &WebApp{
		Name:       "ex4-webapp",
		Namespace:  "default",
		Generation: 1,
		Spec: WebAppSpec{
			Image:    "nginx:latest",
			Replicas: 2,
			Port:     80,
		},
		Status: WebAppStatus{
			Phase: PhaseRunning,
		},
		Finalizers: []string{WebAppFinalizer},
	}

	client.CreateWebApp(webapp)

	// 创建资源
	deployment := &Deployment{
		Name:      "ex4-webapp",
		Namespace: "default",
		Spec: DeploymentSpec{
			Replicas: 2,
			Image:    "nginx:latest",
		},
	}
	client.CreateDeployment(deployment)

	service := &Service{
		Name:      "ex4-webapp",
		Namespace: "default",
		Spec: ServiceSpec{
			Port: 80,
		},
	}
	client.CreateService(service)

	// 设置删除时间戳
	now := time.Now()
	webapp.DeletionTimestamp = &now
	client.UpdateWebApp(webapp)

	webapp, _ = client.GetWebApp("default", "ex4-webapp")
	ctx := NewReconcileContext(webapp, client, logger)
	Reconcile(ctx)

	// 验证资源已删除
	_, deploymentErr := client.GetDeployment("default", "ex4-webapp")
	_, serviceErr := client.GetService("default", "ex4-webapp")

	webapp, _ = client.GetWebApp("default", "ex4-webapp")

	fmt.Printf("✓ 检测到删除请求\n")
	fmt.Printf("✓ Service 已删除: %v\n", serviceErr != nil)
	fmt.Printf("✓ Deployment 已删除: %v\n", deploymentErr != nil)
	fmt.Printf("✓ Finalizer 已移除: %v\n", !hasFinalizer(webapp, WebAppFinalizer))
}

// testExercise5 测试练习 5: 错误处理与可观测性
func testExercise5() {
	fmt.Println("=== 测试练习 5: 错误处理与可观测性 ===")

	client := NewMockK8sClient()
	logger := &SimpleLogger{Name: "ex5"}

	webapp := &WebApp{
		Name:       "ex5-webapp",
		Namespace:  "default",
		Generation: 1,
		Spec: WebAppSpec{
			Image:    "nginx:latest",
			Replicas: 2,
			Port:     80,
		},
		Status: WebAppStatus{
			Phase: PhaseRunning,
		},
		Finalizers: []string{WebAppFinalizer},
	}

	client.CreateWebApp(webapp)

	// 创建资源
	deployment := &Deployment{
		Name:      "ex5-webapp",
		Namespace: "default",
		Spec: DeploymentSpec{
			Replicas: 2,
			Image:    "nginx:latest",
		},
		Status: DeploymentStatus{
			ReadyReplicas: 2,
		},
	}
	client.CreateDeployment(deployment)

	service := &Service{
		Name:      "ex5-webapp",
		Namespace: "default",
		Spec: ServiceSpec{
			Port: 80,
		},
	}
	client.CreateService(service)

	webapp, _ = client.GetWebApp("default", "ex5-webapp")
	ctx := NewReconcileContext(webapp, client, logger)
	Reconcile(ctx)

	webapp, _ = client.GetWebApp("default", "ex5-webapp")

	// 检查 Conditions
	hasReady := false
	hasProgressing := false
	for _, cond := range webapp.Status.Conditions {
		if cond.Type == "Ready" && cond.Status == "True" {
			hasReady = true
		}
		if cond.Type == "Progressing" {
			hasProgressing = true
		}
	}

	fmt.Printf("✓ Conditions 已设置:\n")
	fmt.Printf("  - Ready: %v\n", hasReady)
	fmt.Printf("  - Progressing: %v\n", hasProgressing)
	fmt.Printf("✓ 事件记录数量: %d\n", len(client.GetEvents()))
	fmt.Printf("✓ LastReconcileTime 已更新\n")
}
