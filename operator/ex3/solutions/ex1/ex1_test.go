package main

import (
	"fmt"
	"testing"
)

// TestExercise1_StateMachine 测试练习 1 的状态机实现
func TestExercise1_StateMachine(t *testing.T) {
	fmt.Println("\n=== 测试练习 1: 状态机与基础协调循环 ===")

	// 创建客户端和日志器
	client := NewMockK8sClient()
	logger := &SimpleLogger{Name: "test"}

	// 场景 1: 新创建的资源（无状态）
	t.Run("场景1_处理新创建的资源", func(t *testing.T) {
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

		err := client.CreateWebApp(webapp)
		if err != nil {
			t.Fatalf("创建 WebApp 失败: %v", err)
		}

		webapp, _ = client.GetWebApp("default", "test-webapp")
		ctx := NewReconcileContext(webapp, client, logger)
		result := Reconcile(ctx)

		fmt.Printf("  协调结果: Requeue=%v, Error=%v\n", result.Requeue, result.Error)

		// 验证结果
		if result.Error != nil {
			t.Errorf("协调失败: %v", result.Error)
		}

		if !result.Requeue {
			t.Error("期望 Requeue=true，但得到 false")
		}

		// 验证 Finalizer 已添加
		webapp, _ = client.GetWebApp("default", "test-webapp")
		if !hasFinalizer(webapp, WebAppFinalizer) {
			t.Error("Finalizer 未添加")
		}
		fmt.Printf("  Finalizer 已添加: %v\n", hasFinalizer(webapp, WebAppFinalizer))

		// 验证状态转换
		if webapp.Status.Phase != PhaseCreating {
			t.Errorf("期望状态为 %s，但得到 %s", PhaseCreating, webapp.Status.Phase)
		}
		fmt.Printf("  当前状态: %s\n", webapp.Status.Phase)
		fmt.Printf("  状态消息: %s\n\n", webapp.Status.Message)
	})

	// 场景 2: 再次协调（应该进入 Creating 状态）
	t.Run("场景2_再次协调进入Creating状态", func(t *testing.T) {
		fmt.Println("场景 2: 再次协调（进入 Creating 状态）")

		webapp, _ := client.GetWebApp("default", "test-webapp")
		ctx := NewReconcileContext(webapp, client, logger)
		result := Reconcile(ctx)

		fmt.Printf("  协调结果: Requeue=%v, Error=%v\n", result.Requeue, result.Error)
		fmt.Printf("  当前状态: %s\n\n", webapp.Status.Phase)

		// 验证进入 Creating 状态处理
		if webapp.Status.Phase != PhaseCreating {
			t.Errorf("期望状态为 %s，但得到 %s", PhaseCreating, webapp.Status.Phase)
		}
	})

	// 场景 3: 验证事件记录
	t.Run("场景3_验证事件记录", func(t *testing.T) {
		events := client.GetEvents()
		fmt.Printf("记录的事件 (%d 个):\n", len(events))

		if len(events) < 2 {
			t.Errorf("期望至少 2 个事件，但得到 %d 个", len(events))
		}

		for i, event := range events {
			fmt.Printf("  %d. [%s] %s: %s\n", i+1, event.Type, event.Reason, event.Message)
		}

		// 验证 FinalizerAdded 事件
		foundFinalizerEvent := false
		foundTransitionEvent := false
		for _, event := range events {
			if event.Reason == "FinalizerAdded" {
				foundFinalizerEvent = true
			}
			if event.Reason == "StateTransition" {
				foundTransitionEvent = true
			}
		}

		if !foundFinalizerEvent {
			t.Error("未找到 FinalizerAdded 事件")
		}
		if !foundTransitionEvent {
			t.Error("未找到 StateTransition 事件")
		}
	})
}

// TestExercise1_FinalizerHelpers 测试 Finalizer 辅助函数
func TestExercise1_FinalizerHelpers(t *testing.T) {
	fmt.Println("\n=== 测试 Finalizer 辅助函数 ===")

	webapp := &WebApp{
		Name:       "test",
		Namespace:  "default",
		Finalizers: []string{},
	}

	// 测试 hasFinalizer
	if hasFinalizer(webapp, WebAppFinalizer) {
		t.Error("期望 hasFinalizer 返回 false，但得到 true")
	}

	// 测试 addFinalizer
	addFinalizer(webapp, WebAppFinalizer)
	if !hasFinalizer(webapp, WebAppFinalizer) {
		t.Error("添加 Finalizer 后，hasFinalizer 应该返回 true")
	}

	if len(webapp.Finalizers) != 1 {
		t.Errorf("期望 1 个 Finalizer，但得到 %d 个", len(webapp.Finalizers))
	}

	// 测试重复添加
	addFinalizer(webapp, WebAppFinalizer)
	if len(webapp.Finalizers) != 1 {
		t.Error("重复添加 Finalizer 应该被忽略")
	}

	// 测试 removeFinalizer
	removeFinalizer(webapp, WebAppFinalizer)
	if hasFinalizer(webapp, WebAppFinalizer) {
		t.Error("移除 Finalizer 后，hasFinalizer 应该返回 false")
	}

	if len(webapp.Finalizers) != 0 {
		t.Errorf("期望 0 个 Finalizer，但得到 %d 个", len(webapp.Finalizers))
	}

	fmt.Println("所有 Finalizer 辅助函数测试通过")
}

// TestExercise1_StateMachineDispatch 测试状态机分发逻辑
func TestExercise1_StateMachineDispatch(t *testing.T) {
	fmt.Println("\n=== 测试状态机分发逻辑 ===")

	client := NewMockK8sClient()
	logger := &SimpleLogger{Name: "test"}

	testCases := []struct {
		name          string
		phase         string
		expectRequeue bool
	}{
		{"空状态", "", true},
		{"Pending状态", PhasePending, true},
		{"Creating状态", PhaseCreating, false},
		{"Running状态", PhaseRunning, false},
		{"Failed状态", PhaseFailed, false},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			webapp := &WebApp{
				Name:       "test-webapp-" + tc.phase,
				Namespace:  "default",
				Generation: 1,
				Spec: WebAppSpec{
					Image:    "nginx:latest",
					Replicas: 1,
					Port:     80,
				},
				Status: WebAppStatus{
					Phase: tc.phase,
				},
			}

			client.CreateWebApp(webapp)
			webapp, _ = client.GetWebApp("default", webapp.Name)

			ctx := NewReconcileContext(webapp, client, logger)
			result := Reconcile(ctx)

			fmt.Printf("  状态 %s: Requeue=%v, Error=%v\n", tc.phase, result.Requeue, result.Error)

			// 对于 Pending 和空状态，验证 Requeue
			if (tc.phase == "" || tc.phase == PhasePending) && !result.Requeue {
				t.Errorf("状态 %s 应该返回 Requeue=true", tc.phase)
			}
		})
	}
}
