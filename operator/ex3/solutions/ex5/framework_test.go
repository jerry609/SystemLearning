package main

import (
	"testing"
	"time"
)

// TestMockK8sClient 测试 MockK8sClient 基础功能
func TestMockK8sClient(t *testing.T) {
	client := NewMockK8sClient()

	// 测试创建 WebApp
	webapp := &WebApp{
		Name:      "test-webapp",
		Namespace: "default",
		Spec: WebAppSpec{
			Image:    "nginx:latest",
			Replicas: 2,
			Port:     80,
		},
	}

	err := client.CreateWebApp(webapp)
	if err != nil {
		t.Fatalf("Failed to create webapp: %v", err)
	}

	// 测试获取 WebApp
	retrieved, err := client.GetWebApp("default", "test-webapp")
	if err != nil {
		t.Fatalf("Failed to get webapp: %v", err)
	}

	if retrieved.Name != "test-webapp" {
		t.Errorf("Expected name 'test-webapp', got '%s'", retrieved.Name)
	}

	if retrieved.Spec.Replicas != 2 {
		t.Errorf("Expected replicas 2, got %d", retrieved.Spec.Replicas)
	}

	// 测试更新 Status
	retrieved.Status.Phase = PhaseRunning
	retrieved.Status.Message = "All resources ready"
	err = client.UpdateWebAppStatus(retrieved)
	if err != nil {
		t.Fatalf("Failed to update status: %v", err)
	}

	// 验证更新
	updated, _ := client.GetWebApp("default", "test-webapp")
	if updated.Status.Phase != PhaseRunning {
		t.Errorf("Expected phase 'Running', got '%s'", updated.Status.Phase)
	}
}

// TestDeploymentOperations 测试 Deployment 操作
func TestDeploymentOperations(t *testing.T) {
	client := NewMockK8sClient()

	// 创建 Deployment
	deployment := &Deployment{
		Name:      "test-deployment",
		Namespace: "default",
		Spec: DeploymentSpec{
			Replicas: 3,
			Image:    "nginx:latest",
		},
	}

	err := client.CreateDeployment(deployment)
	if err != nil {
		t.Fatalf("Failed to create deployment: %v", err)
	}

	// 获取 Deployment
	retrieved, err := client.GetDeployment("default", "test-deployment")
	if err != nil {
		t.Fatalf("Failed to get deployment: %v", err)
	}

	if retrieved.Spec.Replicas != 3 {
		t.Errorf("Expected replicas 3, got %d", retrieved.Spec.Replicas)
	}

	// 更新 Deployment
	retrieved.Spec.Replicas = 5
	err = client.UpdateDeployment(retrieved)
	if err != nil {
		t.Fatalf("Failed to update deployment: %v", err)
	}

	// 验证更新
	updated, _ := client.GetDeployment("default", "test-deployment")
	if updated.Spec.Replicas != 5 {
		t.Errorf("Expected replicas 5, got %d", updated.Spec.Replicas)
	}

	// 删除 Deployment
	err = client.DeleteDeployment("default", "test-deployment")
	if err != nil {
		t.Fatalf("Failed to delete deployment: %v", err)
	}

	// 验证删除
	_, err = client.GetDeployment("default", "test-deployment")
	if err == nil {
		t.Error("Expected error when getting deleted deployment")
	}
}

// TestServiceOperations 测试 Service 操作
func TestServiceOperations(t *testing.T) {
	client := NewMockK8sClient()

	// 创建 Service
	service := &Service{
		Name:      "test-service",
		Namespace: "default",
		Spec: ServiceSpec{
			Port: 80,
			Selector: map[string]string{
				"app": "test",
			},
		},
	}

	err := client.CreateService(service)
	if err != nil {
		t.Fatalf("Failed to create service: %v", err)
	}

	// 获取 Service
	retrieved, err := client.GetService("default", "test-service")
	if err != nil {
		t.Fatalf("Failed to get service: %v", err)
	}

	if retrieved.Spec.Port != 80 {
		t.Errorf("Expected port 80, got %d", retrieved.Spec.Port)
	}

	// 删除 Service
	err = client.DeleteService("default", "test-service")
	if err != nil {
		t.Fatalf("Failed to delete service: %v", err)
	}
}

// TestEventRecording 测试事件记录
func TestEventRecording(t *testing.T) {
	client := NewMockK8sClient()

	// 记录事件
	event := Event{
		Type:    EventTypeNormal,
		Reason:  "Created",
		Message: "WebApp created successfully",
		Object:  "default/test-webapp",
	}

	client.RecordEvent(event)

	// 获取事件
	events := client.GetEvents()
	if len(events) != 1 {
		t.Fatalf("Expected 1 event, got %d", len(events))
	}

	if events[0].Reason != "Created" {
		t.Errorf("Expected reason 'Created', got '%s'", events[0].Reason)
	}
}

// TestReconcileContext 测试协调上下文
func TestReconcileContext(t *testing.T) {
	client := NewMockK8sClient()
	logger := &SimpleLogger{Name: "test"}

	webapp := &WebApp{
		Name:      "test-webapp",
		Namespace: "default",
		Status: WebAppStatus{
			Phase: PhasePending,
		},
	}

	ctx := NewReconcileContext(webapp, client, logger)

	if ctx.WebApp.Name != "test-webapp" {
		t.Errorf("Expected webapp name 'test-webapp', got '%s'", ctx.WebApp.Name)
	}

	if ctx.Client == nil {
		t.Error("Expected client to be set")
	}

	if ctx.Logger == nil {
		t.Error("Expected logger to be set")
	}
}

// TestErrorTypes 测试错误类型
func TestErrorTypes(t *testing.T) {
	// 测试 RetryableError
	retryErr := NewRetryableError("temporary failure", 5*time.Second)
	if !IsRetryableError(retryErr) {
		t.Error("Expected IsRetryableError to return true")
	}

	if IsPermanentError(retryErr) {
		t.Error("Expected IsPermanentError to return false")
	}

	// 测试 PermanentError
	permErr := NewPermanentError("configuration error")
	if IsRetryableError(permErr) {
		t.Error("Expected IsRetryableError to return false")
	}

	if !IsPermanentError(permErr) {
		t.Error("Expected IsPermanentError to return true")
	}
}

// TestWebAppCopy 测试 WebApp 深拷贝
func TestWebAppCopy(t *testing.T) {
	original := &WebApp{
		Name:      "test",
		Namespace: "default",
		Spec: WebAppSpec{
			Env: map[string]string{
				"KEY": "VALUE",
			},
		},
		Finalizers: []string{"finalizer1"},
	}

	copy := copyWebApp(original)

	// 修改副本
	copy.Name = "modified"
	copy.Spec.Env["KEY"] = "MODIFIED"
	copy.Finalizers[0] = "modified-finalizer"

	// 验证原始对象未被修改
	if original.Name != "test" {
		t.Error("Original name was modified")
	}

	if original.Spec.Env["KEY"] != "VALUE" {
		t.Error("Original env was modified")
	}

	if original.Finalizers[0] != "finalizer1" {
		t.Error("Original finalizers were modified")
	}
}
