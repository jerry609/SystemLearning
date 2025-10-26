package main

import (
	"fmt"
	"sync"
	"time"
)

// MockK8sClient 模拟 Kubernetes 客户端，使用内存存储
type MockK8sClient struct {
	mu          sync.RWMutex
	WebApps     map[string]*WebApp     // key: "namespace/name"
	Deployments map[string]*Deployment // key: "namespace/name"
	Services    map[string]*Service    // key: "namespace/name"
	Events      []Event                // 事件列表
}

// NewMockK8sClient 创建新的模拟客户端
func NewMockK8sClient() *MockK8sClient {
	return &MockK8sClient{
		WebApps:     make(map[string]*WebApp),
		Deployments: make(map[string]*Deployment),
		Services:    make(map[string]*Service),
		Events:      make([]Event, 0),
	}
}

// makeKey 生成资源的唯一键
func makeKey(namespace, name string) string {
	return fmt.Sprintf("%s/%s", namespace, name)
}

// ============================================================================
// WebApp 操作
// ============================================================================

// GetWebApp 获取 WebApp 资源
func (c *MockK8sClient) GetWebApp(namespace, name string) (*WebApp, error) {
	c.mu.RLock()
	defer c.mu.RUnlock()

	key := makeKey(namespace, name)
	webapp, exists := c.WebApps[key]
	if !exists {
		return nil, fmt.Errorf("webapp %s not found", key)
	}

	// 返回副本以避免外部修改
	return copyWebApp(webapp), nil
}

// UpdateWebAppStatus 更新 WebApp 的 Status 子资源
func (c *MockK8sClient) UpdateWebAppStatus(webapp *WebApp) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	key := makeKey(webapp.Namespace, webapp.Name)
	existing, exists := c.WebApps[key]
	if !exists {
		return fmt.Errorf("webapp %s not found", key)
	}

	// 只更新 Status 字段
	existing.Status = webapp.Status
	return nil
}

// UpdateWebApp 更新 WebApp 资源（包括 Spec 和 Metadata）
func (c *MockK8sClient) UpdateWebApp(webapp *WebApp) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	key := makeKey(webapp.Namespace, webapp.Name)
	_, exists := c.WebApps[key]
	if !exists {
		return fmt.Errorf("webapp %s not found", key)
	}

	// 更新整个资源
	c.WebApps[key] = copyWebApp(webapp)
	return nil
}

// CreateWebApp 创建 WebApp 资源（用于测试）
func (c *MockK8sClient) CreateWebApp(webapp *WebApp) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	key := makeKey(webapp.Namespace, webapp.Name)
	if _, exists := c.WebApps[key]; exists {
		return fmt.Errorf("webapp %s already exists", key)
	}

	// 初始化 Generation
	if webapp.Generation == 0 {
		webapp.Generation = 1
	}

	c.WebApps[key] = copyWebApp(webapp)
	return nil
}

// ============================================================================
// Deployment 操作
// ============================================================================

// GetDeployment 获取 Deployment 资源
func (c *MockK8sClient) GetDeployment(namespace, name string) (*Deployment, error) {
	c.mu.RLock()
	defer c.mu.RUnlock()

	key := makeKey(namespace, name)
	deployment, exists := c.Deployments[key]
	if !exists {
		return nil, fmt.Errorf("deployment %s not found", key)
	}

	return copyDeployment(deployment), nil
}

// CreateDeployment 创建 Deployment 资源
func (c *MockK8sClient) CreateDeployment(deployment *Deployment) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	key := makeKey(deployment.Namespace, deployment.Name)
	if _, exists := c.Deployments[key]; exists {
		return fmt.Errorf("deployment %s already exists", key)
	}

	c.Deployments[key] = copyDeployment(deployment)
	return nil
}

// UpdateDeployment 更新 Deployment 资源
func (c *MockK8sClient) UpdateDeployment(deployment *Deployment) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	key := makeKey(deployment.Namespace, deployment.Name)
	if _, exists := c.Deployments[key]; !exists {
		return fmt.Errorf("deployment %s not found", key)
	}

	c.Deployments[key] = copyDeployment(deployment)
	return nil
}

// DeleteDeployment 删除 Deployment 资源
func (c *MockK8sClient) DeleteDeployment(namespace, name string) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	key := makeKey(namespace, name)
	if _, exists := c.Deployments[key]; !exists {
		return fmt.Errorf("deployment %s not found", key)
	}

	delete(c.Deployments, key)
	return nil
}

// ============================================================================
// Service 操作
// ============================================================================

// GetService 获取 Service 资源
func (c *MockK8sClient) GetService(namespace, name string) (*Service, error) {
	c.mu.RLock()
	defer c.mu.RUnlock()

	key := makeKey(namespace, name)
	service, exists := c.Services[key]
	if !exists {
		return nil, fmt.Errorf("service %s not found", key)
	}

	return copyService(service), nil
}

// CreateService 创建 Service 资源
func (c *MockK8sClient) CreateService(service *Service) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	key := makeKey(service.Namespace, service.Name)
	if _, exists := c.Services[key]; exists {
		return fmt.Errorf("service %s already exists", key)
	}

	c.Services[key] = copyService(service)
	return nil
}

// UpdateService 更新 Service 资源
func (c *MockK8sClient) UpdateService(service *Service) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	key := makeKey(service.Namespace, service.Name)
	if _, exists := c.Services[key]; !exists {
		return fmt.Errorf("service %s not found", key)
	}

	c.Services[key] = copyService(service)
	return nil
}

// DeleteService 删除 Service 资源
func (c *MockK8sClient) DeleteService(namespace, name string) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	key := makeKey(namespace, name)
	if _, exists := c.Services[key]; !exists {
		return fmt.Errorf("service %s not found", key)
	}

	delete(c.Services, key)
	return nil
}

// ============================================================================
// Event 操作
// ============================================================================

// RecordEvent 记录事件
func (c *MockK8sClient) RecordEvent(event Event) {
	c.mu.Lock()
	defer c.mu.Unlock()

	event.Timestamp = time.Now()
	c.Events = append(c.Events, event)
}

// GetEvents 获取所有事件（用于测试和调试）
func (c *MockK8sClient) GetEvents() []Event {
	c.mu.RLock()
	defer c.mu.RUnlock()

	events := make([]Event, len(c.Events))
	copy(events, c.Events)
	return events
}

// ============================================================================
// 辅助函数：深拷贝
// ============================================================================

func copyWebApp(webapp *WebApp) *WebApp {
	if webapp == nil {
		return nil
	}

	copy := &WebApp{
		Name:       webapp.Name,
		Namespace:  webapp.Namespace,
		Generation: webapp.Generation,
		Spec:       webapp.Spec,
		Status:     webapp.Status,
		Finalizers: make([]string, len(webapp.Finalizers)),
	}

	// 拷贝 Finalizers
	for i, f := range webapp.Finalizers {
		copy.Finalizers[i] = f
	}

	// 拷贝 DeletionTimestamp
	if webapp.DeletionTimestamp != nil {
		ts := *webapp.DeletionTimestamp
		copy.DeletionTimestamp = &ts
	}

	// 拷贝 Env map
	if webapp.Spec.Env != nil {
		copy.Spec.Env = make(map[string]string)
		for k, v := range webapp.Spec.Env {
			copy.Spec.Env[k] = v
		}
	}

	// 拷贝 Conditions
	if webapp.Status.Conditions != nil {
		copy.Status.Conditions = make([]Condition, len(webapp.Status.Conditions))
		for i, c := range webapp.Status.Conditions {
			copy.Status.Conditions[i] = c
		}
	}

	return copy
}

func copyDeployment(deployment *Deployment) *Deployment {
	if deployment == nil {
		return nil
	}

	copy := &Deployment{
		Name:            deployment.Name,
		Namespace:       deployment.Namespace,
		Spec:            deployment.Spec,
		Status:          deployment.Status,
		OwnerReferences: make([]OwnerReference, len(deployment.OwnerReferences)),
	}

	// 拷贝 OwnerReferences
	for i, ref := range deployment.OwnerReferences {
		copy.OwnerReferences[i] = ref
	}

	// 拷贝 Env map
	if deployment.Spec.Env != nil {
		copy.Spec.Env = make(map[string]string)
		for k, v := range deployment.Spec.Env {
			copy.Spec.Env[k] = v
		}
	}

	return copy
}

func copyService(service *Service) *Service {
	if service == nil {
		return nil
	}

	copy := &Service{
		Name:            service.Name,
		Namespace:       service.Namespace,
		Spec:            service.Spec,
		OwnerReferences: make([]OwnerReference, len(service.OwnerReferences)),
	}

	// 拷贝 OwnerReferences
	for i, ref := range service.OwnerReferences {
		copy.OwnerReferences[i] = ref
	}

	// 拷贝 Selector map
	if service.Spec.Selector != nil {
		copy.Spec.Selector = make(map[string]string)
		for k, v := range service.Spec.Selector {
			copy.Spec.Selector[k] = v
		}
	}

	return copy
}
