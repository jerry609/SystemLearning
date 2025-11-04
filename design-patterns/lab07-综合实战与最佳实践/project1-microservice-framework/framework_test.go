package main

import (
	"testing"
	"time"
)

// TestServiceRegistry 测试服务注册与发现
func TestServiceRegistry(t *testing.T) {
	factory := NewServiceFactory("http")
	registry := NewServiceRegistry(factory)

	// 测试注册服务
	registry.Register("test-service", "localhost:8001")
	registry.Register("test-service", "localhost:8002")

	// 测试发现服务
	services := registry.Discover("test-service")
	if len(services) != 2 {
		t.Errorf("期望 2 个服务实例，实际 %d 个", len(services))
	}

	// 测试注销服务
	registry.Deregister("test-service", "localhost:8001")
	services = registry.Discover("test-service")
	if len(services) != 1 {
		t.Errorf("期望 1 个服务实例，实际 %d 个", len(services))
	}
}

// TestMiddlewareChain 测试中间件链
func TestMiddlewareChain(t *testing.T) {
	executed := false
	handler := func(ctx *Context) error {
		executed = true
		return nil
	}

	chain := NewMiddlewareChain(handler)
	chain.Use(&LoggingMiddleware{})

	ctx := NewContext("/test")
	err := chain.Execute(ctx)

	if err != nil {
		t.Errorf("中间件链执行失败: %v", err)
	}

	if !executed {
		t.Error("处理器未被执行")
	}
}

// TestAuthMiddleware 测试认证中间件
func TestAuthMiddleware(t *testing.T) {
	handler := func(ctx *Context) error {
		return nil
	}

	chain := NewMiddlewareChain(handler)
	chain.Use(&AuthMiddleware{})

	// 测试无令牌
	ctx := NewContext("/test")
	err := chain.Execute(ctx)
	if err == nil {
		t.Error("期望认证失败，但成功了")
	}

	// 测试有效令牌
	ctx.Token = "token-123"
	err = chain.Execute(ctx)
	if err != nil {
		t.Errorf("认证失败: %v", err)
	}

	if ctx.UserID != "user-001" {
		t.Errorf("期望 UserID 为 user-001，实际 %s", ctx.UserID)
	}
}

// TestRateLimitMiddleware 测试限流中间件
func TestRateLimitMiddleware(t *testing.T) {
	handler := func(ctx *Context) error {
		return nil
	}

	chain := NewMiddlewareChain(handler)
	rateLimiter := NewRateLimitMiddleware(2, time.Second)
	chain.Use(rateLimiter)

	ctx := NewContext("/test")
	ctx.UserID = "test-user"

	// 前两次请求应该成功
	for i := 0; i < 2; i++ {
		err := chain.Execute(ctx)
		if err != nil {
			t.Errorf("请求 %d 失败: %v", i+1, err)
		}
	}

	// 第三次请求应该被限流
	err := chain.Execute(ctx)
	if err == nil {
		t.Error("期望被限流，但请求成功了")
	}
}

// TestEventBus 测试事件总线
func TestEventBus(t *testing.T) {
	bus := NewEventBus()

	handled := false
	handler := EventHandlerFunc(func(event Event) error {
		handled = true
		return nil
	})

	bus.Subscribe("TestEvent", handler)

	event := NewEvent("TestEvent", "test data")
	err := bus.Publish(event)

	if err != nil {
		t.Errorf("发布事件失败: %v", err)
	}

	if !handled {
		t.Error("事件未被处理")
	}
}

// TestRoundRobinBalancer 测试轮询负载均衡
func TestRoundRobinBalancer(t *testing.T) {
	factory := NewServiceFactory("http")
	registry := NewServiceRegistry(factory)
	registry.Register("test-service", "localhost:8001")
	registry.Register("test-service", "localhost:8002")
	services := registry.Discover("test-service")

	balancer := NewRoundRobinBalancer()

	// 测试轮询
	service1 := balancer.Select(services)
	service2 := balancer.Select(services)
	service3 := balancer.Select(services)

	if service1.Address() != "localhost:8001" {
		t.Errorf("期望第一个服务是 localhost:8001，实际 %s", service1.Address())
	}

	if service2.Address() != "localhost:8002" {
		t.Errorf("期望第二个服务是 localhost:8002，实际 %s", service2.Address())
	}

	if service3.Address() != "localhost:8001" {
		t.Errorf("期望第三个服务是 localhost:8001，实际 %s", service3.Address())
	}
}

// TestRandomBalancer 测试随机负载均衡
func TestRandomBalancer(t *testing.T) {
	factory := NewServiceFactory("http")
	registry := NewServiceRegistry(factory)
	registry.Register("test-service", "localhost:8001")
	registry.Register("test-service", "localhost:8002")
	services := registry.Discover("test-service")

	balancer := NewRandomBalancer()

	// 测试随机选择
	for i := 0; i < 10; i++ {
		service := balancer.Select(services)
		if service == nil {
			t.Error("选择的服务为 nil")
		}
	}
}

// TestWeightedBalancer 测试加权负载均衡
func TestWeightedBalancer(t *testing.T) {
	services := []Service{
		NewWeightedService("test-service", "localhost:8001", 3),
		NewWeightedService("test-service", "localhost:8002", 1),
	}

	balancer := NewWeightedBalancer()

	// 统计选择次数
	counts := make(map[string]int)
	for i := 0; i < 100; i++ {
		service := balancer.Select(services)
		counts[service.Address()]++
	}

	// 验证权重比例（允许一定误差）
	ratio := float64(counts["localhost:8001"]) / float64(counts["localhost:8002"])
	if ratio < 2.0 || ratio > 4.0 {
		t.Errorf("权重比例不正确，期望约 3:1，实际 %d:%d", counts["localhost:8001"], counts["localhost:8002"])
	}
}

// TestLoadBalancerFactory 测试负载均衡器工厂
func TestLoadBalancerFactory(t *testing.T) {
	factory := &LoadBalancerFactory{}

	balancers := []struct {
		strategy string
		expected string
	}{
		{"round-robin", "*main.RoundRobinBalancer"},
		{"random", "*main.RandomBalancer"},
		{"weighted", "*main.WeightedBalancer"},
		{"least-connection", "*main.LeastConnectionBalancer"},
	}

	for _, tc := range balancers {
		balancer := factory.CreateBalancer(tc.strategy)
		if balancer == nil {
			t.Errorf("创建 %s 负载均衡器失败", tc.strategy)
		}
	}
}
