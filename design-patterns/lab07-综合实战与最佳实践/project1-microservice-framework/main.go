package main

import (
	"fmt"
	"time"
)

func main() {
	fmt.Println("=== 微服务框架演示 ===\n")

	// 1. 服务注册与发现演示
	demonstrateServiceRegistry()

	// 2. 中间件链演示
	demonstrateMiddlewareChain()

	// 3. 事件系统演示
	demonstrateEventBus()

	// 4. 负载均衡演示
	demonstrateLoadBalancer()

	fmt.Println("\n=== 演示完成 ===")
}

// demonstrateServiceRegistry 演示服务注册与发现
func demonstrateServiceRegistry() {
	fmt.Println("--- 1. 服务注册与发现 ---")

	// 创建服务工厂和注册中心
	factory := NewServiceFactory("http")
	registry := NewServiceRegistry(factory)

	// 注册服务
	registry.Register("user-service", "localhost:8001")
	fmt.Println("注册服务: user-service @ localhost:8001")

	registry.Register("user-service", "localhost:8002")
	fmt.Println("注册服务: user-service @ localhost:8002")

	registry.Register("order-service", "localhost:9001")
	fmt.Println("注册服务: order-service @ localhost:9001")

	// 发现服务
	userServices := registry.Discover("user-service")
	fmt.Printf("发现服务 user-service: %d 个实例\n", len(userServices))

	orderServices := registry.Discover("order-service")
	fmt.Printf("发现服务 order-service: %d 个实例\n", len(orderServices))

	fmt.Println()
}

// demonstrateMiddlewareChain 演示中间件链
func demonstrateMiddlewareChain() {
	fmt.Println("--- 2. 中间件链 ---")

	// 创建业务处理器
	handler := func(ctx *Context) error {
		fmt.Printf("[业务处理器] 处理请求: %s\n", ctx.Path)
		ctx.Data["result"] = "success"
		return nil
	}

	// 创建中间件链
	chain := NewMiddlewareChain(handler)
	chain.Use(&LoggingMiddleware{})
	chain.Use(&AuthMiddleware{})
	chain.Use(NewRateLimitMiddleware(10, time.Minute))

	// 创建请求上下文
	ctx := NewContext("/api/users")
	ctx.Token = "token-123"

	// 执行中间件链
	if err := chain.Execute(ctx); err != nil {
		fmt.Printf("错误: %v\n", err)
	}

	fmt.Println()
}

// demonstrateEventBus 演示事件系统
func demonstrateEventBus() {
	fmt.Println("--- 3. 事件系统 ---")

	// 创建事件总线
	bus := NewEventBus()

	// 订阅事件
	emailHandler := NewEmailEventHandler("email-handler")
	logHandler := NewLogEventHandler("log-handler")

	bus.Subscribe("UserCreated", emailHandler)
	bus.Subscribe("UserCreated", logHandler)
	bus.Subscribe("OrderPlaced", emailHandler)
	bus.Subscribe("OrderPlaced", logHandler)

	// 发布事件
	userEvent := NewEvent("UserCreated", map[string]interface{}{
		"userId": "user-001",
		"email":  "user@example.com",
	})
	fmt.Println("发布事件: UserCreated")
	bus.Publish(userEvent)

	orderEvent := NewEvent("OrderPlaced", map[string]interface{}{
		"orderId": "order-001",
		"amount":  99.99,
	})
	fmt.Println("发布事件: OrderPlaced")
	bus.Publish(orderEvent)

	fmt.Println()
}

// demonstrateLoadBalancer 演示负载均衡
func demonstrateLoadBalancer() {
	fmt.Println("--- 4. 负载均衡 ---")

	// 创建服务列表
	factory := NewServiceFactory("http")
	registry := NewServiceRegistry(factory)
	registry.Register("api-service", "localhost:8001")
	registry.Register("api-service", "localhost:8002")
	services := registry.Discover("api-service")

	// 1. 轮询策略
	fmt.Println("轮询策略:")
	roundRobin := NewRoundRobinBalancer()
	for i := 0; i < 3; i++ {
		service := roundRobin.Select(services)
		fmt.Printf("  选择服务: %s\n", service.Address())
	}

	// 2. 随机策略
	fmt.Println("随机策略:")
	random := NewRandomBalancer()
	for i := 0; i < 3; i++ {
		service := random.Select(services)
		fmt.Printf("  选择服务: %s\n", service.Address())
	}

	// 3. 加权策略
	fmt.Println("加权策略:")
	weightedServices := []Service{
		NewWeightedService("api-service", "localhost:8001", 3),
		NewWeightedService("api-service", "localhost:8002", 1),
	}
	weighted := NewWeightedBalancer()
	for i := 0; i < 3; i++ {
		service := weighted.Select(weightedServices)
		if ws, ok := service.(*WeightedServiceImpl); ok {
			fmt.Printf("  选择服务: %s (权重: %d)\n", ws.Address(), ws.Weight())
		}
	}

	fmt.Println()
}
