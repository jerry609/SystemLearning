package main

import (
	"fmt"
	"time"
)

// 日志装饰器示例
// 本示例展示了如何使用装饰器模式为服务添加日志功能
// 演示了接口装饰器的实现方式

// Service 服务接口
// 定义了基本的服务操作
type Service interface {
	Execute(data string) (string, error)
	Query(id int) (string, error)
	Update(id int, data string) error
}

// BasicService 基础服务实现
type BasicService struct {
	name string
}

// NewBasicService 创建基础服务
func NewBasicService(name string) *BasicService {
	return &BasicService{name: name}
}

func (s *BasicService) Execute(data string) (string, error) {
	// 模拟处理
	time.Sleep(100 * time.Millisecond)
	return fmt.Sprintf("[%s] Processed: %s", s.name, data), nil
}

func (s *BasicService) Query(id int) (string, error) {
	// 模拟查询
	time.Sleep(50 * time.Millisecond)
	return fmt.Sprintf("[%s] Record #%d", s.name, id), nil
}

func (s *BasicService) Update(id int, data string) error {
	// 模拟更新
	time.Sleep(80 * time.Millisecond)
	return nil
}

// LoggingDecorator 日志装饰器
// 为服务添加日志记录功能
type LoggingDecorator struct {
	service Service
	prefix  string
}

// NewLoggingDecorator 创建日志装饰器
func NewLoggingDecorator(service Service, prefix string) *LoggingDecorator {
	return &LoggingDecorator{
		service: service,
		prefix:  prefix,
	}
}

func (d *LoggingDecorator) Execute(data string) (string, error) {
	start := time.Now()
	fmt.Printf("[%s] [INFO] Execute called with data: %s\n", d.prefix, data)

	result, err := d.service.Execute(data)
	duration := time.Since(start)

	if err != nil {
		fmt.Printf("[%s] [ERROR] Execute failed in %v: %v\n", d.prefix, duration, err)
	} else {
		fmt.Printf("[%s] [INFO] Execute succeeded in %v, result: %s\n", d.prefix, duration, result)
	}

	return result, err
}

func (d *LoggingDecorator) Query(id int) (string, error) {
	start := time.Now()
	fmt.Printf("[%s] [INFO] Query called with id: %d\n", d.prefix, id)

	result, err := d.service.Query(id)
	duration := time.Since(start)

	if err != nil {
		fmt.Printf("[%s] [ERROR] Query failed in %v: %v\n", d.prefix, duration, err)
	} else {
		fmt.Printf("[%s] [INFO] Query succeeded in %v, result: %s\n", d.prefix, duration, result)
	}

	return result, err
}

func (d *LoggingDecorator) Update(id int, data string) error {
	start := time.Now()
	fmt.Printf("[%s] [INFO] Update called with id: %d, data: %s\n", d.prefix, id, data)

	err := d.service.Update(id, data)
	duration := time.Since(start)

	if err != nil {
		fmt.Printf("[%s] [ERROR] Update failed in %v: %v\n", d.prefix, duration, err)
	} else {
		fmt.Printf("[%s] [INFO] Update succeeded in %v\n", d.prefix, duration)
	}

	return err
}

// MetricsDecorator 性能监控装饰器
// 记录方法执行时间和调用次数
type MetricsDecorator struct {
	service      Service
	executeCalls int
	queryCalls   int
	updateCalls  int
	totalTime    time.Duration
}

// NewMetricsDecorator 创建性能监控装饰器
func NewMetricsDecorator(service Service) *MetricsDecorator {
	return &MetricsDecorator{service: service}
}

func (d *MetricsDecorator) Execute(data string) (string, error) {
	start := time.Now()
	result, err := d.service.Execute(data)
	duration := time.Since(start)

	d.executeCalls++
	d.totalTime += duration
	fmt.Printf("[METRICS] Execute: calls=%d, duration=%v\n", d.executeCalls, duration)

	return result, err
}

func (d *MetricsDecorator) Query(id int) (string, error) {
	start := time.Now()
	result, err := d.service.Query(id)
	duration := time.Since(start)

	d.queryCalls++
	d.totalTime += duration
	fmt.Printf("[METRICS] Query: calls=%d, duration=%v\n", d.queryCalls, duration)

	return result, err
}

func (d *MetricsDecorator) Update(id int, data string) error {
	start := time.Now()
	err := d.service.Update(id, data)
	duration := time.Since(start)

	d.updateCalls++
	d.totalTime += duration
	fmt.Printf("[METRICS] Update: calls=%d, duration=%v\n", d.updateCalls, duration)

	return err
}

// PrintStats 打印统计信息
func (d *MetricsDecorator) PrintStats() {
	fmt.Println("\n=== 性能统计 ===")
	fmt.Printf("Execute 调用次数: %d\n", d.executeCalls)
	fmt.Printf("Query 调用次数: %d\n", d.queryCalls)
	fmt.Printf("Update 调用次数: %d\n", d.updateCalls)
	fmt.Printf("总调用次数: %d\n", d.executeCalls+d.queryCalls+d.updateCalls)
	fmt.Printf("总耗时: %v\n", d.totalTime)
	if total := d.executeCalls + d.queryCalls + d.updateCalls; total > 0 {
		fmt.Printf("平均耗时: %v\n", d.totalTime/time.Duration(total))
	}
}

// ErrorHandlingDecorator 错误处理装饰器
// 为服务添加错误处理和重试功能
type ErrorHandlingDecorator struct {
	service    Service
	maxRetries int
}

// NewErrorHandlingDecorator 创建错误处理装饰器
func NewErrorHandlingDecorator(service Service, maxRetries int) *ErrorHandlingDecorator {
	return &ErrorHandlingDecorator{
		service:    service,
		maxRetries: maxRetries,
	}
}

func (d *ErrorHandlingDecorator) Execute(data string) (string, error) {
	var err error
	var result string

	for i := 0; i <= d.maxRetries; i++ {
		if i > 0 {
			fmt.Printf("[ERROR_HANDLER] Retry attempt %d/%d for Execute\n", i, d.maxRetries)
			time.Sleep(time.Duration(i*100) * time.Millisecond)
		}

		result, err = d.service.Execute(data)
		if err == nil {
			return result, nil
		}
	}

	return "", fmt.Errorf("failed after %d retries: %w", d.maxRetries, err)
}

func (d *ErrorHandlingDecorator) Query(id int) (string, error) {
	var err error
	var result string

	for i := 0; i <= d.maxRetries; i++ {
		if i > 0 {
			fmt.Printf("[ERROR_HANDLER] Retry attempt %d/%d for Query\n", i, d.maxRetries)
			time.Sleep(time.Duration(i*100) * time.Millisecond)
		}

		result, err = d.service.Query(id)
		if err == nil {
			return result, nil
		}
	}

	return "", fmt.Errorf("failed after %d retries: %w", d.maxRetries, err)
}

func (d *ErrorHandlingDecorator) Update(id int, data string) error {
	var err error

	for i := 0; i <= d.maxRetries; i++ {
		if i > 0 {
			fmt.Printf("[ERROR_HANDLER] Retry attempt %d/%d for Update\n", i, d.maxRetries)
			time.Sleep(time.Duration(i*100) * time.Millisecond)
		}

		err = d.service.Update(id, data)
		if err == nil {
			return nil
		}
	}

	return fmt.Errorf("failed after %d retries: %w", d.maxRetries, err)
}

func main() {
	fmt.Println("=== 日志装饰器示例 ===\n")

	// 示例 1: 基础服务 + 日志装饰器
	fmt.Println("--- 示例 1: 基础服务 + 日志装饰器 ---")
	service1 := NewBasicService("UserService")
	loggedService := NewLoggingDecorator(service1, "LOG")

	result, _ := loggedService.Execute("user data")
	fmt.Printf("返回结果: %s\n\n", result)

	result, _ = loggedService.Query(123)
	fmt.Printf("返回结果: %s\n\n", result)

	_ = loggedService.Update(123, "updated data")
	fmt.Println()

	// 示例 2: 基础服务 + 性能监控装饰器
	fmt.Println("--- 示例 2: 基础服务 + 性能监控装饰器 ---")
	service2 := NewBasicService("OrderService")
	metricsService := NewMetricsDecorator(service2)

	metricsService.Execute("order data")
	metricsService.Query(456)
	metricsService.Update(456, "updated order")
	metricsService.Execute("another order")
	metricsService.Query(789)

	metricsService.PrintStats()
	fmt.Println()

	// 示例 3: 多层装饰器组合
	fmt.Println("--- 示例 3: 多层装饰器组合 ---")
	fmt.Println("组合顺序: 基础服务 -> 错误处理 -> 性能监控 -> 日志")
	service3 := NewBasicService("PaymentService")

	// 逐层装饰
	var decoratedService Service = service3
	decoratedService = NewErrorHandlingDecorator(decoratedService, 2)
	metricsDecorator := NewMetricsDecorator(decoratedService)
	decoratedService = metricsDecorator
	decoratedService = NewLoggingDecorator(decoratedService, "PAYMENT")

	// 使用装饰后的服务
	result, _ = decoratedService.Execute("payment data")
	fmt.Printf("最终结果: %s\n\n", result)

	result, _ = decoratedService.Query(999)
	fmt.Printf("最终结果: %s\n\n", result)

	_ = decoratedService.Update(999, "payment updated")

	metricsDecorator.PrintStats()

	fmt.Println("\n=== 示例结束 ===")
	fmt.Println("\n关键点:")
	fmt.Println("1. 装饰器保持与原始服务相同的接口")
	fmt.Println("2. 可以在不修改原始代码的情况下添加新功能")
	fmt.Println("3. 可以灵活组合多个装饰器")
	fmt.Println("4. 装饰器的顺序会影响最终行为")
}
