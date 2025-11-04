package main

import (
	"fmt"
	"sync"
)

// 事件系统 - 观察者模式
//
// 本模块使用观察者模式实现事件驱动架构
// 支持事件发布和订阅，实现组件间的解耦

// Event 事件接口
type Event interface {
	Type() string
	Data() interface{}
}

// BaseEvent 基础事件实现
type BaseEvent struct {
	eventType string
	data      interface{}
}

func NewEvent(eventType string, data interface{}) Event {
	return &BaseEvent{
		eventType: eventType,
		data:      data,
	}
}

func (e *BaseEvent) Type() string {
	return e.eventType
}

func (e *BaseEvent) Data() interface{} {
	return e.data
}

// EventHandler 事件处理器接口
type EventHandler interface {
	Handle(event Event) error
}

// EventHandlerFunc 事件处理器函数类型
type EventHandlerFunc func(event Event) error

func (f EventHandlerFunc) Handle(event Event) error {
	return f(event)
}

// EventBus 事件总线
type EventBus struct {
	mu          sync.RWMutex
	subscribers map[string][]EventHandler
}

func NewEventBus() *EventBus {
	return &EventBus{
		subscribers: make(map[string][]EventHandler),
	}
}

// Subscribe 订阅事件
func (b *EventBus) Subscribe(eventType string, handler EventHandler) {
	b.mu.Lock()
	defer b.mu.Unlock()

	b.subscribers[eventType] = append(b.subscribers[eventType], handler)
}

// Unsubscribe 取消订阅
func (b *EventBus) Unsubscribe(eventType string, handler EventHandler) {
	b.mu.Lock()
	defer b.mu.Unlock()

	handlers := b.subscribers[eventType]
	for i, h := range handlers {
		// 注意：这里的比较可能不准确，实际应用中可能需要使用 ID
		if fmt.Sprintf("%p", h) == fmt.Sprintf("%p", handler) {
			b.subscribers[eventType] = append(handlers[:i], handlers[i+1:]...)
			break
		}
	}
}

// Publish 发布事件（同步）
func (b *EventBus) Publish(event Event) error {
	b.mu.RLock()
	handlers := b.subscribers[event.Type()]
	b.mu.RUnlock()

	for _, handler := range handlers {
		if err := handler.Handle(event); err != nil {
			return err
		}
	}

	return nil
}

// PublishAsync 发布事件（异步）
func (b *EventBus) PublishAsync(event Event) {
	b.mu.RLock()
	handlers := b.subscribers[event.Type()]
	b.mu.RUnlock()

	for _, handler := range handlers {
		go func(h EventHandler) {
			_ = h.Handle(event)
		}(handler)
	}
}

// EmailEventHandler 邮件事件处理器
type EmailEventHandler struct {
	name string
}

func NewEmailEventHandler(name string) *EmailEventHandler {
	return &EmailEventHandler{name: name}
}

func (h *EmailEventHandler) Handle(event Event) error {
	switch event.Type() {
	case "UserCreated":
		fmt.Printf("[邮件处理器] 处理事件: %s - 发送欢迎邮件\n", event.Type())
	case "OrderPlaced":
		fmt.Printf("[邮件处理器] 处理事件: %s - 发送订单确认邮件\n", event.Type())
	default:
		fmt.Printf("[邮件处理器] 处理事件: %s\n", event.Type())
	}
	return nil
}

// LogEventHandler 日志事件处理器
type LogEventHandler struct {
	name string
}

func NewLogEventHandler(name string) *LogEventHandler {
	return &LogEventHandler{name: name}
}

func (h *LogEventHandler) Handle(event Event) error {
	switch event.Type() {
	case "UserCreated":
		fmt.Printf("[日志处理器] 处理事件: %s - 记录用户创建日志\n", event.Type())
	case "OrderPlaced":
		fmt.Printf("[日志处理器] 处理事件: %s - 记录订单日志\n", event.Type())
	default:
		fmt.Printf("[日志处理器] 处理事件: %s - 数据: %v\n", event.Type(), event.Data())
	}
	return nil
}

// MetricsEventHandler 指标事件处理器
type MetricsEventHandler struct {
	name string
}

func NewMetricsEventHandler(name string) *MetricsEventHandler {
	return &MetricsEventHandler{name: name}
}

func (h *MetricsEventHandler) Handle(event Event) error {
	fmt.Printf("[指标处理器] 处理事件: %s - 更新指标\n", event.Type())
	return nil
}
