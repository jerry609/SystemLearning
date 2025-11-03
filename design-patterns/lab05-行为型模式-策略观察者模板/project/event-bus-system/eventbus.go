package main

import (
	"fmt"
	"sync"
	"time"
)

// Event 事件结构
type Event struct {
	Type      string
	Data      interface{}
	Timestamp time.Time
}

// EventHandler 事件处理器接口
type EventHandler interface {
	Handle(event Event)
	GetID() string
}

// EventFilter 事件过滤器函数类型
type EventFilter func(event Event) bool

// subscription 订阅信息
type subscription struct {
	handler EventHandler
	filter  EventFilter
}

// EventBus 事件总线
type EventBus struct {
	subscribers map[string][]subscription
	mu          sync.RWMutex
	msgChan     chan Event
	stopChan    chan struct{}
	history     []Event
	historySize int
}

// NewEventBus 创建新的事件总线
func NewEventBus() *EventBus {
	bus := &EventBus{
		subscribers: make(map[string][]subscription),
		msgChan:     make(chan Event, 100),
		stopChan:    make(chan struct{}),
		history:     make([]Event, 0),
		historySize: 100,
	}
	
	// 启动消息处理协程
	go bus.processMessages()
	
	return bus
}

// Subscribe 订阅事件
func (eb *EventBus) Subscribe(eventType string, handler EventHandler) {
	eb.SubscribeWithFilter(eventType, handler, nil)
}

// SubscribeWithFilter 带过滤器的订阅
func (eb *EventBus) SubscribeWithFilter(eventType string, handler EventHandler, filter EventFilter) {
	eb.mu.Lock()
	defer eb.mu.Unlock()
	
	sub := subscription{
		handler: handler,
		filter:  filter,
	}
	
	eb.subscribers[eventType] = append(eb.subscribers[eventType], sub)
	
	if filter != nil {
		fmt.Printf("✓ [%s] 订阅了事件: %s (带过滤器)\n", handler.GetID(), eventType)
	} else {
		fmt.Printf("✓ [%s] 订阅了事件: %s\n", handler.GetID(), eventType)
	}
}

// Unsubscribe 取消订阅
func (eb *EventBus) Unsubscribe(eventType string, handler EventHandler) {
	eb.mu.Lock()
	defer eb.mu.Unlock()
	
	subs := eb.subscribers[eventType]
	for i, sub := range subs {
		if sub.handler.GetID() == handler.GetID() {
			eb.subscribers[eventType] = append(subs[:i], subs[i+1:]...)
			fmt.Printf("✗ [%s] 取消订阅事件: %s\n", handler.GetID(), eventType)
			break
		}
	}
}

// Publish 同步发布事件
func (eb *EventBus) Publish(event Event) {
	eb.msgChan <- event
}

// PublishAsync 异步发布事件
func (eb *EventBus) PublishAsync(event Event) {
	go func() {
		eb.msgChan <- event
	}()
}

// processMessages 处理消息队列
func (eb *EventBus) processMessages() {
	for {
		select {
		case event := <-eb.msgChan:
			eb.deliverEvent(event)
		case <-eb.stopChan:
			return
		}
	}
}

// deliverEvent 投递事件给订阅者
func (eb *EventBus) deliverEvent(event Event) {
	// 添加到历史记录
	eb.addToHistory(event)
	
	eb.mu.RLock()
	subs := make([]subscription, len(eb.subscribers[event.Type]))
	copy(subs, eb.subscribers[event.Type])
	eb.mu.RUnlock()
	
	fmt.Printf("\n📢 发布事件: %s\n", event.Type)
	fmt.Printf("   数据: %v\n", event.Data)
	fmt.Printf("   订阅者数量: %d\n\n", len(subs))
	
	// 并发投递给所有订阅者
	var wg sync.WaitGroup
	for _, sub := range subs {
		wg.Add(1)
		go func(s subscription) {
			defer wg.Done()
			
			// 应用过滤器
			if s.filter != nil && !s.filter(event) {
				fmt.Printf("  [%s] 事件被过滤\n", s.handler.GetID())
				return
			}
			
			// 处理事件
			s.handler.Handle(event)
		}(sub)
	}
	wg.Wait()
}

// addToHistory 添加事件到历史记录
func (eb *EventBus) addToHistory(event Event) {
	eb.mu.Lock()
	defer eb.mu.Unlock()
	
	eb.history = append(eb.history, event)
	
	// 限制历史记录大小
	if len(eb.history) > eb.historySize {
		eb.history = eb.history[1:]
	}
}

// GetHistory 获取事件历史
func (eb *EventBus) GetHistory() []Event {
	eb.mu.RLock()
	defer eb.mu.RUnlock()
	
	history := make([]Event, len(eb.history))
	copy(history, eb.history)
	return history
}

// GetSubscriberCount 获取订阅者数量
func (eb *EventBus) GetSubscriberCount(eventType string) int {
	eb.mu.RLock()
	defer eb.mu.RUnlock()
	return len(eb.subscribers[eventType])
}

// Stop 停止事件总线
func (eb *EventBus) Stop() {
	close(eb.stopChan)
	close(eb.msgChan)
}

// 全局事件总线实例（单例模式）
var (
	globalBus     *EventBus
	globalBusOnce sync.Once
)

// GlobalEventBus 获取全局事件总线实例
func GlobalEventBus() *EventBus {
	globalBusOnce.Do(func() {
		globalBus = NewEventBus()
	})
	return globalBus
}
