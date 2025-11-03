package main

import (
	"sync"
	"testing"
	"time"
)

// 测试用的事件处理器
type TestHandler struct {
	id       string
	received []Event
	mu       sync.Mutex
}

func NewTestHandler(id string) *TestHandler {
	return &TestHandler{
		id:       id,
		received: make([]Event, 0),
	}
}

func (t *TestHandler) Handle(event Event) {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.received = append(t.received, event)
}

func (t *TestHandler) GetID() string {
	return t.id
}

func (t *TestHandler) GetReceivedCount() int {
	t.mu.Lock()
	defer t.mu.Unlock()
	return len(t.received)
}

// 测试基本订阅和发布
func TestBasicSubscribeAndPublish(t *testing.T) {
	bus := NewEventBus()
	defer bus.Stop()
	
	handler := NewTestHandler("test-handler")
	bus.Subscribe("test.event", handler)
	
	event := Event{
		Type:      "test.event",
		Data:      "test data",
		Timestamp: time.Now(),
	}
	
	bus.Publish(event)
	time.Sleep(100 * time.Millisecond)
	
	if handler.GetReceivedCount() != 1 {
		t.Errorf("Expected 1 event, got %d", handler.GetReceivedCount())
	}
}

// 测试多个订阅者
func TestMultipleSubscribers(t *testing.T) {
	bus := NewEventBus()
	defer bus.Stop()
	
	handler1 := NewTestHandler("handler-1")
	handler2 := NewTestHandler("handler-2")
	handler3 := NewTestHandler("handler-3")
	
	bus.Subscribe("test.event", handler1)
	bus.Subscribe("test.event", handler2)
	bus.Subscribe("test.event", handler3)
	
	event := Event{
		Type:      "test.event",
		Data:      "test data",
		Timestamp: time.Now(),
	}
	
	bus.Publish(event)
	time.Sleep(100 * time.Millisecond)
	
	if handler1.GetReceivedCount() != 1 {
		t.Errorf("Handler1: Expected 1 event, got %d", handler1.GetReceivedCount())
	}
	if handler2.GetReceivedCount() != 1 {
		t.Errorf("Handler2: Expected 1 event, got %d", handler2.GetReceivedCount())
	}
	if handler3.GetReceivedCount() != 1 {
		t.Errorf("Handler3: Expected 1 event, got %d", handler3.GetReceivedCount())
	}
}

// 测试取消订阅
func TestUnsubscribe(t *testing.T) {
	bus := NewEventBus()
	defer bus.Stop()
	
	handler := NewTestHandler("test-handler")
	bus.Subscribe("test.event", handler)
	
	event := Event{
		Type:      "test.event",
		Data:      "test data",
		Timestamp: time.Now(),
	}
	
	bus.Publish(event)
	time.Sleep(100 * time.Millisecond)
	
	if handler.GetReceivedCount() != 1 {
		t.Errorf("Expected 1 event before unsubscribe, got %d", handler.GetReceivedCount())
	}
	
	bus.Unsubscribe("test.event", handler)
	
	bus.Publish(event)
	time.Sleep(100 * time.Millisecond)
	
	if handler.GetReceivedCount() != 1 {
		t.Errorf("Expected 1 event after unsubscribe, got %d", handler.GetReceivedCount())
	}
}

// 测试事件过滤
func TestEventFilter(t *testing.T) {
	bus := NewEventBus()
	defer bus.Stop()
	
	handler := NewTestHandler("test-handler")
	
	// 只接收 priority 为 "high" 的事件
	filter := func(event Event) bool {
		if data, ok := event.Data.(map[string]interface{}); ok {
			if priority, exists := data["priority"]; exists {
				return priority == "high"
			}
		}
		return false
	}
	
	bus.SubscribeWithFilter("test.event", handler, filter)
	
	// 发布低优先级事件
	bus.Publish(Event{
		Type: "test.event",
		Data: map[string]interface{}{
			"priority": "low",
		},
		Timestamp: time.Now(),
	})
	time.Sleep(100 * time.Millisecond)
	
	if handler.GetReceivedCount() != 0 {
		t.Errorf("Expected 0 events for low priority, got %d", handler.GetReceivedCount())
	}
	
	// 发布高优先级事件
	bus.Publish(Event{
		Type: "test.event",
		Data: map[string]interface{}{
			"priority": "high",
		},
		Timestamp: time.Now(),
	})
	time.Sleep(100 * time.Millisecond)
	
	if handler.GetReceivedCount() != 1 {
		t.Errorf("Expected 1 event for high priority, got %d", handler.GetReceivedCount())
	}
}

// 测试异步发布
func TestAsyncPublish(t *testing.T) {
	bus := NewEventBus()
	defer bus.Stop()
	
	handler := NewTestHandler("test-handler")
	bus.Subscribe("test.event", handler)
	
	// 异步发布多个事件
	for i := 0; i < 10; i++ {
		bus.PublishAsync(Event{
			Type:      "test.event",
			Data:      i,
			Timestamp: time.Now(),
		})
	}
	
	time.Sleep(500 * time.Millisecond)
	
	if handler.GetReceivedCount() != 10 {
		t.Errorf("Expected 10 events, got %d", handler.GetReceivedCount())
	}
}

// 测试事件历史
func TestEventHistory(t *testing.T) {
	bus := NewEventBus()
	defer bus.Stop()
	
	handler := NewTestHandler("test-handler")
	bus.Subscribe("test.event", handler)
	
	// 发布多个事件
	for i := 0; i < 5; i++ {
		bus.Publish(Event{
			Type:      "test.event",
			Data:      i,
			Timestamp: time.Now(),
		})
	}
	
	time.Sleep(200 * time.Millisecond)
	
	history := bus.GetHistory()
	if len(history) != 5 {
		t.Errorf("Expected 5 events in history, got %d", len(history))
	}
}

// 测试订阅者数量
func TestGetSubscriberCount(t *testing.T) {
	bus := NewEventBus()
	defer bus.Stop()
	
	handler1 := NewTestHandler("handler-1")
	handler2 := NewTestHandler("handler-2")
	
	bus.Subscribe("test.event", handler1)
	bus.Subscribe("test.event", handler2)
	
	count := bus.GetSubscriberCount("test.event")
	if count != 2 {
		t.Errorf("Expected 2 subscribers, got %d", count)
	}
	
	bus.Unsubscribe("test.event", handler1)
	
	count = bus.GetSubscriberCount("test.event")
	if count != 1 {
		t.Errorf("Expected 1 subscriber after unsubscribe, got %d", count)
	}
}

// 测试全局事件总线
func TestGlobalEventBus(t *testing.T) {
	bus1 := GlobalEventBus()
	bus2 := GlobalEventBus()
	
	if bus1 != bus2 {
		t.Error("GlobalEventBus should return the same instance")
	}
}

// 并发测试
func TestConcurrentPublish(t *testing.T) {
	bus := NewEventBus()
	defer bus.Stop()
	
	handler := NewTestHandler("test-handler")
	bus.Subscribe("test.event", handler)
	
	var wg sync.WaitGroup
	eventCount := 100
	
	// 并发发布事件
	for i := 0; i < eventCount; i++ {
		wg.Add(1)
		go func(index int) {
			defer wg.Done()
			bus.Publish(Event{
				Type:      "test.event",
				Data:      index,
				Timestamp: time.Now(),
			})
		}(i)
	}
	
	wg.Wait()
	time.Sleep(500 * time.Millisecond)
	
	if handler.GetReceivedCount() != eventCount {
		t.Errorf("Expected %d events, got %d", eventCount, handler.GetReceivedCount())
	}
}

// 性能测试
func BenchmarkPublish(b *testing.B) {
	bus := NewEventBus()
	defer bus.Stop()
	
	handler := NewTestHandler("test-handler")
	bus.Subscribe("test.event", handler)
	
	event := Event{
		Type:      "test.event",
		Data:      "test data",
		Timestamp: time.Now(),
	}
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		bus.Publish(event)
	}
}

// 性能测试：多订阅者
func BenchmarkPublishMultipleSubscribers(b *testing.B) {
	bus := NewEventBus()
	defer bus.Stop()
	
	// 添加 10 个订阅者
	for i := 0; i < 10; i++ {
		handler := NewTestHandler("handler-" + string(rune(i)))
		bus.Subscribe("test.event", handler)
	}
	
	event := Event{
		Type:      "test.event",
		Data:      "test data",
		Timestamp: time.Now(),
	}
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		bus.Publish(event)
	}
}
