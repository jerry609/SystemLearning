package main

import (
	"fmt"
	"sync"
	"time"
)

// 事件类型
type EventType string

const (
	UserRegistered EventType = "user.registered"
	UserLoggedIn   EventType = "user.logged_in"
	OrderCreated   EventType = "order.created"
	OrderPaid      EventType = "order.paid"
	OrderShipped   EventType = "order.shipped"
)

// 事件数据
type Event struct {
	Type      EventType
	Data      interface{}
	Timestamp time.Time
}

// 事件处理器接口
type EventHandler interface {
	Handle(event Event)
	GetID() string
}

// 事件总线
type EventBus struct {
	handlers map[EventType][]EventHandler
	mu       sync.RWMutex
}

func NewEventBus() *EventBus {
	return &EventBus{
		handlers: make(map[EventType][]EventHandler),
	}
}

// 订阅事件
func (eb *EventBus) Subscribe(eventType EventType, handler EventHandler) {
	eb.mu.Lock()
	defer eb.mu.Unlock()
	
	eb.handlers[eventType] = append(eb.handlers[eventType], handler)
	fmt.Printf("✓ [%s] 订阅了事件: %s\n", handler.GetID(), eventType)
}

// 取消订阅
func (eb *EventBus) Unsubscribe(eventType EventType, handler EventHandler) {
	eb.mu.Lock()
	defer eb.mu.Unlock()
	
	handlers := eb.handlers[eventType]
	for i, h := range handlers {
		if h.GetID() == handler.GetID() {
			eb.handlers[eventType] = append(handlers[:i], handlers[i+1:]...)
			fmt.Printf("✗ [%s] 取消订阅事件: %s\n", handler.GetID(), eventType)
			break
		}
	}
}

// 发布事件
func (eb *EventBus) Publish(event Event) {
	eb.mu.RLock()
	handlers := make([]EventHandler, len(eb.handlers[event.Type]))
	copy(handlers, eb.handlers[event.Type])
	eb.mu.RUnlock()
	
	fmt.Printf("\n📢 发布事件: %s (时间: %s)\n", event.Type, event.Timestamp.Format("15:04:05"))
	fmt.Printf("   数据: %v\n", event.Data)
	fmt.Printf("   通知 %d 个订阅者...\n\n", len(handlers))
	
	for _, handler := range handlers {
		handler.Handle(event)
	}
}

// 异步发布事件
func (eb *EventBus) PublishAsync(event Event) {
	go eb.Publish(event)
}

// 邮件通知处理器
type EmailHandler struct {
	id    string
	email string
}

func NewEmailHandler(id, email string) *EmailHandler {
	return &EmailHandler{id: id, email: email}
}

func (e *EmailHandler) Handle(event Event) {
	fmt.Printf("  📧 [邮件通知-%s] 发送邮件到 %s\n", e.id, e.email)
	fmt.Printf("     事件: %s, 数据: %v\n", event.Type, event.Data)
	time.Sleep(100 * time.Millisecond) // 模拟发送邮件
}

func (e *EmailHandler) GetID() string {
	return e.id
}

// 短信通知处理器
type SMSHandler struct {
	id    string
	phone string
}

func NewSMSHandler(id, phone string) *SMSHandler {
	return &SMSHandler{id: id, phone: phone}
}

func (s *SMSHandler) Handle(event Event) {
	fmt.Printf("  📱 [短信通知-%s] 发送短信到 %s\n", s.id, s.phone)
	fmt.Printf("     事件: %s, 数据: %v\n", event.Type, event.Data)
	time.Sleep(100 * time.Millisecond) // 模拟发送短信
}

func (s *SMSHandler) GetID() string {
	return s.id
}

// 日志处理器
type LogHandler struct {
	id string
}

func NewLogHandler(id string) *LogHandler {
	return &LogHandler{id: id}
}

func (l *LogHandler) Handle(event Event) {
	fmt.Printf("  📝 [日志-%s] 记录事件日志\n", l.id)
	fmt.Printf("     [%s] %s: %v\n", 
		event.Timestamp.Format("2006-01-02 15:04:05"), 
		event.Type, 
		event.Data)
}

func (l *LogHandler) GetID() string {
	return l.id
}

// 数据分析处理器
type AnalyticsHandler struct {
	id string
}

func NewAnalyticsHandler(id string) *AnalyticsHandler {
	return &AnalyticsHandler{id: id}
}

func (a *AnalyticsHandler) Handle(event Event) {
	fmt.Printf("  📊 [数据分析-%s] 收集分析数据\n", a.id)
	fmt.Printf("     事件类型: %s, 时间戳: %s\n", 
		event.Type, 
		event.Timestamp.Format("15:04:05"))
}

func (a *AnalyticsHandler) GetID() string {
	return a.id
}

// 推送通知处理器
type PushHandler struct {
	id     string
	device string
}

func NewPushHandler(id, device string) *PushHandler {
	return &PushHandler{id: id, device: device}
}

func (p *PushHandler) Handle(event Event) {
	fmt.Printf("  📲 [推送通知-%s] 推送到设备 %s\n", p.id, p.device)
	fmt.Printf("     事件: %s, 数据: %v\n", event.Type, event.Data)
	time.Sleep(100 * time.Millisecond) // 模拟推送
}

func (p *PushHandler) GetID() string {
	return p.id
}

func main() {
	fmt.Println("=== 事件总线模式示例 ===\n")
	
	// 创建事件总线
	bus := NewEventBus()
	
	// 创建各种事件处理器
	emailHandler := NewEmailHandler("email-001", "user@example.com")
	smsHandler := NewSMSHandler("sms-001", "138****8888")
	logHandler := NewLogHandler("log-001")
	analyticsHandler := NewAnalyticsHandler("analytics-001")
	pushHandler := NewPushHandler("push-001", "iPhone-12")
	
	// 场景 1: 用户注册事件
	fmt.Println("【场景 1: 用户注册事件】\n")
	
	// 订阅用户注册事件
	bus.Subscribe(UserRegistered, emailHandler)
	bus.Subscribe(UserRegistered, smsHandler)
	bus.Subscribe(UserRegistered, logHandler)
	bus.Subscribe(UserRegistered, analyticsHandler)
	
	// 发布用户注册事件
	fmt.Println()
	bus.Publish(Event{
		Type: UserRegistered,
		Data: map[string]interface{}{
			"user_id":  "12345",
			"username": "alice",
			"email":    "alice@example.com",
		},
		Timestamp: time.Now(),
	})
	
	// 场景 2: 订单创建事件
	fmt.Println("\n\n【场景 2: 订单创建事件】\n")
	
	// 订阅订单创建事件
	bus.Subscribe(OrderCreated, emailHandler)
	bus.Subscribe(OrderCreated, pushHandler)
	bus.Subscribe(OrderCreated, logHandler)
	
	// 发布订单创建事件
	fmt.Println()
	bus.Publish(Event{
		Type: OrderCreated,
		Data: map[string]interface{}{
			"order_id": "ORD-001",
			"user_id":  "12345",
			"amount":   299.99,
			"items":    []string{"商品A", "商品B"},
		},
		Timestamp: time.Now(),
	})
	
	// 场景 3: 订单支付事件
	fmt.Println("\n\n【场景 3: 订单支付事件】\n")
	
	// 订阅订单支付事件
	bus.Subscribe(OrderPaid, emailHandler)
	bus.Subscribe(OrderPaid, smsHandler)
	bus.Subscribe(OrderPaid, pushHandler)
	bus.Subscribe(OrderPaid, logHandler)
	bus.Subscribe(OrderPaid, analyticsHandler)
	
	// 发布订单支付事件
	fmt.Println()
	bus.Publish(Event{
		Type: OrderPaid,
		Data: map[string]interface{}{
			"order_id":     "ORD-001",
			"payment_id":   "PAY-001",
			"amount":       299.99,
			"payment_type": "alipay",
		},
		Timestamp: time.Now(),
	})
	
	// 场景 4: 取消订阅
	fmt.Println("\n\n【场景 4: 取消订阅】\n")
	
	// 取消短信通知订阅
	bus.Unsubscribe(OrderPaid, smsHandler)
	
	// 再次发布订单支付事件
	fmt.Println()
	bus.Publish(Event{
		Type: OrderPaid,
		Data: map[string]interface{}{
			"order_id":     "ORD-002",
			"payment_id":   "PAY-002",
			"amount":       199.99,
			"payment_type": "wechat",
		},
		Timestamp: time.Now(),
	})
	
	// 场景 5: 异步事件发布
	fmt.Println("\n\n【场景 5: 异步事件发布】\n")
	
	// 订阅订单发货事件
	bus.Subscribe(OrderShipped, emailHandler)
	bus.Subscribe(OrderShipped, smsHandler)
	bus.Subscribe(OrderShipped, pushHandler)
	
	fmt.Println("异步发布多个事件...\n")
	
	// 异步发布多个事件
	for i := 1; i <= 3; i++ {
		bus.PublishAsync(Event{
			Type: OrderShipped,
			Data: map[string]interface{}{
				"order_id":       fmt.Sprintf("ORD-%03d", i),
				"tracking_number": fmt.Sprintf("TRACK-%d", 1000+i),
				"courier":        "顺丰快递",
			},
			Timestamp: time.Now(),
		})
	}
	
	// 等待异步事件处理完成
	time.Sleep(2 * time.Second)
	
	// 场景 6: 用户登录事件（演示多个事件类型）
	fmt.Println("\n\n【场景 6: 用户登录事件】\n")
	
	// 订阅用户登录事件
	bus.Subscribe(UserLoggedIn, logHandler)
	bus.Subscribe(UserLoggedIn, analyticsHandler)
	
	// 发布用户登录事件
	fmt.Println()
	bus.Publish(Event{
		Type: UserLoggedIn,
		Data: map[string]interface{}{
			"user_id":    "12345",
			"ip_address": "192.168.1.100",
			"device":     "Chrome/Windows",
		},
		Timestamp: time.Now(),
	})
	
	fmt.Println("\n=== 示例结束 ===")
	fmt.Println("\n💡 事件总线的优势:")
	fmt.Println("- 松耦合: 发布者和订阅者互不依赖")
	fmt.Println("- 可扩展: 轻松添加新的事件处理器")
	fmt.Println("- 灵活性: 支持同步和异步事件处理")
	fmt.Println("- 可维护: 事件处理逻辑集中管理")
}
