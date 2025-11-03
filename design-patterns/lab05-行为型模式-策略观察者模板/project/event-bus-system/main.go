package main

import (
	"fmt"
	"time"
)

// 用户服务处理器
type UserServiceHandler struct {
	id string
}

func NewUserServiceHandler() *UserServiceHandler {
	return &UserServiceHandler{id: "用户服务"}
}

func (u *UserServiceHandler) Handle(event Event) {
	fmt.Printf("  [%s] 处理事件: %s\n", u.id, event.Type)
	
	if data, ok := event.Data.(map[string]interface{}); ok {
		if username, exists := data["username"]; exists {
			fmt.Printf("     创建用户资料: %v\n", username)
		}
	}
}

func (u *UserServiceHandler) GetID() string {
	return u.id
}

// 邮件服务处理器
type EmailServiceHandler struct {
	id string
}

func NewEmailServiceHandler() *EmailServiceHandler {
	return &EmailServiceHandler{id: "邮件服务"}
}

func (e *EmailServiceHandler) Handle(event Event) {
	fmt.Printf("  [%s] 处理事件: %s\n", e.id, event.Type)
	
	if data, ok := event.Data.(map[string]interface{}); ok {
		if email, exists := data["email"]; exists {
			fmt.Printf("     发送欢迎邮件到: %v\n", email)
		}
	}
}

func (e *EmailServiceHandler) GetID() string {
	return e.id
}

// 日志服务处理器
type LogServiceHandler struct {
	id string
}

func NewLogServiceHandler() *LogServiceHandler {
	return &LogServiceHandler{id: "日志服务"}
}

func (l *LogServiceHandler) Handle(event Event) {
	fmt.Printf("  [%s] 处理事件: %s\n", l.id, event.Type)
	
	if data, ok := event.Data.(map[string]interface{}); ok {
		if username, exists := data["username"]; exists {
			fmt.Printf("     记录日志: 用户 %v 已创建\n", username)
		}
	}
}

func (l *LogServiceHandler) GetID() string {
	return l.id
}

// 订单服务处理器
type OrderServiceHandler struct {
	id string
}

func NewOrderServiceHandler() *OrderServiceHandler {
	return &OrderServiceHandler{id: "订单服务"}
}

func (o *OrderServiceHandler) Handle(event Event) {
	fmt.Printf("  [%s] 处理事件: %s\n", o.id, event.Type)
	
	if data, ok := event.Data.(map[string]interface{}); ok {
		if orderID, exists := data["order_id"]; exists {
			fmt.Printf("     处理订单: %v\n", orderID)
		}
	}
}

func (o *OrderServiceHandler) GetID() string {
	return o.id
}

// 通知服务处理器
type NotificationServiceHandler struct {
	id string
}

func NewNotificationServiceHandler() *NotificationServiceHandler {
	return &NotificationServiceHandler{id: "通知服务"}
}

func (n *NotificationServiceHandler) Handle(event Event) {
	fmt.Printf("  [%s] 处理事件: %s\n", n.id, event.Type)
	
	if data, ok := event.Data.(map[string]interface{}); ok {
		if orderID, exists := data["order_id"]; exists {
			fmt.Printf("     发送订单通知: %v\n", orderID)
		}
	}
}

func (n *NotificationServiceHandler) GetID() string {
	return n.id
}

// VIP用户服务处理器
type VIPUserServiceHandler struct {
	id string
}

func NewVIPUserServiceHandler() *VIPUserServiceHandler {
	return &VIPUserServiceHandler{id: "VIP用户服务"}
}

func (v *VIPUserServiceHandler) Handle(event Event) {
	fmt.Printf("  [%s] 处理事件: %s\n", v.id, event.Type)
	
	if data, ok := event.Data.(map[string]interface{}); ok {
		if username, exists := data["username"]; exists {
			fmt.Printf("     为VIP用户 %v 提供特殊服务\n", username)
		}
	}
}

func (v *VIPUserServiceHandler) GetID() string {
	return v.id
}

func main() {
	fmt.Println("=== 事件总线系统示例 ===")
	
	// 创建事件总线
	bus := NewEventBus()
	defer bus.Stop()
	
	// 场景 1: 基本事件发布订阅
	fmt.Println("\n【场景 1: 基本事件发布订阅】\n")
	
	userService := NewUserServiceHandler()
	emailService := NewEmailServiceHandler()
	logService := NewLogServiceHandler()
	
	bus.Subscribe("user.created", userService)
	bus.Subscribe("user.created", emailService)
	bus.Subscribe("user.created", logService)
	
	bus.Publish(Event{
		Type: "user.created",
		Data: map[string]interface{}{
			"user_id":  "12345",
			"username": "alice",
			"email":    "alice@example.com",
		},
		Timestamp: time.Now(),
	})
	
	time.Sleep(500 * time.Millisecond)
	
	// 场景 2: 异步事件发布
	fmt.Println("\n\n【场景 2: 异步事件发布】\n")
	
	orderService := NewOrderServiceHandler()
	notificationService := NewNotificationServiceHandler()
	
	bus.Subscribe("order.created", orderService)
	bus.Subscribe("order.created", notificationService)
	
	fmt.Println("异步发布 3 个订单事件...\n")
	
	for i := 1; i <= 3; i++ {
		bus.PublishAsync(Event{
			Type: "order.created",
			Data: map[string]interface{}{
				"order_id": fmt.Sprintf("ORD-%03d", i),
				"amount":   99.99,
			},
			Timestamp: time.Now(),
		})
	}
	
	time.Sleep(1 * time.Second)
	
	// 场景 3: 事件过滤
	fmt.Println("\n\n【场景 3: 事件过滤】\n")
	
	vipService := NewVIPUserServiceHandler()
	
	// 只处理 VIP 用户的事件
	vipFilter := func(event Event) bool {
		if data, ok := event.Data.(map[string]interface{}); ok {
			if vip, exists := data["vip"]; exists {
				if vipBool, ok := vip.(bool); ok {
					return vipBool
				}
			}
		}
		fmt.Println("  [VIP用户服务] 事件被过滤: 非VIP用户")
		return false
	}
	
	bus.SubscribeWithFilter("user.created", vipService, vipFilter)
	
	// 发布非 VIP 用户事件
	bus.Publish(Event{
		Type: "user.created",
		Data: map[string]interface{}{
			"user_id":  "12346",
			"username": "bob",
			"email":    "bob@example.com",
			"vip":      false,
		},
		Timestamp: time.Now(),
	})
	
	time.Sleep(300 * time.Millisecond)
	
	// 发布 VIP 用户事件
	bus.Publish(Event{
		Type: "user.created",
		Data: map[string]interface{}{
			"user_id":  "12347",
			"username": "charlie",
			"email":    "charlie@example.com",
			"vip":      true,
		},
		Timestamp: time.Now(),
	})
	
	time.Sleep(300 * time.Millisecond)
	
	// 场景 4: 取消订阅
	fmt.Println("\n\n【场景 4: 取消订阅】\n")
	
	bus.Unsubscribe("user.created", emailService)
	
	bus.Publish(Event{
		Type: "user.created",
		Data: map[string]interface{}{
			"user_id":  "12348",
			"username": "david",
			"email":    "david@example.com",
		},
		Timestamp: time.Now(),
	})
	
	time.Sleep(300 * time.Millisecond)
	
	// 场景 5: 查看事件历史
	fmt.Println("\n\n【场景 5: 事件历史】\n")
	
	history := bus.GetHistory()
	fmt.Printf("事件历史记录 (共 %d 条):\n", len(history))
	for i, event := range history {
		fmt.Printf("  %d. %s - %s\n", i+1, event.Type, event.Timestamp.Format("15:04:05"))
	}
	
	// 场景 6: 订阅统计
	fmt.Println("\n\n【场景 6: 订阅统计】\n")
	
	fmt.Println("各事件类型的订阅者数量:")
	eventTypes := []string{"user.created", "order.created"}
	for _, eventType := range eventTypes {
		count := bus.GetSubscriberCount(eventType)
		fmt.Printf("  %s: %d 个订阅者\n", eventType, count)
	}
	
	fmt.Println("\n=== 示例结束 ===")
	fmt.Println("\n💡 事件总线的优势:")
	fmt.Println("- 松耦合: 发布者和订阅者互不依赖")
	fmt.Println("- 可扩展: 轻松添加新的事件处理器")
	fmt.Println("- 灵活性: 支持同步和异步事件处理")
	fmt.Println("- 过滤器: 支持事件过滤，只处理感兴趣的事件")
}
