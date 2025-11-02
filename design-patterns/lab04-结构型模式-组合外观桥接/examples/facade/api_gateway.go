package main

import (
	"fmt"
	"time"
)

// 外观模式示例：API 网关
// 本示例展示了如何使用外观模式简化微服务的访问

// 子系统 1: 用户服务
type UserService struct{}

func (u *UserService) GetUser(userID string) map[string]interface{} {
	fmt.Printf("  [UserService] 获取用户信息: %s\n", userID)
	time.Sleep(50 * time.Millisecond)
	return map[string]interface{}{
		"id":    userID,
		"name":  "张三",
		"email": "zhangsan@example.com",
	}
}

func (u *UserService) ValidateUser(userID string) bool {
	fmt.Printf("  [UserService] 验证用户: %s\n", userID)
	time.Sleep(30 * time.Millisecond)
	return true
}

// 子系统 2: 订单服务
type OrderService struct{}

func (o *OrderService) GetOrders(userID string) []map[string]interface{} {
	fmt.Printf("  [OrderService] 获取用户订单: %s\n", userID)
	time.Sleep(80 * time.Millisecond)
	return []map[string]interface{}{
		{"id": "order-001", "amount": 299.99, "status": "已完成"},
		{"id": "order-002", "amount": 599.99, "status": "配送中"},
	}
}

func (o *OrderService) CreateOrder(userID string, items []string) string {
	fmt.Printf("  [OrderService] 创建订单: 用户=%s, 商品=%v\n", userID, items)
	time.Sleep(100 * time.Millisecond)
	return "order-003"
}

// 子系统 3: 支付服务
type PaymentService struct{}

func (p *PaymentService) ProcessPayment(orderID string, amount float64) bool {
	fmt.Printf("  [PaymentService] 处理支付: 订单=%s, 金额=%.2f\n", orderID, amount)
	time.Sleep(120 * time.Millisecond)
	return true
}

func (p *PaymentService) GetPaymentHistory(userID string) []map[string]interface{} {
	fmt.Printf("  [PaymentService] 获取支付历史: %s\n", userID)
	time.Sleep(60 * time.Millisecond)
	return []map[string]interface{}{
		{"order_id": "order-001", "amount": 299.99, "time": "2024-01-15"},
		{"order_id": "order-002", "amount": 599.99, "time": "2024-01-20"},
	}
}

// 子系统 4: 库存服务
type InventoryService struct{}

func (i *InventoryService) CheckStock(items []string) bool {
	fmt.Printf("  [InventoryService] 检查库存: %v\n", items)
	time.Sleep(40 * time.Millisecond)
	return true
}

func (i *InventoryService) ReserveStock(items []string) bool {
	fmt.Printf("  [InventoryService] 预留库存: %v\n", items)
	time.Sleep(50 * time.Millisecond)
	return true
}

// 子系统 5: 通知服务
type NotificationService struct{}

func (n *NotificationService) SendEmail(userID, subject, content string) {
	fmt.Printf("  [NotificationService] 发送邮件: 用户=%s, 主题=%s\n", userID, subject)
	time.Sleep(30 * time.Millisecond)
}

func (n *NotificationService) SendSMS(userID, message string) {
	fmt.Printf("  [NotificationService] 发送短信: 用户=%s, 内容=%s\n", userID, message)
	time.Sleep(40 * time.Millisecond)
}

// 外观类：API 网关
type APIGateway struct {
	userService         *UserService
	orderService        *OrderService
	paymentService      *PaymentService
	inventoryService    *InventoryService
	notificationService *NotificationService
}

func NewAPIGateway() *APIGateway {
	return &APIGateway{
		userService:         &UserService{},
		orderService:        &OrderService{},
		paymentService:      &PaymentService{},
		inventoryService:    &InventoryService{},
		notificationService: &NotificationService{},
	}
}

// GetUserProfile 获取用户完整信息（聚合多个服务）
func (a *APIGateway) GetUserProfile(userID string) map[string]interface{} {
	fmt.Println("\n📊 API Gateway: 获取用户完整信息")
	fmt.Println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

	// 调用多个微服务
	user := a.userService.GetUser(userID)
	orders := a.orderService.GetOrders(userID)
	payments := a.paymentService.GetPaymentHistory(userID)

	// 聚合结果
	profile := map[string]interface{}{
		"user":     user,
		"orders":   orders,
		"payments": payments,
	}

	fmt.Println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
	fmt.Println("✅ 用户信息获取完成\n")

	return profile
}

// PlaceOrder 下单（协调多个服务）
func (a *APIGateway) PlaceOrder(userID string, items []string, amount float64) (string, error) {
	fmt.Println("\n🛒 API Gateway: 处理下单请求")
	fmt.Println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

	// 1. 验证用户
	if !a.userService.ValidateUser(userID) {
		fmt.Println("❌ 用户验证失败")
		return "", fmt.Errorf("用户验证失败")
	}

	// 2. 检查库存
	if !a.inventoryService.CheckStock(items) {
		fmt.Println("❌ 库存不足")
		return "", fmt.Errorf("库存不足")
	}

	// 3. 预留库存
	if !a.inventoryService.ReserveStock(items) {
		fmt.Println("❌ 库存预留失败")
		return "", fmt.Errorf("库存预留失败")
	}

	// 4. 创建订单
	orderID := a.orderService.CreateOrder(userID, items)

	// 5. 处理支付
	if !a.paymentService.ProcessPayment(orderID, amount) {
		fmt.Println("❌ 支付失败")
		return "", fmt.Errorf("支付失败")
	}

	// 6. 发送通知
	a.notificationService.SendEmail(userID, "订单确认", "您的订单已创建")
	a.notificationService.SendSMS(userID, "订单创建成功")

	fmt.Println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
	fmt.Printf("✅ 订单创建成功: %s\n\n", orderID)

	return orderID, nil
}

// CancelOrder 取消订单（协调多个服务）
func (a *APIGateway) CancelOrder(userID, orderID string) error {
	fmt.Println("\n❌ API Gateway: 处理取消订单请求")
	fmt.Println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

	// 1. 验证用户
	if !a.userService.ValidateUser(userID) {
		return fmt.Errorf("用户验证失败")
	}

	// 2. 处理退款（简化示例）
	fmt.Printf("  [PaymentService] 处理退款: 订单=%s\n", orderID)

	// 3. 释放库存（简化示例）
	fmt.Printf("  [InventoryService] 释放库存: 订单=%s\n", orderID)

	// 4. 发送通知
	a.notificationService.SendEmail(userID, "订单取消", "您的订单已取消")

	fmt.Println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
	fmt.Println("✅ 订单取消成功\n")

	return nil
}

func main() {
	fmt.Println("=== 外观模式示例：API 网关 ===")

	// 创建 API 网关
	gateway := NewAPIGateway()

	// 场景 1: 获取用户完整信息
	// 客户端只需调用一个方法，网关会协调多个微服务
	userID := "user-123"
	profile := gateway.GetUserProfile(userID)
	fmt.Printf("用户资料: %v\n", profile)

	// 场景 2: 下单
	// 客户端不需要了解下单涉及的复杂流程
	items := []string{"商品A", "商品B"}
	amount := 899.99
	orderID, err := gateway.PlaceOrder(userID, items, amount)
	if err != nil {
		fmt.Printf("下单失败: %v\n", err)
	} else {
		fmt.Printf("订单号: %s\n", orderID)
	}

	// 场景 3: 取消订单
	err = gateway.CancelOrder(userID, orderID)
	if err != nil {
		fmt.Printf("取消订单失败: %v\n", err)
	}

	fmt.Println("=== 示例结束 ===")

	// 说明外观模式的优势
	fmt.Println("\n💡 外观模式的优势")
	fmt.Println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
	fmt.Println("1. 简化客户端调用")
	fmt.Println("   - 客户端只需调用网关的高层接口")
	fmt.Println("   - 不需要了解各个微服务的细节")
	fmt.Println()
	fmt.Println("2. 降低耦合度")
	fmt.Println("   - 客户端与微服务解耦")
	fmt.Println("   - 微服务的变化不影响客户端")
	fmt.Println()
	fmt.Println("3. 统一入口")
	fmt.Println("   - 统一处理认证、授权、限流等")
	fmt.Println("   - 统一的错误处理和日志记录")
	fmt.Println()
	fmt.Println("4. 业务编排")
	fmt.Println("   - 协调多个微服务完成复杂业务")
	fmt.Println("   - 处理服务间的依赖关系")
	fmt.Println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
}

// 输出示例：
// === 外观模式示例：API 网关 ===
//
// 📊 API Gateway: 获取用户完整信息
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
//   [UserService] 获取用户信息: user-123
//   [OrderService] 获取用户订单: user-123
//   [PaymentService] 获取支付历史: user-123
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// ✅ 用户信息获取完成
//
// 用户资料: map[orders:[map[amount:299.99 id:order-001 status:已完成] map[amount:599.99 id:order-002 status:配送中]] payments:[map[amount:299.99 order_id:order-001 time:2024-01-15] map[amount:599.99 order_id:order-002 time:2024-01-20]] user:map[email:zhangsan@example.com id:user-123 name:张三]]
//
// 🛒 API Gateway: 处理下单请求
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
//   [UserService] 验证用户: user-123
//   [InventoryService] 检查库存: [商品A 商品B]
//   [InventoryService] 预留库存: [商品A 商品B]
//   [OrderService] 创建订单: 用户=user-123, 商品=[商品A 商品B]
//   [PaymentService] 处理支付: 订单=order-003, 金额=899.99
//   [NotificationService] 发送邮件: 用户=user-123, 主题=订单确认
//   [NotificationService] 发送短信: 用户=user-123, 内容=订单创建成功
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// ✅ 订单创建成功: order-003
//
// 订单号: order-003
//
// ❌ API Gateway: 处理取消订单请求
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
//   [UserService] 验证用户: user-123
//   [PaymentService] 处理退款: 订单=order-003
//   [InventoryService] 释放库存: 订单=order-003
//   [NotificationService] 发送邮件: 用户=user-123, 主题=订单取消
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// ✅ 订单取消成功
//
// === 示例结束 ===
