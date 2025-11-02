package main

import (
	"fmt"
	"time"
)

// 子系统 1: 用户服务
type UserService struct{}

func (u *UserService) GetUser(userID string) map[string]interface{} {
	fmt.Printf("  [UserService] 获取用户信息: %s\n", userID)
	time.Sleep(30 * time.Millisecond)
	return map[string]interface{}{
		"id":    userID,
		"name":  "张三",
		"email": "zhangsan@example.com",
		"phone": "13800138000",
	}
}

func (u *UserService) ValidateUser(userID string) bool {
	fmt.Printf("  [UserService] 验证用户: %s\n", userID)
	time.Sleep(20 * time.Millisecond)
	return true
}

func (u *UserService) GetUserPreferences(userID string) map[string]interface{} {
	fmt.Printf("  [UserService] 获取用户偏好: %s\n", userID)
	time.Sleep(20 * time.Millisecond)
	return map[string]interface{}{
		"language":     "zh-CN",
		"notification": true,
	}
}

// 子系统 2: 订单服务
type OrderService struct{}

func (o *OrderService) GetOrders(userID string) []map[string]interface{} {
	fmt.Printf("  [OrderService] 获取用户订单: %s\n", userID)
	time.Sleep(50 * time.Millisecond)
	return []map[string]interface{}{
		{"id": "order-001", "amount": 299.99, "status": "已完成"},
		{"id": "order-002", "amount": 599.99, "status": "配送中"},
	}
}

func (o *OrderService) CreateOrder(userID string, items []string) string {
	fmt.Printf("  [OrderService] 创建订单: 用户=%s, 商品=%v\n", userID, items)
	time.Sleep(60 * time.Millisecond)
	return fmt.Sprintf("order-%d", time.Now().Unix()%1000)
}

func (o *OrderService) GetOrderDetails(orderID string) map[string]interface{} {
	fmt.Printf("  [OrderService] 获取订单详情: %s\n", orderID)
	time.Sleep(40 * time.Millisecond)
	return map[string]interface{}{
		"id":     orderID,
		"status": "已完成",
		"amount": 299.99,
	}
}

func (o *OrderService) CancelOrder(orderID string) error {
	fmt.Printf("  [OrderService] 取消订单: %s\n", orderID)
	time.Sleep(40 * time.Millisecond)
	return nil
}

// 子系统 3: 支付服务
type PaymentService struct{}

func (p *PaymentService) ProcessPayment(orderID string, amount float64) bool {
	fmt.Printf("  [PaymentService] 处理支付: 订单=%s, 金额=%.2f\n", orderID, amount)
	time.Sleep(80 * time.Millisecond)
	return true
}

func (p *PaymentService) GetPaymentHistory(userID string) []map[string]interface{} {
	fmt.Printf("  [PaymentService] 获取支付历史: %s\n", userID)
	time.Sleep(40 * time.Millisecond)
	return []map[string]interface{}{
		{"order_id": "order-001", "amount": 299.99, "time": "2024-01-15"},
		{"order_id": "order-002", "amount": 599.99, "time": "2024-01-20"},
	}
}

func (p *PaymentService) Refund(orderID string, amount float64) error {
	fmt.Printf("  [PaymentService] 处理退款: 订单=%s, 金额=%.2f\n", orderID, amount)
	time.Sleep(70 * time.Millisecond)
	return nil
}

// 子系统 4: 库存服务
type InventoryService struct{}

func (i *InventoryService) CheckStock(items []string) bool {
	fmt.Printf("  [InventoryService] 检查库存: %v\n", items)
	time.Sleep(30 * time.Millisecond)
	return true
}

func (i *InventoryService) ReserveStock(items []string) bool {
	fmt.Printf("  [InventoryService] 预留库存: %v\n", items)
	time.Sleep(40 * time.Millisecond)
	return true
}

func (i *InventoryService) ReleaseStock(orderID string) error {
	fmt.Printf("  [InventoryService] 释放库存: 订单=%s\n", orderID)
	time.Sleep(30 * time.Millisecond)
	return nil
}

// 子系统 5: 通知服务
type NotificationService struct{}

func (n *NotificationService) SendEmail(userID, subject, content string) {
	fmt.Printf("  [NotificationService] 发送邮件: 用户=%s, 主题=%s\n", userID, subject)
	time.Sleep(20 * time.Millisecond)
}

func (n *NotificationService) SendSMS(userID, message string) {
	fmt.Printf("  [NotificationService] 发送短信: 用户=%s, 内容=%s\n", userID, message)
	time.Sleep(25 * time.Millisecond)
}

func (n *NotificationService) SendPush(userID, message string) {
	fmt.Printf("  [NotificationService] 发送推送: 用户=%s, 内容=%s\n", userID, message)
	time.Sleep(15 * time.Millisecond)
}

// 外观类: API 网关
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
func (a *APIGateway) GetUserProfile(userID string) (map[string]interface{}, error) {
	fmt.Println("\n📊 API Gateway: 获取用户完整信息")
	fmt.Println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

	// 并行调用多个微服务（简化示例，实际应使用 goroutine）
	user := a.userService.GetUser(userID)
	orders := a.orderService.GetOrders(userID)
	payments := a.paymentService.GetPaymentHistory(userID)
	preferences := a.userService.GetUserPreferences(userID)

	// 聚合结果
	profile := map[string]interface{}{
		"user":        user,
		"orders":      orders,
		"payments":    payments,
		"preferences": preferences,
	}

	fmt.Println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
	fmt.Println("✅ 用户信息获取完成\n")

	return profile, nil
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
		// 回滚：释放库存
		a.inventoryService.ReleaseStock(orderID)
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

	// 2. 获取订单详情
	orderDetails := a.orderService.GetOrderDetails(orderID)

	// 3. 取消订单
	if err := a.orderService.CancelOrder(orderID); err != nil {
		return err
	}

	// 4. 处理退款
	amount := orderDetails["amount"].(float64)
	if err := a.paymentService.Refund(orderID, amount); err != nil {
		return err
	}

	// 5. 释放库存
	if err := a.inventoryService.ReleaseStock(orderID); err != nil {
		return err
	}

	// 6. 发送通知
	a.notificationService.SendEmail(userID, "订单取消", "您的订单已取消")

	fmt.Println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
	fmt.Println("✅ 订单取消成功\n")

	return nil
}

// GetOrderStatus 查询订单状态（简化接口）
func (a *APIGateway) GetOrderStatus(userID, orderID string) (map[string]interface{}, error) {
	fmt.Println("\n🔍 API Gateway: 查询订单状态")
	fmt.Println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

	// 验证用户
	if !a.userService.ValidateUser(userID) {
		return nil, fmt.Errorf("用户验证失败")
	}

	// 获取订单详情
	orderDetails := a.orderService.GetOrderDetails(orderID)

	fmt.Println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
	fmt.Println("✅ 订单状态查询完成\n")

	return orderDetails, nil
}
