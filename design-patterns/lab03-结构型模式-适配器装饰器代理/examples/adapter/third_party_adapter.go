package main

import "fmt"

// 适配器模式示例：第三方库适配
// 本示例展示如何适配不同的第三方支付库到统一的接口

// Target - 统一的支付接口
type PaymentProcessor interface {
	ProcessPayment(amount float64, recipient string) error
	GetProviderName() string
}

// Adaptee 1 - 支付宝 API（第三方库）
type AlipayAPI struct{}

func (a *AlipayAPI) SendPayment(amount float64, account string) error {
	fmt.Printf("[Alipay] Sending %.2f CNY to account: %s\n", amount, account)
	return nil
}

func (a *AlipayAPI) GetBalance(account string) float64 {
	return 1000.0
}

// Adaptee 2 - 微信支付 API（第三方库）
type WeChatPayAPI struct{}

func (w *WeChatPayAPI) Pay(money float64, receiver string) error {
	fmt.Printf("[WeChat Pay] Paying %.2f CNY to: %s\n", money, receiver)
	return nil
}

func (w *WeChatPayAPI) QueryBalance(user string) float64 {
	return 2000.0
}

// Adaptee 3 - PayPal API（第三方库）
type PayPalAPI struct{}

func (p *PayPalAPI) MakePayment(sum float64, recipient string) error {
	fmt.Printf("[PayPal] Making payment of $%.2f to: %s\n", sum, recipient)
	return nil
}

func (p *PayPalAPI) CheckBalance(email string) float64 {
	return 500.0
}

// Adapter 1 - 支付宝适配器
type AlipayAdapter struct {
	alipay *AlipayAPI
}

func NewAlipayAdapter() *AlipayAdapter {
	return &AlipayAdapter{alipay: &AlipayAPI{}}
}

func (a *AlipayAdapter) ProcessPayment(amount float64, recipient string) error {
	return a.alipay.SendPayment(amount, recipient)
}

func (a *AlipayAdapter) GetProviderName() string {
	return "Alipay"
}

// Adapter 2 - 微信支付适配器
type WeChatPayAdapter struct {
	wechat *WeChatPayAPI
}

func NewWeChatPayAdapter() *WeChatPayAdapter {
	return &WeChatPayAdapter{wechat: &WeChatPayAPI{}}
}

func (w *WeChatPayAdapter) ProcessPayment(amount float64, recipient string) error {
	return w.wechat.Pay(amount, recipient)
}

func (w *WeChatPayAdapter) GetProviderName() string {
	return "WeChat Pay"
}

// Adapter 3 - PayPal 适配器
type PayPalAdapter struct {
	paypal *PayPalAPI
}

func NewPayPalAdapter() *PayPalAdapter {
	return &PayPalAdapter{paypal: &PayPalAPI{}}
}

func (p *PayPalAdapter) ProcessPayment(amount float64, recipient string) error {
	return p.paypal.MakePayment(amount, recipient)
}

func (p *PayPalAdapter) GetProviderName() string {
	return "PayPal"
}

// PaymentService - 支付服务（客户端）
type PaymentService struct {
	processor PaymentProcessor
}

func NewPaymentService(processor PaymentProcessor) *PaymentService {
	return &PaymentService{processor: processor}
}

func (s *PaymentService) Pay(amount float64, recipient string) error {
	fmt.Printf("Using %s for payment...\n", s.processor.GetProviderName())
	return s.processor.ProcessPayment(amount, recipient)
}

// 支付工厂（可选）
func CreatePaymentProcessor(provider string) PaymentProcessor {
	switch provider {
	case "alipay":
		return NewAlipayAdapter()
	case "wechat":
		return NewWeChatPayAdapter()
	case "paypal":
		return NewPayPalAdapter()
	default:
		return nil
	}
}

func main() {
	fmt.Println("=== 第三方库适配器示例 ===\n")

	// 方式 1: 直接创建适配器
	fmt.Println("方式 1: 直接创建适配器")
	fmt.Println("---")

	alipayService := NewPaymentService(NewAlipayAdapter())
	alipayService.Pay(100.0, "user@example.com")

	fmt.Println()

	wechatService := NewPaymentService(NewWeChatPayAdapter())
	wechatService.Pay(200.0, "user@example.com")

	fmt.Println()

	paypalService := NewPaymentService(NewPayPalAdapter())
	paypalService.Pay(50.0, "user@example.com")

	// 方式 2: 使用工厂创建
	fmt.Println("\n方式 2: 使用工厂创建")
	fmt.Println("---")

	providers := []string{"alipay", "wechat", "paypal"}
	amounts := []float64{150.0, 250.0, 75.0}

	for i, provider := range providers {
		processor := CreatePaymentProcessor(provider)
		if processor != nil {
			service := NewPaymentService(processor)
			service.Pay(amounts[i], "customer@example.com")
			fmt.Println()
		}
	}

	// 演示统一接口的好处
	fmt.Println("演示统一接口的好处:")
	fmt.Println("---")

	// 可以将不同的支付处理器放在同一个切片中
	processors := []PaymentProcessor{
		NewAlipayAdapter(),
		NewWeChatPayAdapter(),
		NewPayPalAdapter(),
	}

	// 统一处理
	for i, processor := range processors {
		fmt.Printf("Payment %d:\n", i+1)
		service := NewPaymentService(processor)
		service.Pay(100.0, "merchant@example.com")
		fmt.Println()
	}

	fmt.Println("=== 示例结束 ===")
}
