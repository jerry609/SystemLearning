package main

import (
	"errors"
	"fmt"
	"time"
)

// 策略模式示例：支付策略
// 本示例展示了如何使用策略模式实现多种支付方式

// PaymentStrategy 支付策略接口
// 定义所有支付方式的公共接口
type PaymentStrategy interface {
	Pay(amount float64) error
	GetName() string
}

// AlipayStrategy 支付宝支付策略
type AlipayStrategy struct {
	Account string
}

func (a *AlipayStrategy) Pay(amount float64) error {
	fmt.Printf("【支付宝支付】\n")
	fmt.Printf("  账户: %s\n", a.Account)
	fmt.Printf("  金额: ¥%.2f\n", amount)
	fmt.Printf("  状态: 支付成功\n")
	fmt.Printf("  时间: %s\n", time.Now().Format("2006-01-02 15:04:05"))
	return nil
}

func (a *AlipayStrategy) GetName() string {
	return "支付宝"
}

// WechatStrategy 微信支付策略
type WechatStrategy struct {
	OpenID string
}

func (w *WechatStrategy) Pay(amount float64) error {
	fmt.Printf("【微信支付】\n")
	fmt.Printf("  OpenID: %s\n", w.OpenID)
	fmt.Printf("  金额: ¥%.2f\n", amount)
	fmt.Printf("  状态: 支付成功\n")
	fmt.Printf("  时间: %s\n", time.Now().Format("2006-01-02 15:04:05"))
	return nil
}

func (w *WechatStrategy) GetName() string {
	return "微信支付"
}

// BankCardStrategy 银行卡支付策略
type BankCardStrategy struct {
	CardNumber string
	BankName   string
}

func (b *BankCardStrategy) Pay(amount float64) error {
	// 模拟银行卡支付需要验证
	if len(b.CardNumber) < 16 {
		return errors.New("无效的银行卡号")
	}

	fmt.Printf("【银行卡支付】\n")
	fmt.Printf("  银行: %s\n", b.BankName)
	fmt.Printf("  卡号: %s\n", maskCardNumber(b.CardNumber))
	fmt.Printf("  金额: ¥%.2f\n", amount)
	fmt.Printf("  状态: 支付成功\n")
	fmt.Printf("  时间: %s\n", time.Now().Format("2006-01-02 15:04:05"))
	return nil
}

func (b *BankCardStrategy) GetName() string {
	return "银行卡"
}

// maskCardNumber 隐藏银行卡号中间部分
func maskCardNumber(cardNumber string) string {
	if len(cardNumber) < 8 {
		return cardNumber
	}
	return cardNumber[:4] + "********" + cardNumber[len(cardNumber)-4:]
}

// CryptoStrategy 加密货币支付策略
type CryptoStrategy struct {
	WalletAddress string
	CryptoType    string
}

func (c *CryptoStrategy) Pay(amount float64) error {
	fmt.Printf("【加密货币支付】\n")
	fmt.Printf("  币种: %s\n", c.CryptoType)
	fmt.Printf("  钱包地址: %s\n", c.WalletAddress)
	fmt.Printf("  金额: ¥%.2f\n", amount)
	fmt.Printf("  状态: 等待区块确认...\n")
	fmt.Printf("  时间: %s\n", time.Now().Format("2006-01-02 15:04:05"))
	return nil
}

func (c *CryptoStrategy) GetName() string {
	return c.CryptoType + "支付"
}

// PaymentContext 支付上下文
// 维护对支付策略的引用，并提供统一的支付接口
type PaymentContext struct {
	strategy PaymentStrategy
	orderID  string
}

// NewPaymentContext 创建支付上下文
func NewPaymentContext(orderID string) *PaymentContext {
	return &PaymentContext{
		orderID: orderID,
	}
}

// SetStrategy 设置支付策略
func (p *PaymentContext) SetStrategy(strategy PaymentStrategy) {
	p.strategy = strategy
}

// ExecutePayment 执行支付
func (p *PaymentContext) ExecutePayment(amount float64) error {
	if p.strategy == nil {
		return errors.New("未设置支付策略")
	}

	fmt.Printf("\n========== 订单支付 ==========\n")
	fmt.Printf("订单号: %s\n", p.orderID)
	fmt.Printf("支付方式: %s\n", p.strategy.GetName())
	fmt.Println("------------------------------")

	err := p.strategy.Pay(amount)
	if err != nil {
		fmt.Printf("支付失败: %v\n", err)
		return err
	}

	fmt.Println("==============================\n")
	return nil
}

// 使用函数类型实现策略模式的另一种方式
// PaymentFunc 支付函数类型
type PaymentFunc func(amount float64) error

// PaymentProcessor 使用函数类型的支付处理器
type PaymentProcessor struct {
	paymentFunc PaymentFunc
	methodName  string
}

// NewPaymentProcessor 创建支付处理器
func NewPaymentProcessor() *PaymentProcessor {
	return &PaymentProcessor{}
}

// SetPaymentFunc 设置支付函数
func (p *PaymentProcessor) SetPaymentFunc(fn PaymentFunc, methodName string) {
	p.paymentFunc = fn
	p.methodName = methodName
}

// Process 处理支付
func (p *PaymentProcessor) Process(amount float64) error {
	if p.paymentFunc == nil {
		return errors.New("未设置支付函数")
	}

	fmt.Printf("\n使用 %s 支付 ¥%.2f\n", p.methodName, amount)
	return p.paymentFunc(amount)
}

// 具体的支付函数
func AlipayPayment(amount float64) error {
	fmt.Printf("支付宝支付成功: ¥%.2f\n", amount)
	return nil
}

func WechatPayment(amount float64) error {
	fmt.Printf("微信支付成功: ¥%.2f\n", amount)
	return nil
}

func main() {
	fmt.Println("=== 策略模式示例：支付策略 ===\n")

	// 方式 1: 使用接口实现策略模式
	fmt.Println("【方式 1: 使用接口实现】")

	// 创建支付上下文
	payment := NewPaymentContext("ORDER-2024-001")

	// 使用支付宝支付
	alipay := &AlipayStrategy{
		Account: "user@example.com",
	}
	payment.SetStrategy(alipay)
	payment.ExecutePayment(299.99)

	// 切换到微信支付
	wechat := &WechatStrategy{
		OpenID: "oX1234567890abcdef",
	}
	payment.SetStrategy(wechat)
	payment.ExecutePayment(199.50)

	// 切换到银行卡支付
	bankCard := &BankCardStrategy{
		CardNumber: "6222021234567890123",
		BankName:   "中国工商银行",
	}
	payment.SetStrategy(bankCard)
	payment.ExecutePayment(1299.00)

	// 切换到加密货币支付
	crypto := &CryptoStrategy{
		WalletAddress: "0x742d35Cc6634C0532925a3b844Bc9e7595f0bEb",
		CryptoType:    "USDT",
	}
	payment.SetStrategy(crypto)
	payment.ExecutePayment(99.99)

	// 方式 2: 使用函数类型实现策略模式
	fmt.Println("\n【方式 2: 使用函数类型实现】")

	processor := NewPaymentProcessor()

	// 使用支付宝支付函数
	processor.SetPaymentFunc(AlipayPayment, "支付宝")
	processor.Process(88.88)

	// 使用微信支付函数
	processor.SetPaymentFunc(WechatPayment, "微信支付")
	processor.Process(66.66)

	// 使用匿名函数作为策略
	processor.SetPaymentFunc(func(amount float64) error {
		fmt.Printf("现金支付成功: ¥%.2f\n", amount)
		return nil
	}, "现金")
	processor.Process(100.00)

	fmt.Println("\n=== 示例结束 ===")
}
