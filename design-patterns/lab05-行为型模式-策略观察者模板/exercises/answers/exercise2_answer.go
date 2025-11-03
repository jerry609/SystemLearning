package main

import (
	"fmt"
	"sync"
)

// 练习 2: 股票价格监控系统 - 参考答案
//
// 设计思路:
// 1. 定义 Subject 和 Observer 接口
// 2. Stock 作为具体主题，维护观察者列表
// 3. 实现多种观察者（邮件、短信、App推送、日志）
// 4. 支持条件通知和阈值设置
//
// 使用的设计模式: 观察者模式
// 模式应用位置: Stock (Subject) 和各种 Observer 实现

// 观察者接口
type Observer interface {
	Update(stock *Stock)
	GetID() string
}

// 股票（主题）
type Stock struct {
	symbol        string
	name          string
	currentPrice  float64
	previousPrice float64
	observers     []Observer
	mu            sync.RWMutex
}

func NewStock(symbol, name string, initialPrice float64) *Stock {
	return &Stock{
		symbol:        symbol,
		name:          name,
		currentPrice:  initialPrice,
		previousPrice: initialPrice,
		observers:     make([]Observer, 0),
	}
}

func (s *Stock) Attach(observer Observer) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.observers = append(s.observers, observer)
	fmt.Printf("✓ [%s] 已订阅股票: %s\n", observer.GetID(), s.symbol)
}

func (s *Stock) Detach(observer Observer) {
	s.mu.Lock()
	defer s.mu.Unlock()
	for i, obs := range s.observers {
		if obs.GetID() == observer.GetID() {
			s.observers = append(s.observers[:i], s.observers[i+1:]...)
			fmt.Printf("✗ [%s] 已取消订阅股票: %s\n", observer.GetID(), s.symbol)
			break
		}
	}
}

func (s *Stock) Notify() {
	s.mu.RLock()
	observers := make([]Observer, len(s.observers))
	copy(observers, s.observers)
	s.mu.RUnlock()
	
	for _, observer := range observers {
		observer.Update(s)
	}
}

func (s *Stock) SetPrice(newPrice float64) {
	s.mu.Lock()
	s.previousPrice = s.currentPrice
	s.currentPrice = newPrice
	s.mu.Unlock()
	
	change := newPrice - s.previousPrice
	changePercent := (change / s.previousPrice) * 100
	
	fmt.Printf("\n📈 股票价格变化:\n")
	fmt.Printf("  股票: %s (%s)\n", s.symbol, s.name)
	fmt.Printf("  原价格: $%.2f\n", s.previousPrice)
	fmt.Printf("  新价格: $%.2f\n", newPrice)
	fmt.Printf("  涨跌: %+.2f (%+.2f%%)\n\n", change, changePercent)
	
	s.Notify()
}

func (s *Stock) GetSymbol() string {
	return s.symbol
}

func (s *Stock) GetName() string {
	return s.name
}

func (s *Stock) GetCurrentPrice() float64 {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.currentPrice
}

func (s *Stock) GetPreviousPrice() float64 {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.previousPrice
}

func (s *Stock) GetChangePercent() float64 {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return ((s.currentPrice - s.previousPrice) / s.previousPrice) * 100
}

// 邮件通知观察者
type EmailObserver struct {
	id    string
	email string
}

func NewEmailObserver(email string) *EmailObserver {
	return &EmailObserver{
		id:    fmt.Sprintf("邮件通知-%s", email),
		email: email,
	}
}

func (e *EmailObserver) Update(stock *Stock) {
	fmt.Printf("  📧 [邮件通知] 发送到 %s\n", e.email)
	fmt.Printf("     %s 价格变化: $%.2f -> $%.2f (%+.2f%%)\n",
		stock.GetSymbol(),
		stock.GetPreviousPrice(),
		stock.GetCurrentPrice(),
		stock.GetChangePercent())
}

func (e *EmailObserver) GetID() string {
	return e.id
}

// 短信通知观察者
type SMSObserver struct {
	id    string
	phone string
}

func NewSMSObserver(phone string) *SMSObserver {
	return &SMSObserver{
		id:    fmt.Sprintf("短信通知-%s", phone),
		phone: phone,
	}
}

func (s *SMSObserver) Update(stock *Stock) {
	fmt.Printf("  📱 [短信通知] 发送到 %s\n", s.phone)
	fmt.Printf("     %s 价格变化: $%.2f -> $%.2f (%+.2f%%)\n",
		stock.GetSymbol(),
		stock.GetPreviousPrice(),
		stock.GetCurrentPrice(),
		stock.GetChangePercent())
}

func (s *SMSObserver) GetID() string {
	return s.id
}

// App 推送观察者
type AppPushObserver struct {
	id     string
	device string
}

func NewAppPushObserver(device string) *AppPushObserver {
	return &AppPushObserver{
		id:     fmt.Sprintf("App推送-%s", device),
		device: device,
	}
}

func (a *AppPushObserver) Update(stock *Stock) {
	fmt.Printf("  📲 [App推送] 推送到 %s\n", a.device)
	fmt.Printf("     %s 价格变化: $%.2f -> $%.2f (%+.2f%%)\n",
		stock.GetSymbol(),
		stock.GetPreviousPrice(),
		stock.GetCurrentPrice(),
		stock.GetChangePercent())
}

func (a *AppPushObserver) GetID() string {
	return a.id
}

// 价格警报观察者（条件通知）
type PriceAlertObserver struct {
	id        string
	email     string
	threshold float64
	basePrice float64
}

func NewPriceAlertObserver(email string, threshold float64) *PriceAlertObserver {
	return &PriceAlertObserver{
		id:        fmt.Sprintf("价格警报-%s", email),
		email:     email,
		threshold: threshold,
	}
}

func (p *PriceAlertObserver) Update(stock *Stock) {
	if p.basePrice == 0 {
		p.basePrice = stock.GetPreviousPrice()
	}
	
	changePercent := ((stock.GetCurrentPrice() - p.basePrice) / p.basePrice) * 100
	absChange := changePercent
	if absChange < 0 {
		absChange = -absChange
	}
	
	if absChange >= p.threshold {
		fmt.Printf("  🚨 [价格警报] 发送到 %s\n", p.email)
		fmt.Printf("     %s 价格变化超过阈值！\n", stock.GetSymbol())
		fmt.Printf("     变化幅度: %.2f%% (阈值: %.1f%%)\n", changePercent, p.threshold)
		p.basePrice = stock.GetCurrentPrice() // 重置基准价格
	} else {
		fmt.Printf("  ⚠️  [价格警报] 变化幅度未达到阈值 (%.2f%% < %.1f%%)\n", absChange, p.threshold)
	}
}

func (p *PriceAlertObserver) GetID() string {
	return p.id
}

func main() {
	fmt.Println("=== 股票价格监控系统 ===")
	
	// 场景 1: 基本价格监控
	fmt.Println("\n【场景 1: 基本价格监控】\n")
	stock := NewStock("AAPL", "Apple Inc.", 150.00)
	
	emailObserver := NewEmailObserver("user@example.com")
	smsObserver := NewSMSObserver("138****8888")
	appObserver := NewAppPushObserver("iPhone-12")
	
	stock.Attach(emailObserver)
	stock.Attach(smsObserver)
	stock.Attach(appObserver)
	
	stock.SetPrice(155.00)
	stock.SetPrice(160.00)
	
	// 场景 2: 取消订阅
	fmt.Println("\n【场景 2: 取消订阅】\n")
	stock.Detach(smsObserver)
	stock.SetPrice(158.00)
	
	// 场景 3: 条件通知
	fmt.Println("\n\n【场景 3: 条件通知（价格警报）】\n")
	stock2 := NewStock("GOOGL", "Google", 140.00)
	alertObserver := NewPriceAlertObserver("alert@example.com", 5.0)
	stock2.Attach(alertObserver)
	
	stock2.SetPrice(142.00) // 涨幅 1.43%，不通知
	stock2.SetPrice(148.00) // 涨幅 5.71%，通知
	
	fmt.Println("\n=== 示例结束 ===")
}
